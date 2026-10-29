"""NLI judge + frozen shared evidence pools (evaluation core, Layer 3).

Every factuality number in the write-up flows through ``Judge.support``: a
claim is supported when the cross-encoder's argmax label over the *best*
passage of the pool is ``entailment``.  Two conventions are part of the
experiment contract:

- **Label by name, never by position.**  The entailment index is resolved in
  every forward pass from ``model.config.id2label`` (case-insensitive).  A
  hard-coded ``0`` would silently invert all decisions if a checkpoint used a
  different label order.
- **Temperature scaling on the entailment logit.**  ``fit_temperature``
  fits a single scalar ``T`` on dev pairs by minimising binary NLL over
  ``log(T)`` (bounded to ``[0.05, 20]``); ``support`` applies ``T`` before the
  softmax.  The score reported is still the raw entailment probability
  (T = 1) unless a temperature has explicitly been fitted.  Best-passage
  *selection* is by tempered entailment probability in both the fit and the
  decision paths, so calibration happens on the same passage that is scored.

``build_pool`` freezes the evidence each response is judged against: the gold
knowledge first, then the union of what every system retrieved (deduped, order
stable), so all systems are scored against identical inputs.

Batching: all (claim, passage) combinations of one ``_scores`` call are run
through the cross-encoder in batches of ``BATCH_SIZE`` -- ``.support()`` keeps
its single-call contract, ``eval_contrast_pairs`` simply calls the same helper
with bigger claim lists.

Known limits (deliberate, prototype scope):

- The binary logit used for temperature fitting is
  ``entailment_logit - logsumexp(other_logits)`` of the *best* passage; no
  per-passage aggregation is attempted beyond argmax selection.
- CPU-only throughput is model-dependent: DeBERTa-v3-base's disentangled
  attention runs the pinned 2 x 100 contrast eval in minutes but is ~20x
  slower per pair than the small JUDGE2 checkpoint, so scaled sweeps
  (~10k pairs) should run ``JUDGE2`` under the same protocol.
"""

import math

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from healgen.claims import decompose

BATCH_SIZE = 16

_TEMPERATURE_BOUNDS = (0.05, 20.0)


def build_pool(
    gold_knowledge: str | None, all_system_passages: list[str]
) -> list[str]:
    """Frozen shared evidence pool: gold first, then the deduped passage union.

    Duplicates (within ``all_system_passages`` and between a passage and the
    gold knowledge) are dropped, and the gold knowledge always leads the pool
    even if a system retrieved the same text.  Order is stable: passages keep
    the order of their first occurrence.  An empty/blank gold knowledge
    contributes no leading entry.
    """
    pool: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            pool.append(text)

    if gold_knowledge:
        add(gold_knowledge)
    for passage in all_system_passages:
        if passage:
            add(passage)
    return pool


class Judge:
    """Cross-encoder NLI judge over a list of evidence passages."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.temperature = 1.0
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()

    # --- label resolution -----------------------------------------------------

    def entailment_index(self) -> int:
        """Position of 'entailment', resolved by name on every call.

        Re-reading ``config.id2label`` each time is cheap and keeps the
        mapping test honest: monkeypatching the config flips decisions
        immediately instead of hitting a cached index.
        """
        id2label = self.model.config.id2label
        if not id2label:
            raise ValueError(f"{self.model_id}: model config has no id2label")
        for key, name in id2label.items():
            if str(name).strip().lower() == "entailment":
                return int(key)
        raise ValueError(
            f"{self.model_id}: no 'entailment' label in id2label={id2label}"
        )

    # --- forward passes ---------------------------------------------------------

    def _logits(self, claims: list[str], passages: list[str]) -> torch.Tensor:
        """Tensor of logits with shape (n_claims, n_passages, n_labels).

        Every claim is paired with every passage; pairs run through the
        cross-encoder in batches of :data:`BATCH_SIZE`.  Blank/whitespace
        premises are dropped first, and the tensor is keyed on the *filtered*
        premise count -- including the degenerate no-passage case, whose shape
        stays ``(n_claims, 0, n_labels)`` so callers never see a shape that
        disagrees with this convention.
        """
        premises = [p.strip() for p in passages if p and p.strip()]
        if not premises and claims:
            return torch.empty((len(claims), 0, self.model.config.num_labels))
        if not claims:
            return torch.empty(0)
        batch: list[tuple[str, str]] = []
        for claim in claims:
            for passage in premises:
                batch.append((claim, passage))
        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(batch), BATCH_SIZE):
                chunk = batch[start : start + BATCH_SIZE]
                encoded = self.tokenizer(
                    [claim for claim, _ in chunk],
                    [passage for _, passage in chunk],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                logits = self.model(**encoded).logits
                outputs.append(logits.detach().cpu())
        stacked = torch.cat(outputs, dim=0)
        return stacked.view(len(claims), len(premises), -1)

    def _scores(
        self, claims: list[str], passages: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(entailment_probs, argmax_indices), both shaped (n_claims, n_premises).

        Premises are filtered the same way :meth:`_logits` filters them
        (blank/whitespace texts dropped), and both returned tensors are keyed
        on that *filtered* count -- the empty branch too, with every cell
        filled deterministically (prob 0.0, argmax -1), so no caller can ever
        read an uninitialised value as a score.
        """
        premises = [p.strip() for p in passages if p and p.strip()]
        if not claims or not premises:
            empty_probs = torch.zeros((len(claims), len(premises)))
            empty_argmax = torch.full(
                (len(claims), len(premises)), -1, dtype=torch.long
            )
            return empty_probs, empty_argmax
        logits = self._logits(claims, premises)
        if logits.numel() == 0:
            n_premises = len(premises)
            degenerate_probs = torch.zeros((len(claims), n_premises))
            degenerate_argmax = torch.full(
                (len(claims), n_premises), -1, dtype=torch.long
            )
            return degenerate_probs, degenerate_argmax
        tempered = logits / self.temperature
        entail = self.entailment_index()
        probs = torch.softmax(tempered, dim=-1)
        return probs[:, :, entail], probs.argmax(dim=-1)

    # --- public API ---------------------------------------------------------------

    def support(self, claim: str, evidence: list[str]) -> tuple[bool, float]:
        """(supported, entailment_score) for one claim against the evidence pool.

        The boolean follows the argmax rule on the *best* passage (highest
        entailment score) -- not a ``>= 0.5`` read-off, which would differ on
        flat distributions.  The float is the max entailment score over the
        pool.  No evidence -> ``(False, 0.0)``.
        """
        if not isinstance(claim, str):
            raise TypeError("claim must be a string")
        probability, argmax = self._scores([claim], evidence)
        if argmax.numel() == 0:
            return False, 0.0
        best = int(torch.argmax(probability[0]).item())
        score = float(probability[0][best].item())
        supported = int(argmax[0][best].item()) == self.entailment_index()
        return supported, score

    def evaluate_claims(
        self, claims: list[str], evidence: list[str]
    ) -> list[tuple[bool, float]]:
        """Batched version of ``support``: one (supported, score) per claim.

        Each claim is judged against every pool passage; the returned tuple
        mirrors ``support`` exactly (argmax rule on the best passage).  Used
        by :func:`eval_contrast_pairs` so all claims of an answer share one
        batched call.
        """
        if not claims:
            return []
        probability, argmax = self._scores(claims, evidence)
        if probability.shape[1] == 0:
            # Blank-only evidence: no premise was scored, so every claim is
            # deterministically unsupported with the neutral score 0.0.
            return [(False, 0.0) for _ in claims]
        out: list[tuple[bool, float]] = []
        entail = self.entailment_index()
        for i in range(probability.shape[0]):
            best = int(torch.argmax(probability[i]).item())
            out.append(
                (
                    int(argmax[i][best].item()) == entail,
                    float(probability[i][best].item()),
                )
            )
        return out

    # --- temperature fitting -------------------------------------------------

    def _best_logits(self, claims: list[str], passages: list[str]) -> torch.Tensor:
        """1D tensor: per claim, (entail_logit - logsumexp(others)) of the best passage.

        "Best" means the passage with the highest **tempered entailment
        probability** -- exactly how ``support``/``evaluate_claims`` pick the
        scored passage -- so a fitted temperature is calibrated on the very
        same passage that is later judged.  (Selecting by the raw entailment
        logit can disagree with the probability argmax, and fitting on a
        passage the decision path would not even pick is meaningless.)  The
        returned logit itself stays untempered: the fit needs the raw logit
        to divide by the unknown scalar ``T``.
        """
        raw = self._logits(claims, passages)
        if raw.numel() == 0:
            return torch.empty(0)
        entail = self.entailment_index()
        # Selection mirrors the decision path (tempered entailment
        # probability argmax), NOT the raw entailment logit -- the two
        # argmaxes can disagree, and fit/eval must agree on the passage.
        tempered_probs = torch.softmax(raw / self.temperature, dim=-1)
        best = tempered_probs[:, :, entail].argmax(dim=1)
        selected = raw[torch.arange(raw.shape[0]), best].numpy()
        others = np.delete(selected, entail, axis=1)
        return torch.as_tensor(selected[:, entail] - logsumexp(others, axis=1))

    def fit_temperature(self, dev_pairs: list[dict]) -> float:
        """Fit scalar T on dev pairs by binary NLL over log(T); sets ``self.temperature``.

        ``dev_pairs``: ``{"claim": str, "evidence": list[str], "gold": bool}``.
        The binary logit of a pair is the entailment logit minus the
        logsumexp of the other logits of the *best* (max-entailment) passage,
        divided by ``T``; gold is 1 for supported, 0 for not.  Minimising NLL
        over ``log(T)`` (bounded so ``T`` stays in ``[0.05, 20]``) keeps a
        single, reportable scalar.
        """
        targets: list[int] = []
        binary_logits: list[float] = []
        for pair in dev_pairs:
            gold = int(bool(pair["gold"]))
            logit = self._best_logits([pair["claim"]], pair["evidence"])
            if logit.numel() == 0:
                continue  # blank evidence/claim -> no candidate logits to calibrate
            targets.append(gold)
            binary_logits.append(float(logit.item()))
        if not targets:
            raise ValueError("no usable dev pairs for temperature fitting")

        golds = np.asarray(targets, dtype=float)
        zs = np.asarray(binary_logits, dtype=float)

        def nll(log_t: float) -> float:
            p = np.clip(expit(zs / np.exp(log_t)), 1e-7, 1.0 - 1e-7)
            loss = -(golds * np.log(p) + (1.0 - golds) * np.log1p(-p))
            return float(loss.mean())

        lower, upper = _TEMPERATURE_BOUNDS
        fit = minimize_scalar(
            nll,
            bounds=(math.log(lower), math.log(upper)),
            method="bounded",
        )
        self.temperature = float(np.exp(fit.x))
        return self.temperature


# --- contrast-pair evaluation --------------------------------------------------


def eval_contrast_pairs(judge: Judge, pairs: list[dict]) -> dict:
    """Judge faithful vs. hallucinated answers on the gold-only condition.

    ``pairs``: dicts with ``question``, ``right_answer``,
    ``hallucinated_answer``, ``knowledge`` (the judge slice from
    ``load_examples("judge")``).  For each item, both answers are decomposed
    into claims and every claim is judged against ``[knowledge]``.  A response
    is *predicted faithful* iff all its claims are supported; its score is the
    **min** entailment score across its claims (soft-AND).

    Returns ``{"accuracy", "auc", "n"}`` over the 2 x len(pairs) responses
    (``n`` counts the scored ones; zero-claim answers -- e.g. empty strings --
    are skipped with a counted note printed).  ``accuracy`` compares
    predictions with gold faithful (True for ``right_answer``, False for
    ``hallucinated_answer``); ``auc`` is the ROC AUC of the response scores
    against the same labels.
    """
    golds: list[int] = []
    scores: list[float] = []
    predicted: list[bool] = []
    skipped = 0

    for pair in pairs:
        knowledge = str(pair.get("knowledge", "")).strip()
        for key, gold in (("right_answer", True), ("hallucinated_answer", False)):
            claims = decompose(pair.get(key, "") or "")
            if not claims:
                skipped += 1
                continue
            responses = judge.evaluate_claims(claims, [knowledge])
            supported_all = all(supported for supported, _ in responses)
            min_score = min(score for _, score in responses)
            predicted.append(supported_all)
            scores.append(min_score)
            golds.append(int(gold))

    if skipped:
        print(f"eval_contrast_pairs: skipped {skipped} zero-claim responses")

    if not scores:
        raise ValueError("no responses scored: every answer decomposed to zero claims")

    accuracy = sum(
        p == bool(g) for p, g in zip(predicted, golds, strict=True)
    ) / len(predicted)
    auc = float(roc_auc_score(golds, scores))
    return {"accuracy": accuracy, "auc": auc, "n": len(scores)}
