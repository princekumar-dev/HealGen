"""NLI judge + evidence pool tests (downloads the pinned judge on first run).

The judge is the evaluation core of the prototype: every factuality number in
the write-up flows through ``Judge.support``.  The label convention tests pin
that ``entailment`` is resolved by name via ``model.config.id2label`` -- never
by hard-coded position -- so swapping the name mapping must flip decisions.
"""

import time

import pytest
import torch

from healgen import config
from healgen import judge as judge_module
from healgen.data import load_examples
from healgen.judge import Judge, build_pool


@pytest.fixture(scope="module")
def judge():
    """One judge instance for the module (~700 MB download, first run only)."""
    return Judge(config.JUDGE_MODEL)


# --- build_pool -------------------------------------------------------------


def test_pool_contains_gold_and_union():
    """Pool = [gold] + union of passages, deduped, order stable."""
    gold = "Paris is the capital of France."
    passages = [
        "Paris is the largest city of France.",
        gold,
        "The Eiffel Tower stands in Paris.",
        "Paris is the largest city of France.",
    ]
    pool = build_pool(gold, passages)
    assert pool == [
        gold,
        "Paris is the largest city of France.",
        "The Eiffel Tower stands in Paris.",
    ]
    assert len(pool) == len(set(pool))


# --- Judge.support (uses the real pinned judge) -----------------------------


@pytest.mark.slow
def test_support_entailment(judge):
    supported, prob = judge.support(
        "Paris is the capital of France.",
        ["Paris is the capital of France."],
    )
    assert supported is True
    assert prob > 0.5, f"expected clear entailment, got {prob}"


@pytest.mark.slow
def test_contradiction_not_supported(judge):
    supported, prob = judge.support(
        "Paris is in Germany.",
        ["Paris is the capital of France."],
    )
    assert supported is False
    assert 0.0 <= prob < 1.0


def test_label_mapping_not_positional(judge, monkeypatch):
    """Renaming id2label so a different index carries 'entailment' flips it.

    Uses an NLI-path pair (claim not a substring of the evidence) because
    substring matches short-circuit the model entirely.
    """
    claim = "The capital of France is Paris."
    evidence = ["Paris is the capital of France."]
    before = judge.support(claim, evidence)[0]

    id2label = dict(judge.model.config.id2label)
    entail_index = judge.entailment_index()
    other_index = (entail_index + 1) % len(id2label)
    swapped = dict(id2label)
    swapped[entail_index] = id2label[other_index]
    swapped[other_index] = id2label[entail_index]
    monkeypatch.setattr(judge.model.config, "id2label", swapped)

    after = judge.support(claim, evidence)[0]
    assert before is True
    assert after is False


@pytest.mark.slow
def test_support_empty_evidence(judge):
    """No evidence -> cannot support, no crash."""
    supported, prob = judge.support("Paris is the capital of France.", [])
    assert supported is False
    assert prob == 0.0


@pytest.mark.slow
def test_support_blank_evidence_deterministic(judge):
    """Blank-only evidence -> (False, 0.0) every call; never garbage memory.

    Pins the empty branch of ``Judge._scores``: previously it returned a
    fully-shaped *uninitialised* tensor (``torch.empty``), so blank-knowledge
    records could read arbitrary floats as entailment scores.
    """
    outcomes = {
        judge.support("Paris is the capital of France.", ["", "   "])
        for _ in range(4)
    }
    assert outcomes == {(False, 0.0)}


@pytest.mark.slow
def test_support_minus1_argmax_never_surfaces(judge):
    """The -1 argmax filler must not equal a real label index (no 0..k-1 read)."""
    _probability, argmax = judge._scores(
        ["Paris is the capital of France."], ["", "   "]
    )
    assert argmax.shape == (1, 0)
    assert bool((argmax == -1).all())


@pytest.mark.slow
def test_support_tokenizes_passage_then_claim(judge, monkeypatch):
    """Model-card order: tokenizer(premise=evidence, hypothesis=claim).

    Spy on the tokenizer: the FIRST argument must be the evidence passage,
    the SECOND the short claim.  Regression for the Task-5 direction bug
    (claim was passed as premise, collapsing support to ~3%).
    """
    evidence = ["Paris is the capital of France."]
    claim = "The capital of France is Paris."

    calls = []
    real_tokenizer = judge.tokenizer

    class _SpyTokenizer:
        def __call__(self, first, second, **kwargs):
            calls.append((list(first), list(second)))
            return real_tokenizer(first, second, **kwargs)

    monkeypatch.setattr(judge, "tokenizer", _SpyTokenizer())
    supported, prob = judge.support(claim, evidence)

    assert len(calls) == 1
    premise_args, hypothesis_args = calls[0]
    assert premise_args == evidence, "first tokenizer arg must be the passage"
    assert hypothesis_args == [claim], "second tokenizer arg must be the claim"
    assert supported is True, "paraphrase must be entailed in the right direction"


@pytest.mark.slow
def test_support_hypothesis_longer_than_premise_not_supported(judge):
    """A claim adding a fact outside the evidence is never entailed.

    Discriminator for the direction bug (verified empirically both ways):
    the model card direction asks "does the evidence entail the claim?" ->
    the invented fact 'It was completed in 1889.' is NEUTRAL (p_ent 0.007).
    The broken claim-as-premise order asks the reverse and wrongly returns
    ENTAILED (p_ent 0.998).  Substring rule does not fire (claim is longer).
    """
    supported, prob = judge.support(
        "The Eiffel Tower is in Paris. It was completed in 1889.",
        ["The Eiffel Tower is in Paris."],
    )
    assert supported is False
    assert prob < 0.5


@pytest.mark.slow
def test_substring_claim_supported_score_one(judge):
    """Claim contained in (normalized) evidence -> supported with score 1.0.

    The NLI path on this pair returns a near-1 but NOT exactly 1.0 score, so
    the exact equality pins the lexical branch.
    """
    supported, prob = judge.support(
        "Paris is the capital of France.",
        ["Founded in antiquity, Paris is the capital of France. It is in Europe."],
    )
    assert supported is True
    assert prob == 1.0


@pytest.mark.slow
def test_substring_claim_short_circuits_nli(judge, monkeypatch):
    """Lexical match must skip the model forward entirely."""
    calls = []
    real_tokenizer = judge.tokenizer

    class _SpyTokenizer:
        def __call__(self, first, second, **kwargs):
            calls.append((first, second))
            return real_tokenizer(first, second, **kwargs)

    monkeypatch.setattr(judge, "tokenizer", _SpyTokenizer())
    supported, prob = judge.support(
        "Paris is the capital of France.",
        ["Founded in antiquity, Paris is the capital of France. It is in Europe."],
    )
    assert calls == [], "substring match must short-circuit the NLI forward"
    assert (supported, prob) == (True, 1.0)


@pytest.mark.slow
def test_swapped_entity_claim_judged_by_nli_only(judge):
    """Entity-substituted claim is not a substring -> falls through to NLI."""
    supported, prob = judge.support(
        "London is the capital of France.",
        ["Paris is the capital of France."],
    )
    assert supported is False
    assert 0.0 <= prob < 1.0


@pytest.mark.slow
def test_support_blank_entries_among_real_evidence(judge):
    """Blank strings are dropped, real premises in the same pool still judged."""
    supported, prob = judge.support(
        "Paris is the capital of France.",
        ["", "Paris is the capital of France.", "   "],
    )
    assert supported is True
    assert prob > 0.5


@pytest.mark.slow
def test_eval_blank_knowledge_deterministic(judge):
    """A blank-knowledge record scores (False, 0.0) claims -- repeat runs agree.

    This is the injected-record path from the review: one malformed row must
    not poison accuracy/AUC with nondeterministic garbage.
    """
    pairs = [
        {
            "question": "What is the capital of France?",
            "right_answer": "Paris is the capital of France.",
            "hallucinated_answer": "Paris is the capital of Germany.",
            "knowledge": "   ",
        }
    ]
    runs = [judge_module.eval_contrast_pairs(judge, pairs) for _ in range(3)]
    assert all(run == runs[0] for run in runs), "results must be deterministic"
    assert runs[0]["n"] == 2
    assert 0.0 <= runs[0]["accuracy"] <= 1.0
    assert 0.0 <= runs[0]["auc"] <= 1.0


@pytest.mark.slow
def test_best_logits_selection_matches_eval_path(judge, monkeypatch):
    """``_best_logits`` must pick the passage by tempered-prob argmax.

    Crafted logits make the raw entailment-logit argmax (passage 0) disagree
    with the probability argmax (passage 1): the old raw-logit selection
    calibrated the temperature on passage 0's logits, the decision path
    scored passage 1.  The binary logit built here from the *eval-selected*
    passage is the only acceptable fit value.
    """
    entail = judge.entailment_index()
    crafted = torch.tensor(
        [[[3.0, 10.0, 0.0], [2.9, 0.0, 0.0], [2.0, 0.0, 0.0]]]
    )

    def fake_logits(claims, passages):
        assert len(claims) == 1
        return crafted

    monkeypatch.setattr(judge, "_logits", fake_logits)

    from scipy.special import logsumexp

    probabilities = torch.softmax(crafted, dim=-1)
    eval_best = int(probabilities[0, :, entail].argmax().item())
    assert eval_best == 1, "probability argmax must pick passage 1"
    raw_entail_best = int(crafted[0, :, entail].argmax().item())
    assert raw_entail_best == 0, "raw-logit argmax picks passage 0 (the trap)"

    binary = judge._best_logits(["c"], ["p0", "p1", "p2"])
    selected_row = crafted[0, eval_best].numpy()
    expected = selected_row[entail] - logsumexp(
        selected_row[[i for i in range(3) if i != entail]]
    )
    assert float(binary.item()) == pytest.approx(float(expected), abs=1e-6)
    # The old implementation returned the passage-0 value (3 - logsumexp(10, 0)),
    # a completely different logit; pin that it is NOT that any more.
    wrong_row = crafted[0, 0].numpy()
    wrong_value = float(
        wrong_row[entail] - logsumexp(wrong_row[[i for i in range(3) if i != entail]])
    )
    assert abs(float(binary.item()) - wrong_value) > 1.0


# --- fit_temperature ---------------------------------------------------------


@pytest.mark.slow
def test_fit_temperature_sets_scalar(judge):
    """Fitting on a small dev slice stores a finite T in the bounded range."""
    pairs = [
        {
            "claim": "Paris is the capital of France.",
            "evidence": ["Paris is the capital of France."],
            "gold": True,
        },
        {
            "claim": "Paris is in Germany.",
            "evidence": ["Paris is the capital of France."],
            "gold": False,
        },
    ]
    judge.fit_temperature(pairs)
    assert 0.05 <= judge.temperature <= 20.0
    assert judge.temperature > 0


# --- eval_contrast_pairs ------------------------------------------------------


@pytest.mark.slow
def test_eval_contrast_pairs_smoke(judge):
    """Runs the real eval on a small slice; checks result shape + prints rate."""
    pairs = load_examples("judge")[:4]
    start = time.perf_counter()
    result = judge_module.eval_contrast_pairs(judge, pairs)
    elapsed = time.perf_counter() - start

    assert set(result) == {"accuracy", "auc", "n"}
    assert result["n"] == 8, "two answers per pair -> 8 scored responses"
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["auc"] <= 1.0
    pairs_per_sec = result["n"] / elapsed
    print(f"NLI pairs/sec: {pairs_per_sec:.1f} ({result['n']} in {elapsed:.2f}s)")


@pytest.mark.slow
def test_eval_contrast_pairs_skips_zero_claims(judge):
    """Empty/hallucinated-empty answers are skipped without crashing."""
    pairs = [
        {
            "question": "What is the capital of France?",
            "right_answer": "Paris",
            "hallucinated_answer": "",
            "knowledge": "The capital of France is Paris.",
        },
        {
            "question": "Who wrote Hamlet?",
            "right_answer": "",
            "hallucinated_answer": "Dan Brown",
            "knowledge": "William Shakespeare wrote the play Hamlet.",
        },
    ]
    result = judge_module.eval_contrast_pairs(judge, pairs)
    assert result["n"] == 2, "empty answers yield zero claims -> skipped"
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["auc"] <= 1.0
