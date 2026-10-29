"""Adaptive Evidence Sufficiency Predictor (AESP, Layer 4).

Assesses whether current retrieved evidence is sufficient to verify or support
a given claim using embedding similarity (MiniLM).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from healgen import config
from healgen.risk import get_embedder


class Sufficiency:
    """Predicts whether an evidence bundle is sufficient for a claim.

    Attributes:
        tau: Decision threshold for sufficiency (score >= tau -> sufficient).
    """

    def __init__(self, tau: float = 0.5, embedder: Any | None = None):
        self.tau = float(tau)
        self._embedder = embedder

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def score(self, claim: str, evidence: list[str]) -> float:
        """Compute maximum cosine similarity between claim and each evidence passage.

        Args:
            claim: Standalone atomic claim string.
            evidence: List of retrieved evidence passages.

        Returns:
            Max cosine similarity in [-1.0, 1.0], or 0.0 if evidence/claim is empty.
        """
        if not isinstance(claim, str):
            raise TypeError("claim must be a string")
        if not evidence:
            return 0.0

        claim_clean = claim.strip()
        if not claim_clean:
            return 0.0

        valid_evidence = [e.strip() for e in evidence if isinstance(e, str) and e.strip()]
        if not valid_evidence:
            return 0.0

        # Embed claim and evidence passages
        claim_emb = self.embedder.encode([claim_clean])[0]
        ev_embs = self.embedder.encode(valid_evidence)

        # Norms and dot products for cosine similarity
        claim_norm = np.linalg.norm(claim_emb)
        if claim_norm < 1e-12:
            return 0.0

        ev_norms = np.linalg.norm(ev_embs, axis=1)
        # Avoid division by zero
        valid_mask = ev_norms >= 1e-12
        if not np.any(valid_mask):
            return 0.0

        sims = np.dot(ev_embs, claim_emb) / (ev_norms * claim_norm)
        return float(np.max(sims))

    def is_sufficient(self, claim: str, evidence: list[str]) -> bool:
        """Return True if evidence sufficiency score meets or exceeds tau."""
        return self.score(claim, evidence) >= self.tau


def tune_tau(
    dev: Sequence[tuple[str, list[str], int | bool]],
    embedder: Any | None = None,
    output_path: Path | str | None = None,
    candidate_taus: Sequence[float] = (0.3, 0.5, 0.7),
) -> float:
    """Tune sufficiency threshold tau on dev split to maximize F1 against labels.

    Args:
        dev: List of (claim, evidence_list, label) triples.
             label: 1/True if evidence is sufficient, 0/False if insufficient.
        embedder: Optional sentence embedder instance.
        output_path: Destination for tau_sensitivity.csv. Defaults to config results dir.
        candidate_taus: Grid of candidate thresholds (default: 0.3, 0.5, 0.7).

    Returns:
        The tau value achieving the highest F1 score (deterministic tie-breaking).
    """
    if not dev:
        raise ValueError("Cannot tune tau on an empty dev set")

    suff = Sufficiency(embedder=embedder)

    # Precompute scores for all dev items
    scores: list[float] = []
    labels: list[int] = []
    for claim, evidence, label in dev:
        scores.append(suff.score(claim, evidence))
        labels.append(1 if bool(label) else 0)

    results: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_acc = -1.0
    best_tau = float(candidate_taus[0])

    for tau_val in candidate_taus:
        tau = float(tau_val)
        tp = fp = fn = tn = 0
        for s, y in zip(scores, labels, strict=True):
            pred = 1 if s >= tau else 0
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 1:
                fn += 1
            else:
                tn += 1

        total = len(dev)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / total if total > 0 else 0.0

        results.append(
            {
                "tau": tau,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "accuracy": round(accuracy, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "n": total,
            }
        )

        # Maximize F1, tie-break on accuracy then smaller distance to 0.5
        if (
            f1 > best_f1
            or (f1 == best_f1 and accuracy > best_acc)
            or (f1 == best_f1 and accuracy == best_acc and abs(tau - 0.5) < abs(best_tau - 0.5))
        ):
            best_f1 = f1
            best_acc = accuracy
            best_tau = tau

    # Mark selected tau in rows
    for r in results:
        r["selected"] = bool(r["tau"] == best_tau)

    # Write sensitivity table
    csv_file = (
        Path(output_path)
        if output_path is not None
        else config.PATHS["results"] / "tau_sensitivity.csv"
    )
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tau",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "tp",
            "fp",
            "fn",
            "tn",
            "n",
            "selected",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return best_tau
