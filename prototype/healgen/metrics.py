"""Metrics and statistical testing suite (Task 11).

Implements evaluation metrics:
- SP_macro and SP_micro (Supported-claim Precision)
- Claim-level and response-level hallucination rates
- Exact Match (EM) and token-level F1
- Expected Calibration Error (ECE) and Brier score
- Percentile bootstrap confidence intervals
- Paired bootstrap hypothesis tests (Koehn 2004) with Holm-Bonferroni correction
"""

from __future__ import annotations

from collections import Counter
import re
import string
from typing import Any, Callable

import numpy as np


# --- Claim-level & Response-level metrics ------------------------------------

def sp_macro(records: list[dict[str, Any]]) -> float:
    """Macro-averaged supported-claim precision across responses."""
    if not records:
        return 0.0
    precisions: list[float] = []
    for r in records:
        claims = r.get("claims", [])
        if not claims:
            precisions.append(1.0)
            continue
        n_sup = sum(1.0 for c in claims if c.get("supported"))
        precisions.append(n_sup / len(claims))
    return float(np.mean(precisions))


def sp_micro(records: list[dict[str, Any]]) -> float:
    """Micro-averaged supported-claim precision (total supported / total claims)."""
    total_claims = 0
    total_supported = 0
    for r in records:
        claims = r.get("claims", [])
        total_claims += len(claims)
        total_supported += sum(1 for c in claims if c.get("supported"))
    if total_claims == 0:
        return 0.0
    return float(total_supported / total_claims)


def hallucination_rate(records: list[dict[str, Any]]) -> float:
    """Claim-level hallucination rate: HR = 1 - SP_micro."""
    return 1.0 - sp_micro(records)


def any_hallucination_rate(records: list[dict[str, Any]]) -> float:
    """Response-level any-hallucination rate: fraction of responses with >=1 unsupported claim."""
    if not records:
        return 0.0
    has_hallucination = 0
    for r in records:
        claims = r.get("claims", [])
        if any(not c.get("supported") for c in claims):
            has_hallucination += 1
    return float(has_hallucination / len(records))


# --- Judge-free metrics (EM & F1) --------------------------------------------

def _normalize_answer(s: str) -> str:
    """Standard SQuAD text normalization (lowercase, strip punct/articles/whitespace)."""
    s = s.lower()
    # Strip punctuation
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    # Strip articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Collapse whitespace
    return " ".join(s.split()).strip()


def _f1_score(pred: str, gold: str) -> float:
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def em_f1(preds: list[str], golds: list[str]) -> dict[str, float]:
    """Compute Exact Match (EM) and token-level F1 averaged over pairs."""
    if not preds or len(preds) != len(golds):
        return {"em": 0.0, "f1": 0.0}
    ems: list[float] = []
    f1s: list[float] = []
    for p, g in zip(preds, golds):
        norm_p = _normalize_answer(p)
        norm_g = _normalize_answer(g)
        ems.append(1.0 if norm_p == norm_g else 0.0)
        f1s.append(_f1_score(p, g))
    return {
        "em": float(np.mean(ems)),
        "f1": float(np.mean(f1s)),
    }


# --- Calibration metrics (ECE & Brier) ---------------------------------------

def ece(
    confs: list[float] | np.ndarray,
    correct: list[bool | int] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-width binning in [0, 1]."""
    confs_arr = np.asarray(confs, dtype=float)
    correct_arr = np.asarray(correct, dtype=float)
    if len(confs_arr) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = len(confs_arr)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confs_arr >= lower) & (confs_arr <= upper)
        else:
            in_bin = (confs_arr >= lower) & (confs_arr < upper)

        count = np.sum(in_bin)
        if count > 0:
            bin_acc = np.mean(correct_arr[in_bin])
            bin_conf = np.mean(confs_arr[in_bin])
            ece_val += (count / n) * np.abs(bin_acc - bin_conf)

    return float(ece_val)


def brier(
    confs: list[float] | np.ndarray,
    correct: list[bool | int] | np.ndarray,
) -> float:
    """Brier score: mean squared error between confidence probabilities and labels."""
    confs_arr = np.asarray(confs, dtype=float)
    correct_arr = np.asarray(correct, dtype=float)
    if len(confs_arr) == 0:
        return 0.0
    return float(np.mean((confs_arr - correct_arr) ** 2))


# --- Statistical testing -----------------------------------------------------

def bootstrap_ci(
    values: list[float] | np.ndarray,
    n: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute percentile bootstrap confidence interval for the mean."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n, len(arr)))
    boot_means = np.mean(arr[indices], axis=1)
    lo = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def paired_bootstrap(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    metric_fn: Callable[[list[dict[str, Any]]], float],
    n: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Pairwise bootstrap hypothesis test (Koehn 2004) at the example level.

    Computes observed delta, two-sided empirical p-value, and 95% bootstrap CI.
    """
    if len(records_a) != len(records_b):
        raise ValueError("records_a and records_b must have the same length")
    m = len(records_a)
    if m == 0:
        return {"p_value": 1.0, "delta": 0.0, "ci": (0.0, 0.0)}

    orig_a = metric_fn(records_a)
    orig_b = metric_fn(records_b)
    obs_delta = orig_a - orig_b

    rng = np.random.default_rng(seed)
    deltas = np.empty(n, dtype=float)

    for i in range(n):
        idxs = rng.integers(0, m, size=m)
        sample_a = [records_a[idx] for idx in idxs]
        sample_b = [records_b[idx] for idx in idxs]
        deltas[i] = metric_fn(sample_a) - metric_fn(sample_b)

    # Two-sided empirical p-value test against null delta = 0
    # Koehn 2004: 2 * min(P(d* <= 0), P(d* >= 0))
    p_le = np.mean(deltas <= 0.0)
    p_ge = np.mean(deltas >= 0.0)
    p_value = min(1.0, 2.0 * min(p_le, p_ge))

    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))

    return {
        "p_value": float(p_value),
        "delta": float(obs_delta),
        "ci": (ci_lo, ci_hi),
    }


def holm_bonferroni(
    pvals: list[float] | dict[str, float],
) -> list[float] | dict[str, float]:
    """Holm-Bonferroni step-down correction for multiple comparisons."""
    is_dict = isinstance(pvals, dict)
    if is_dict:
        keys = list(pvals.keys())
        raw_list = [pvals[k] for k in keys]
    else:
        keys = []
        raw_list = list(pvals)

    m = len(raw_list)
    if m == 0:
        return {} if is_dict else []

    sorted_indices = sorted(range(m), key=lambda i: raw_list[i])
    adjusted = [0.0] * m

    cum_max = 0.0
    for rank, orig_idx in enumerate(sorted_indices):
        multiplier = m - rank
        adj = raw_list[orig_idx] * multiplier
        cum_max = max(cum_max, adj)
        adjusted[orig_idx] = min(1.0, cum_max)

    if is_dict:
        return {k: adjusted[i] for i, k in enumerate(keys)}
    return adjusted
