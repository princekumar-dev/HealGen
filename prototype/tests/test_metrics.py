"""Unit tests for metrics and statistical testing (Task 11)."""

import pytest
import numpy as np

from healgen.metrics import (
    sp_macro,
    sp_micro,
    hallucination_rate,
    any_hallucination_rate,
    em_f1,
    ece,
    brier,
    bootstrap_ci,
    paired_bootstrap,
    holm_bonferroni,
)


def test_sp_macro():
    """test_sp_macro: resp1 claims [1,0,1] -> 2/3; resp2 [1,1] -> 1.0; avg = 5/6 ~= 0.8333."""
    records = [
        {
            "qid": "1",
            "claims": [
                {"supported": True},
                {"supported": False},
                {"supported": True},
            ],
        },
        {
            "qid": "2",
            "claims": [
                {"supported": True},
                {"supported": True},
            ],
        },
    ]
    val = sp_macro(records)
    expected = (2.0 / 3.0 + 1.0) / 2.0
    assert pytest.approx(val, rel=1e-4) == expected


def test_sp_micro():
    """sp_micro computes total supported / total claims across all responses."""
    records = [
        {
            "qid": "1",
            "claims": [
                {"supported": True},
                {"supported": False},
                {"supported": True},
            ],
        },
        {
            "qid": "2",
            "claims": [
                {"supported": True},
                {"supported": True},
            ],
        },
    ]
    val = sp_micro(records)
    assert pytest.approx(val, rel=1e-4) == 4.0 / 5.0
    assert pytest.approx(hallucination_rate(records), rel=1e-4) == 1.0 / 5.0


def test_any_hallucination_rate():
    """any_hallucination_rate is the fraction of responses having at least one unsupported claim."""
    records = [
        {"claims": [{"supported": True}, {"supported": False}]},  # has unsupported
        {"claims": [{"supported": True}, {"supported": True}]},   # fully supported
    ]
    assert pytest.approx(any_hallucination_rate(records)) == 0.5


def test_em_f1():
    """Exact match and token-level F1 with standard normalization."""
    preds = ["Paris, France", "The capital is Rome"]
    golds = ["paris france", "Rome"]
    res = em_f1(preds, golds)
    assert "em" in res and "f1" in res
    assert res["em"] == 0.5  # First is exact match after punctuation/case strip
    assert res["f1"] > 0.5


def test_ece_known():
    """2 bins with perfect calibration yield ECE == 0.0."""
    # Bin 1: conf 0.2, 1 out of 5 positive (accuracy 0.2)
    # Bin 2: conf 0.8, 4 out of 5 positive (accuracy 0.8)
    confs = [0.2] * 5 + [0.8] * 5
    correct = [1, 0, 0, 0, 0] + [1, 1, 1, 1, 0]
    score = ece(confs, correct, n_bins=2)
    assert pytest.approx(score, abs=1e-6) == 0.0


def test_brier():
    """Brier score is mean squared error between confidences and binary outcomes."""
    confs = [0.8, 0.2]
    correct = [1, 0]
    # (0.8 - 1)^2 = 0.04; (0.2 - 0)^2 = 0.04; mean = 0.04
    assert pytest.approx(brier(confs, correct), rel=1e-4) == 0.04


def test_bootstrap_ci():
    """Bootstrap confidence interval is bounded within data range and deterministic with seed."""
    vals = [10.0, 11.0, 12.0, 10.5, 11.5]
    lo1, hi1 = bootstrap_ci(vals, n=1000, seed=42)
    lo2, hi2 = bootstrap_ci(vals, n=1000, seed=42)
    assert lo1 == lo2 and hi1 == hi2
    assert min(vals) <= lo1 <= hi1 <= max(vals)


def test_paired_bootstrap_deterministic():
    """Same seed -> identical p; swapped args -> p symmetric and delta negated."""
    recs_a = [{"qid": str(i), "claims": [{"supported": True}]} for i in range(20)]
    recs_b = [{"qid": str(i), "claims": [{"supported": i % 2 == 0}]} for i in range(20)]

    res1 = paired_bootstrap(recs_a, recs_b, sp_macro, n=500, seed=123)
    res2 = paired_bootstrap(recs_a, recs_b, sp_macro, n=500, seed=123)
    res_swapped = paired_bootstrap(recs_b, recs_a, sp_macro, n=500, seed=123)

    assert res1["p_value"] == res2["p_value"]
    assert res1["delta"] == -res_swapped["delta"]
    assert pytest.approx(res1["p_value"], abs=0.05) == res_swapped["p_value"]


def test_holm():
    """Holm-Bonferroni step-down correction on p-values."""
    pvals = [0.01, 0.04, 0.03]
    # Sorted: p1=0.01 (* 3 = 0.03), p2=0.03 (* 2 = 0.06), p3=0.04 (* 1 = 0.04 -> monotonic 0.06)
    # Original order restored: [0.03, 0.06, 0.06]
    adj = holm_bonferroni(pvals)
    assert pytest.approx(adj[0], rel=1e-4) == 0.03
    assert pytest.approx(adj[1], rel=1e-4) == 0.06
    assert pytest.approx(adj[2], rel=1e-4) == 0.06

    # Dict input format
    pdict = {"c1": 0.01, "c2": 0.04, "c3": 0.03}
    adj_dict = holm_bonferroni(pdict)
    assert set(adj_dict.keys()) == set(pdict.keys())
    assert pytest.approx(adj_dict["c1"], rel=1e-4) == 0.03
