"""Unit tests for the Adaptive Evidence Sufficiency Predictor (AESP, Layer 4)."""

from pathlib import Path
import pytest
from healgen.sufficiency import Sufficiency, tune_tau


def test_empty_evidence_score_zero():
    """Empty or blank evidence list yields score 0.0 and insufficient."""
    suff = Sufficiency(tau=0.5)
    assert suff.score("Paris is the capital of France.", []) == 0.0
    assert not suff.is_sufficient("Paris is the capital of France.", [])
    assert suff.score("Paris is the capital of France.", ["", "   "]) == 0.0


def test_tau_deterministic(tmp_path: Path):
    """Tuning tau on identical input yields the exact same tau value.

    output_path must be redirected to tmp_path: tune_tau's default writes to
    results/tau_sensitivity.csv, a committed paper-input artifact -- a fixture
    call without output_path used to clobber it on every suite run (found by
    the final whole-branch review; RED: git status showed the file modified
    after this test ran).
    """
    dev = [
        ("Paris is the capital of France.", ["Paris is the capital of France."], 1),
        ("The Moon is made of green cheese.", ["The Moon is Earth's only natural satellite."], 0),
        ("Water boils at 100 degrees Celsius.", ["Water has a boiling point of 100 °C at 1 atm."], 1),
        ("Elephants can fly.", ["Elephants are the largest existing land animals."], 0),
    ]
    out_csv = tmp_path / "tau_sensitivity.csv"
    tau1 = tune_tau(dev, output_path=out_csv, candidate_taus=(0.3, 0.5, 0.7))
    tau2 = tune_tau(dev, output_path=out_csv, candidate_taus=(0.3, 0.5, 0.7))
    assert tau1 == tau2
    assert tau1 in {0.3, 0.5, 0.7}
    assert out_csv.exists()


def test_tune_tau_writes_csv(tmp_path: Path):
    """tune_tau writes candidate tau metrics to CSV file."""
    dev = [
        ("Paris is the capital of France.", ["Paris is the capital of France."], 1),
        ("The Moon is made of green cheese.", ["The Moon is Earth's only natural satellite."], 0),
    ]
    out_csv = tmp_path / "tau_sensitivity.csv"
    chosen = tune_tau(dev, output_path=out_csv, candidate_taus=(0.3, 0.5, 0.7))
    assert out_csv.exists()
    content = out_csv.read_text(encoding="utf-8")
    assert "tau" in content
    assert "f1" in content
    assert str(chosen) in content


@pytest.mark.slow
def test_gold_evidence_sufficient():
    """Claim matched with its gold passage yields score > 0.5 and is sufficient."""
    suff = Sufficiency(tau=0.5)
    claim = "Douglas Corrigan was an American aviator born in Galveston, Texas."
    evidence = [
        "Douglas Corrigan (January 22, 1907 – December 9, 1995) was an American "
        "aviator born in Galveston, Texas."
    ]
    score = suff.score(claim, evidence)
    assert score > 0.5
    assert suff.is_sufficient(claim, evidence)


@pytest.mark.slow
def test_irrelevant_evidence_insufficient():
    """Claim matched with unrelated evidence yields low score and is insufficient."""
    suff = Sufficiency(tau=0.5)
    claim = "Douglas Corrigan was an American aviator born in Galveston, Texas."
    unrelated_evidence = [
        "Astrix, is an israeli trance music dj and producer specializing in "
        "the subgenre of Full On psychedelic trance."
    ]
    score = suff.score(claim, evidence=unrelated_evidence)
    assert score < 0.5
    assert not suff.is_sufficient(claim, evidence=unrelated_evidence)
