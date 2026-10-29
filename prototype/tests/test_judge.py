"""NLI judge + evidence pool tests (downloads the pinned judge on first run).

The judge is the evaluation core of the prototype: every factuality number in
the write-up flows through ``Judge.support``.  The label convention tests pin
that ``entailment`` is resolved by name via ``model.config.id2label`` -- never
by hard-coded position -- so swapping the name mapping must flip decisions.
"""

import time

import pytest

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
    """Renaming id2label so a different index carries 'entailment' flips it."""
    claim = "Paris is the capital of France."
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
