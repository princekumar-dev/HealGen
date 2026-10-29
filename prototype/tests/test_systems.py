"""Unit tests for systems orchestration (Task 10).

Verifies the 4 paired systems (vanilla, rag, cove, healgen), the record
schema contract, gold-leakage prevention, and retrieval behaviors using fakes.
"""

from dataclasses import dataclass
from typing import Any
import pytest

from healgen.systems import Deps, run_system

REQUIRED_RECORD_KEYS = {
    "qid",
    "system",
    "question",
    "answer",
    "claims",
    "num_bm25_calls",
    "retrieved",
    "latency_s",
    "abstained",
}

REQUIRED_CLAIM_KEYS = {
    "id",
    "text",
    "risk",
    "sufficient",
    "score",
    "supported",
    "action",
}


class FakeGenerator:
    """Mock generator tracking prompts received."""

    def __init__(self, response: str = "Paris is the capital of France. It has the Eiffel Tower."):
        self.response = response
        self.prompts: list[str] = []
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self.call_count += 1
        return self.response


class FakeJudge:
    """Mock judge with controllable support decisions."""

    def __init__(self, default_support: bool = True, default_score: float = 0.95):
        self.default_support = default_support
        self.default_score = default_score
        self.calls: list[tuple[str, list[str]]] = []

    def support(self, claim: str, evidence: list[str]) -> tuple[bool, float]:
        self.calls.append((claim, evidence))
        return self.default_support, self.default_score


class FakeBM25:
    """Mock BM25 index tracking search queries."""

    def __init__(self, passages: list[str] | None = None):
        self.passages = passages or ["Paris is in France.", "The Eiffel Tower is in Paris."]
        self.queries: list[str] = []

    def search(self, query: str, k: int = 3) -> list[tuple[int, str, float]]:
        self.queries.append(query)
        return [(i, p, 1.0 / (i + 1)) for i, p in enumerate(self.passages[:k])]


class FakeEstimator:
    """Mock RiskEstimator."""

    def __init__(self, default_risk: float = 0.25):
        self.default_risk = default_risk

    def predict_proba(self, claim: str, question: str, bm25: Any, emb: Any = None) -> float:
        return self.default_risk


class FakeSufficiency:
    """Mock Sufficiency predictor."""

    def __init__(self, sufficient: bool = True):
        self.sufficient = sufficient

    def is_sufficient(self, claim: str, evidence: list[str]) -> bool:
        return self.sufficient

    def score(self, claim: str, evidence: list[str]) -> float:
        return 0.8 if self.sufficient else 0.2


@pytest.fixture
def sample_example() -> dict:
    return {
        "id": "ex_001",
        "question": "What is the capital of France and what famous landmark is there?",
        "right_answer": "Paris is the capital of France and has the Eiffel Tower.",
        "hallucinated_answer": "Lyon is the capital of France and has the Colosseum.",
        "knowledge": "Paris is the capital and largest city of France. The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
    }


@pytest.fixture
def fake_deps() -> Deps:
    return Deps(
        generator=FakeGenerator(),
        judge=FakeJudge(),
        bm25=FakeBM25(),
        estimator=FakeEstimator(),
        sufficiency=FakeSufficiency(),
    )


@pytest.mark.parametrize("system", ["vanilla", "rag", "cove", "healgen"])
def test_record_schema(system: str, sample_example: dict, fake_deps: Deps):
    """Every system produces records strictly matching the pinned schema."""
    rec = run_system(system, sample_example, fake_deps)

    assert set(rec.keys()) == REQUIRED_RECORD_KEYS
    assert rec["qid"] == "ex_001"
    assert rec["system"] == system
    assert rec["question"] == sample_example["question"]
    assert isinstance(rec["answer"], str)
    assert isinstance(rec["num_bm25_calls"], int)
    assert isinstance(rec["retrieved"], list)
    assert isinstance(rec["latency_s"], float)
    assert isinstance(rec["abstained"], bool)
    assert isinstance(rec["claims"], list)

    assert len(rec["claims"]) > 0
    for c in rec["claims"]:
        assert set(c.keys()) == REQUIRED_CLAIM_KEYS
        assert isinstance(c["id"], str)
        assert isinstance(c["text"], str)
        assert isinstance(c["risk"], float)
        assert isinstance(c["sufficient"], bool)
        assert isinstance(c["score"], float)
        assert isinstance(c["supported"], bool)
        assert isinstance(c["action"], str)


def test_paired_coverage(fake_deps: Deps):
    """Running all 4 systems on a dataset preserves exact qid pairing."""
    examples = [
        {"id": "q1", "question": "Question 1?", "knowledge": "Info 1"},
        {"id": "q2", "question": "Question 2?", "knowledge": "Info 2"},
    ]
    systems = ["vanilla", "rag", "cove", "healgen"]

    results = {sys: [run_system(sys, ex, fake_deps) for ex in examples] for sys in systems}

    for sys in systems:
        qids = [r["qid"] for r in results[sys]]
        assert qids == ["q1", "q2"]


def test_no_gold_in_prompt_inputs(sample_example: dict):
    """Gold knowledge is NEVER leaked into generator prompts directly."""
    gold_secret = "TOP_SECRET_GOLD_PARAGRAPH_NEVER_LEAK"
    ex = dict(sample_example)
    ex["knowledge"] = gold_secret

    systems = ["vanilla", "rag", "cove", "healgen"]
    for sys in systems:
        gen = FakeGenerator()
        deps = Deps(
            generator=gen,
            judge=FakeJudge(),
            bm25=FakeBM25(passages=["Public passage 1", "Public passage 2"]),
            estimator=FakeEstimator(),
            sufficiency=FakeSufficiency(),
        )
        run_system(sys, ex, deps)

        for prompt in gen.prompts:
            assert gold_secret not in prompt, f"System {sys} leaked gold knowledge into prompt!"


def test_healgen_retrieves_less_than_rag(sample_example: dict):
    """When evidence is already sufficient, HealGen avoids redundant BM25 calls."""
    deps_healgen = Deps(
        generator=FakeGenerator(),
        judge=FakeJudge(),
        bm25=FakeBM25(),
        estimator=FakeEstimator(),
        sufficiency=FakeSufficiency(sufficient=True),
    )
    deps_rag = Deps(
        generator=FakeGenerator(),
        judge=FakeJudge(),
        bm25=FakeBM25(),
        estimator=FakeEstimator(),
        sufficiency=FakeSufficiency(sufficient=True),
    )

    rec_healgen = run_system("healgen", sample_example, deps_healgen)
    rec_rag = run_system("rag", sample_example, deps_rag)

    assert rec_rag["num_bm25_calls"] >= 1
    assert rec_healgen["num_bm25_calls"] < rec_rag["num_bm25_calls"]


def test_cove_regenerates_once_only(sample_example: dict):
    """CoVe regenerates the entire response at most once when claims are unsupported."""
    gen = FakeGenerator()
    # Judge flags claims as unsupported
    judge = FakeJudge(default_support=False, default_score=0.1)
    deps = Deps(
        generator=gen,
        judge=judge,
        bm25=FakeBM25(),
        estimator=FakeEstimator(),
        sufficiency=FakeSufficiency(),
    )

    rec = run_system("cove", sample_example, deps)

    # CoVe generates initial draft (call 1) + regenerates once (call 2)
    assert gen.call_count == 2
    assert rec["system"] == "cove"
    assert rec["num_bm25_calls"] > 0

def test_ablation_flags_isolated(sample_example: dict):
    """Each ablation variant modifies exactly one subsystem behavior."""
    # 1. healgen-noCRE sets uniform risk = 1.0 regardless of estimator
    deps_nocre = Deps(
        generator=FakeGenerator(),
        judge=FakeJudge(),
        bm25=FakeBM25(),
        estimator=FakeEstimator(default_risk=0.12),
        sufficiency=FakeSufficiency(sufficient=True),
    )
    rec_nocre = run_system("healgen-noCRE", sample_example, deps_nocre)
    for c in rec_nocre["claims"]:
        assert c["risk"] == 1.0

    # 2. healgen-noAESP bypasses sufficiency check and always retrieves
    deps_noaesp = Deps(
        generator=FakeGenerator(),
        judge=FakeJudge(),
        bm25=FakeBM25(),
        estimator=FakeEstimator(),
        sufficiency=FakeSufficiency(sufficient=True),
    )
    rec_noaesp = run_system("healgen-noAESP", sample_example, deps_noaesp)
    assert rec_noaesp["num_bm25_calls"] > 0

    # 3. healgen-noLSHE performs confidence reduction rather than rephrasing
    gen = FakeGenerator()
    judge = FakeJudge(default_support=False, default_score=0.1)
    deps_nolshe = Deps(
        generator=gen,
        judge=judge,
        bm25=FakeBM25(),
        estimator=FakeEstimator(),
        sufficiency=FakeSufficiency(),
    )
    rec_nolshe = run_system("healgen-noLSHE", sample_example, deps_nolshe)
    for c in rec_nolshe["claims"]:
        assert c["action"] in {"reduce_confidence", "express_uncertainty", "unsupported"}
