"""Unit tests for the Localized Self-Healing Engine (LSHE, Layer 7)."""

import pytest
from healgen.repair import repair_answer


class FakeGenerator:
    """Mock generator that replaces text with content containing evidence tokens."""

    def __init__(self, replacement_text: str = "Sentence two was repaired with evidence token_xyz."):
        self.replacement_text = replacement_text
        self.prompt_received = ""

    def generate(self, prompt: str) -> str:
        self.prompt_received = prompt
        return self.replacement_text


def test_repair_preserves_untouched_sentences():
    """Repaired answer keeps unsupported-adjacent sentences byte-identical."""
    s1 = "The Eiffel Tower is in Paris."
    s2 = "It was completed in 1999."  # unsupported fact
    s3 = "It attracts millions of visitors annually."
    answer = f"{s1} {s2} {s3}"
    claims = [s1, s2, s3]
    unsupported_ids = ["c1"]  # corresponds to s2 (0-indexed c1)
    evidence = ["The Eiffel Tower was completed on 31 March 1889."]

    generator = FakeGenerator("The Eiffel Tower was completed in 1889.")
    repaired, records = repair_answer(
        answer=answer,
        claims=claims,
        unsupported_ids=unsupported_ids,
        evidence=evidence,
        generator=generator,
    )

    # Sentences 1 and 3 must be byte-identical
    assert s1 in repaired
    assert s3 in repaired
    assert s2 not in repaired
    assert "1889" in repaired


def test_rephrase_uses_evidence():
    """Rephrased sentence uses facts/tokens from the provided evidence."""
    answer = "The founder was Alice."
    claims = ["The founder was Alice."]
    unsupported_ids = ["c0"]
    evidence = ["The company was founded in 1995 by Bob Smith."]

    generator = FakeGenerator("The founder was Bob Smith.")
    repaired, records = repair_answer(
        answer=answer,
        claims=claims,
        unsupported_ids=unsupported_ids,
        evidence=evidence,
        generator=generator,
    )

    assert "Bob Smith" in repaired
    assert records[0]["action"] == "rephrase"
    assert "Alice" in records[0]["before"]
    assert "Bob Smith" in records[0]["after"]


def test_unchanged_when_all_supported():
    """When no claims are unsupported, the answer is returned unchanged."""
    answer = "Paris is the capital of France. It is in Europe."
    claims = ["Paris is the capital of France.", "It is in Europe."]
    generator = FakeGenerator("Different text.")

    repaired, records = repair_answer(
        answer=answer,
        claims=claims,
        unsupported_ids=[],
        evidence=["Paris is the capital of France."],
        generator=generator,
    )

    assert repaired == answer
    assert all(r["action"] == "supported" for r in records)


def test_record_actions():
    """repair_answer produces structured action records for every claim."""
    answer = "First fact. Second fact."
    claims = ["First fact.", "Second fact."]
    generator = FakeGenerator("Corrected second fact.")

    repaired, records = repair_answer(
        answer=answer,
        claims=claims,
        unsupported_ids=["c1"],
        evidence=["Evidence for second fact."],
        generator=generator,
    )

    assert len(records) == 2
    for r in records:
        assert set(r.keys()) == {"id", "action", "before", "after"}
    assert records[0]["action"] == "supported"
    assert records[1]["action"] in {"rephrase", "reduce_confidence", "express_uncertainty"}
