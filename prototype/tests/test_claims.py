"""Claim decomposition tests (Layer 2)."""

from healgen import claims


def test_splits_conjunction():
    """A pronoun-led second clause is split off and keeps the first subject."""
    result = claims.decompose(
        "Paris is the capital of France and it has the Eiffel Tower."
    )
    assert len(result) == 2
    assert "Paris" in result[1]


def test_pronoun_copied():
    """Comma-conjunction split copies the previous subject onto the pronoun."""
    result = claims.decompose(
        "Marie Curie was born in Warsaw, and she later studied in Paris."
    )
    assert len(result) == 2
    assert result[1].startswith("Marie Curie ")
    assert "she" not in result[1].lower()


def test_single_sentence():
    result = claims.decompose("The sky is blue.")
    assert result == ["The sky is blue."]


def test_empty_returns_empty():
    assert claims.decompose("") == []
    assert claims.decompose("   \n\t ") == []
