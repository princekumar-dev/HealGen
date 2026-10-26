"""Shared T5 generator tests (uses the smoke model to keep runs short)."""

import time

import pytest

from healgen import generator


@pytest.mark.slow
def test_generate_nonempty_and_deterministic():
    """Same prompt twice -> identical non-empty string; prints CPU tokens/sec."""
    gen = generator.get_generator(smoke=True)
    prompt = "What is the capital of France?"
    gen.generate(prompt)  # prime the model; excluded from timing

    # The pair pinned by the brief: two calls, identical non-empty strings.
    first = gen.generate(prompt)
    second = gen.generate(prompt)
    assert first.strip() != ""
    assert first == second

    # More timed calls than the pair above: ~20 tokens is far too small a
    # sample for a stable rate (one re-run can double it).  Determinism is
    # re-asserted across every timed repeat.
    n_timed = 16
    tokens_before = gen.n_tokens
    start = time.perf_counter()
    answers = [gen.generate(prompt) for _ in range(n_timed)]
    elapsed = time.perf_counter() - start
    assert all(a == first for a in answers)

    tokens = gen.n_tokens - tokens_before
    rate = tokens / elapsed if elapsed > 0 else float("nan")
    print(
        f"tokens/sec: {rate:.1f} ({tokens} tokens in {elapsed:.2f}s, "
        "greedy, smoke model)"
    )
