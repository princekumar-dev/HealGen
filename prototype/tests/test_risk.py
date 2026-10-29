"""Claim Risk Estimator (CRE) tests.

Unit tests stay fast: they use a tiny hand-built BM25 index over three
sentences and a fake sentence embedder (deterministic vectors, no model
downloads).  The real pinned models are only exercised by the training run
(``experiments/train_cre.py``), not here.

The three contract tests from the task brief:

- ``test_features_shape``: 6 finite features in the pinned order.
- ``test_fit_predict_roundtrip``: fit on synthetic separable rows, proba in
  (0,1), save/load round-trips identically (pure JSON, no pickle).
- ``test_train_split_disjoint``: the seeded 1,000-item train pool shares no
  id with dev, test or judge.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from healgen import risk
from healgen.data import BM25Index, build_corpus, load_examples


class FakeEmbedder:
    """Deterministic unit-norm vectors keyed on the tokens of each text.

    Mimics the MiniLM wrapper surface used by ``risk.features`` (``.encode``
    with ``normalize_embeddings=True``) without downloading anything.
    """

    _DIMS = 8

    def encode(self, texts, normalize_embeddings=True):  # noqa: ARG002
        vectors = [self._vector(text) for text in texts]
        return np.asarray(vectors, dtype=float)

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._DIMS, dtype=float)
        for index, token in enumerate(str(text).lower().split()):
            vec[hash(token) % self._DIMS] += 1.0 + (index % 3)
        return vec


@pytest.fixture()
def fake_emb():
    return FakeEmbedder()


@pytest.fixture()
def tiny_bm25():
    corpus = [
        "Paris is the capital of France and of the French republic.",
        "The Eiffel Tower stands in Paris and was finished in 1889.",
        "William Shakespeare wrote the play Hamlet around 1600.",
    ]
    return BM25Index(corpus)


# --- features ----------------------------------------------------------------


def test_features_shape(tiny_bm25, fake_emb):
    """6 features, all finite, in the pinned feature order."""
    feats = risk.features(
        "Paris is the capital of FRANCE and has 3 bridges.",
        "What is the capital of France?",
        tiny_bm25,
        emb=fake_emb,
    )
    assert len(feats) == len(risk.FEATURE_NAMES) == 6
    assert all(math.isfinite(value) for value in feats)
    assert all(isinstance(value, float) for value in feats)


def test_features_lazy_embedder(tiny_bm25, fake_emb, monkeypatch):
    """``features(..., emb=None)`` must route through the lazy singleton."""
    monkeypatch.setattr(risk, "get_embedder", lambda: fake_emb)
    feats = risk.features(
        "Paris is the capital of France.",
        "What is the capital of France?",
        tiny_bm25,
        emb=None,
    )
    assert len(feats) == 6


def test_features_rarity_flag(tiny_bm25, fake_emb):
    """All-caps token (len>1) or rare/unknown token flips the rarity flag."""
    rare_flag = risk.features(
        "Xqzwvk lives in Paris.", "What is in Paris?", tiny_bm25, emb=fake_emb
    )
    assert rare_flag[5] == 1.0  # unknown token -> document frequency 0 <= 2


def test_features_are_floats_not_nan(tiny_bm25, fake_emb):
    """Zero-norm embeddings must degrade to 0.0 cosine, never NaN."""
    feats = risk.features("", "", tiny_bm25, emb=fake_emb)
    assert all(math.isfinite(value) for value in feats)


# --- RiskEstimator -----------------------------------------------------------


def _separable_rows(n: int = 60) -> list[dict]:
    """Synthetic rows perfectly separable on feature 0 (claim length)."""
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(n):
        length = float(rng.integers(0, 12))
        features = [
            length,
            float(rng.integers(0, 4)),
            float(rng.uniform(0, 10)),
            float(rng.uniform(0, 1)),
            float(rng.uniform(0, 1)),
            float(rng.integers(0, 2)),
        ]
        rows.append({"features": features, "label": int(length >= 6)})
    return rows


def test_fit_predict_roundtrip(tmp_path, tiny_bm25, fake_emb):
    """Fit on separable rows -> proba in (0,1); JSON save/load identical."""
    estimator = risk.RiskEstimator()
    estimator.fit(_separable_rows())

    for features in ([0.0, 0.0, 0.0, 0.5, 0.5, 0.0], [11.0, 3.0, 9.0, 0.2, 0.8, 1.0]):
        proba = estimator.score(features)
        assert 0.0 < proba < 1.0
        assert math.isfinite(proba)

    # full inference path through predict_proba with fakes
    proba = estimator.predict_proba(
        "Paris is the capital of France and has many bridges.",
        "What is the capital of France?",
        tiny_bm25,
        emb=fake_emb,
    )
    assert 0.0 < proba < 1.0

    path = tmp_path / "cre_model.json"
    estimator.save(path)
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["feature_names"] == risk.FEATURE_NAMES  # pure JSON, no pickle

    reloaded = risk.RiskEstimator.load(path)
    for features in ([0.0, 0.0, 0.0, 0.5, 0.5, 0.0], [11.0, 3.0, 9.0, 0.2, 0.8, 1.0]):
        assert reloaded.score(features) == pytest.approx(
            estimator.score(features), abs=1e-6
        )


def test_reject_pickle_file(tmp_path):
    """The saved artefact must be plain JSON (no pickle bytes)."""
    estimator = risk.RiskEstimator()
    estimator.fit(_separable_rows(20))
    path = tmp_path / "cre_model.json"
    estimator.save(path)
    raw = path.read_bytes()
    assert raw[:1] == b"{"  # JSON object, not a pickle protocol stream


# --- train pool --------------------------------------------------------------


def test_train_split_disjoint():
    """Train ids must not overlap dev, test or judge."""
    train_ids = {ex["id"] for ex in risk.sample_train_items()}
    excluded: set[str] = set()
    for split in ("dev", "test", "judge"):
        excluded |= {ex["id"] for ex in load_examples(split)}
    assert train_ids.isdisjoint(excluded)


def test_train_sample_deterministic():
    """Pinned seed: two calls return the identical, sorted sample."""
    first = risk.sample_train_items()
    second = risk.sample_train_items()
    assert [ex["id"] for ex in first] == [ex["id"] for ex in second]
    ids = [ex["id"] for ex in first]
    assert ids == sorted(ids)


def test_train_sample_size():
    assert len(risk.sample_train_items()) == 1000
