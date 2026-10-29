"""Claim Risk Estimator (CRE) -- Layer 3 of the HealGen pipeline.

For every decomposed claim the estimator predicts the probability that the
NLI judge will find it *unsupported* (label 1), i.e. how likely the claim is
a hallucination.  Training labels come from judge-vs-gold on 1,000 HaluEval
items outside dev/test/judge (constructed in ``experiments/train_cre.py``);
at inference the model is pure feature-space logistic regression -- the judge
is never needed to score claims.

The six features, in the pinned :data:`FEATURE_NAMES` order:

1. ``claim_words``          -- whitespace word count of the claim.
2. ``digit_cap_count``      -- tokens containing a digit plus capitalised
                               tokens, excluding the sentence-initial token.
3. ``max_bm25``             -- best BM25 score of the claim against the
                               knowledge corpus (top-3 retrieval).
4. ``claim_question_cos``   -- MiniLM cosine claim vs question.
5. ``novelty``              -- 1 - max MiniLM cosine claim vs its top-3
                               retrieved passages (knowledge-free novelty).
6. ``rarity_flag``          -- 1.0 when the claim carries an ALL-CAPS token
                               (length > 1) or any token whose corpus
                               document frequency is <= 2 (unseen tokens
                               count as frequency 0), else 0.0.

Design constraints:

- ``healgen.risk`` imports without loading any Hugging Face model; the MiniLM
  embedder is a lazy singleton (``get_embedder``) and ``bm25``/``emb`` are
  injectable everywhere, so tests run on fakes.
- The fitted artefact is pure JSON (scaler mean/scale, coefficients,
  intercept, metadata) -- no pickle -- so results files stay readable and
  diffable.
- ``sample_train_items`` pins the train pool: pool minus dev/test/judge ids,
   sampled with ``random.Random(99)``, then sorted by id for determinism.
"""

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from healgen import config

# --- feature extraction -------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

FEATURE_NAMES = [
    "claim_words",
    "digit_cap_count",
    "max_bm25",
    "claim_question_cos",
    "novelty",
    "rarity_flag",
]

N_CLAIM_FEATURES = len(FEATURE_NAMES)

TRAIN_SAMPLE_SEED = 99
TRAIN_SAMPLE_SIZE = 1000


def _bag_tokens(text: str) -> list[str]:
    """Case-sensitive alphanumeric tokens (feature tokenisation)."""
    return _WORD_RE.findall(text)


def _cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine of two vectors; zero-norm inputs degrade to 0.0 (never NaN)."""
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class MiniLMEmbedder:
    """Sentence-transformers wrapper exposing the ``encode`` contract."""

    def __init__(self, model_id: str):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self._model = SentenceTransformer(model_id)

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts),
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False,
            ),
            dtype=float,
        )


_EMBEDDER: MiniLMEmbedder | None = None


def get_embedder() -> MiniLMEmbedder:
    """Lazy MiniLM singleton -- the pinned :attr:`config.EMB_MODEL`."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = MiniLMEmbedder(config.EMB_MODEL)
    return _EMBEDDER


def _document_frequencies(bm25) -> Counter[str]:
    """Lowercase token -> corpus document frequency, memoised on the index.

    The counter is cached once per ``BM25Index`` instance (an attribute set on
    the instance itself, so a new index recomputes and nothing can go stale).
    """
    cached = getattr(bm25, "_cre_df", None)
    if cached is not None:
        return cached
    df: Counter[str] = Counter()
    for text in bm25._texts:
        df.update(set(re.findall(r"[a-z0-9]+", text.lower())))
    bm25._cre_df = df
    return df


def features(
    claim: str, question: str, bm25, emb: Any = None
) -> list[float]:
    """The 6 pinned features of one claim, in :data:`FEATURE_NAMES` order.

    ``emb`` must expose ``encode(texts, normalize_embeddings=...)``; ``None``
    routes through the lazy :func:`get_embedder` singleton.
    """
    claim = str(claim)
    question = str(question)

    tokens = _bag_tokens(claim)
    claim_words = float(len(claim.split()))
    digit_cap_count = float(
        sum(
            1
            for index, token in enumerate(tokens)
            if index > 0
            and (
                any(char.isdigit() for char in token)
                or (token[:1].isalpha() and token[:1].isupper())
            )
        )
    )

    retrieved = list(bm25.search(claim, k=3))
    max_bm25 = float(retrieved[0][2]) if retrieved else 0.0
    passages = [text for _, text, _ in retrieved]

    if emb is None:
        emb = get_embedder()
    texts = [claim, question, *passages]
    vectors = np.asarray(emb.encode(texts, normalize_embeddings=True), dtype=float)
    claim_vec = vectors[0]
    question_vec = vectors[1]
    claim_question_cos = _cosine(claim_vec, question_vec)
    passage_cosines = (
        [_cosine(claim_vec, vectors[2 + i]) for i in range(len(passages))]
        if passages
        else []
    )
    novelty = 1.0 - max(passage_cosines) if passage_cosines else 1.0

    df = _document_frequencies(bm25)
    lower_tokens = {token.lower() for token in tokens}
    rarity_flag = 1.0 if (
        any(token.isupper() and len(token) > 1 for token in tokens)
        or any(df.get(token, 0) <= 2 for token in lower_tokens)
    ) else 0.0

    return [
        claim_words,
        digit_cap_count,
        max_bm25,
        claim_question_cos,
        novelty,
        rarity_flag,
    ]


# --- train pool ----------------------------------------------------------------


def sample_train_items(n: int = TRAIN_SAMPLE_SIZE, seed: int = TRAIN_SAMPLE_SEED) -> list[dict]:
    """The pinned 1,000-item train pool, disjoint from dev/test/judge.

    Candidates are the filtered pool minus the id sets of the dev/test/judge
    splits, sampled with ``random.Random(seed)`` and then sorted by id so the
    downstream cache order is deterministic.
    """
    from healgen.data import load_examples

    excluded: set[str] = set()
    for split in ("dev", "test", "judge"):
        excluded |= {ex["id"] for ex in load_examples(split)}
    candidates = [ex for ex in load_examples("all") if ex["id"] not in excluded]
    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} candidate items for the train pool")
    sampled = random.Random(seed).sample(candidates, n)
    sampled.sort(key=lambda ex: ex["id"])
    return sampled


# --- estimator -------------------------------------------------------------------


class RiskEstimator:
    """LogisticRegression over standardised claim features (pure-JSON model).

    ``fit`` takes **feature rows**: dicts with ``features`` (the 6 floats from
    :func:`features`) and ``label`` (0 supported / 1 risky).  ``predict_proba``
    is the full inference path (claim/question -> features -> probability);
    ``score`` consumes a precomputed feature vector.
    """

    def __init__(self) -> None:
        self.feature_names: list[str] = list(FEATURE_NAMES)
        self.scaler_mean: list[float] | None = None
        self.scaler_scale: list[float] | None = None
        self.coef: list[float] | None = None
        self.intercept: float | None = None
        self.metadata: dict = {}

    # --- training ----------------------------------------------------------------

    def fit(self, train_rows: list[dict]) -> "RiskEstimator":
        """Fit StandardScaler + LogisticRegression on the given rows."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if not train_rows:
            raise ValueError("train_rows is empty")
        x = np.asarray([row["features"] for row in train_rows], dtype=float)
        y = np.asarray([int(row["label"]) for row in train_rows], dtype=int)
        if x.shape[1] != N_CLAIM_FEATURES:
            raise ValueError(
                f"expected {N_CLAIM_FEATURES} features per row, got {x.shape[1]}"
            )
        if len(np.unique(y)) < 2:
            raise ValueError("labels must contain both classes to fit")

        scaler = StandardScaler().fit(x)
        classifier = LogisticRegression(max_iter=1000, random_state=0)
        classifier.fit(scaler.transform(x), y)

        self.scaler_mean = [float(value) for value in scaler.mean_]
        self.scaler_scale = [float(value) for value in scaler.scale_]
        self.coef = [float(value) for value in classifier.coef_[0]]
        self.intercept = float(classifier.intercept_[0])
        predicted = self.predict_labels(x)
        self.metadata = {
            "n_train": int(len(y)),
            "positive_rate": float(y.mean()),
            "train_accuracy": float((predicted == y).mean()),
        }
        return self

    # --- inference -----------------------------------------------------------------

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        assert self.scaler_mean is not None and self.scaler_scale is not None
        return (np.asarray(x, dtype=float) - np.asarray(self.scaler_mean)) / np.asarray(
            self.scaler_scale
        )

    def _logits(self, x_scaled: np.ndarray) -> np.ndarray:
        assert self.coef is not None and self.intercept is not None
        return x_scaled @ np.asarray(self.coef) + self.intercept

    def predict_labels(self, x: np.ndarray) -> np.ndarray:
        return (self._logits(self._standardise(x)) >= 0.0).astype(int)

    def score(self, feature_vector: list[float]) -> float:
        """Probability in the open interval (0, 1) for a feature vector."""
        if self.coef is None:
            raise RuntimeError("estimator is not fitted (call fit() or load())")
        logit = float(self._logits(self._standardise([feature_vector]))[0])
        probability = 1.0 / (1.0 + np.exp(-logit))
        return float(min(max(probability, 1e-12), 1.0 - 1e-12))

    def predict_proba(
        self, claim: str, question: str, bm25, emb: Any = None
    ) -> float:
        """Full inference path: build the features, then read the probability."""
        return self.score(features(claim, question, bm25, emb=emb))

    # --- persistence ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "scaler": {"mean": self.scaler_mean, "scale": self.scaler_scale},
            "coef": self.coef,
            "intercept": self.intercept,
            "model": "LogisticRegression(max_iter=1000, random_state=0) on StandardScaler",
            "metadata": self.metadata,
        }

    def save(self, path) -> None:
        """Write the fitted model as pure JSON (no pickle)."""
        if self.coef is None:
            raise RuntimeError("estimator is not fitted (call fit() or load())")
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path) -> "RiskEstimator":
        """Restore an estimator from a ``save``d JSON file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        estimator = cls()
        estimator.feature_names = list(payload["feature_names"])
        estimator.scaler_mean = [float(v) for v in payload["scaler"]["mean"]]
        estimator.scaler_scale = [float(v) for v in payload["scaler"]["scale"]]
        estimator.coef = [float(v) for v in payload["coef"]]
        estimator.intercept = float(payload["intercept"])
        estimator.metadata = dict(payload.get("metadata", {}))
        return estimator
