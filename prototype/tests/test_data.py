"""Data layer tests: HaluEval download/cache, filtering, seeded splits, BM25.

The qa_data.json download happens once, on the first test that needs raw
records, and is cached under ``config.PATHS["data_cache"]`` for later runs
(network is only hit when the cache file is absent).
"""

from healgen import config, data

PUBLIC_KEYS = {"id", "question", "right_answer", "hallucinated_answer", "knowledge"}

SPLIT_SIZES = {
    "dev": config.N_DEV,
    "test": config.N_TEST,
    "judge": config.N_JUDGE_PAIRS,
}


def test_splits_match_pinned_sizes_and_public_keys():
    for split, expected_n in SPLIT_SIZES.items():
        items = data.load_examples(split)
        assert len(items) == expected_n, split
        ids = [ex["id"] for ex in items]
        assert len(set(ids)) == len(ids), f"{split} has duplicate ids"
        for ex in items:
            assert set(ex) == PUBLIC_KEYS, split
            assert all(isinstance(v, str) for v in ex.values()), split


def test_sample_deterministic():
    """Two calls under the same pinned seeds give identical ids; dev ∩ test = ∅."""
    first = [data.load_examples(s) for s in ("dev", "test")]
    second = [data.load_examples(s) for s in ("dev", "test")]
    for a, b in zip(first, second):
        assert [ex["id"] for ex in a] == [ex["id"] for ex in b]
    dev_ids = {ex["id"] for ex in first[0]}
    test_ids = {ex["id"] for ex in first[1]}
    assert len(dev_ids) == config.N_DEV
    assert len(test_ids) == config.N_TEST
    assert dev_ids.isdisjoint(test_ids)


def test_filters():
    """Every dev/test item has non-empty knowledge and a <=8-token right answer."""
    for split in ("dev", "test"):
        for ex in data.load_examples(split):
            assert ex["knowledge"].strip() != "", ex["id"]
            assert len(ex["right_answer"].split()) <= 8, ex["id"]


def test_corpus_distinct():
    """Corpus is the distinct knowledge paragraphs over the whole filtered set."""
    corpus = data.build_corpus(data.load_examples("all"))
    assert len(corpus) > 1_000
    assert len(corpus) == len(set(corpus))


def test_bm25_returns_k():
    corpus = data.build_corpus(data.load_examples("all"))
    index = data.BM25Index(corpus)
    hits = index.search("American literary magazine")
    assert len(hits) == config.TOPK
    for idx, text, score in hits:
        assert isinstance(idx, int)
        assert text == corpus[idx]
        assert isinstance(score, float)
    scores = [score for _, _, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_bm25_ranking():
    """Known fact: the corpus paragraph describing Arthur's Magazine calls it a
    'literary periodical'.  With the single-term query "periodical" every
    returned paragraph must contain the keyword (documents without the term
    score zero and cannot outrank containing documents), and the Arthur's
    Magazine paragraph itself must be among the hits.
    """
    corpus = data.build_corpus(data.load_examples("all"))
    index = data.BM25Index(corpus)
    hits = index.search("periodical")
    assert len(hits) == config.TOPK
    for _, text, _score in hits:
        assert "periodical" in text.lower()
    assert "19" in corpus[0] and "literary periodical" in corpus[0].lower()
    assert corpus[0] in {text for _, text, _score in hits}
