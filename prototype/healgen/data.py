"""HaluEval QA data layer: download/cache, filter, seeded sampling, BM25 index.

Verified against the real file (2026-10-26): the canonical URL below serves
``qa_data.json`` as JSON *Lines* -- one JSON object per line, 10,000 records --
not as a single JSON document with a top-level ``qa_data`` list.  Each record
carries exactly four string fields (knowledge, question, right_answer,
hallucinated_answer) and no id, so stable ids are synthesised from the file
line index.  ``_read_records`` still accepts a plain JSON list or a
``{"qa_data": [...]}`` wrapper in case the upstream file shape changes.

Public API (experiment contract):
- ``load_examples(split)``: dev/test/judge dicts with keys
  id, question, right_answer, hallucinated_answer, knowledge, or ``"all"``
  for the full filtered pool (used to build the retrieval corpus).
- ``build_corpus(examples_all)``: distinct knowledge paragraphs, file order.
- ``BM25Index``: BM25Okapi wrapper with ``search(query, k=config.TOPK)``.
"""

import json
import random
import re
from pathlib import Path

import requests
from rank_bm25 import BM25Okapi

from healgen import config

QA_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
CACHE_FILE = config.PATHS["data_cache"] / "qa_data.json"

REQUIRED_FIELDS = ("knowledge", "question", "right_answer", "hallucinated_answer")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_RAW_RECORDS: list[dict] | None = None  # lazily memoised raw records


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _ensure_download() -> Path:
    """Return the local qa_data.json path, downloading it only if missing."""
    cache_dir = Path(config.PATHS["data_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not CACHE_FILE.exists():
        response = requests.get(QA_URL, timeout=120)
        response.raise_for_status()
        CACHE_FILE.write_bytes(response.content)
    return CACHE_FILE


def _read_records(path: Path) -> list[dict]:
    """Parse the cached file into the list of raw QA records.

    Accepts the observed shape (JSON Lines -- one JSON object per line, which
    is not a whole-document ``json.loads``), plus the defensive variants
    (plain JSON list or {"qa_data": [...]} or a single-record object) named in
    the task brief.
    """
    text = path.read_text(encoding="utf-8")
    records: list[dict]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None  # falls through to the JSON Lines branch
    if isinstance(parsed, dict):
        if "qa_data" in parsed:
            records = parsed["qa_data"]
        elif all(field in parsed for field in REQUIRED_FIELDS):
            records = [parsed]
        else:
            raise RuntimeError(f"{path.name}: unexpected top-level JSON object")
        if not isinstance(records, list):
            raise RuntimeError(f"{path.name}: unexpected top-level JSON object")
    elif isinstance(parsed, list):
        records = parsed
    else:
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path.name}, line {line_no}: {exc}") from exc
            if not isinstance(rec, dict):
                raise RuntimeError(f"{path.name}, line {line_no}: record is not an object")
            records.append(rec)
    if not records:
        raise RuntimeError(f"{path.name}: no QA records found")
    if not all(isinstance(rec, dict) for rec in records):
        raise RuntimeError(f"{path.name}: records must be JSON objects")
    for i, rec in enumerate(records):
        missing = [field for field in REQUIRED_FIELDS if field not in rec]
        if missing:
            raise RuntimeError(f"{path.name}, record {i}: missing fields {missing}")
    return records


def _raw_records() -> list[dict]:
    global _RAW_RECORDS
    if _RAW_RECORDS is None:
        _RAW_RECORDS = _read_records(_ensure_download())
    return _RAW_RECORDS


def _example(line_no: int, record: dict) -> dict:
    """Map a raw record onto the public id/question/answers/knowledge keys."""
    return {
        "id": f"qa_{line_no:05d}",
        "question": str(record["question"]).strip(),
        "right_answer": str(record["right_answer"]).strip(),
        "hallucinated_answer": str(record["hallucinated_answer"]).strip(),
        "knowledge": str(record["knowledge"]).strip(),
    }


def _filtered_examples() -> list[dict]:
    """Filter (non-empty knowledge, <=8-token right answer) and dedup by question."""
    examples = [
        _example(line_no, record) for line_no, record in enumerate(_raw_records())
    ]
    passed = [
        ex
        for ex in examples
        if ex["knowledge"] and len(ex["right_answer"].split()) <= 8
    ]
    seen_questions: set[str] = set()
    unique: list[dict] = []
    for ex in passed:
        if ex["question"] not in seen_questions:
            seen_questions.add(ex["question"])
            unique.append(ex)
    return unique


def _shuffled(pool: list[dict], seed: int) -> list[dict]:
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def load_examples(split: str) -> list[dict]:
    """Deterministic split sampling under the pinned config seeds.

    dev    = seed 43, first :attr:`config.N_DEV` items after filters+dedup.
    test   = seed 42, first :attr:`config.N_TEST` items of the seed-42 order
             whose ids are not in dev (dev and test are disjoint).
    judge  = seed 44, first :attr:`config.N_JUDGE_PAIRS` items.
    "all"  = the full filtered pool in file order (corpus construction).
    """
    pool = _filtered_examples()
    if split == "dev":
        return _shuffled(pool, config.SEED_DEV)[: config.N_DEV]
    if split == "test":
        dev_ids = {ex["id"] for ex in load_examples("dev")}
        sampled: list[dict] = []
        for ex in _shuffled(pool, config.SEED_TEST):
            if ex["id"] in dev_ids:
                continue
            sampled.append(ex)
            if len(sampled) == config.N_TEST:
                break
        return sampled
    if split == "judge":
        return _shuffled(pool, config.SEED_JUDGE)[: config.N_JUDGE_PAIRS]
    if split == "all":
        return list(pool)
    raise ValueError(
        f"unknown split {split!r}; expected 'dev', 'test', 'judge' or 'all'"
    )


def build_corpus(examples_all: list[dict]) -> list[str]:
    """Distinct non-empty knowledge paragraphs over the given examples (file order)."""
    paragraphs: list[str] = []
    seen: set[str] = set()
    for ex in examples_all:
        knowledge = ex["knowledge"].strip()
        if knowledge and knowledge not in seen:
            seen.add(knowledge)
            paragraphs.append(knowledge)
    return paragraphs


class BM25Index:
    """BM25Okapi wrapper over distinct knowledge paragraphs.

    Uses the pinned ``config.BM25_K1``/``config.BM25_B`` and a shared
    lowercase ``[a-z0-9]+`` tokenizer for corpus and query.  ``search``
    returns ``[(corpus_index, paragraph, score), ...]`` sorted by
    non-increasing score (ties broken by file order).
    """

    def __init__(self, corpus: list[str]):
        if not corpus:
            raise ValueError("corpus must be non-empty")
        self._texts = list(corpus)
        # Pre-tokenised (tokenizer=None) so rank_bm25 uses the corpus of token
        # lists as-is and skips its multiprocessing tokenization helper.
        self._bm25 = BM25Okapi(
            [_tokenize(text) for text in self._texts],
            k1=config.BM25_K1,
            b=config.BM25_B,
        )

    def search(self, query: str, k: int = config.TOPK) -> list[tuple[int, str, float]]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            range(len(self._texts)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [
            (i, self._texts[i], float(scores[i])) for i in ranked
        ]
