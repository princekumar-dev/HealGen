"""Train the Claim Risk Estimator (CRE): features, judge labels, logistic fit.

Run from the repository root::

    python prototype/experiments/train_cre.py

Pipeline (all checkpointed -- safe to interrupt and re-run):

1. Label pool: the pinned 1,000-item train pool (``risk.sample_train_items``)
   and the 50-item dev split.  Both answers of every item are decomposed into
   claims; each claim is judged against ``[knowledge]`` with the pinned NLI
   judge (supported -> label 0, unsupported -> label 1).  Identical
   ``(question, claim)`` pairs are judged once and share the label, but every
   ``(qid, answer_type, claim_idx)`` row is kept.
2. Cache: labeled rows append incrementally to
   ``results/cre_train_cache.csv`` / ``results/cre_dev_cache.csv`` (columns
   ``qid, answer_type, claim_idx, claim, knowledge, label``).  Re-runs load
   the cache and only judge the missing keys, so an interrupted labeling run
   resumes where it stopped.
3. Features: the 6 pinned per-claim features (``healgen.risk.features``).
4. Fit: StandardScaler + ``LogisticRegression(max_iter=1000, random_state=0)``
   on the train rows; report train accuracy vs the majority (always-0)
   baseline and dev accuracy; save the pure-JSON model to
   ``results/cre_model.json``.  If train accuracy does not beat the baseline
   the script says so explicitly and still saves the model -- nothing is
   tuned on the test split.
"""

import csv
import sys
import time
from pathlib import Path

PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
if str(PROTOTYPE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT))

from healgen import claims, config, data, risk  # noqa: E402  (path above)
from healgen.judge import Judge  # noqa: E402

CACHE_COLUMNS = (
    "qid",
    "answer_type",
    "claim_idx",
    "claim",
    "knowledge",
    "label",
)

TRAIN_CACHE = config.PATHS["results"] / "cre_train_cache.csv"
DEV_CACHE = config.PATHS["results"] / "cre_dev_cache.csv"
MODEL_PATH = config.PATHS["results"] / "cre_model.json"

ANSWER_KEYS = (
    ("right", "right_answer"),
    ("hallucinated", "hallucinated_answer"),
)

PROGRESS_EVERY = 25


# --- claim-row construction ----------------------------------------------------


def item_claim_rows(example: dict) -> list[dict]:
    """Unlabeled rows ``(qid, answer_type, claim_idx, claim, knowledge)``."""
    rows: list[dict] = []
    for answer_type, key in ANSWER_KEYS:
        for claim_idx, claim_text in enumerate(claims.decompose(example[key])):
            rows.append(
                {
                    "qid": example["id"],
                    "answer_type": answer_type,
                    "claim_idx": claim_idx,
                    "claim": claim_text,
                    "knowledge": example["knowledge"],
                }
            )
    return rows


# --- resume-safe cache ------------------------------------------------------------


def load_cache(path: Path) -> dict[tuple[str, str, int], dict]:
    """Read a cache CSV keyed on ``(qid, answer_type, claim_idx)``.

    Tolerant of a torn final line (an interrupted append): malformed rows are
    dropped and will be re-labeled on the next run, never half-trusted.
    """
    if not path.exists():
        return {}
    cache: dict[tuple[str, str, int], dict] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                key = (str(raw["qid"]), str(raw["answer_type"]), int(raw["claim_idx"]))
                label = int(raw["label"])
            except (KeyError, TypeError, ValueError):
                continue  # torn/blank line -> re-label
            cache[key] = {
                "claim": str(raw["claim"]),
                "knowledge": str(raw["knowledge"]),
                "label": label,
            }
    return cache


def append_cache_rows(path: Path, rows: list[dict]) -> None:
    """Append labeled rows (one flush per call keeps resumability cheap)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(CACHE_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row["qid"],
                    row["answer_type"],
                    row["claim_idx"],
                    row["claim"],
                    row["knowledge"],
                    row["label"],
                ]
            )
        handle.flush()


def label_split(judge: Judge, examples: list[dict], cache_path: Path, tag: str) -> dict:
    """Judge-label every claim of every example into ``cache_path`` (resumable).

    Return ``{"labeled": rows_in_cache, "judged": rows_judged_this_run,
    "items": item_count}``.  Identical ``(question, claim)`` texts are judged
    once per run (memoised) but kept as separate rows.
    """
    cache = load_cache(cache_path)
    question_of = {example["id"]: example["question"] for example in examples}
    label_memo: dict[tuple[str, str], int] = {
        (question_of[row_key[0]], row["claim"]): row["label"]
        for row_key, row in cache.items()
        if row_key[0] in question_of
    }
    judged = 0
    started = time.perf_counter()
    for index, example in enumerate(examples, start=1):
        rows = item_claim_rows(example)
        pending = [row for row in rows if _cache_key(row) not in cache]
        if pending:
            # Identical (question, claim) pairs are judged once: rows whose
            # label is already known inherit it and are appended directly.
            todo = []
            for row in pending:
                pair = (example["question"], row["claim"])
                if pair in label_memo:
                    row["label"] = label_memo[pair]
                else:
                    todo.append(row)
            if todo:
                for row in _judge_rows(judge, todo, example):
                    label_memo[(example["question"], row["claim"])] = row["label"]
                judged += len(todo)
            append_cache_rows(cache_path, pending)
            for row in pending:
                cache[_cache_key(row)] = row
        if index % PROGRESS_EVERY == 0 or index == len(examples):
            rate = judged / max(time.perf_counter() - started, 1e-9)
            print(
                f"[{tag}] {index}/{len(examples)} items, "
                f"{judged} claims judged this run ({rate:.2f} rows/s), "
                f"cache={len(cache)} rows",
                flush=True,
            )
    return {"labeled": len(cache), "judged": judged, "items": len(examples)}


def _cache_key(row: dict) -> tuple[str, str, int]:
    return (row["qid"], row["answer_type"], row["claim_idx"])


def _judge_rows(judge: Judge, pending: list[dict], example: dict) -> list[dict]:
    """Fill in labels for a list of rows sharing the item's evidence pool.

    All pending claims go through one batched ``evaluate_claims`` call
    against ``[knowledge]`` (the gold-only condition of the experiment).
    """
    supported = judge.evaluate_claims(
        [row["claim"] for row in pending], [example["knowledge"]]
    )
    for row, (is_supported, _score) in zip(pending, supported, strict=True):
        row["label"] = 0 if is_supported else 1
    return pending


# --- features + fit ----------------------------------------------------------------


def build_feature_rows(cache_path: Path, examples: list[dict], bm25, emb) -> list[dict]:
    """Row dicts ``{"features": [...], "label": int}`` from a labeled cache."""
    cache = load_cache(cache_path)
    question_of = {example["id"]: example["question"] for example in examples}
    rows = []
    for key in sorted(cache, key=lambda k: (k[0], k[1], k[2])):
        row = cache[key]
        question = question_of.get(key[0], "")
        rows.append(
            {
                "features": risk.features(row["claim"], question, bm25, emb=emb),
                "label": row["label"],
            }
        )
    return rows


def main() -> None:
    started = time.perf_counter()
    results_dir = Path(config.PATHS["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    train_items = risk.sample_train_items()
    dev_items = data.load_examples("dev")
    print(
        f"train pool: {len(train_items)} items | dev: {len(dev_items)} items "
        f"(pool build {time.perf_counter() - started:.1f}s)",
        flush=True,
    )

    judge_start = time.perf_counter()
    judge = Judge(config.JUDGE_MODEL)
    print(f"judge {config.JUDGE_MODEL} loaded ({time.perf_counter() - judge_start:.1f}s)", flush=True)

    train_stats = label_split(judge, train_items, TRAIN_CACHE, "train")
    print(f"train cache: {train_stats}", flush=True)
    dev_stats = label_split(judge, dev_items, DEV_CACHE, "dev")
    print(f"dev cache: {dev_stats}", flush=True)

    # --- features (embedder downloads on first use) ------------------------------
    feature_start = time.perf_counter()
    bm25 = data.BM25Index(data.build_corpus(data.load_examples("all")))
    emb = risk.get_embedder()
    train_rows = build_feature_rows(TRAIN_CACHE, train_items, bm25, emb)
    dev_rows = build_feature_rows(DEV_CACHE, dev_items, bm25, emb)
    print(
        f"features built: {len(train_rows)} train + {len(dev_rows)} dev rows "
        f"({time.perf_counter() - feature_start:.1f}s)",
        flush=True,
    )

    labels = [row["label"] for row in train_rows]
    positive_rate = sum(1 for label in labels if label == 1) / len(labels)
    baseline = max(sum(labels) / len(labels), 1.0 - sum(labels) / len(labels))
    print(f"label stats: n={len(labels)}, positive (risky) rate={positive_rate:.3f}")

    estimator = risk.RiskEstimator().fit(train_rows)
    train_accuracy = estimator.metadata["train_accuracy"]
    print(f"train accuracy: {train_accuracy:.3f}")
    print(f"majority baseline: {baseline:.3f}")
    if train_accuracy <= baseline:
        print(
            "WARNING: train accuracy does not beat the majority baseline -- "
            "reported honestly; no test-set tuning was performed."
        )

    if dev_rows:
        dev_labels = [row["label"] for row in dev_rows]
        dev_x = [row["features"] for row in dev_rows]
        predictions = estimator.predict_labels(dev_x)
        dev_accuracy = float(
            sum(int(pred == lab) for pred, lab in zip(predictions, dev_labels, strict=True))
            / len(dev_labels)
        )
        print(f"dev accuracy: {dev_accuracy:.3f} (n={len(dev_labels)})")
    else:
        dev_accuracy = None

    estimator.metadata.update(
        {
            "baseline": float(baseline),
            "dev_accuracy": dev_accuracy,
            "n_dev": len(dev_rows),
            "judge_model": config.JUDGE_MODEL,
            "emb_model": config.EMB_MODEL,
            "wall_seconds": round(time.perf_counter() - started, 1),
        }
    )
    estimator.save(MODEL_PATH)
    print(f"model saved to {MODEL_PATH}")
    print(f"total wall time: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
