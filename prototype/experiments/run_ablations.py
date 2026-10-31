"""Ablation experiments (Task 15) — the novelty defense.

Runs three HealGen ablation variants on the same 150 test qids as the main run:
- healgen-noCRE   : uniform risk = 1.0 (CRE bypassed, always-verify behavior)
- healgen-noAESP  : sufficiency bypassed (always retrieve, AESP off)
- healgen-noLSHE  : unsupported claims detected but only hedged (no rephrase)

Produces:
- results/ablations.jsonl (raw records, resumable, 150 x 3 = 450 rows)
- results/ablations.csv   (flattened per-example metrics, 450 rows;
  shared-pool evaluation labels via healgen.rescore -- ledger R12)
Full-HealGen references come from results/main.jsonl (same qids, paired);
evidence pools come from results/evidence_pools.json (written by run_main),
so ablations are judged against the same frozen pools as the main systems.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
if str(PROTOTYPE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT))

from healgen import config, data, generator as gen_module  # noqa: E402
from healgen.judge import Judge  # noqa: E402
from healgen.metrics import em_f1  # noqa: E402
from healgen.rescore import attach_eval_labels, label_view, load_pools  # noqa: E402
from healgen.risk import RiskEstimator  # noqa: E402
from healgen.sufficiency import Sufficiency  # noqa: E402
from healgen.systems import Deps, run_system  # noqa: E402

ABLATION_SYSTEMS = ["healgen-noCRE", "healgen-noAESP", "healgen-noLSHE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HealGen Ablations (N=150 x 3)")
    parser.add_argument("--n", type=int, default=config.N_TEST, help="Number of test examples")
    parser.add_argument(
        "--model",
        choices=["small", "base"],
        default="base",
        help="Generator model — must match the main run (default: base)",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=config.PATHS["results"] / "ablations.jsonl",
        help="Path for raw ablation records JSONL",
    )
    return parser.parse_args()


def load_existing_records(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    records: list[dict[str, Any]] = []
    completed: set[tuple[str, str]] = set()
    if not path.exists():
        return records, completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            completed.add((rec["qid"], rec["system"]))
    return records, completed


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print(f"HEALGEN ABLATION RUN: N={args.n}, Generator={args.model}, systems={ABLATION_SYSTEMS}")
    print("=" * 70)

    # 1. Load data — identical sampling to the main run (seed 42, dev excluded)
    print("\n[1/4] Loading data and building frozen BM25 index...")
    dev_exs = data.load_examples("dev")
    test_exs = data.load_examples("test")[: args.n]
    all_exs = dev_exs + test_exs
    corpus = data.build_corpus(all_exs)
    bm25 = data.BM25Index(corpus)
    ex_by_id = {ex["id"]: ex for ex in test_exs}

    # 2. Initialize models — same components as the main run
    print(f"\n[2/4] Initializing generator ({args.model})...")
    gen = gen_module.get_generator(smoke=(args.model == "small"))

    print(f"\n[3/4] Initializing judge ({config.JUDGE_MODEL})...")
    judge = Judge(config.JUDGE_MODEL)

    print("\n[4/4] Initializing CRE risk estimator and AESP sufficiency predictor...")
    cre_path = config.PATHS["results"] / "cre_model.json"
    estimator = RiskEstimator.load(cre_path) if cre_path.exists() else RiskEstimator()
    sufficiency = Sufficiency(tau=0.7)

    deps = Deps(
        generator=gen,
        judge=judge,
        bm25=bm25,
        estimator=estimator,
        sufficiency=sufficiency,
    )

    # 3. Resumable execution over the paired qid set
    existing_records, completed_pairs = load_existing_records(args.out_jsonl)
    print(f"\nFound {len(existing_records)} existing records ({len(completed_pairs)} pairs).")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl_file = args.out_jsonl.open("a", encoding="utf-8")

    all_records = list(existing_records)
    total_to_run = len(test_exs) * len(ABLATION_SYSTEMS) - len(completed_pairs)
    print(f"Executing remaining {total_to_run} ablation runs...")

    run_count = 0
    t0 = time.perf_counter()
    try:
        for idx, ex in enumerate(test_exs):
            qid = ex["id"]
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  Processing example {idx + 1}/{len(test_exs)} (qid={qid})...")
            for sys_name in ABLATION_SYSTEMS:
                if (qid, sys_name) in completed_pairs:
                    continue
                rec = run_system(sys_name, ex, deps)
                all_records.append(rec)
                jsonl_file.write(json.dumps(rec) + "\n")
                jsonl_file.flush()
                completed_pairs.add((qid, sys_name))
                run_count += 1
    finally:
        jsonl_file.close()

    elapsed = time.perf_counter() - t0
    print(f"\nFinished {run_count} new runs in {elapsed:.1f}s. Total records: {len(all_records)}")

    # 4. Frozen-pool evaluation rescoring (ledger R12): ablations score
    #    against the SAME shared pools as the four main systems
    #    (results/evidence_pools.json, written by run_main.py) so the
    #    ablation deltas are paired against an identical verdict source.
    #    Judged claims = decompose(answer), the emitted response.
    print("\nRescoring ablation claims against frozen evidence pools (R12)...")
    pools_path = config.PATHS["results"] / "evidence_pools.json"
    pools = load_pools(pools_path)
    gold_by_qid = {qid: str(ex.get("knowledge", "")) for qid, ex in ex_by_id.items()}
    attach_eval_labels(all_records, pools, gold_by_qid, judge)
    ev_records = label_view(all_records, "ev")
    n_eval = sum(len(r["eval_claims"]) for r in all_records)
    print(f"Rescored {n_eval} emitted claims for {len(pools)} pooled qids.")

    # 5. Flattened per-example metrics (same columns as main.csv; shared-pool labels)
    csv_path = config.PATHS["results"] / "ablations.csv"
    rows: list[dict[str, Any]] = []
    for r in ev_records:
        ex = ex_by_id.get(r["qid"], {})
        gold_answer = ex.get("right_answer", "")
        claims = r.get("claims", [])
        num_claims = len(claims)
        num_sup = sum(1 for c in claims if c.get("supported"))
        sp_m = (num_sup / num_claims) if num_claims > 0 else 1.0
        em_res = em_f1([r["answer"]], [gold_answer])
        rows.append(
            {
                "qid": r["qid"],
                "system": r["system"],
                "question": r["question"],
                "answer": r["answer"],
                "num_claims": num_claims,
                "num_supported_claims": num_sup,
                "sp_macro": round(sp_m, 4),
                "any_hallucination": int(any(not c.get("supported") for c in claims)),
                "num_bm25_calls": r.get("num_bm25_calls", 0),
                "latency_s": round(r.get("latency_s", 0.0), 3),
                "abstained": int(r.get("abstained", False)),
                "em": round(em_res["em"], 4),
                "f1": round(em_res["f1"], 4),
            }
        )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")

    # Guard: exact 450-row contract (150 qids x 3 variants)
    expected = len(test_exs) * len(ABLATION_SYSTEMS)
    assert len(rows) == expected, f"Expected {expected} rows, got {len(rows)}"
    print(f"Row-count guard PASSED: {len(rows)} == {expected}")


if __name__ == "__main__":
    main()
