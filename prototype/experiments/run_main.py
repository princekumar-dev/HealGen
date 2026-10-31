"""Main experiment run (Task 14).

Runs the four paired systems on the full test split (N=150 examples, seed 42)
with resumability, computes headline evaluation metrics, ECE/Brier calibration,
bootstrap confidence intervals, and 6 Holm-corrected pairwise hypothesis tests.

Produces:
- results/main.jsonl (600 paired records -- raw system records, unmodified)
- results/evidence_pools.json (frozen shared pool per qid: gold ∪ union of all
  four systems' retrievals -- ledger R12, the plan's Global Constraint)
- results/main.csv (flattened per-example metrics; shared-pool eval labels)
- results/claims.csv (emitted claims: shared-pool verdict + gold-only verdict
  + pipeline metadata joined by text + the evidence pool shown to the judge)
- results/system_summary.csv (system-level aggregates with CIs + a
  gold-only SP column for the plan's second evaluation condition)
- results/pairwise_tests.csv (pairwise bootstrap significance tests)

All claim verdicts come from rescore.attach_eval_labels: the judge scores
the *emitted* answer's claims against the frozen pools -- never against
system-private evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
if str(PROTOTYPE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT))

from healgen import config, data, generator as gen_module  # noqa: E402
from healgen.judge import Judge  # noqa: E402
from healgen.metrics import (  # noqa: E402
    any_hallucination_rate,
    bootstrap_ci,
    brier,
    ece,
    em_f1,
    hallucination_rate,
    holm_bonferroni,
    paired_bootstrap,
    sp_macro,
    sp_micro,
)
from healgen.risk import RiskEstimator  # noqa: E402
from healgen.rescore import (  # noqa: E402
    attach_eval_labels,
    build_pools,
    label_view,
    save_pools,
)
from healgen.sufficiency import Sufficiency  # noqa: E402
from healgen.systems import Deps, run_system  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HealGen Main Experiment (N=150)")
    parser.add_argument("--n", type=int, default=config.N_TEST, help="Number of test examples")
    parser.add_argument(
        "--model",
        choices=["small", "base"],
        default="small",
        help="Generator model: small (flan-t5-small) or base (flan-t5-base)",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=config.PATHS["results"] / "main.jsonl",
        help="Path for main JSONL output",
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
    print(f"HEALGEN MAIN BENCHMARK RUN: N={args.n}, Generator={args.model}")
    print("=" * 70)

    # 1. Load data
    print("\n[1/5] Loading data and building frozen BM25 index...")
    dev_exs = data.load_examples("dev")
    test_exs = data.load_examples("test")[: args.n]
    all_exs = dev_exs + test_exs
    corpus = data.build_corpus(all_exs)
    bm25 = data.BM25Index(corpus)
    ex_by_id = {ex["id"]: ex for ex in test_exs}

    # 2. Initialize models
    print(f"\n[2/5] Initializing generator ({args.model})...")
    gen = gen_module.get_generator(smoke=(args.model == "small"))

    print(f"\n[3/5] Initializing judge ({config.JUDGE_MODEL})...")
    judge = Judge(config.JUDGE_MODEL)

    print("\n[4/5] Initializing CRE risk estimator and AESP sufficiency predictor...")
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

    # 3. Resumable execution
    systems = ["vanilla", "rag", "cove", "healgen"]
    existing_records, completed_pairs = load_existing_records(args.out_jsonl)
    print(f"\nFound {len(existing_records)} existing records ({len(completed_pairs)} pairs).")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl_file = args.out_jsonl.open("a", encoding="utf-8")

    all_records = list(existing_records)
    total_to_run = len(test_exs) * len(systems) - len(completed_pairs)
    print(f"Executing remaining {total_to_run} system runs...")

    run_count = 0
    t0_main = time.perf_counter()

    try:
        for idx, ex in enumerate(test_exs):
            qid = ex["id"]
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  Processing example {idx + 1}/{len(test_exs)} (qid={qid})...")
            for sys_name in systems:
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

    elapsed_main = time.perf_counter() - t0_main
    print(f"\nFinished running {run_count} runs in {elapsed_main:.1f}s.")
    print(f"Total dataset records: {len(all_records)}")

    # 4. Frozen-pool evaluation rescoring (ledger R12: the plan's Global
    #    Constraint -- "Judge never scores against system-private evidence").
    #    Verdicts are attached on the *emitted* claims (decompose(answer)),
    #    against the per-qid shared pool (gold ∪ union of retrievals) and a
    #    gold-only condition.  main.jsonl keeps the raw records untouched.
    print("\n[5/5] Rescoring claims against frozen evidence pools (R12)...")
    gold_by_qid = {qid: str(ex.get("knowledge", "")) for qid, ex in ex_by_id.items()}
    pools = build_pools(all_records, gold_by_qid)
    pools_path = config.PATHS["results"] / "evidence_pools.json"
    save_pools(pools, pools_path)
    print(f"Wrote frozen pools for {len(pools)} qids to {pools_path}")
    attach_eval_labels(all_records, pools, gold_by_qid, judge)
    ev_records = label_view(all_records, "ev")
    gold_records = label_view(all_records, "gold")
    n_eval_claims = sum(len(r["eval_claims"]) for r in all_records)
    print(f"Rescored {n_eval_claims} emitted claims (shared pool + gold-only).")

    # 5. Generate results/main.csv (per-example metrics; shared-pool labels)
    main_csv_path = config.PATHS["results"] / "main.csv"
    per_example_rows: list[dict[str, Any]] = []

    for r in ev_records:
        qid = r["qid"]
        ex = ex_by_id.get(qid, {})
        gold_answer = ex.get("right_answer", "")
        pred_answer = r["answer"]

        claims = r.get("claims", [])
        num_claims = len(claims)
        num_sup = sum(1 for c in claims if c.get("supported"))
        sp_m = (num_sup / num_claims) if num_claims > 0 else 1.0
        any_halluc = int(any(not c.get("supported") for c in claims))

        em_res = em_f1([pred_answer], [gold_answer])

        per_example_rows.append(
            {
                "qid": qid,
                "system": r["system"],
                "question": r["question"],
                "answer": pred_answer,
                "num_claims": num_claims,
                "num_supported_claims": num_sup,
                "sp_macro": round(sp_m, 4),
                "any_hallucination": any_halluc,
                "num_bm25_calls": r.get("num_bm25_calls", 0),
                "latency_s": round(r.get("latency_s", 0.0), 3),
                "abstained": int(r.get("abstained", False)),
                "em": round(em_res["em"], 4),
                "f1": round(em_res["f1"], 4),
            }
        )

    with main_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_example_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_example_rows)
    print(f"Wrote {len(per_example_rows)} rows to {main_csv_path}")

    # Claim-level rows (feeds Task 16's reliability diagram + spot-check
    # export): one row per *emitted* claim with the evaluation verdicts
    # (shared-pool `supported`/`score` + gold-only condition) and the
    # pipeline metadata joined by exact text match (blank for rephrased
    # claims, whose emitted text has no draft-side counterpart).
    claims_csv_path = config.PATHS["results"] / "claims.csv"
    claim_rows: list[dict[str, Any]] = []
    for r in all_records:
        pipeline_by_text: dict[str, dict] = {
            str(c.get("text", "")): c for c in r.get("claims", [])
        }
        pool_text = " || ".join(pools.get(r["qid"], []))
        for ec in r.get("eval_claims", []):
            pipe = pipeline_by_text.get(str(ec["text"]), {})
            claim_rows.append(
                {
                    "qid": r["qid"],
                    "system": r["system"],
                    "claim_id": ec["id"],
                    "claim_text": ec["text"],
                    "score": round(float(ec["ev_score"]), 4),
                    "supported": int(bool(ec["ev_supported"])),
                    "gold_score": round(float(ec["gold_score"]), 4),
                    "gold_supported": int(bool(ec["gold_supported"])),
                    "pipeline_risk": (
                        round(float(pipe["risk"]), 4) if "risk" in pipe else ""
                    ),
                    "pipeline_sufficient": (
                        int(bool(pipe["sufficient"])) if "sufficient" in pipe else ""
                    ),
                    "pipeline_action": pipe.get("action", ""),
                    "evidence_pool": pool_text,
                }
            )
    with claims_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(claim_rows[0].keys()))
        writer.writeheader()
        writer.writerows(claim_rows)
    print(f"Wrote {len(claim_rows)} claim rows to {claims_csv_path}")

    # 6. System Summary Table (system_summary.csv; shared-pool labels + a
    #    gold-only SP column for the plan's second evaluation condition)
    summary_path = config.PATHS["results"] / "system_summary.csv"
    summary_rows: list[dict[str, Any]] = []

    sys_records: dict[str, list[dict]] = {s: [] for s in systems}
    for r in ev_records:
        if r["system"] in sys_records:
            sys_records[r["system"]].append(r)

    for s in systems:
        recs = sys_records[s]
        n_items = len(recs)
        if n_items == 0:
            continue

        sp_mac = sp_macro(recs)
        # Bootstrap CI on sp_macro
        per_rec_sp = [
            (sum(1.0 for c in r.get("claims", []) if c.get("supported")) / len(r.get("claims", [])))
            if r.get("claims")
            else 1.0
            for r in recs
        ]
        ci_lo, ci_hi = bootstrap_ci(per_rec_sp, n=2000, seed=config.SEED_TEST)

        sp_mic = sp_micro(recs)
        hr = hallucination_rate(recs)
        any_hr = any_hallucination_rate(recs)

        # Second evaluation condition (plan line 19: "plus a gold-only
        # condition"): same records, labels judged against gold alone.
        gold_recs = [g for g in gold_records if g["system"] == s]
        sp_gold = sp_macro(gold_recs) if gold_recs else 0.0

        # QA metrics EM and F1
        preds = [r["answer"] for r in recs]
        golds = [ex_by_id.get(r["qid"], {}).get("right_answer", "") for r in recs]
        qa_metrics = em_f1(preds, golds)

        # Calibration on all claims of this system
        all_confs = [float(c.get("score", 0.0)) for r in recs for c in r.get("claims", [])]
        all_correct = [int(bool(c.get("supported", False))) for r in recs for c in r.get("claims", [])]
        ece_val = ece(all_confs, all_correct, n_bins=10)
        brier_val = brier(all_confs, all_correct)

        bm25_calls = [r.get("num_bm25_calls", 0) for r in recs]
        lats = [r.get("latency_s", 0.0) for r in recs]
        abstains = [int(bool(r.get("abstained", False))) for r in recs]

        summary_rows.append(
            {
                "system": s,
                "n_examples": n_items,
                "sp_macro": round(sp_mac, 4),
                "sp_macro_ci_lo": round(ci_lo, 4),
                "sp_macro_ci_hi": round(ci_hi, 4),
                "sp_micro": round(sp_mic, 4),
                "sp_macro_goldonly": round(sp_gold, 4),
                "hallucination_rate": round(hr, 4),
                "any_hallucination_rate": round(any_hr, 4),
                "em": round(qa_metrics["em"], 4),
                "f1": round(qa_metrics["f1"], 4),
                "ece": round(ece_val, 4),
                "brier": round(brier_val, 4),
                "mean_bm25_calls": round(float(np.mean(bm25_calls)), 2),
                "mean_latency_s": round(float(np.mean(lats)), 2),
                "p50_latency_s": round(float(np.percentile(lats, 50)), 2),
                "p95_latency_s": round(float(np.percentile(lats, 95)), 2),
                "abstention_rate": round(float(np.mean(abstains)), 4),
            }
        )

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote system summary to {summary_path}")

    # 7. Pairwise Hypothesis Tests (pairwise_tests.csv; shared-pool labels)
    pairwise_path = config.PATHS["results"] / "pairwise_tests.csv"
    comparisons = [
        ("healgen", "vanilla"),
        ("healgen", "rag"),
        ("healgen", "cove"),
        ("cove", "rag"),
        ("cove", "vanilla"),
        ("rag", "vanilla"),
    ]

    pairwise_raw: list[dict[str, Any]] = []
    pvals_to_correct: list[float] = []

    for sys_a, sys_b in comparisons:
        recs_a = sys_records[sys_a]
        recs_b = sys_records[sys_b]
        test_res = paired_bootstrap(recs_a, recs_b, sp_macro, n=2000, seed=config.SEED_TEST)
        pvals_to_correct.append(test_res["p_value"])
        pairwise_raw.append(
            {
                "comparison": f"{sys_a} vs {sys_b}",
                "system_a": sys_a,
                "system_b": sys_b,
                "metric": "sp_macro",
                "delta": round(test_res["delta"], 4),
                "ci_lo": round(test_res["ci"][0], 4),
                "ci_hi": round(test_res["ci"][1], 4),
                "raw_p_value": round(test_res["p_value"], 5),
            }
        )

    adj_pvals = holm_bonferroni(pvals_to_correct)
    for i, row in enumerate(pairwise_raw):
        row["adjusted_p_value"] = round(adj_pvals[i], 5)

    with pairwise_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairwise_raw[0].keys()))
        writer.writeheader()
        writer.writerows(pairwise_raw)
    print(f"Wrote pairwise significance tests to {pairwise_path}")

    # Print summary tables
    print("\n" + "=" * 100)
    print("MAIN BENCHMARK SYSTEM SUMMARY (frozen shared-pool labels; SP_gold = gold-only condition)")
    print("=" * 100)
    print(
        f"{'System':<12} | {'SP_Macro [95% CI]':<22} | {'SP_gold':<8} | {'HR':<6} | {'EM':<6} | {'F1':<6} | {'BM25 Calls':<10}"
    )
    print("-" * 100)
    for sr in summary_rows:
        ci_str = f"{sr['sp_macro']:.3f} [{sr['sp_macro_ci_lo']:.3f}, {sr['sp_macro_ci_hi']:.3f}]"
        print(
            f"{sr['system']:<12} | {ci_str:<22} | {sr['sp_macro_goldonly']:<8.3f} | {sr['hallucination_rate']:<6.3f} | "
            f"{sr['em']:<6.3f} | {sr['f1']:<6.3f} | {sr['mean_bm25_calls']:<10.2f}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
