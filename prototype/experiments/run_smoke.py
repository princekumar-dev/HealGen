"""Smoke integration gate (Task 12).

Runs all 4 systems (vanilla, rag, cove, healgen) on N examples with real models
to verify the full end-to-end pipeline, write results/smoke.jsonl, and project
total wall-clock runtime for the main N=150 run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
if str(PROTOTYPE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT))

from healgen import config, data, generator as gen_module  # noqa: E402
from healgen.judge import Judge  # noqa: E402
from healgen.risk import RiskEstimator  # noqa: E402
from healgen.sufficiency import Sufficiency  # noqa: E402
from healgen.systems import Deps, run_system  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HealGen Smoke Integration Gate")
    parser.add_argument("--n", type=int, default=5, help="Number of examples to evaluate")
    parser.add_argument(
        "--model",
        choices=["small", "base"],
        default="small",
        help="Generator model size: small (flan-t5-small) or base (flan-t5-base)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=config.PATHS["results"] / "smoke.jsonl",
        help="Output path for smoke records JSONL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print(f"HEALGEN SMOKE GATE: N={args.n}, Generator={args.model}")
    print("=" * 70)

    t_start = time.perf_counter()

    # 1. Load data & corpus
    print("\n[1/5] Loading data and building BM25 index...")
    dev_exs = data.load_examples("dev")
    test_exs = data.load_examples("test")
    all_exs = dev_exs + test_exs
    corpus = data.build_corpus(all_exs)
    bm25 = data.BM25Index(corpus)
    eval_examples = test_exs[: args.n]
    print(f"Loaded {len(eval_examples)} evaluation examples, corpus size = {len(corpus)}")

    # 2. Load models
    print(f"\n[2/5] Initializing generator ({args.model})...")
    gen = gen_module.get_generator(smoke=(args.model == "small"))

    print(f"\n[3/5] Initializing judge ({config.JUDGE_MODEL})...")
    judge = Judge(config.JUDGE_MODEL)

    print("\n[4/5] Initializing CRE risk estimator and AESP sufficiency predictor...")
    cre_path = config.PATHS["results"] / "cre_model.json"
    if cre_path.exists():
        estimator = RiskEstimator.load(cre_path)
    else:
        print("Warning: cre_model.json not found, using uninitialized estimator")
        estimator = RiskEstimator()

    # Load optimal tau from sensitivity table if available, else 0.7
    tau = 0.7
    sufficiency = Sufficiency(tau=tau)

    deps = Deps(
        generator=gen,
        judge=judge,
        bm25=bm25,
        estimator=estimator,
        sufficiency=sufficiency,
    )

    # 3. Run systems
    systems = ["vanilla", "rag", "cove", "healgen"]
    records: list[dict] = []
    latencies: dict[str, list[float]] = {s: [] for s in systems}

    print(f"\n[5/5] Executing 4 systems across {args.n} examples...")
    for idx, ex in enumerate(eval_examples):
        qid = ex["id"]
        print(f"  Example {idx + 1}/{args.n} (qid={qid})...")
        for sys_name in systems:
            rec = run_system(sys_name, ex, deps)
            records.append(rec)
            latencies[sys_name].append(rec["latency_s"])

    # 4. Save JSONL
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nSuccessfully wrote {len(records)} records to {args.out}")

    # 5. Timing and Projection
    total_elapsed = time.perf_counter() - t_start
    total_gen_tokens = gen.n_tokens
    tok_per_sec = total_gen_tokens / total_elapsed if total_elapsed > 0 else 0.0

    print("\n" + "=" * 70)
    print("SMOKE TEST RESULTS & RUNTIME PROJECTIONS")
    print("=" * 70)
    print(f"{'System':<15} | {'Mean Latency (s)':<18} | {'Min (s)':<10} | {'Max (s)':<10}")
    print("-" * 62)
    sum_mean_latencies = 0.0
    for s in systems:
        lats = latencies[s]
        mean_l = sum(lats) / len(lats)
        sum_mean_latencies += mean_l
        print(f"{s:<15} | {mean_l:<18.2f} | {min(lats):<10.2f} | {max(lats):<10.2f}")
    print("-" * 62)

    projected_seconds_150 = sum_mean_latencies * 150
    projected_hours_150 = projected_seconds_150 / 3600.0

    print(f"\nTotal smoke elapsed time : {total_elapsed:.1f} s")
    print(f"Generated tokens count   : {total_gen_tokens} tokens")
    print(f"Mean Generation Speed    : {tok_per_sec:.2f} tokens/s")
    print(f"Projected N=150 runtime  : {projected_seconds_150:.1f} s (~{projected_hours_150:.2f} hours)")

    gate_passed = projected_hours_150 < 12.0
    print(f"Gate Condition (< 12 h)  : {'PASSED [OK]' if gate_passed else 'FAILED [EXCEEDED]'}")
    print("=" * 70)

    if not gate_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
