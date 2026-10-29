"""Judge validation experiment (Task 13).

Validates the primary and secondary NLI judges on HaluEval contrast pairs
(seed 44, N=100 items = 200 responses) under the gold-only condition.
Produces results/judge_validation.json and enforces the honesty gate (AUC >= 0.70).
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

from healgen import config, data  # noqa: E402
from healgen.judge import Judge, eval_contrast_pairs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge Validation Gate")
    parser.add_argument(
        "--out",
        type=Path,
        default=config.PATHS["results"] / "judge_validation.json",
        help="Path for output validation results JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("JUDGE VALIDATION EXPERIMENT (Task 13 Honesty Gate)")
    print("=" * 70)

    # 1. Load judge split (seed 44, 100 contrast pairs)
    print("\n[1/3] Loading judge split (N=100 pairs, seed 44)...")
    judge_examples = data.load_examples("judge")
    print(f"Loaded {len(judge_examples)} contrast pairs.")

    results: dict[str, dict] = {}

    # 2. Evaluate Primary Judge
    print(f"\n[2/3] Evaluating Primary Judge: {config.JUDGE_MODEL}...")
    t0 = time.perf_counter()
    judge1 = Judge(config.JUDGE_MODEL)
    res1 = eval_contrast_pairs(judge1, judge_examples)
    res1["time_s"] = round(time.perf_counter() - t0, 2)
    res1["model"] = config.JUDGE_MODEL
    results["judge1"] = res1
    print(
        f"  Primary -> Accuracy: {res1['accuracy']:.4f}, AUC: {res1['auc']:.4f}, "
        f"N: {res1['n']}, Time: {res1['time_s']}s"
    )

    # 3. Evaluate Secondary Judge
    print(f"\n[3/3] Evaluating Secondary Judge: {config.JUDGE2_MODEL}...")
    t0 = time.perf_counter()
    judge2 = Judge(config.JUDGE2_MODEL)
    res2 = eval_contrast_pairs(judge2, judge_examples)
    res2["time_s"] = round(time.perf_counter() - t0, 2)
    res2["model"] = config.JUDGE2_MODEL
    results["judge2"] = res2
    print(
        f"  Secondary -> Accuracy: {res2['accuracy']:.4f}, AUC: {res2['auc']:.4f}, "
        f"N: {res2['n']}, Time: {res2['time_s']}s"
    )

    # 4. Check gate condition (AUC >= 0.70)
    gate_primary = res1["auc"] >= 0.70
    gate_secondary = res2["auc"] >= 0.70
    gate_passed = gate_primary or gate_secondary

    summary = {
        "judge1": res1,
        "judge2": res2,
        "gate_threshold_auc": 0.70,
        "gate_passed": gate_passed,
        "primary_selected": config.JUDGE_MODEL if gate_primary else config.JUDGE2_MODEL,
    }

    # 5. Save results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved validation results to {args.out}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Primary ({config.JUDGE_MODEL}):")
    print(f"  Accuracy: {res1['accuracy']:.4f} | AUC: {res1['auc']:.4f}")
    print(f"Secondary ({config.JUDGE2_MODEL}):")
    print(f"  Accuracy: {res2['accuracy']:.4f} | AUC: {res2['auc']:.4f}")
    print(f"Gate Condition (AUC >= 0.70): {'PASSED [OK]' if gate_passed else 'FAILED'}")
    print(f"Selected Primary Judge for Benchmark: {summary['primary_selected']}")
    print("=" * 70)

    if not gate_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
