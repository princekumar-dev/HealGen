"""Tune the AESP sufficiency threshold tau on the HaluEval dev split.

Run from the repository root:
    python prototype/experiments/tune_tau.py

Pipeline:
1. Load the 50 pinned dev examples (seed 43).
2. Decompose both right and hallucinated answers into atomic claims.
3. Judge each claim against gold knowledge (and contrast with unrelated passages)
   using the pinned NLI Judge to establish gold sufficiency ground truth.
4. Tune tau in {0.3, 0.5, 0.7} to maximize F1.
5. Save the sensitivity table to `results/tau_sensitivity.csv`.
"""

import csv
from pathlib import Path
import sys
import time

PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
if str(PROTOTYPE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT))

from healgen import claims, config, data  # noqa: E402
from healgen.judge import Judge  # noqa: E402
from healgen.sufficiency import tune_tau  # noqa: E402


def build_dev_pairs(examples: list[dict], judge: Judge) -> list[tuple[str, list[str], int]]:
    """Build (claim, evidence_list, label) triples from dev split."""
    pairs: list[tuple[str, list[str], int]] = []
    total_claims = 0
    t0 = time.perf_counter()

    for idx, ex in enumerate(examples):
        gold_knowledge = [ex["knowledge"]]
        
        # 1. Right answer claims against gold knowledge
        right_claims = claims.decompose(ex["right_answer"])
        for c in right_claims:
            if not c.strip():
                continue
            is_supp, _ = judge.support(c, gold_knowledge)
            pairs.append((c, gold_knowledge, 1 if is_supp else 0))
            total_claims += 1

        # 2. Hallucinated answer claims against gold knowledge
        halluc_claims = claims.decompose(ex["hallucinated_answer"])
        for c in halluc_claims:
            if not c.strip():
                continue
            is_supp, _ = judge.support(c, gold_knowledge)
            pairs.append((c, gold_knowledge, 1 if is_supp else 0))
            total_claims += 1

        # 3. Negative contrast: right answer claim paired with an unrelated dev passage
        if right_claims and len(examples) > 1:
            unrelated_idx = (idx + 1) % len(examples)
            unrelated_knowledge = [examples[unrelated_idx]["knowledge"]]
            pairs.append((right_claims[0], unrelated_knowledge, 0))
            total_claims += 1

    elapsed = time.perf_counter() - t0
    print(f"Constructed {len(pairs)} dev evaluation pairs in {elapsed:.2f}s")
    return pairs


def main() -> None:
    print("Loading dev split (N=50, seed 43)...")
    dev_examples = data.load_examples("dev")

    print(f"Loading judge ({config.JUDGE_MODEL})...")
    judge = Judge(config.JUDGE_MODEL)

    print("Building and labeling dev pairs...")
    dev_pairs = build_dev_pairs(dev_examples, judge)

    out_csv = config.PATHS["results"] / "tau_sensitivity.csv"
    print(f"Tuning tau over candidate set {{0.3, 0.5, 0.7}} -> {out_csv}...")
    chosen_tau = tune_tau(
        dev_pairs,
        output_path=out_csv,
        candidate_taus=(0.3, 0.5, 0.7),
    )

    print(f"\nOptimal tau selected: {chosen_tau}")
    print("\n--- Sensitivity Results ---")
    with out_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            marker = " *" if row.get("selected") == "True" else ""
            print(
                f"tau={row['tau']}: Precision={row['precision']}, "
                f"Recall={row['recall']}, F1={row['f1']}, Acc={row['accuracy']}{marker}"
            )


if __name__ == "__main__":
    main()
