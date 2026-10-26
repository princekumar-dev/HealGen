"""Export 30 seeded test-split responses for manual claim-decomposition audit.

The sample is drawn with ``random.Random(0)`` from the pinned
``data.load_examples("test")`` order over each example's ``right_answer``.
Output is ``results/audit_claims.csv`` with columns::

    id, response, claims_json, manual_verdict, notes

``manual_verdict`` and ``notes`` are left empty for the human reviewer.  The
script only creates the results directory when missing and writes this one
file; it never deletes or rewrites other artefacts.  Re-running it is
deterministic (same ids, same claims).

Run from the repository root::

    python prototype/experiments/audit_claims.py
"""

import csv
import json
import random
import sys
from pathlib import Path

PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
if str(PROTOTYPE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT))

from healgen import claims, config, data  # noqa: E402  (path set above)

N_SAMPLE = 30
SEED = 0
OUTPUT_NAME = "audit_claims.csv"
COLUMNS = ("id", "response", "claims_json", "manual_verdict", "notes")


def sample_examples(n: int = N_SAMPLE, seed: int = SEED) -> list[dict]:
    """Deterministic sample of test-split examples (pinned order, own RNG)."""
    examples = data.load_examples("test")
    rng = random.Random(seed)
    return rng.sample(examples, min(n, len(examples)))


def build_rows(examples: list[dict]) -> list[dict]:
    """One audit row per example, with the decomposition serialised as JSON."""
    rows: list[dict] = []
    for example in examples:
        response = example["right_answer"]
        rows.append(
            {
                "id": example["id"],
                "response": response,
                "claims_json": json.dumps(
                    claims.decompose(response), ensure_ascii=False
                ),
                "manual_verdict": "",
                "notes": "",
            }
        )
    return rows


def main() -> None:
    results_dir = Path(config.PATHS["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / OUTPUT_NAME

    rows = build_rows(sample_examples())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
