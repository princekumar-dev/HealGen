"""Export the human spot-check sample (Task 16).

Reads results/claims.csv (produced by run_main.py) and samples 100 claims
stratified by system (25 per system, deterministic stride) with the frozen
evidence pool shown alongside and empty columns for the human rater. The
rater fills `human_supported` (ideally WITHOUT consulting `judge_supported`
-- blindness keeps the agreement estimate honest); inter-annotator
agreement (Cohen's kappa vs. the judge) is computed later from this file.

Output: results/spotcheck_sample.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]
if str(PROTOTYPE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_ROOT))

from healgen import config  # noqa: E402

PER_SYSTEM = 25  # 25 x 4 systems = 100 claims


def main() -> None:
    src = config.PATHS["results"] / "claims.csv"
    if not src.exists():
        sys.exit(f"ERROR: required results file missing: {src}")
    with src.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_system: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_system.setdefault(r["system"], []).append(r)

    sample: list[dict[str, str]] = []
    for system in sorted(by_system):
        pool = by_system[system]
        # Deterministic even-spaced stride over the system's rows (stable under
        # row-order changes from resumable runs; no RNG state to pin).
        n = min(PER_SYSTEM, len(pool))
        if n == 0:
            continue
        stride = max(1, len(pool) // n)
        picked = pool[::stride][:n]
        for r in picked:
            sample.append(
                {
                    "qid": r["qid"],
                    "system": r["system"],
                    "claim_id": r["claim_id"],
                    "claim_text": r["claim_text"],
                    "evidence_pool": r.get("evidence_pool", ""),
                    "judge_supported": r["supported"],
                    "judge_score": r["score"],
                    "human_supported": "",  # rater fills: 1 / 0 / NA
                    "rater_notes": "",
                }
            )

    out = config.PATHS["results"] / "spotcheck_sample.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample[0].keys()))
        writer.writeheader()
        writer.writerows(sample)
    print(f"Wrote {len(sample)} spot-check claims to {out}")


if __name__ == "__main__":
    main()
