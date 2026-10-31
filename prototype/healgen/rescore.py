"""Frozen-pool evaluation rescoring (ledger ruling R12).

The plan's Global Constraint (prototype plan line 19): *"Judge never scores
against system-private evidence: frozen shared pool per question = gold
knowledge ∪ union of all four systems' top-3 passages (plus a gold-only
condition)."*

``run_system`` legitimately records each system's *in-pipeline* judgments
(they drive HealGen's control flow), but those labels are not valid
evaluation verdicts: vanilla judges against an empty pool (structurally
unsupported) and RAG judges against the very passages it generated from
(self-retrieval circularity).  Worse, HealGen's recorded claims are derived
from the *draft* while its emitted answer is post-repair (73/150 records
diverge after localized repair), so scoring them would measure the
pre-repair state.  This module attaches evaluation verdicts after the fact,
so generation is never re-run:

- ``build_pools(records, gold_by_qid)`` -- the frozen shared pool per qid,
  gold first, deduped (delegates to :func:`healgen.judge.build_pool`).
- ``attach_eval_labels(records, pools, gold_by_qid, judge)`` -- judges
  ``decompose(record["answer"])`` (the *emitted* response's claims), batched
  per qid through ``judge.evaluate_claims`` (identical semantics to
  ``judge.support``, ledger R8 hybrid rule): one call against the shared
  pool (``ev_*``) and one against gold-only (``gold_*``), attached to a new
  ``eval_claims`` field.  Recorded claims are never mutated.
- ``label_view(records, prefix)`` -- metric-ready copies whose ``claims``
  are the eval claims with ``supported``/``score`` overridden from the
  chosen label set, so the committed metrics functions (which read
  ``supported``) consume the evaluation verdicts unchanged.
- ``save_pools``/``load_pools`` -- ``results/evidence_pools.json`` is the
  hand-off: run_main writes it, run_ablations reads it so ablation variants
  score against the *same* pools as the four main systems (paired fairness).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from healgen.claims import decompose
from healgen.judge import build_pool


def build_pools(
    records: list[dict[str, Any]], gold_by_qid: dict[str, str]
) -> dict[str, list[str]]:
    """Frozen shared pool per qid: gold first, then the deduped passage union.

    Passages are unioned across every record passed in -- for the main run
    that is the four systems' retrievals, exactly as the plan pins it.
    Qids with no records do not appear.
    """
    union_by_qid: dict[str, list[str]] = {}
    for rec in records:
        qid = str(rec["qid"])
        passages = rec.get("retrieved") or []
        union_by_qid.setdefault(qid, []).extend(p for p in passages if p)
    return {
        qid: build_pool(gold_by_qid.get(qid, ""), passages)
        for qid, passages in union_by_qid.items()
    }


def _emitted_texts(record: dict[str, Any]) -> list[str]:
    """Atomic claims of the emitted answer (what evaluation must judge)."""
    answer = str(record.get("answer", ""))
    return [t for t in decompose(answer) if t and t.strip()]


def attach_eval_labels(
    records: list[dict[str, Any]],
    pools: dict[str, list[str]],
    gold_by_qid: dict[str, str],
    judge: Any,
) -> None:
    """Attach ``eval_claims`` with ``ev_*`` (shared pool) and ``gold_*`` labels.

    Every record gets an ``eval_claims`` list -- one entry per atomic claim
    of its emitted answer, ``{"id", "text", "ev_supported", "ev_score",
    "gold_supported", "gold_score"}``.  Claims are batched per qid: one
    ``evaluate_claims`` call for the shared pool and one for gold-only.
    Raises ``KeyError`` when a record's qid has no pool -- callers must
    build pools first.  Idempotent: re-running overwrites with the same
    values.  Recorded pipeline claims are never mutated.
    """
    # 1. Derive the emitted claims per record (deterministic, model-free).
    items_by_qid: dict[str, list[tuple[dict[str, Any], list[str]]]] = {}
    for rec in records:
        qid = str(rec["qid"])
        texts = _emitted_texts(rec)
        rec["eval_claims"] = []  # every record gets the key, even empty answers
        if texts:
            items_by_qid.setdefault(qid, []).append((rec, texts))

    # 2. Batched judge calls per qid: shared pool, then gold-only.
    for qid, items in items_by_qid.items():
        pool = pools[qid]  # KeyError if the qid was never pooled (intended)
        flat = [t for _, texts in items for t in texts]
        gold_pool = build_pool(gold_by_qid.get(qid, ""), [])

        shared = judge.evaluate_claims(flat, pool)
        gold = judge.evaluate_claims(flat, gold_pool)

        k = 0
        for rec, texts in items:
            eval_claims = []
            for text in texts:
                ev, gd = shared[k], gold[k]
                k += 1
                eval_claims.append(
                    {
                        "id": f"e{len(eval_claims)}",
                        "text": text,
                        "ev_supported": bool(ev[0]),
                        "ev_score": float(ev[1]),
                        "gold_supported": bool(gd[0]),
                        "gold_score": float(gd[1]),
                    }
                )
            rec["eval_claims"] = eval_claims


def label_view(records: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    """Metric-ready copies of records with ``claims`` from the chosen labels.

    ``prefix`` is ``"ev"`` (shared-pool evaluation) or ``"gold"`` (gold-only
    condition): ``claims`` becomes the record's ``eval_claims`` with
    ``supported``/``score`` overridden from ``<prefix>_supported``/
    ``<prefix>_score``.  ``prefix == ""`` falls back to the recorded
    pipeline claims (identity copy).  Originals are never mutated.
    """
    out: list[dict[str, Any]] = []
    for rec in records:
        rec2 = dict(rec)
        if prefix:
            if "eval_claims" not in rec:
                raise ValueError(
                    "record has no eval_claims -- run attach_eval_labels first"
                )
            rec2["claims"] = [
                {
                    "id": ec["id"],
                    "text": ec["text"],
                    "supported": bool(ec[f"{prefix}_supported"]),
                    "score": float(ec[f"{prefix}_score"]),
                }
                for ec in rec["eval_claims"]
            ]
        else:
            rec2["claims"] = [dict(c) for c in rec.get("claims", [])]
        out.append(rec2)
    return out


def save_pools(pools: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pools, indent=1, sort_keys=True), encoding="utf-8")


def load_pools(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run run_main.py first (it writes the frozen "
            "evidence pools before any downstream run may score claims)."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): [str(p) for p in v] for k, v in data.items()}
