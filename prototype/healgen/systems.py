"""Systems orchestration layer (Task 10).

Orchestrates the four paired benchmark systems:
- vanilla: generation only, no retrieval.
- rag: standard RAG (BM25 top-3 always -> generation with context).
- cove: Chain-of-Verification (draft -> per-claim verification -> single full regeneration if unsupported).
- healgen: localized self-healing pipeline (draft -> per-claim risk & sufficiency ->
  targeted re-retrieval -> DCEG -> LSHE localized repair). Also supports ablation systems:
  healgen-noCRE, healgen-noAESP, healgen-noLSHE.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from healgen import config
from healgen.claims import decompose
from healgen.graph import DCEG
from healgen.repair import repair_answer


@dataclass
class Deps:
    """Dependency bundle for injecting real models or test fakes."""

    generator: Any
    judge: Any = None
    bm25: Any = None
    estimator: Any = None
    sufficiency: Any = None


def _extract_passages(hits: Any) -> list[str]:
    """Extract plain text passages from BM25 hits (tuples or raw strings)."""
    passages: list[str] = []
    if not hits:
        return passages
    for h in hits:
        if isinstance(h, tuple) and len(h) >= 2:
            passages.append(str(h[1]))
        elif isinstance(h, str):
            passages.append(h)
    return passages


def _dedup_passages(existing: list[str], new_passages: list[str]) -> list[str]:
    """Append new passages preserving order and uniqueness."""
    seen = set(existing)
    combined = list(existing)
    for p in new_passages:
        if p not in seen:
            seen.add(p)
            combined.append(p)
    return combined


def _generate_answer(generator: Any, question: str, passages: list[str] | None = None) -> str:
    """Prompt the generator with or without retrieved passage context."""
    if passages:
        context = "\n".join(passages)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"
    return generator.generate(prompt)


def run_system(system: str, ex: dict, deps: Deps) -> dict:
    """Execute a system on an example and produce a validated record.

    Parameters
    ----------
    system : str
        One of "vanilla", "rag", "cove", "healgen", or ablation variants
        ("healgen-noCRE", "healgen-noAESP", "healgen-noLSHE").
    ex : dict
        Example containing "id" (or "qid") and "question". Gold "knowledge"
        is strictly ignored at inference time.
    deps : Deps
        Injected dependencies (generator, judge, bm25, estimator, sufficiency).

    Returns
    -------
    dict
        Record matching the pinned schema.
    """
    valid_systems = {
        "vanilla",
        "rag",
        "cove",
        "healgen",
        "healgen-noCRE",
        "healgen-noAESP",
        "healgen-noLSHE",
    }
    if system not in valid_systems:
        raise ValueError(f"Unknown system '{system}'. Must be one of {valid_systems}")

    qid = str(ex.get("qid", ex.get("id", "")))
    question = str(ex["question"])

    t0 = time.perf_counter()

    num_bm25_calls = 0
    retrieved: list[str] = []
    claims_data: list[dict[str, Any]] = []

    if system == "vanilla":
        answer = deps.generator.generate(f"Question: {question}\nAnswer:")
        claim_texts = decompose(answer)
        for idx, c in enumerate(claim_texts):
            risk = (
                float(deps.estimator.predict_proba(c, question, deps.bm25))
                if deps.estimator
                else 0.0
            )
            score = 0.0
            supported = False
            if deps.judge:
                supported, score = deps.judge.support(c, [])
            claims_data.append(
                {
                    "id": f"c{idx}",
                    "text": c,
                    "risk": float(risk),
                    "sufficient": False,
                    "score": float(score),
                    "supported": bool(supported),
                    "action": "none",
                }
            )

    elif system == "rag":
        if deps.bm25:
            hits = deps.bm25.search(question, k=config.TOPK)
            num_bm25_calls += 1
            retrieved = _extract_passages(hits)
        answer = _generate_answer(deps.generator, question, retrieved)
        claim_texts = decompose(answer)
        for idx, c in enumerate(claim_texts):
            risk = (
                float(deps.estimator.predict_proba(c, question, deps.bm25))
                if deps.estimator
                else 0.0
            )
            sufficient = (
                deps.sufficiency.is_sufficient(c, retrieved)
                if deps.sufficiency
                else bool(retrieved)
            )
            score = 0.0
            supported = False
            if deps.judge:
                supported, score = deps.judge.support(c, retrieved)
            action = "supported" if supported else "none"
            claims_data.append(
                {
                    "id": f"c{idx}",
                    "text": c,
                    "risk": float(risk),
                    "sufficient": bool(sufficient),
                    "score": float(score),
                    "supported": bool(supported),
                    "action": action,
                }
            )

    elif system == "cove":
        draft = deps.generator.generate(f"Question: {question}\nAnswer:")
        initial_claims = decompose(draft)
        claim_evals: list[tuple[str, bool, float]] = []

        for c in initial_claims:
            claim_passages: list[str] = []
            if deps.bm25:
                hits = deps.bm25.search(c, k=config.TOPK)
                num_bm25_calls += 1
                claim_passages = _extract_passages(hits)
                retrieved = _dedup_passages(retrieved, claim_passages)
            if deps.judge:
                sup, sc = deps.judge.support(c, claim_passages)
            else:
                sup, sc = (True, 1.0)
            claim_evals.append((c, sup, sc))

        any_unsupported = any(not sup for _, sup, _ in claim_evals)
        if any_unsupported:
            answer = _generate_answer(deps.generator, question, retrieved)
            final_claims = decompose(answer)
            for idx, c in enumerate(final_claims):
                risk = (
                    float(deps.estimator.predict_proba(c, question, deps.bm25))
                    if deps.estimator
                    else 0.0
                )
                sufficient = (
                    deps.sufficiency.is_sufficient(c, retrieved)
                    if deps.sufficiency
                    else bool(retrieved)
                )
                if deps.judge:
                    sup, sc = deps.judge.support(c, retrieved)
                else:
                    sup, sc = (True, 1.0)
                action = "supported" if sup else "rephrase"
                claims_data.append(
                    {
                        "id": f"c{idx}",
                        "text": c,
                        "risk": float(risk),
                        "sufficient": bool(sufficient),
                        "score": float(sc),
                        "supported": bool(sup),
                        "action": action,
                    }
                )
        else:
            answer = draft
            for idx, (c, sup, sc) in enumerate(claim_evals):
                risk = (
                    float(deps.estimator.predict_proba(c, question, deps.bm25))
                    if deps.estimator
                    else 0.0
                )
                sufficient = (
                    deps.sufficiency.is_sufficient(c, retrieved)
                    if deps.sufficiency
                    else bool(retrieved)
                )
                claims_data.append(
                    {
                        "id": f"c{idx}",
                        "text": c,
                        "risk": float(risk),
                        "sufficient": bool(sufficient),
                        "score": float(sc),
                        "supported": bool(sup),
                        "action": "supported" if sup else "none",
                    }
                )

    else:
        # HealGen and ablation variants
        no_cre = system == "healgen-noCRE"
        no_aesp = system == "healgen-noAESP"
        no_lshe = system == "healgen-noLSHE"

        draft = deps.generator.generate(f"Question: {question}\nAnswer:")
        claims = decompose(draft)

        claim_states: list[dict[str, Any]] = []
        for c in claims:
            if no_cre:
                risk = 1.0
            elif deps.estimator:
                risk = float(deps.estimator.predict_proba(c, question, deps.bm25))
            else:
                risk = 0.0

            if no_aesp:
                is_suff = False
            elif deps.sufficiency:
                is_suff = deps.sufficiency.is_sufficient(c, retrieved)
            else:
                is_suff = bool(retrieved)

            if not is_suff and deps.bm25:
                hits = deps.bm25.search(c, k=config.TOPK)
                num_bm25_calls += 1
                passages = _extract_passages(hits)
                retrieved = _dedup_passages(retrieved, passages)
                if deps.sufficiency and not no_aesp:
                    is_suff = deps.sufficiency.is_sufficient(c, retrieved)
                else:
                    is_suff = True

            claim_states.append(
                {
                    "text": c,
                    "risk": risk,
                    "sufficient": is_suff,
                }
            )

        dceg = DCEG()
        for idx, cs in enumerate(claim_states):
            dceg.add_claim(f"c{idx}", cs["text"], risk=cs["risk"])

        for j, ev in enumerate(retrieved):
            dceg.add_evidence(f"e{j}", ev)

        claim_best_score: dict[str, float] = {}
        for idx, cs in enumerate(claim_states):
            cid = f"c{idx}"
            best_sc = 0.0
            for j, ev in enumerate(retrieved):
                eid = f"e{j}"
                if deps.judge:
                    _, sup_score = deps.judge.support(cs["text"], [ev])
                else:
                    sup_score = 1.0
                dceg.link(cid, eid, support=float(sup_score))
                if sup_score > best_sc:
                    best_sc = float(sup_score)
            claim_best_score[cid] = best_sc

        unsupported_cids = dceg.unsupported()

        if unsupported_cids:
            if no_lshe:
                repaired_answer, repair_records = repair_answer(
                    answer=draft,
                    claims=claims,
                    unsupported_ids=unsupported_cids,
                    evidence=[],
                    generator=deps.generator,
                )
            else:
                repaired_answer, repair_records = repair_answer(
                    answer=draft,
                    claims=claims,
                    unsupported_ids=unsupported_cids,
                    evidence=retrieved,
                    generator=deps.generator,
                )
        else:
            repaired_answer = draft
            repair_records = [
                {"id": f"c{i}", "action": "supported", "before": claims[i], "after": claims[i]}
                for i in range(len(claims))
            ]

        action_map = {r["id"]: r["action"] for r in repair_records}

        for idx, cs in enumerate(claim_states):
            cid = f"c{idx}"
            is_supported = cid not in unsupported_cids
            action = action_map.get(cid, "supported" if is_supported else "unsupported")
            sc = claim_best_score.get(cid, 0.0)
            claims_data.append(
                {
                    "id": cid,
                    "text": cs["text"],
                    "risk": float(cs["risk"]),
                    "sufficient": bool(cs["sufficient"]),
                    "score": float(sc),
                    "supported": bool(is_supported),
                    "action": str(action),
                }
            )

        answer = repaired_answer

    latency_s = time.perf_counter() - t0
    abstained = not bool(answer.strip())

    return {
        "qid": qid,
        "system": system,
        "question": question,
        "answer": answer,
        "claims": claims_data,
        "num_bm25_calls": num_bm25_calls,
        "retrieved": retrieved,
        "latency_s": float(latency_s),
        "abstained": abstained,
    }
