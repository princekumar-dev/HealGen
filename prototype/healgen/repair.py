"""Localized Self-Healing Engine (LSHE, Layer 7).

Executes localized, sentence-level repairs on an answer when claims are
flagged as unsupported by the Dynamic Claim-Evidence Graph (DCEG).
Untouched sentences are preserved byte-identically.
"""

from __future__ import annotations

import re
from typing import Any
from healgen.claims import _split_sentences


def _parse_claim_index(cid: str | int, claims: list[str]) -> int:
    """Resolve claim identifier (integer, 'c1', 'claim_1', or text) to 0-based index."""
    if isinstance(cid, int):
        return cid
    cid_str = str(cid).strip()
    if cid_str.isdigit():
        return int(cid_str)
    if cid_str.lower().startswith("c") and cid_str[1:].isdigit():
        return int(cid_str[1:])
    if cid_str.lower().startswith("claim_") and cid_str[6:].isdigit():
        return int(cid_str[6:])
    if cid_str in claims:
        return claims.index(cid_str)
    # Fallback: substring search in claims
    for idx, c in enumerate(claims):
        if cid_str in c or c in cid_str:
            return idx
    return -1


def _find_sentence_for_claim(claim: str, sentences: list[str]) -> int:
    """Identify which sentence contains or best matches the claim."""
    claim_norm = re.sub(r"[^\w\s]", "", claim.lower())
    claim_words = set(claim_norm.split())
    best_idx = 0
    best_overlap = -1

    for idx, s in enumerate(sentences):
        if claim in s or s in claim:
            return idx
        s_norm = re.sub(r"[^\w\s]", "", s.lower())
        s_words = set(s_norm.split())
        overlap = len(claim_words & s_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    return best_idx


def repair_answer(
    answer: str,
    claims: list[str],
    unsupported_ids: list[str | int],
    evidence: list[str],
    generator: Any,
) -> tuple[str, list[dict[str, str]]]:
    """Execute localized repair on unsupported claim segments within an answer.

    Args:
        answer: Full original generated answer string.
        claims: Decomposed atomic claims from the answer.
        unsupported_ids: List of claim identifiers determined to be unsupported.
        evidence: Retrieved evidence passages available for rephrasing.
        generator: Language model generator exposing `.generate(prompt: str) -> str`.

    Returns:
        (repaired_answer, records)
        where each record is: `{'id': cid, 'action': action, 'before': text, 'after': text}`
    """
    if not answer or not claims:
        return answer, []

    sentences = _split_sentences(answer)
    if not sentences:
        sentences = [answer]

    # Convert unsupported IDs to indices
    unsupported_claim_indices: set[int] = set()
    for uid in unsupported_ids:
        idx = _parse_claim_index(uid, claims)
        if 0 <= idx < len(claims):
            unsupported_claim_indices.add(idx)

    # If all claims are supported, return original answer untouched
    if not unsupported_claim_indices:
        records = [
            {
                "id": f"c{i}",
                "action": "supported",
                "before": claims[i],
                "after": claims[i],
            }
            for i in range(len(claims))
        ]
        return answer, records

    # Map each claim to its parent sentence
    claim_to_sentence: dict[int, int] = {}
    for i, claim in enumerate(claims):
        if len(sentences) == len(claims):
            claim_to_sentence[i] = i
        else:
            claim_to_sentence[i] = _find_sentence_for_claim(claim, sentences)

    # Collect sentence-level modifications
    sentence_replacements: dict[int, str] = {}
    records: list[dict[str, str]] = []

    valid_evidence = [e.strip() for e in evidence if isinstance(e, str) and e.strip()]
    ev_context = " ".join(valid_evidence)

    for i, claim in enumerate(claims):
        cid = f"c{i}"
        if i not in unsupported_claim_indices:
            records.append(
                {
                    "id": cid,
                    "action": "supported",
                    "before": claim,
                    "after": claim,
                }
            )
            continue

        s_idx = claim_to_sentence[i]
        orig_sentence = sentences[s_idx]

        if valid_evidence:
            # Rephrase using evidence
            prompt = (
                f"Fix the sentence based on the facts provided.\n"
                f"Facts: {ev_context}\n"
                f"Sentence: {orig_sentence}\n"
                f"Corrected sentence:"
            )
            generated_fix = generator.generate(prompt).strip()
            if generated_fix and generated_fix != orig_sentence:
                action = "rephrase"
                new_sentence = generated_fix
            else:
                action = "reduce_confidence"
                new_sentence = f"{orig_sentence.rstrip('.!?')} (unconfirmed)."
        else:
            action = "express_uncertainty"
            new_sentence = f"{orig_sentence.rstrip('.!?')} (uncertain)."

        sentence_replacements[s_idx] = new_sentence
        records.append(
            {
                "id": cid,
                "action": action,
                "before": claim,
                "after": new_sentence,
            }
        )

    # Reconstruct the answer preserving untouched sentences byte-identically
    reconstructed_sentences: list[str] = []
    for s_idx, original_s in enumerate(sentences):
        if s_idx in sentence_replacements:
            reconstructed_sentences.append(sentence_replacements[s_idx])
        else:
            reconstructed_sentences.append(original_s)

    repaired_answer = " ".join(reconstructed_sentences)
    return repaired_answer, records
