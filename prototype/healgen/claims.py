"""Rule-based claim decomposition (Layer 2 of the HealGen pipeline).

``decompose(text)`` turns a free-text answer into a list of standalone
declarative claims by applying three purely syntactic rules in order:

1. **Sentence split** -- cut after ``.``, ``!`` or ``?`` followed by a space.
   A small abbreviation guard (``Mr.``, ``Dr.``, ``e.g.``, ``i.e.``, ``U.S.``,
   single-letter initials) rejoins pieces the naive split would cut in half.
2. **Conjunctive clause split** -- cut before ``and``/``but``/``while``/
   ``whereas`` when the conjunction follows a comma or semicolon.  A *bare*
   conjunction is also cut when the clause it introduces opens with a
   pronoun from the resolution set (see below); without that narrow guard the
   plan's own example, ``"... France and it has ..."``, would stay one claim,
   while noun phrases such as ``"salt and pepper"`` are still left alone.
3. **Pronoun resolution** -- a clause that opens with he/she/it/they/this/
   that/these/those inherits the subject of the preceding clause in the same
   sentence, so every claim can be verified in isolation.  The subject is
   approximated as the tokens before the first recognised finite verb (a
   closed list of auxiliaries/copulas), else a leading determiner plus the
   following noun phrase, else the leading capitalised run or first token.

Claims are whitespace-normalised, stripped of dangling ``;``/``,`` and given
terminal punctuation, so cached downstream artefacts stay stable.

Known limits (deliberate; this is a rule-based approximation, not parsing):

- Subject detection is a closed-list heuristic.  Long noun phrases, passive
  constructions and verbs outside the list can truncate the copied subject.
- Only the leading pronoun of a split clause is rewritten.  Object and
  possessive pronouns (``him``, ``its``, ``their``) and cross-sentence
  anaphora (``"A is B. It is C."``) are left untouched.  A semicolon without
  a following conjunction is not treated as a clause boundary either.
- The abbreviation guard can over-merge a genuinely sentence-final
  abbreviation with the next sentence (``"He lives in the U.S. It is big."``).
- Verb agreement is not adjusted: a plural subject planted onto a singular
  clause can read ungrammatically (``"they have"`` -> ``"Paris have"``).
- Conjunctions are split on surface form only; no semantic parse decides
  whether two clauses really are separable propositions.

The module is stdlib-only (``re``) by controller decision -- no spaCy, whose
Python 3.14 wheels are a deployment risk for this prototype.
"""

import re

PRONOUNS = frozenset({"he", "she", "it", "they", "this", "that", "these", "those"})

_DETERMINERS = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "his",
        "her",
        "its",
        "their",
        "our",
        "my",
        "your",
    }
)

# Closed list used only to find where the subject of a clause ends.  Keeping it
# short deliberately: an unknown verb falls back to the noun-phrase heuristics
# below instead of guessing.
_VERBS = frozenset(
    {
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "contains",
        "includes",
        "provides",
        "offers",
        "gives",
        "makes",
        "uses",
        "refers",
        "means",
        "lies",
        "stands",
        "becomes",
        "became",
        "remains",
        "seems",
        "appears",
        "belongs",
        "consists",
        "measures",
        "covers",
        "spans",
        "borders",
        "hosts",
        "features",
        "serves",
        "holds",
        "produces",
        "creates",
        "allows",
        "requires",
    }
)

_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "st",
        "jr",
        "sr",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "u.s",
        "u.k",
    }
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?]) ")
_COMMA_CONJUNCTION = re.compile(r"(?<=[,;]) (?:and|but|while|whereas) ")
_BARE_PRONOUN_CONJUNCTION = re.compile(
    r" (?:and|but|while|whereas) (?=(?:he|she|it|they|this|that|these|those)\b)",
    re.IGNORECASE,
)
_TERMINAL_PUNCTUATION = ".!?"


def _ends_with_abbreviation(text: str) -> bool:
    """True when ``text`` ends in a period that probably is not a full stop."""
    match = re.search(r"([A-Za-z.]+)\.$", text)
    if match is None:
        return False
    token = match.group(1).lower()
    if token in _ABBREVIATIONS:
        return True
    return len(token) == 1 and token.isalpha()


def _split_sentences(text: str) -> list[str]:
    """Split on the pinned boundary, then rejoin abbreviation false positives."""
    sentences: list[str] = []
    for part in _SENTENCE_SPLIT.split(text):
        part = part.strip()
        if not part:
            continue
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)
    return sentences


def _split_clauses(sentence: str) -> list[str]:
    """Split comma/semicolon conjunctions and pronoun-led bare conjunctions."""
    clauses: list[str] = []
    for chunk in _COMMA_CONJUNCTION.split(sentence):
        clauses.extend(_BARE_PRONOUN_CONJUNCTION.split(chunk))
    return [clause.strip() for clause in clauses if clause.strip()]


def _bare_word(word: str) -> str:
    """Lowercase a token and drop surrounding punctuation."""
    return re.sub(r"[^a-z']", "", word.lower())


def _strip_token(word: str) -> str:
    return word.strip("\"'(").rstrip(",;:")


def _subject(clause: str) -> str:
    """Approximate the subject of a clause (documented heuristic, not parsing)."""
    words = clause.split()
    if not words:
        return ""

    for index, word in enumerate(words):
        if index and _bare_word(word) in _VERBS:
            return " ".join(words[:index]).rstrip(",;:")

    first = _strip_token(words[0])
    if first.lower() in _DETERMINERS and len(words) > 1:
        span = [first]
        for word in words[1:]:
            core = _strip_token(word)
            if core[:1].isupper():
                span.append(core)
            else:
                if len(span) == 1:
                    span.append(core)
                break
        return " ".join(span)

    capitalized: list[str] = []
    for word in words:
        core = _strip_token(word)
        if core[:1].isupper():
            capitalized.append(core)
        else:
            break
    if len(capitalized) >= 2:
        return " ".join(capitalized)
    return first


def _replace_leading_pronoun(clause: str, subject: str) -> str:
    """Replace a clause-initial pronoun with ``subject``; leave anything else."""
    if not subject:
        return clause
    match = re.match(r"([\"'(]?)([A-Za-z]+)\b", clause)
    if match is None:
        return clause
    prefix, word = match.groups()
    if word.lower() not in PRONOUNS:
        return clause
    if word[:1].isupper() and subject[:1].islower():
        subject = subject[:1].upper() + subject[1:]
    return f"{prefix}{subject}{clause[match.end():]}"


def _resolve_pronouns(clauses: list[str]) -> list[str]:
    """Copy the previous clause's subject onto pronoun-led follow-on clauses."""
    resolved: list[str] = []
    for clause in clauses:
        if resolved:
            clause = _replace_leading_pronoun(clause, _subject(resolved[-1]))
        resolved.append(clause)
    return resolved


def _finalize(claim: str) -> str:
    """Normalise spacing/punctuation so each claim is a standalone sentence."""
    claim = re.sub(r"\s+", " ", claim).strip().rstrip(",;").strip()
    if not claim:
        return ""
    if claim[-1] not in _TERMINAL_PUNCTUATION:
        claim += "."
    return claim


def decompose(text: str) -> list[str]:
    """Return the standalone declarative claims contained in ``text``.

    Empty (or whitespace-only) input yields ``[]``.  Output order follows the
    source text.  See the module docstring for the rules and their limits.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    claims: list[str] = []
    for sentence in _split_sentences(normalized):
        for clause in _resolve_pronouns(_split_clauses(sentence)):
            finalized = _finalize(clause)
            if finalized:
                claims.append(finalized)
    return claims
