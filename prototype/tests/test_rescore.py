"""Unit tests for frozen-pool evaluation rescoring (ledger ruling R12).

The plan's Global Constraint (prototype plan line 19): *"Judge never scores
against system-private evidence: frozen shared pool per question = gold
knowledge ∪ union of all four systems' top-3 passages (plus a gold-only
condition)."*

Contract under test:
- ``build_pools``: gold first, deduped union across the passed records.
- ``attach_eval_labels``: judges ``decompose(record["answer"])`` -- the
  claims of the *emitted response*, not the recorded pipeline claims (for
  HealGen those are draft-derived and diverge from the answer in 73/150
  records after localized repair) -- attaching ``ev_*`` (shared pool) and
  ``gold_*`` (gold-only) to a new ``eval_claims`` field.  Recorded claims
  are never mutated.
- ``label_view``: metric-ready copies whose ``claims`` are the eval claims
  with ``supported``/``score`` overridden from the chosen label set.

Pure-fake tests: no models are loaded.
"""

from healgen.rescore import attach_eval_labels, build_pools, label_view

TWO_SENTENCE = "Paris is the capital of France. It has the Eiffel Tower."


def _record(qid, system, retrieved, answer=TWO_SENTENCE, supported=True, score=0.5):
    return {
        "qid": qid,
        "system": system,
        "question": "q?",
        "answer": answer,
        "claims": [
            {"id": "c0", "text": "draft claim text", "risk": 0.1, "sufficient": True,
             "score": score, "supported": supported, "action": "supported"}
        ],
        "num_bm25_calls": len(retrieved),
        "retrieved": list(retrieved),
        "latency_s": 0.1,
        "abstained": False,
    }


class FakeJudge:
    """Records every evaluate_claims call; verdicts derived from evidence size."""

    def __init__(self):
        self.calls: list[tuple[list[str], list[str]]] = []

    def evaluate_claims(self, claims, evidence):
        self.calls.append((list(claims), list(evidence)))
        # Larger pools -> "more supported", deterministic and distinguishable.
        verdict = (len(evidence) >= 3, 0.9 if len(evidence) >= 3 else 0.2)
        return [verdict for _ in claims]


# --- build_pools --------------------------------------------------------------

def test_build_pools_gold_first_then_deduped_union():
    records = [
        _record("q1", "vanilla", []),
        _record("q1", "rag", ["P1", "P2"]),
        _record("q1", "cove", ["P2", "P3"]),
        _record("q1", "healgen", ["P4"]),
        _record("q2", "rag", ["X1"]),
    ]
    gold = {"q1": "GOLD1", "q2": "GOLD2"}
    pools = build_pools(records, gold)
    # gold leads, duplicates dropped, first-occurrence order stable
    assert pools["q1"] == ["GOLD1", "P1", "P2", "P3", "P4"]
    assert pools["q2"] == ["GOLD2", "X1"]


def test_build_pools_empty_gold_contributes_no_entry():
    records = [_record("q1", "rag", ["P1"])]
    pools = build_pools(records, {"q1": "   "})
    assert pools["q1"] == ["P1"]


def test_build_pools_skips_blank_passages():
    records = [
        _record("q1", "rag", ["", "  ", "P1"]),
        _record("q1", "cove", ["P1"]),
    ]
    pools = build_pools(records, {"q1": "GOLD"})
    assert pools["q1"] == ["GOLD", "P1"]


# --- attach_eval_labels -------------------------------------------------------

def test_attach_eval_labels_judges_emitted_answer_not_pipeline_claims():
    rec = _record("q1", "rag", ["P1", "P2"],
                  answer="Alpha one. Beta two.")  # decomposes to 2 claims
    pools = {"q1": ["GOLD1", "P1", "P2", "P3"]}  # 4 entries -> fake: supported
    judge = FakeJudge()

    attach_eval_labels([rec], pools, {"q1": "GOLD1"}, judge)

    eval_claims = rec["eval_claims"]
    assert [c["text"] for c in eval_claims] == ["Alpha one.", "Beta two."]
    # recorded pipeline claims untouched (no ev_* keys added to them)
    assert "ev_supported" not in rec["claims"][0]
    assert rec["claims"][0]["text"] == "draft claim text"
    # shared-pool verdict attached to the emitted claims
    for c in eval_claims:
        assert c["ev_supported"] is True
        assert c["ev_score"] == 0.9
        assert c["gold_supported"] is False  # gold pool has 1 entry
        assert c["gold_score"] == 0.2


def test_attach_eval_labels_shared_pool_and_gold_condition_calls():
    records = [
        _record("q1", "rag", ["P1", "P2"]),
        _record("q1", "healgen", ["P3"]),
    ]
    pools = {"q1": ["GOLD1", "P1", "P2", "P3"]}
    gold = {"q1": "GOLD1"}
    judge = FakeJudge()

    attach_eval_labels(records, pools, gold, judge)

    shared_calls = [c for c in judge.calls if len(c[1]) == 4]
    gold_calls = [c for c in judge.calls if c[1] == ["GOLD1"]]
    # one batched shared-pool call covering ALL emitted claims of the qid
    assert len(shared_calls) == 1
    assert len(shared_calls[0][0]) == 4  # 2 records x 2-sentence answers
    assert len(gold_calls) == 1
    assert len(gold_calls[0][0]) == 4


def test_attach_eval_labels_requires_pool_for_every_qid():
    records = [_record("q1", "rag", ["P1"])]
    try:
        attach_eval_labels(records, {}, {"q1": "GOLD"}, FakeJudge())
    except KeyError:
        pass
    else:
        raise AssertionError("missing pool for qid must raise KeyError")


def test_attach_eval_labels_idempotent():
    records = [_record("q1", "rag", ["P1", "P2"])]
    pools = {"q1": ["GOLD", "P1", "P2"]}
    gold = {"q1": "GOLD"}
    attach_eval_labels(records, pools, gold, FakeJudge())
    first = list(records[0]["eval_claims"])
    attach_eval_labels(records, pools, gold, FakeJudge())
    assert records[0]["eval_claims"] == first


def test_attach_eval_labels_empty_answer_makes_no_judge_calls():
    records = [_record("q1", "rag", ["P1"], answer="")]
    pools = {"q1": ["GOLD", "P1"]}
    judge = FakeJudge()
    attach_eval_labels(records, pools, {"q1": "GOLD"}, judge)
    assert records[0]["eval_claims"] == []
    assert judge.calls == []


# --- label_view ---------------------------------------------------------------

def test_label_view_builds_metric_claims_from_eval_labels():
    records = [_record("q1", "rag", ["P1", "P2"])]  # ev: 4-pool -> True/0.9
    pools = {"q1": ["GOLD1", "P1", "P2", "P3"]}
    attach_eval_labels(records, pools, {"q1": "GOLD1"}, FakeJudge())

    ev = label_view(records, "ev")
    assert len(ev[0]["claims"]) == 2  # emitted claims, not the 1 pipeline claim
    assert all(c["supported"] is True for c in ev[0]["claims"])
    assert all(c["score"] == 0.9 for c in ev[0]["claims"])
    assert ev[0]["claims"][0]["text"] == "Paris is the capital of France."
    # originals untouched
    assert records[0]["claims"][0]["supported"] is True
    assert "supported" not in records[0]["eval_claims"][0]  # raw key name is ev_supported

    gold = label_view(records, "gold")
    assert all(c["supported"] is False for c in gold[0]["claims"])
    assert all(c["score"] == 0.2 for c in gold[0]["claims"])


def test_label_view_empty_prefix_uses_record_claims():
    records = [_record("q1", "rag", ["P1"], supported=False, score=0.3)]
    view = label_view(records, "")
    # no eval claims attached -> falls back to the recorded pipeline claims
    assert len(view[0]["claims"]) == 1
    assert view[0]["claims"][0]["supported"] is False
    assert view[0]["claims"][0]["score"] == 0.3
    assert view[0] is not records[0]
    assert view[0]["claims"][0] is not records[0]["claims"][0]
