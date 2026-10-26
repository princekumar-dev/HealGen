# HealGen Prototype + Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only, reproducible prototype of the 9-layer HealGen self-healing inference architecture and run the paired experiment (4 systems + 3 ablations, N=150 HaluEval) that produces every number the arXiv paper will report.

**Architecture:** One Python package `prototype/healgen/` with one module per pipeline stage (claims, risk, sufficiency, graph, repair, judge, metrics) plus an orchestrator exposing four systems (vanilla, RAG, CoVe, HealGen) that all share one generator and one BM25 index. Experiments are thin scripts writing JSONL/CSV to `prototype/results/`; tables/figures are generated from those files, never hand-written.

**Tech Stack:** Python 3.14, torch 2.14 CPU, transformers 5.20, sentence-transformers, rank_bm25, networkx, scikit-learn, datasets/pandas/matplotlib/scipy, pytest. No API keys, no GPU.

**Spec:** `docs/superpowers/specs/2026-10-26-verigen-arxiv-design.md` (design) + `research/03-eval-methodology.md` §RECOMMENDED EVAL DESIGN (eval decisions) + `research/02-differentiation.md` (what the ablations must defend). Read all three before implementing.

## Global Constraints

- CPU-only, no API keys anywhere (design §8.4). All models local HF downloads: generator `google/flan-t5-base` (smoke: `google/flan-t5-small`), judge `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, second judge `cross-encoder/nli-deberta-v3-small`, embeddings `sentence-transformers/all-MiniLM-L6-v2`.
- Seeds fixed: test=42, dev=43, judge-slice=44; N_test=150, N_dev=50, N_judge_pairs=100. Dataset: HaluEval QA (`RUCAIBox/HaluEval` `data/qa_data.json`), filters per research §RECOMMENDED EVAL DESIGN (non-empty knowledge, right_answer ≤ 8 tokens, dedup by question).
- BM25: `rank_bm25` Okapi, k1=1.5, b=0.75, top-k=3, corpus = distinct knowledge paragraphs of the 10k HaluEval QA set.
- Decoding identical for all systems: greedy, max_new_tokens=64.
- Judge never scores against system-private evidence: frozen shared pool per question = gold knowledge ∪ union of all four systems' top-3 passages (plus a gold-only condition). Gold `knowledge` is never given directly to any system at inference (BM25 retrieval is the only path to it).
- Honesty: no result value may be written into tables/figures by hand; `make_tables.py` reads CSVs only. Every test/example ID appears in all systems (paired design).
- Windows/PowerShell commands; run from repo root `C:\Users\princ\Downloads\builder\verigen`.

## Review Focus

1. **Evaluation-loop leakage** — gold knowledge must reach systems only via BM25, and the judge must use the shared frozen pool: pinned by `test_pool_contains_gold_and_union` (Task 5) and `test_no_gold_in_prompt_inputs` (Task 10).
2. **Judge circularity inflating HealGen** — judge validation on known-label contrast pairs must run before the main experiment and be reported: pinned by Task 13's gate (accuracy printed and saved).
3. **LSHE must actually be localized** — repaired answers must keep unsupported-adjacent sentences byte-identical: pinned by `test_repair_preserves_untouched_sentences` (Task 9).
4. **Determinism** — same seed ⇒ same sampled IDs, same BM25 index, same bootstrap CIs: pinned by `test_sample_deterministic`, `test_paired_bootstrap_deterministic` (Tasks 2, 11).
5. **CPU runtime blowup** — no main run before the smoke gate measures tok/s and estimates total wall time: pinned by Task 12 gate (estimated total printed, must be < 12 h or plan re-scoped).

---

### Task 1: Repo scaffold + config

**Files:**
- Create: `prototype/healgen/__init__.py`, `prototype/healgen/config.py`, `prototype/pytest.ini`, `prototype/tests/__init__.py`, `prototype/tests/test_config.py`
- Modify: none (git repo root = repo root)

**Interfaces:**
- Produces: module `healgen.config` exposing pinned constants: `SEED_TEST=42, SEED_DEV=43, SEED_JUDGE=44, N_TEST=150, N_DEV=50, N_JUDGE_PAIRS=100, BM25_K1=1.5, BM25_B=0.75, TOPK=3, MAX_NEW_TOKENS=64, GEN_MODEL, GEN_SMOKE, JUDGE_MODEL, JUDGE2_MODEL, EMB_MODEL, PATHS (data_cache, results, figures dirs)`.

- [ ] **Step 1: Write failing test** (`tests/test_config.py`): assert `config.SEED_TEST == 42`, `config.TOPK == 3`, `config.GEN_MODEL == "google/flan-t5-base"`, and that `config.RESULTS_DIR.mkdir(exist_ok=True)` creates the folder.
- [ ] **Step 2: Run test, expect FAIL** — `python -m pytest prototype/tests/test_config.py -v` (module not found).
- [ ] **Step 3: Implement `config.py`** with exactly those constants; `pytest.ini` with `testpaths = tests`, marker `slow` for model tests.
- [ ] **Step 4: Run test, expect PASS.**
- [ ] **Step 5:** `git init` (if absent), add `.gitignore` (`__pycache__/`, `prototype/data_cache/`, `prototype/results/raw/`, `*.pyc`, HF cache refs), `git add -A && git commit -m "scaffold: healgen package + config"`.

### Task 2: Data layer — HaluEval load, sampling, corpus, BM25

**Files:**
- Create: `prototype/healgen/data.py`, `prototype/tests/test_data.py`

**Interfaces:**
- Consumes: `config` constants; downloads `https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json` to `data_cache/` if missing (requests, verify JSON parses and has `qa_data` list — field names verified against the real file at implementation time; adapt to actual schema, keeping the public API).
- Produces:
  - `load_examples(split: str) -> list[dict]` with keys `id, question, right_answer, hallucinated_answer, knowledge` — deterministic: dev = seed 43 first 50 after filters+dedup; test = seed 42 first 150 excluding dev ids; judge pairs = seed 44, 100 items.
  - `build_corpus(examples_all) -> list[str]` distinct knowledge paragraphs.
  - `BM25Index` class: `__init__(corpus: list[str])`, `.search(query: str, k: int = TOPK) -> list[tuple[int, str, float]]` (index, text, score).

- [ ] **Step 1: Write failing tests** (`tests/test_data.py`): `test_sample_deterministic` (two calls, same seed ⇒ identical ids; dev∩test = ∅); `test_filters` (every dev/test item has non-empty knowledge and ≤8-token right_answer); `test_corpus_distinct` (no duplicate paragraphs); `test_bm25_returns_k` (search returns 3 results, scores non-increasing, known query about a fact returns the paragraph containing its keyword in `test_bm25_ranking`).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement `data.py`** (download with cache, filters per Global Constraints, dedup by question text, `rank_bm25.BM25Okapi` wrapper).
- [ ] **Step 4: Run, expect PASS** (first run downloads QA json ~50 MB — network OK).
- [ ] **Step 5: Commit** `feat: halueval sampling + bm25 index`.

### Task 3: Claim decomposition (Layer 2)

**Files:**
- Create: `prototype/healgen/claims.py`, `prototype/tests/test_claims.py`, `prototype/experiments/audit_claims.py`

**Interfaces:**
- Consumes: nothing but stdlib (`re`) — no spaCy (Python 3.14 wheel risk; regex splitter is the builder's call, §6.1).
- Produces: `decompose(text: str) -> list[str]` — sentence split on `(?<=[.!?]) ` + conjunctive clause split on `(?<=[,;]) (and|but|while|whereas) ` + pronoun resolution (copy subject onto clauses beginning with he/she/it/they/this/that). Each claim is a standalone declarative string. Also `audit_claims.py` exports 30 sampled responses with manual-label columns to `results/audit_claims.csv`.

- [ ] **Step 1: Write failing tests:** `test_splits_conjunction` ("Paris is the capital of France and it has the Eiffel Tower." ⇒ 2 claims, second contains "Paris"); `test_pronoun_copied`; `test_single_sentence`; `test_empty_returns_empty`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement `claims.py`.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat: claim decomposition + audit export`.

### Task 4: Generator wrapper (shared by all systems)

**Files:**
- Create: `prototype/healgen/generator.py`, `prototype/tests/test_generator.py`

**Interfaces:**
- Consumes: `config.GEN_MODEL/GEN_SMOKE/MAX_NEW_TOKENS`.
- Produces: class `Generator(model_id: str, smoke: bool = False)` with `.generate(prompt: str) -> str` (greedy, `max_new_tokens=64`, `torch.manual_seed(0)` set at init) and `.n_tokens` counter for throughput; module-level `get_generator(smoke: bool)` cached singleton.

- [ ] **Step 1: Failing test** (`@pytest.mark.slow`): `test_generate_nonempty_and_deterministic` — two `generate("What is the capital of France?")` calls return identical strings, non-empty; prints tokens/sec.
- [ ] **Step 2: Run, expect FAIL** (downloads `flan-t5-small` ~300 MB on first pass).
- [ ] **Step 3: Implement `generator.py`** (AutoTokenizer/AutoModelForSeq2SeqLM, pad token fix).
- [ ] **Step 4: Run, expect PASS**; record printed tok/s in session notes (resolves research [unverified] item).
- [ ] **Step 5: Commit** `feat: shared t5 generator`.

### Task 5: NLI judge + frozen evidence pools (evaluation core)

**Files:**
- Create: `prototype/healgen/judge.py`, `prototype/tests/test_judge.py`

**Interfaces:**
- Consumes: examples, system retrievals (passed in, not imported from systems).
- Produces:
  - `build_pool(gold_knowledge: str, all_system_passages: list[str]) -> list[str]` — dedup, gold first.
  - `class Judge(model_id: str)` with `.support(claim: str, evidence: list[str]) -> tuple[bool, float]` — max entailment prob over pool passages; `.temperature` attribute set by `.fit_temperature(dev_pairs)` (scaler on logits vs. gold labels). Label convention: `entailment` index read from `config.id2label` (never hard-coded position).
  - `eval_contrast_pairs(judge, pairs) -> dict(accuracy, auc, n)`.

- [ ] **Step 1: Failing tests:** `test_pool_contains_gold_and_union` (pool = [gold] ∪ set(union of passages), order stable); `test_support_entailment` (claim "Paris is the capital of France." vs gold ["Paris is the capital of France."] ⇒ supported, prob > 0.5); `test_contradiction_not_supported` ("Paris is in Germany." vs same ⇒ not supported); `test_label_mapping_not_positional` (monkeypatch id2label swap ⇒ decision flips).
- [ ] **Step 2: Run, expect FAIL** (downloads ~700 MB judge on first pass).
- [ ] **Step 3: Implement `judge.py`.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat: nli judge with frozen shared evidence pools`.

### Task 6: Claim Risk Estimator (CRE, Layer 3)

**Files:**
- Create: `prototype/healgen/risk.py`, `prototype/tests/test_risk.py`, `prototype/experiments/train_cre.py`

**Interfaces:**
- Consumes: `decompose`, corpus (for BM25 feature), EMB_MODEL, judge (training-label construction only, never at inference).
- Produces: `features(claim, question, bm25) -> list[float]` (6 features: claim length, count of digits/proper-noun tokens, max BM25 score vs corpus, claim–question cosine (MiniLM), claim–knowledge-free novelty = 1 − max cosine vs top-3 retrieved, uppercase/rarity flag); `class RiskEstimator` with `.fit(train_rows)`, `.predict_proba(claim, question, bm25) -> float`, `.save/.load` (`results/cre_model.json` — coefficients JSON, no pickle). Training data: claims decomposed from 1,000 HaluEval right/hallucinated answers **outside test/dev/judge splits**, labeled by judge-vs-gold (faithful claims → 0, judge-unsupported claims → 1) in `train_cre.py`.

- [ ] **Step 1: Failing tests:** `test_features_shape` (len == 6, all finite); `test_fit_predict_roundtrip` (fit on synthetic separable rows ⇒ proba in (0,1), saves/loads with identical proba to 1e-6); `test_train_split_disjoint` (train ids ∩ {dev, test, judge} = ∅).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement `risk.py` + `train_cre.py`.**
- [ ] **Step 4: Run tests PASS; run `python prototype/experiments/train_cre.py`** → prints train accuracy vs majority baseline (must beat baseline; if not, log honestly and fall back to feature-ablation note in paper — report, don't tune on test).
- [ ] **Step 5: Commit** `feat: claim risk estimator`.

### Task 7: Adaptive Evidence Sufficiency Predictor (AESP, Layer 4)

**Files:**
- Create: `prototype/healgen/sufficiency.py`, `prototype/tests/test_sufficiency.py`

**Interfaces:**
- Consumes: EMB_MODEL.
- Produces: `class Sufficiency(tau: float)` with `.score(claim: str, evidence: list[str]) -> float` (max cosine MiniLM claim↔each passage) and `.is_sufficient(...) -> bool`; `tune_tau(dev: list[tuple[claim, evidence, label]]) -> float` choosing tau ∈ {0.3, 0.5, 0.7} maximizing F1 on dev labels (judge-vs-gold), returns chosen tau — sensitivity at all three recorded to `results/tau_sensitivity.csv`.

- [ ] **Step 1: Failing tests:** `test_gold_evidence_sufficient` (claim entailed by its gold para ⇒ score > 0.5 with tuned tau); `test_irrelevant_evidence_insufficient` (shuffled unrelated paragraphs ⇒ insufficient); `test_tau_deterministic`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement `sufficiency.py`.**
- [ ] **Step 4: PASS; run tuning script on dev, save tau.**
- [ ] **Step 5: Commit** `feat: sufficiency predictor + tau tuning`.

### Task 8: Dynamic Claim-Evidence Graph (DCEG, Layer 6)

**Files:**
- Create: `prototype/healgen/graph.py`, `prototype/tests/test_graph.py`

**Interfaces:**
- Consumes: networkx, judge (for contradiction edges — optional flag to skip NLI in unit tests).
- Produces: `class DCEG` with `.add_claim(id, text, risk)`, `.add_evidence(eid, text, freshness=1.0, weight=0.0)`, `.link(cid, eid, support: float)`, `.add_contradiction(cid, cid2)`, `.unsupported() -> list[str]` (claims with no supporting edge ≥ 0.5 or with contradiction), `.to_dot() -> str`, `.draw(path)` (matplotlib spring layout PNG → `figures/dceg_example.png`).

- [ ] **Step 1: Failing tests:** `test_unsupported_detection` (claim with only weak link ⇒ in unsupported; strong link ⇒ not); `test_contradiction_flag`; `test_dot_export_contains_nodes`; `test_draw_writes_png` (tmp path exists, size > 0).
- [ ] **Step 2: FAIL → Step 3: implement → Step 4: PASS.**
- [ ] **Step 5: Commit** `feat: claim-evidence graph + figure export`.

### Task 9: Localized Self-Healing Engine (LSHE, Layer 7) — the differentiator

**Files:**
- Create: `prototype/healgen/repair.py`, `prototype/tests/test_repair.py`

**Interfaces:**
- Consumes: `Generator`, DCEG, evidence bundles, judge scores.
- Produces: `repair_answer(answer: str, claims: list[str], unsupported_ids: list[str], evidence: list[str], generator) -> tuple[str, list[dict]]` implementing actions in priority order: (1) `retrieve_more` if AESP insufficient and pool allows (returns same answer + extra passage request flag), (2) `rephrase` — regenerate **only the sentence(s) that produced unsupported claims**, splicing back into the untouched answer, (3) `reduce_confidence` / (4) `express_uncertainty` when rephrase still unsupported (append hedge clause to that claim's sentence only), (5) `reconnect_claim` (drop or attribute as uncertain). Returns (repaired_answer, record per claim: `{id, action, before, after}`).

- [ ] **Step 1: Failing tests:** `test_repair_preserves_untouched_sentences` (answer = 3 sentences, only sentence 2 claimed unsupported ⇒ output sentences 1 & 3 byte-identical, only 2 changed); `test_rephrase_uses_evidence` (new sentence contains a token from provided evidence for a fact the original lacked); `test_unchanged_when_all_supported`; `test_record_actions`.
- [ ] **Step 2: FAIL → Step 3: implement (splice via sentence offsets of `claims`' source sentences) → Step 4: PASS.**
- [ ] **Step 5: Commit** `feat: localized self-healing engine`.

### Task 10: Systems orchestration (4 systems, paired records)

**Files:**
- Create: `prototype/healgen/systems.py`, `prototype/tests/test_systems.py`

**Interfaces:**
- Consumes: all modules above.
- Produces: `run_system(system: str, ex: dict, deps: Deps) -> dict` for `system ∈ {"vanilla","rag","cove","healgen"}`:
  - `vanilla`: generate(question) only, no retrieval.
  - `rag`: BM25 top-3 always → generate(question + passages).
  - `cove`: generate → decompose → verify ALL claims (judge) → if any unsupported, regenerate whole answer once with evidence.
  - `healgen`: Layer1 intent flags (trivial: passthrough) → decompose **question-relevant claims from draft** (draft first via question-only generation, then claim-state loop: risk (CRE) → sufficiency (AESP) → retrieve only if insufficient (count BM25 calls) → DCEG → LSHE localized repair → calibrated record).
  Record schema (JSONL, one per system×example): `{qid, system, question, answer, claims:[{id,text,risk,sufficient,score,supported,action}], num_bm25_calls, retrieved:[...], latency_s, abstained}`.
  `Deps` dataclass bundles generator/judge/bm25/estimator/sufficiency (injectable for tests with fakes).

- [ ] **Step 1: Failing tests (fakes, no models):** `test_record_schema` (all required keys, `set(claims[i]) == {...}`); `test_paired_coverage` (same qid list for all 4 systems); `test_no_gold_in_prompt_inputs` (instrumented fake generator records prompts; assert gold knowledge string appears only when it arrived via bm25 fake, never injected directly for vanilla/rag-of-empty cases); `test_healgen_retrieves_less_than_rag` (on a fixture where evidence already sufficient, healgen `num_bm25_calls < rag`); `test_cove_regenerates_once_only`.
- [ ] **Step 2: FAIL → Step 3: implement `systems.py` → Step 4: PASS.**
- [ ] **Step 5: Commit** `feat: four paired systems with record schema`.

### Task 11: Metrics + statistics

**Files:**
- Create: `prototype/healgen/metrics.py`, `prototype/tests/test_metrics.py`

**Interfaces:**
- Consumes: record schema; numpy/scipy.
- Produces: `sp_macro(records) -> float`, `sp_micro(records) -> float`, `hallucination_rate(...)`, `any_hallucination_rate(...)`, `em_f1(preds, golds)`, `ece(confs, correct, n_bins=10)`, `brier(...)`, `bootstrap_ci(values, n=2000, seed=0) -> (lo, hi)`, `paired_bootstrap(records_a, records_b, metric, n=2000, seed=0) -> dict(p_value, delta, ci)`, `holm_bonferroni(pvals)`.

- [ ] **Step 1: Failing tests with hand-computed values:** `test_sp_macro` (resp1 claims [1,0,1] ⇒ 0.667; resp2 [1,1] ⇒ avg over responses = 0.8335 — exact asserted); `test_ece_known` (2 bins perfect calibration ⇒ 0.0); `test_paired_bootstrap_deterministic` (same seed ⇒ identical p; swapped args ⇒ p symmetric ≈ 1−p within 0.02); `test_holm`.
- [ ] **Step 2: FAIL → Step 3: implement → Step 4: PASS.**
- [ ] **Step 5: Commit** `feat: metrics + paired bootstrap`.

### Task 12: Smoke integration gate (models × full pipeline)

**Files:**
- Create: `prototype/experiments/run_smoke.py`
- Modify: none

**Interfaces:**
- Consumes: everything; `--n 5 --model small` flags.
- Produces: `results/smoke.jsonl` (4 systems × 5 examples, full record schema) + printed table: tok/s, NLI pairs/s, mean latency per system, **projected wall-clock for N=150**.

- [ ] **Step 1: Implement `run_smoke.py`** (argparse; writes JSONL; prints projection).
- [ ] **Step 2: Run** `python prototype/experiments/run_smoke.py --n 5 --model small`. Expect: JSONL exists, all 4 systems present, no exceptions.
- [ ] **Step 3: Gate:** projected N=150 wall-clock < 12 h; if not, re-scope (raise smoke→base is expected here; the number that matters is total time) and record decision in plan checkboxes.
- [ ] **Step 4: Commit** `feat: smoke gate`.

### Task 13: Judge validation (honesty gate)

**Files:**
- Create: `prototype/experiments/run_judge_validation.py`

**Interfaces:**
- Consumes: `Judge.eval_contrast_pairs`, judge slice (seed 44).
- Produces: `results/judge_validation.json` (accuracy, AUC, n, both judges).

- [ ] **Step 1: Implement + run.** Expect accuracy/AUC printed; **gate: AUC ≥ 0.70 required to proceed** (if below, swap to `JUDGE2_MODEL` primary and rerun; document either way).
- [ ] **Step 2: Commit** `feat: judge validation gate`.

### Task 14: Main experiment run

**Files:**
- Create: `prototype/experiments/run_main.py`

**Interfaces:**
- Produces: `results/main.jsonl` (150 × 4 systems), `results/main.csv` (flattened metrics per example), `results/system_summary.csv` (per-system aggregates incl. retrieval calls, latency mean/p50/p95, ECE, Brier, EM/F1, SP_macro/micro with bootstrap CIs), `results/pairwise_tests.csv` (6 paired bootstrap comparisons, Holm-corrected).

- [ ] **Step 1: Implement** (resumable: skip qid×system pairs already in JSONL; progress printed every 10).
- [ ] **Step 2: Run in background**; verify completion: JSONL row count == 600, all qids paired.
- [ ] **Step 3: Commit** results files (they are the paper's source of truth) `exp: main run N=150`.

### Task 15: Ablations (the novelty defense)

**Files:**
- Create: `prototype/experiments/run_ablations.py`

**Interfaces:**
- Produces: `results/ablations.csv` — three variants, each vs full HealGen on the same 150 qids: `healgen-noCRE` (uniform risk → always-verify), `healgen-noAESP` (always retrieve = sufficiency bypass), `healgen-noLSHE` (detect unsupported but only `reduce_confidence`, no rephrase). Reuses `run_system` with flags.

- [ ] **Step 1: Implement** with `system` names `healgen-noCRE` etc. in schema (tests: `test_ablation_flags_isolated` asserts each flag changes exactly one behavior in an instrumented fake run).
- [ ] **Step 2: Run; verify 450 rows; commit.**

### Task 16: Tables, figures, spot-check export

**Files:**
- Create: `prototype/experiments/make_tables.py`, `prototype/experiments/export_spotcheck.py`
- Produces: `paper/tables/*.tex` (main table, ablation table, judge-validation, tau sensitivity), `paper/figures/reliability_diagram.png`, `paper/figures/latency_retrieval.png`, `figures/dceg_example.png` (from Task 8), `results/spotcheck_sample.csv` (100 claims stratified by system, for the human spot-check with κ — partner's task later).

- [ ] **Step 1: Implement `make_tables.py`** reading ONLY `results/*.csv`; every `\input{}` table cell must trace to a CSV column (spot-check: pick one cell, grep the CSV).
- [ ] **Step 2: Run; LaTeX files compile-check via tectonic later (paper plan).**
- [ ] **Step 3: Commit** `feat: tables + figures from results`.

---

## Scope note (per writing-plans skill)

The design spec spans research (DONE — `research/*.md`), prototype (this plan), and paper (separate plan, written after Task 16 because its tables must exist first). This plan alone yields working, testable software.

## Self-review record

- Spec coverage: design §6.1 modules → Tasks 3–10; §6.2 datasets/metrics → 2, 11, 13, 14; §6.2 judge → 5, 13; ablation requirement (02-differentiation §5) → 15; honesty (§4) → 14, 16; reproducibility (§6.5) → seeds in config, resumable runs, committed CSVs. Research [unverified] items (CPU tok/s) → resolved in Tasks 4/12.
- Type consistency: record schema defined once (Task 10) and consumed unchanged in 11/14/16; `Deps` injection used by all system tests.
- Review Focus: each line has its pinning test/task as listed.
