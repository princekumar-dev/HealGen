# 03 — Evaluation Methodology Research

Design decisions for evaluating a **claim-level self-healing RAG** pipeline on a CPU-only Windows machine, no API keys, ~100–200 test examples, small generator (flan-t5 family). All claims sourced from live web/HF checks on 2026-00-26; anything I could not directly confirm is marked **[unverified]**.

---

## 1. Benchmark

### 1.1 FaithDial — verified facts

Checked: <https://huggingface.co/datasets/McGill-NLP/FaithDial> (HF page + GitHub + TACL paper page).

| Property | Value (verified) |
|---|---|
| HF dataset id | `McGill-NLP/FaithDial` (config: `plain_text`) |
| License | **MIT** |
| Size | 32.3k rows total. Splits: train 18.4k, validation 3.42k (`valid_random_split` 1.67k + `valid_topic_split` 1.75k), **test 3.54k** (`test_random_split` 1.72k + `test_topic_split` 1.82k) |
| Source | Amended hallucinated turns of Wizard-of-Wikipedia via AMT; human-validated: 94.4% faithful / 5.6% hallucinated (TACL 2022, Dziri et al.) |
| Columns | `dialog_idx`, `response` (edited/faithful target), `original_response` (hallucinated original), `history` (prior turns), **`knowledge`** (per-turn gold knowledge blurb), `BEGIN` (per-sentence belief-tracker labels: Hallucination / Entailment / Uncooperative / Generic), `VRM` (dialogue act labels) |

**Can it be used without the original belief-tracker pipeline?** Yes. The knowledge source and the per-sentence hallucination/entailment labels are *materialized in the dataset* — you never need to rerun the belief tracker or the WOW preprocessing. The `BEGIN` field gives automatic (noisy) sentence-level support labels, which is a nice free sanity signal for your claim-level scorer.

**Does it fit claim-level factuality eval of generated responses?** Partially, with real caveats:

- ✅ Ships per-turn gold evidence (`knowledge`) — exactly what a claim-support judge needs.
- ✅ MIT, small, CPU-friendly, includes hallucinated counterparts (`original_response`) usable as contrastive/calibration data.
- ⚠️ It is a **dialogue** benchmark: the "question" is the last user utterance plus a variable-length `history`. Defining the input to your generator requires either truncating to the last turn or templating history — an awkward wrapper for a QA pipeline.
- ⚠️ Responses are short conversational turns (16–361 chars in the viewer) → typically only 2–4 atomic claims per example, so per-example signal is thin.
- ⚠️ The `BEGIN` labels are *automatic* belief-tracker output, not human per-claim labels (human validation was done at the response level only).

**Verdict:** usable, but dialogue-shaped for a QA-shaped pipeline. If you want FaithDial (e.g., your contribution is framed on knowledge-grounded dialogue), it works out of the box. For claim-level factuality of *answers*, a QA benchmark is cleaner — see below.

### 1.2 Alternatives

**HaluEval QA subset** (EMNLP 2023, RUCAIBox) — verified via GitHub repo + HF mirrors:

- 10,000 QA samples seeded from HotpotQA; each sample has `knowledge` (Wikipedia text, gold evidence), `question`, `right_answer`, `hallucinated_answer` (ChatGPT-generated, then filtered). License: **MIT** (repo `LICENSE`). Official repo `RUCAIBox/HaluEval`; community HF mirrors exist (`pminervini/HaluEval`, `nehalecky/halu-qa-eval-dataset`) — exact canonical HF id **[unverified]**; plan to download from GitHub `data/qa_data.json` if HF mirrors look unofficial.
- The paired (faithful, hallucinated) answers per question are unusually valuable for a CPU/weak-judge paper: they give *known-supported* and *known-unsupported* answers with the exact gold evidence — a free validation set for your NLI judge and an oracle upper-bound experiment (repair a hallucinated answer with gold knowledge).

**HotpotQA** — verified: 113k multi-hop QA pairs; `supporting_facts` are sentence-level gold evidence; distractor setting supplies 10 paragraphs/question (2 gold + 8 distractors); CC BY-SA 4.0. Great retrieval benchmark, but multi-hop is likely beyond flan-t5-small/base (they'd produce degenerate answers, floor effects), and it needs you to build the corpus yourself.

**FEVER** — verified: 185,445 claims, 3-way SUPPORTS/REFUTED/NOT_ENOUGH_INFO with evidence sentences; Fleiss κ 0.684. It is *claim classification*, not generation — you'd have to invert it to make a generation benchmark, which breaks the RAG story. Use only as NLI-judge calibration data (its dev claims with gold evidence are ideal for measuring your judge's accuracy on in-domain verification). Dataset license **[unverified]** (Wikipedia-derived content is CC BY-SA; check before redistribution).

**StrategyQA** — implicit, multi-hop, no per-claim evidence shipped; awkward for claim-support grounding. Not recommended.

### 1.3 Decision & sampling plan

**Primary: HaluEval QA.** It is QA (matches the pipeline), ships gold evidence per example, is MIT, and ships hallucinated answers for judge calibration and oracle experiments. **FaithDial stays as an optional secondary/robustness set** (50 dialogue turns) since it is the canonical knowledge-grounded-dialogue hallucination benchmark and reviewers may expect it.

Corpus for BM25: pool all distinct `knowledge` texts across the full 10k HaluEval QA set (roughly 10k short Wikipedia paragraphs; exact unique count **[unverified]** — measure after dedup). Gold evidence for each test question = its own source `knowledge` (guaranteed present in the corpus by construction).

Sampling (concrete, reproducible):

- **Test set: N = 150**, `random.seed(42)`, sampled from HaluEval QA after filtering: non-empty `knowledge`; `right_answer` ≤ 8 tokens (flan-t5-feasible answer lengths); dedup by normalized question string.
- **Dev set: N = 50** (seed 43) from the same filtered pool, disjoint from test — used for calibrator fitting, NLI-threshold selection, and prompt/threshold tuning of the risk predictor. Never report on it.
- **Judge-validation slice:** 100 (question, `right_answer`) pairs + the same 100 questions' (question, `hallucinated_answer`) pairs (seed 44) → ~200–400 atomic claims with *known* support status. Report NLI-judge accuracy/AUC on this slice in the paper.
- **Optional dialogue robustness:** 50 turns from FaithDial `test_random_split` (seed 42), input = last user utterance + knowledge (history omitted or templated), noted as a secondary result.
- If HaluEval's `knowledge` field proves too long/missing on inspection, **fallback**: HotpotQA `distractor` dev, N=150 (seed 42), filter `level=medium, type=bridge, answer≤8 tokens`; gold evidence = supporting-fact sentences; corpus = pooled paragraphs of the sampled questions.

Why N=150: at N≈150 examples × ~3 claims × 4 systems you get ~1,500–2,000 claim verdicts per system — enough for bootstrap CIs to be informative on 3–5 point differences, small enough that the NLI judge workload (~5k–10k sentence-pairs total including retries/repair passes) runs in minutes-to-low-hours on CPU.

---

## 2. Factuality judge without an LLM

### 2.1 Candidate NLI models (verified on HF/sbert.net)

| Model | Params | MNLI-mm acc | Notes | License |
|---|---|---|---|---|
| `microsoft/deberta-v3-base-mnli-fever` | ~184M (86M backbone + 98M 128K-vocab embeddings) | ~89–90 **[unverified exact]** | The classic ALCE (Gao et al., 2023) factual-precision judge. **Fetch returned HTTP 401 on 2026-01-26 → currently gated or unavailable without auth.** Do not depend on it. | MIT (DeBERTa base) |
| **`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`** | ~184M | mnli-m/mm 0.903, fever-nli 0.777, anli 0.579 (per model card) | Trained on MNLI + FEVER-NLI + ANLI (763,913 pairs). Not gated, MIT, ONNX-friendly, used by dozens of hallucination-detection apps. Same recipe family as the ALCE judge. | **MIT** |
| `cross-encoder/nli-deberta-v3-base` | ~184M | 90.04 (sbert.net table) | SNLI+MNLI only (no FEVER), but the sbert `CrossEncoder` API is the simplest CPU path. | Apache-2.0 **[verify on card]** |
| `cross-encoder/nli-deberta-v3-small` | ~142M (44M backbone + embeddings) **[computed]** | 87.55 | ~2× faster, good backup/second-judge. | Apache-2.0 **[verify]** |
| `cross-encoder/nli-deberta-v3-xsmall` | ~22M backbone **[computed]** | 87.77 | Fastest credible option; MNLI-only. | Apache-2.0 **[verify]** |
| `facebook/bart-large-mnli` | 406M | — | Seq2seq → much slower per claim on CPU; no advantage over DeBERTa here. Skip. | MIT |
| `tasksource/ModernBERT-base-nli` | 149M | — (multi-NLI suite; card shows nli_fever 0.78 among recasts) | 8k context, strong reasoning claims, but it is a *generative-classifier* (MLM-head) usage — awkward for simple CPU probability extraction; license **[unverified]**. Interesting ablation, not a primary. | **[unverified]** |

Recent supporting evidence that NLI cross-encoders are a legitimate claim-verification method: *Coarse-Grained Hallucination Detection via NLI Fine-Tuning* (SDLP @ ACL 2025) finds fine-tuned NLI cross-encoders (best: DeBERTa-v3-large) competitive with few-shot LLM judges, and explicitly notes Mor-Lan & Levi (2024) and Sankararaman et al. (2024) establishing NLI cross-encoders for scientific-claim verification.

**Primary judge: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`.** Rationale: FEVER-NLI training matches the claim-verification setting; MNLI+ANLI keeps generalization; MIT; not gated; ~184M runs comfortably on CPU (tens of claims/sec single-threaded-batch; low hundreds with ONNX Runtime int8 via `optimum` — measure on your box **[unverified estimate]**; for ~10k pairs even 20 pairs/sec is <10 min).

**Backup / second judge: `cross-encoder/nli-deberta-v3-small`** (~142M, Apache-2.0) — for (a) judge-agreement reporting (see §2.3), (b) fallback if the MoritzLaurer model has loader issues with your transformers version (DeBERTa-v3 tokenizers were historically fragile on old transformers; model card warns transformers ≥4.13).

Known weaknesses to state honestly in the paper: NLI judges are overconfident (softmax poorly calibrated — you must temperature-scale before using probabilities for ECE), confuse *neutral* with weak *entailment* on paraphrases, are strong at catching entity/number swaps (good for hallucination detection) but weak on multi-hop/compositional claims and on claims requiring knowledge absent from the given evidence (they should output neutral; plausibility biases them toward entailment). Judge sees only the evidence you give it — hence the judge-symmetry protocol below.

### 2.2 How FActScore / HALoGEN handle judge bias (and what we copy)

- **FActScore** (Min et al., EMNLP 2023): ChatGPT judge per atomic fact against retrieved Wikipedia sentences; validity established by a large human study — reported agreement with human judgments at the response level is high (≈90%+ **[exact figures: check paper Table; mark verify]**). They also release an open "LLAMA+NP" automatic variant (LLaMA-2-70B decomposition + noun-phrase filtering) whose labels are compared against ChatGPT's.
- **HALoGEN** (Kang et al., 2024): GPT-4 judge, but with *predicate-specific verification prompts* (each atomic fact is verified under a declared predicate type), and they validate the judge against a human-annotated subsample, reporting judge error rates explicitly rather than assuming oracle status.
- Neither paper treats the LLM judge as free of bias; both buy credibility with (i) human agreement studies on a subsample and (ii) transparent judge prompts/artifacts.

### 2.3 Judge-bias mitigation for a weak (NLI) judge — validated & extended

Your proposed ideas are sound; here is the sharpened protocol (each item is cheap and paper-reportable):

1. **Shared gold evidence pool, enforced symmetrically.** Every system's claims are verified against the *identical* evidence set: gold `knowledge` ∪ the union of top-k BM25 passages retrieved by any system for that question (frozen once per question, before scoring any system). This removes "system A looked smarter because the judge saw better evidence" as a confound. *Addition:* report a second, stricter condition — **gold-only evidence** (judge sees only the gold knowledge blurb) — which isolates generation quality from retrieval quality. If a claim is unsupported even under gold-only, it is a genuine hallucination.
2. **Report judge-free metrics** alongside judged ones: exact-match/F1 of the answer against `right_answer` (QA needs no judge), abstention/"I don't know" rate, retrieval calls, latency. These are the ground truth for the efficiency claims and are judge-independent.
3. **Human spot-check protocol (powered, stratified).** Don't just eyeball: sample **100 claims total**, stratified × {system} × {judge verdict}, ~10 per cell over 4 systems × 2 verdicts; one author labels supported/unsupported against the frozen evidence; report judge precision/recall vs human and Cohen's κ (**[power caveat:** at n=100 the CI on judge accuracy is roughly ±10 points; state this**]**). Release the sampled claims in the appendix — reviewers trust auditable artifacts.
4. **Judge validation on known labels (new, cheap).** Use the HaluEval contrast pairs (§1.3): decompose 100 `right_answer` and 100 `hallucinated_answer` and measure the NLI judge's accuracy/AUC. Publish this number — it converts "weak judge" from a hidden liability into a measured instrument property.
5. **Threshold sensitivity.** Report headline metrics at entailment thresholds 0.3 / 0.5 / 0.7 (or argmax labels) and state whether system rankings flip. A result that only exists at one threshold is not a result.
6. **Two-judge agreement.** Run the backup judge on a 20% subsample; report verdict agreement %. Disagreement cases go into the human spot-check preferentially.
7. **Calibrate the judge before using its probabilities** (temperature scaling on the dev set, §1.3) — otherwise ECE numbers measure DeBERTa's overconfidence, not your pipeline's.

---

## 3. Atomic claim decomposition without an LLM

**How FActScore does it:** few-shot prompting of an LLM (GPT-3.5/ChatGPT; their open variant uses LLaMA-2-70B-chat) to decompose a passage into *atomic facts* — each fact must be self-contained (pronouns resolved), single-attribute, verifiable against one evidence sentence. The LLM variant is the accuracy ceiling; Wanner et al. (2024) (DecompScore) show decomposition method materially changes final scores, so any approximation must be disclosed and ideally audited.

**Rule-based / small-model approximations (acceptable for a CPU paper):**

1. **Sentence split** (spaCy `sentencizer` or regex; no model download) — the floor. Weakness: multi-clause sentences stay conflated, so one hallucinated clause poisons an entire sentence-level verdict; inflates measured support per claim.
2. **Sentence split + conjunctive clause split** — split on coordinating conjunctions (`and`, `but`, `while`, `although`, `, which`, `, and`) via dependency parse (spaCy `en_core_web_sm`, CPU, ~15MB). Then **pronoun resolution by subject-copying**: if a clause's subject is a pronoun, copy the antecedent from the previous clause (regex over pronoun lists + capitalization heuristics). This is the recommended primary: cheap, deterministic, auditable. Documented weaknesses: over-splitting relative clauses that are genuinely atomic; broken quantifier scope ("not all X are Y" → "all X are Y"); imperfect coreference (named-entity antecedents OK, nested pronouns not).
3. **NP-based fact filtering** (FActScore's own LLAMA+NP trick, minus LLAMA): drop clauses whose main predicate is non-verifiable (opinion/hedging verbs) using a small stop-verb list; optionally emit each (subject, relation, object) triple as a claim via SVO extraction. Weakness: silently drops hedged-but-checkable claims.
4. **Small instruction model option:** `Qwen2.5-0.5B-Instruct` (Apache-2.0) — verified CPU ballpark: ~120 tok/s on a desktop CPU in one public table (elbruno/LocalLLMs), llama.cpp Q4_K_M ~10–30 tok/s across CPU classes **[both community figures, verify on your box]**; a 150–250-token decomposition output → ~1–5 s/example → 150 examples ≈ 3–12 min. Feasible. `flan-t5-large` (780M) is a worse fit: encoder-decoder CPU generation is typically ~1–4 tok/s **[unverified]** → 25–100 s/example → hours for the full set; and 0.5B-class models' decomposition quality is weak and *variance-prone* (they drop clauses and mangle pronouns at measurable rates **[documented weakness, quantified rates unverified]**).

**Recommendation:** primary = rule-based (2), with a 30-response human audit reporting (a) missing-claim rate and (b) hallucinated-claim-miss rate vs manual decomposition; optional ablation = Qwen2.5-0.5B-Instruct prompted decomposition on the same 30-response audit to show the trade-off. Do **not** use flan-t5-small/base for decomposition — at 80M/250M they are unreliable instruction followers, and a decomposition failure would corrupt every downstream metric.

---

## 4. Metrics — precise definitions & statistics

Let example *i* decompose into claims *c_ij*, j = 1..*n_i*; judge verdict *s_ij* ∈ {0,1} (1 = supported by the frozen evidence pool under the entailment rule); system confidence *p_ij* ∈ [0,1] that the claim is supported.

- **Supported-claim precision (FActScore-style).** Micro: `SP_micro = Σ_i Σ_j s_ij / Σ_i n_i`. Macro (recommended headline, weights examples equally): `SP_macro = (1/N) Σ_i (Σ_j s_ij / n_i)`. Report both; FActScore uses the per-generation mean (macro).
- **Hallucination rate.** `HR = 1 − SP` (claim level). Also report **response-level any-hallucination rate** `RHR = (1/N) Σ_i 1[Σ_j (1−s_ij) > 0]` — this is the user-facing number and is more robust to decomposition granularity differences between systems.
- **Retrieval calls.** Mean BM25 queries per example; % of claims triggering retrieval; mean evidence sentences fetched. For vanilla baseline = 0 by construction; for CoVe = #claims.
- **Latency.** Wall-clock per example, batch size 1, same machine, report mean / p50 / p95, and a one-line hardware note (CPU model, threads, torch version). Warm up models before timing.
- **ECE** (Guo et al., 2017 calibration; 10 equal-width bins recommended over 15 for n≈1,500 claims):  
  `ECE = Σ_{b=1}^{B} (|B_b| / M) · | acc(B_b) − conf(B_b) |`, where M = total claims, `B_b` = claims with *p_ij* in bin *b*, `acc(B_b)` = mean *s_ij* in the bin, `conf(B_b)` = mean *p_ij*. Report with a reliability diagram (2 panels: risk predictor vs NLI-judge confidence).
- **Brier score.** `BS = (1/M) Σ_ij (p_ij − s_ij)^2`. (Strictly proper; lower is better; report alongside ECE since ECE with few bins is unstable at small M.)
- **Bootstrap CIs** (example-level resampling to respect within-example claim correlation): **2,000 resamples** for 95% percentile CIs on every headline metric (10,000 is nearly free on CPU for count-based metrics — use it if you like; Koehn 2004 showed bootstrap CIs remain accurate even on ~300-item test sets). Report point estimate + [CI_lo, CI_hi] in every table cell.
- **Significance test for paired system comparison:** **paired bootstrap resampling** (Koehn 2004) — resample the N examples with replacement 2,000×, recompute both systems' metrics, p = fraction of resamples where the observed ordering reverses (two-sided: use |Δ|). This is the MT/NLP standard and is what SacreBLEU supports. **Approximate randomization** (Riezler & Maxwell 2005; the ACL 2005 WS paper shows AR estimates p-values *more conservatively* than bootstrap on equivalent systems) is the recognized alternative — report AR as a robustness check for your most important claim (self-healing vs standard RAG on SP_macro) if reviewers might push. With N=150, only effects of roughly ≳3–5 SP points reach p<0.05 **[back-of-envelope, unverified]** — say so; emphasize CIs over binary significance, and run the 4 systems on the *same* 150 examples (paired design is what buys power).
- Multiple comparisons: 6 pairwise comparisons among 4 systems → report raw p and note Holm–Bonferroni-adjusted thresholds; don't silently cherry-pick.

---

## 5. Generator choice sanity check

| Model | Params | CPU tok/s (documented ranges) | Knowledge-grounded QA quality notes |
|---|---|---|---|
| `google/flan-t5-small` | 80M | **[unverified]** ~5–15 tok/s typical CPU (community figures) | Weak instruction follower & weak knowledge holder (flan scaling paper: small is far below base/large on MMLU/QA). Floor-risk: may not follow "answer from context" well enough for vanilla-vs-RAG contrast to be clean — but it *will* hallucinate, which gives headroom. |
| `google/flan-t5-base` | 250M | **[unverified]** ~2–8 tok/s CPU | MMLU ≈ 35.9 (community blog citing the flan paper) — barely above chance; fine for short knowledge-grounded answers. Historically the workhorse of knowledge-grounded dialogue papers (WoW/FaithDial-era T5 variants). Good size/speed trade-off. |
| `google/flan-t5-large` | 780M | **[unverified]** ~1–4 tok/s CPU | Noticeably better instruction following; likely too slow for 4 systems × 150 examples × multi-pass self-healing on a laptop CPU. |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | ~120 tok/s (community desktop-CPU table); GGUF Q4_K_M ~10–30 tok/s **[both unverified]**; Ollama/HF widely mirrored; Apache-2.0 | Better instruction following & chat formatting than flan-t5-base at similar size; decoder-only so per-token cost higher for equal params; known to be "fluent but confidently wrong" — arguably *ideal* hallucination substrate, but reviewer familiarity with flan-T5 as a controlled encoder-decoder baseline is higher. |
| `Qwen2.5-1.5B-Instruct` | 1.5B | ~10–30 tok/s (llama.cpp Q4_K_M, community) **[unverified]**; official speed bench is GPU-only | Probably CPU-viable for generation-only, too slow for a 3-pass pipeline. |

**Recommendation: `google/flan-t5-base` as the single generator for all four systems** (same weights, same decoding params — the controlled variable). It is the standard small knowledge-grounded baseline in exactly this literature, is Apache-2.0, and its mediocre factual recall is a *feature*: vanilla generation should measurably hallucinate against the HaluEval gold evidence, giving the self-healing pipeline something to fix. Use `flan-t5-small` for smoke tests during development. Report a 20-example `Qwen2.5-0.5B-Instruct` replication if time permits (it strengthens the "method is not model-specific" claim cheaply). **Measure real tok/s on your machine first** — the paper should quote your measurement, not these community numbers.

---

## 6. BM25-on-small-corpus pitfalls & controls

Documented/standard concerns and the controls to adopt:

1. **Lexical-overlap leakage / trivially-answerable questions.** HaluEval/HotpotQA knowledge paragraphs share surface vocabulary with their questions, so BM25 often puts the gold paragraph at rank 1 → RAG looks magic and the *retrieval* contribution is untestable. **Controls:** (a) report **retrieval recall@k** (k=3) of the gold `knowledge` per question — if recall@3 > 95%, say plainly that retrieval is near-saturated and the contribution is *selective abstention/repair*, not retrieval quality; (b) pool distractor paragraphs from *other* questions' knowledge into the corpus (this happens automatically when you pool all 10k) and consider adding 1–2 random paragraphs per question explicitly as distractors, reporting the with/without-distractor delta; (c) report an **oracle-RAG upper bound** (gold knowledge always given) to bound achievable gains.
2. **Train/test contamination of the generator.** flan-t5 may have memorized common Wikipedia facts → "vanilla" baseline secretly knows answers, compressing the RAG-vs-vanilla gap. **Controls:** report vanilla vs RAG delta per question and analyze the subset where vanilla is wrong (where the method should win); note contamination as a threat-to-validity; HaluEval knowledge skews to less-central HotpotQA entities which partially mitigates this **[unverified]**.
3. **BM25 degeneracy on tiny corpora.** IDF collapses for terms appearing in many short paragraphs; Okapi parameters tuned for corpora of 10⁶+ docs behave differently at 10⁴. **Controls:** use `rank_bm25.BM25Okapi` (k1=1.5, b=0.75) or Lucene-standard (k1=1.2, b=0.75 — pyserini/Lucene defaults), fix and report parameters; paragraph-level chunking (one doc = one knowledge blurb); report a random-retrieval ablation (retrieval replaced by random corpus paragraphs) to show the pipeline's repair step isn't just noise.
4. **Evaluation-loop leakage.** The *risk predictor* and *repair* must never see gold `knowledge` at inference — gold is used only by (a) the frozen judge evidence pool, (b) the oracle upper bound, clearly marked. State this explicitly; it is the difference between a defensible and an indefensible RAG eval.
5. **Distractor-induced false supports.** With a pooled corpus, two blurbs about the same entity can conflict; the NLI judge may find "support" in a wrong-but-similar paragraph. **Control:** the frozen evidence pool includes all top-k retrieved passages *plus gold*, and the gold-only condition (§2.3.1) disambiguates — a claim supported only by a distractor is flagged in error analysis.

---

# RECOMMENDED EVAL DESIGN

The concrete, implementable decision set:

- **Dataset:** HaluEval QA subset (GitHub `RUCAIBox/HaluEval` `data/qa_data.json`, MIT). **Test N=150** (seed 42), **dev N=50** (seed 43), filters: non-empty `knowledge`, `right_answer` ≤ 8 tokens, dedup by question. Judge-validation slice: 100 questions × (right_answer + hallucinated_answer) (seed 44). Optional secondary: 50 FaithDial `test_random_split` turns (MIT) for dialogue robustness.
- **Retrieval corpus:** all distinct `knowledge` paragraphs of the 10k HaluEval QA set, paragraph granularity, `rank_bm25` Okapi (k1=1.5, b=0.75) **[pin and report]**; top-k=3.
- **Generator:** `google/flan-t5-base` (Apache-2.0) for all four systems; identical decoding (greedy, max_new_tokens=64); `flan-t5-small` for dev smoke tests; optional 20-example `Qwen2.5-0.5B-Instruct` (Apache-2.0) replication.
- **Claim decomposition:** spaCy sentence-split + conjunctive clause split + subject-copying pronoun resolution; audit vs manual decomposition on 30 responses (report missing/extra-claim rates). Optional ablation: Qwen2.5-0.5B-Instruct decomposition. No flan-t5 for decomposition.
- **Judge:** primary `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (MIT, ~184M, MNLI+FEVER-NLI+ANLI); backup/second-judge `cross-encoder/nli-deberta-v3-small` (~142M). Evidence pool frozen per question = gold `knowledge` ∪ union of all systems' top-3 BM25 passages; second condition gold-only. Temperature-scale judge probs on dev before using them.
- **Systems compared (same 150 examples, paired):** vanilla generation; standard RAG (top-3 always); CoVe-style post-hoc verification (verify all claims, regenerate unsupported); self-healing claim-level pipeline (proposed).
- **Metrics:** SP_macro (headline) + SP_micro; HR = 1−SP; response-level any-hallucination rate; judge-free: answer EM/F1 vs `right_answer`, abstention rate; efficiency: BM25 calls/example, evidence fetches, latency mean/p50/p95; calibration: ECE (10 bins) + Brier, with reliability diagrams, for the risk predictor and (separately) the NLI judge.
- **Statistics:** example-level bootstrap 95% percentile CIs, **2,000 resamples** (10,000 if free); pairwise system tests via **paired bootstrap resampling** (Koehn 2004), 2,000 resamples, two-sided; approximate randomization as robustness check for the headline comparison; Holm–Bonferroni note for the 6 pairwise tests; explicitly acknowledge N=150 power limits (only ≳3–5 pt effects detectable **[unverified estimate — measure]**).
- **Judge-honesty protocol:** (1) shared frozen evidence pool + gold-only condition; (2) judge-free metrics reported alongside; (3) judge accuracy/AUC on the HaluEval contrast-pair slice, reported in-paper; (4) 100-claim stratified human spot-check with precision/recall/κ, artifacts released; (5) threshold sensitivity at 0.3/0.5/0.7; (6) two-judge agreement on a 20% subsample.
- **Leakage controls:** retrieval recall@3 reported; oracle-RAG upper bound; random-retrieval ablation; gold `knowledge` never given to any system at inference; generator-contamination caveat stated.

**Open/unverified items to resolve on the machine:** actual tok/s for flan-t5-{small,base} and NLI pairs/sec on your CPU; canonical HF id/license-terms check for the HaluEval mirror if GitHub download is inconvenient; FEVER dataset license version if used for judge calibration; transformers version pin for DeBERTa-v3 tokenizer stability.
