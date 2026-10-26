# HealGen Differentiation Analysis

Companion to `01-prior-art.md`. All rows verified against the sources cited there (arXiv API, 2026-10-26).
"Claim-level" = operates on decomposed atomic claims/facts; "span-localized" = edits only offending text
regions; "sufficiency" = judges whether available evidence is *enough* (not merely whether it is relevant);
"calibration" = emits a per-claim confidence aligned with correctness probability.

## 1. Feature comparison table — classical core (14 methods + HealGen)

| Method | Retrieval timing | Claim-level granularity | Evidence-sufficiency check | Repair: localized vs full regen. | Confidence calibration | Structured claim–evidence rep. | Post-hoc vs during-generation |
|---|---|---|---|---|---|---|---|
| **RAG** (Lewis'20) | Pre-generation (fixed) | No | No | None | No | No | Neither (one-shot) |
| **Self-RAG** (Asai'24) | During (on-demand via reflection tokens) | Per-segment support critiques (soft claims) | Relevance/support critique, not sufficiency | Full regeneration guided by critique | No (reflection tokens uncalibrated) | No | During-generation |
| **CRAG** (Yan'24) | Pre-generation + corrective re-retrieval on doc-level confidence | No (document-level) | Doc-quality confidence (retrieval-side, not claim-side) | None (fixes context, not output) | No | No | Pre-generation |
| **CoVe** (Dhuliawala'23) | None by default (verification questions answered from parametric knowledge) | Verification questions ≈ claims | Implicit (questions answered independently) | **Full regeneration** of final answer | No | No | Post-hoc |
| **RARR** (Gao'23) | Post-hoc research (search per unsupported region) | Implicit (edits unsupported content) | Implicit via attribution search | **Span-localized edits with preservation constraint** | No | Attribution links only, no graph | Post-hoc |
| **Self-Refine** (Madaan'23) | None | No | No | Full regeneration each round | No | No | Post-hoc (iterative) |
| **Reflexion** (Shinn'23) | None (verbal self-feedback) | No | No | Full re-attempt per trial | No | No | Post-hoc across trials |
| **CRITIC** (Gou'24) | Post-hoc tool calls (search/interpreter) | No | Tool-based fact check (output-level) | Mostly full revision | No | No | Post-hoc |
| **DRAGIN** (Su'24) | **During** (real-time info-need triggers) | Info-need events ≈ soft claims | No (triggers on uncertainty/info-need, not sufficiency) | None | No | No | During-generation |
| **Adaptive-RAG** (Jeong'24) | Pre (strategy selection by query complexity) | No | No | None | No | No | Pre-generation |
| **Rowen** (Ding'25) | Post-detection (cross-lingual/model consistency triggers retrieval) | No (response-level detection) | No | Revision of detected outputs (not span-localized) | No | No | Post-hoc |
| **FActScore** (Min'23) | Post-hoc (per atomic fact) | **Yes (atomic facts)** | Per-fact support = evaluation | None (metric only) | No | No | Post-hoc (evaluation) |
| **SAFE** (Wei'24) | Post-hoc (Google Search per fact) | **Yes (atomic facts)** | Per-fact support = evaluation | None (metric only) | No | No | Post-hoc (evaluation) |
| **HALoGEN** (Ravichander'25) | Post-hoc (verifiers) | **Yes (atomic units)** | Per-unit verification = evaluation | None (benchmark) | No | No | Post-hoc (evaluation) |
| **FaithDial** (Dziri'22) | n/a (dataset) | Utterance-level critic | Trains a faithfulness critic | Training-time data edit | No | No | Training-time |
| **HealGen (proposed)** | Pre (claim decomposition) + **during** (AESP-driven re-retrieval) | **Yes — claims drive the whole pipeline** | **Yes — per-claim sufficiency predicted mid-generation** | **Span/segment-localized repair only** | **Yes — per-claim calibrated confidence on final output** | **Yes — dynamic claim-evidence graph (contradiction/freshness/confidence)** | **Both: during-generation retrieval + post-generation localized healing** |

## 1b. Feature comparison table — 2025–2026 challengers (the ones that matter)

| Method | Retrieval timing | Claim granularity | Sufficiency check | Localized repair | Per-claim calibration | Claim–evidence structure | Timing |
|---|---|---|---|---|---|---|---|
| **FACTOR** (arXiv:2606.22474) | Post-hoc verification | Yes (risk-adaptive per claim) | Verification effort allocated by claim risk | No (re-ranking) | Uncertainty-driven, not calibrated output | No | Post-hoc |
| **SURE-RAG** (arXiv:2605.03534) | Pre (selective answering gate) | Yes (claim×passage verifier) | **Yes (set-level: coverage/conflict)** | None (abstain/answer) | Auditable selective score | Per-pair relations, aggregated | Post-hoc gate |
| **SimulRAG** (arXiv:2509.25459) | Pre + selective post-hoc grounding | **Yes (claims generated with UE)** | Simulator-boundary assessment | **Selective claim update** | Uncertainty estimation per claim (uncalibrated) | No | Both (mostly pre/post) |
| **DoctorRAG** (arXiv:2604.00865) | During (agentic) | Trajectory-level | **Yes (diagnosis stage)** | **Yes (prefix reuse, local intervention)** | No | No | Post-hoc repair of trajectories |
| **ADORE** (arXiv:2601.18267) | During (coverage-audit loop) | Section/claim-level | **Yes (evidence-coverage audit)** | No (regenerates from memory bank) | No | **Yes — Claim-Evidence Graph memory bank** | During-generation |
| **CURE** (arXiv:2604.12046) | None | **Yes (atomic claims)** | Abstention on low confidence | No (trained-in) | **Yes (trained claim-level calibration)** | No | Training-time |
| **DSVD** (arXiv:2503.03149) | None | Token/span-level | Inline self-verification | **Yes (rollback + local re-decode)** | No | No | During-decoding |
| **Enoki** (arXiv:2609.00581) | Post-hoc | Yes (relational facts) | Per-fact verification | Localization only (no repair) | No | OpenIE facts projected to spans | Post-hoc (detection) |
| **OPT-Kraken** (arXiv:2610.12529) | Post-hoc | Artifact-level | Fidelity/consistency checks | **Yes (granularity-selected, contracts protect correct content)** | No | No | Post-hoc (domain-specific) |
| **ClaimWriter** (arXiv:2608.28643) | Pre (source facts → outline) | **Yes (source-linked claims)** | Source-linked by construction | **Yes (localized revision on source change)** | No | Source-linked claim representation | Both |

Reading: no single row (including combinations of rows) covers all six HealGen differentiators — but
**ADORE covers DCEG+AESP, SimulRAG covers CRE+localized update, SURE-RAG covers sufficiency at claim level,
and DoctorRAG/OPT-Kraken/DSVD cover localized repair.** HealGen's remaining white space is the *closed loop*:
pre-generation claim decomposition → per-claim risk → mid-generation sufficiency-gated re-retrieval →
graph-structured evidence state → localized healing → calibrated per-claim output, all in one inference
architecture with a shared claim state.

## 2. The three closest competitors

### #1 RARR (Gao et al., ACL 2023, arXiv:2210.08726) — the localized-repair precedent
- **What it does:** Post-hoc attribution via web search + **localized post-editing that fixes unsupported
  content while preserving the original output** (explicit preservation objective). Achieves large attribution
  gains with minimal wording churn.
- **What it still lacks:** (a) repair is strictly post-hoc — no risk estimation or evidence acquisition during
  generation, so wasted compute on re-searching things the model was already confident about; (b) no pre-
  generation claim decomposition that *steers* generation; (c) attribution links are ephemeral — no persistent
  claim-evidence graph tracking contradictions, evidence freshness, or confidence; (d) output carries no
  calibrated per-claim reliability signal; (e) unsupported-content detection relies on attribution search
  itself rather than a cheap predicted risk score.
- **Implication for HealGen:** HealGen must cite RARR as the origin of "repair the span, not the response" and
  differentiate on *when* repair decisions are made, the structured evidence state, and calibrated output.

### #2 DRAGIN (Su et al., ACL 2024, arXiv:2403.10081) — the during-generation retrieval precedent
- **What it does:** Dedicated real-time information-need estimation during decoding decides **when** to
  retrieve, and constructs **what** to retrieve from the full context span where the need arose; beats static
  when-to-retrieve heuristics on four knowledge-intensive datasets.
- **What it still lacks:** (a) it targets *missing information*, not *hallucination risk* — no per-claim risk
  estimate; (b) no notion of whether already-retrieved evidence is *sufficient* (its trigger is uncertainty/
  information-need, not sufficiency shortfall); (c) no verification or repair of generated claims; (d) no
  claim-level bookkeeping or calibrated output.
- **Implication for HealGen:** HealGen's AESP is not "dynamic retrieval" in general — it must be framed as
  *sufficiency-prediction-driven* re-retrieval conditioned on the claim state, and evaluated against DRAGIN
  as the strongest dynamic-retrieval baseline.

### #3 CRAG / Self-RAG cluster (Yan et al. ICLR 2024; Asai et al. ICLR 2024) — the adaptive-retrieval-and-
support-checking precedent
- **What they do:** Self-RAG: on-demand retrieval + reflection tokens that critique support/relevance of
  generated segments at inference time. CRAG: retrieval-evaluator confidence gating with corrective web search
  and decompose-then-recompose document filtering. Together they establish "decide adaptively whether/when
  retrieval is warranted and check support" as standard practice.
- **What they still lack:** (a) granularity is document/segment-level, not atomic claims with identity across
  the pipeline; (b) corrections happen at the *context* level (retrieve again) or via full regeneration — no
  span-localized repair; (c) no structured, persistent claim-evidence state (no contradiction or freshness
  tracking across evidence); (d) no calibrated per-claim confidence in the final response; (e) Self-RAG requires
  fine-tuning with reflection tokens, while HealGen is (presumably) a training-free inference architecture —
  state this explicitly.
- **Runner-up worth naming:** **Rowen** (Ding et al., SIGIR-AP 2025) gates retrieval on cross-response
  consistency, i.e., an uncertainty-triggered retrieval trigger without claims.

**Verification of the brief's expectation:** the anticipated "RARR + DRAGIN + CRAG/Self-RAG" cluster is
confirmed as the closest classical competition. However, the *most dangerous* papers for HealGen's novelty are
actually from **2026**: FACTOR (CRE), SURE-RAG (AESP at claim level), ADORE (DCEG + coverage-driven retrieval),
SimulRAG (claim-level uncertainty-gated selective verification/update), and DoctorRAG/OPT-Kraken/DSVD
(localized repair). None of them is the whole architecture; together they cover nearly all of it.

## 3. Differentiation statement (novelty claim — honest, defensible)

> Existing systems each own one stage of the factuality loop: RARR repairs unsupported spans but only
> post-hoc; DRAGIN and the Self-RAG/CRAG family decide adaptively *when* to retrieve but neither model claim
> risk nor repair output; FActScore/SAFE/HALoGEN, FACTOR, and SURE-RAG operate at atomic-claim granularity but
> only to *evaluate or gate*, not to heal. HealGen's claim to novelty is architectural, not mechanistic: it is,
> to our knowledge, the first inference architecture in which a **pre-generation decomposition of the query
> into atomic claims** maintains a **dynamic claim–evidence graph** that is continuously updated by
> **per-claim risk estimates (CRE)** and **mid-generation evidence-sufficiency predictions (AESP)**, and that
> drives **span-localized repair of only unsupported claims (LSHE)** plus a **calibrated per-claim confidence
> for the final response** — a closed loop in which the same claim state governs retrieval, verification,
> repair, and output reliability. Each component has precedent; the closed claim-level control loop does not,
> and we show empirically that the whole materially outperforms the sum of its parts.

Honesty notes to preserve in the paper: (1) do **not** claim novelty for "claim decomposition" (FActScore,
SAFE, HALoGEN), "adaptive retrieval" (Self-RAG, DRAGIN, Adaptive-RAG, Rowen), "evidence sufficiency judging"
(SURE-RAG, REVEAL, DoctorRAG), "localized repair" (RARR, OPT-Kraken, DoctorRAG), or "per-claim confidence"
(CURE, Abbasli et al. 2026) individually; (2) the burden of proof is an ablation showing each stage contributes
and the closed loop beats strong compose-together baselines (e.g., FActScore-style verify + RARR-style revise,
DRAGIN retrieval, SURE-RAG gate).

### Draft related-work paragraph (for the paper)

> **Related work.** *Adaptive retrieval* decides whether and when to consult external evidence: Self-RAG trains
> reflection tokens to retrieve and self-critique on demand (Asai et al., 2024), CRAG gates generation on a
> retrieval-quality evaluator with corrective web search (Yan et al., 2024), Adaptive-RAG routes queries by
> predicted complexity (Jeong et al., 2024), Rowen triggers retrieval on cross-response inconsistency (Ding et
> al., 2025), and DRAGIN estimates the decoder's real-time information need to retrieve mid-generation (Su et
> al., 2024). None of these estimate *per-claim hallucination risk* or judge whether retrieved evidence is
> *sufficient* for the claims being written. *Claim-level factuality* has been formalized as evaluation: FActScore
> decomposes generations into atomic facts (Min et al., 2023), SAFE verifies each fact with search-augmented
> agents (Wei et al., 2024), and HALoGEN adds high-precision atomic verifiers and an error taxonomy (Ravichander
> et al., 2025); FACTOR adaptively allocates verification by claim-level risk (Hassan et al., 2026) and SURE-RAG
> verifies claim-level evidence sufficiency for selective answering (Qiu et al., 2026) — but these systems
> evaluate or abstain rather than repair. *Self-correction* ranges from full-response iteration (Self-Refine,
> Madaan et al., 2023; Reflexion, Shinn et al., 2023; CoVe, Dhuliawala et al., 2023) to tool-grounded revision
> (CRITIC, Gou et al., 2024); RARR remains the closest precedent, attributing and **locally** editing unsupported
> content post-hoc while preserving wording (Gao et al., 2023), and DoctorRAG similarly localizes failures and
> repairs only the diagnosed point of an agentic trajectory (Jiao et al., 2026). Structured claim–evidence state
> appears in enterprise deep-research systems as a Claim-Evidence Graph memory bank (You et al., 2026) and in
> fact verification (Zhang et al., 2024; Guo et al., 2025), but not as a live control structure for inference.
> Calibrated claim-level confidence is emerging as a trained capability (Liu & Wang, 2026; Abbasli et al., 2026)
> rather than an inference-time byproduct. **HealGen** is, to our knowledge, the first architecture to unify
> pre-generation claim decomposition, per-claim risk estimation, mid-generation sufficiency-gated retrieval, a
> dynamic claim–evidence graph, span-localized healing, and calibrated per-claim output confidence in a single
> closed inference loop.

*(Author-year placeholders above map 1:1 to the BibTeX keys in `01-prior-art.md`; expand when building
refs.bib. "Hassan et al., 2026" = FACTOR, arXiv:2606.22474; "Qiu et al., 2026" = SURE-RAG, arXiv:2605.03534;
"You et al., 2026" = ADORE, arXiv:2601.18267; "Jiao et al., 2026" = DoctorRAG, arXiv:2604.00865;
"Liu & Wang, 2026" = CURE, arXiv:2604.12046; "Abbasli et al., 2026" = arXiv:2608.22483.)*

## 4. Papers that weaken or obsolete HealGen's novelty claim (2025–2026)

None fully obsoletes HealGen, but four materially narrow the claim and **must** be cited and beaten:

1. **ADORE (arXiv:2601.18267, 2026)** — weakestens the DCEG+AESP pairing hardest: it maintains an explicit
   Claim-Evidence Graph memory bank, runs an evidence-coverage audit loop that triggers targeted re-retrieval,
   and stops on an evidence-driven criterion. Counter: it is an agentic enterprise pipeline (no per-claim risk,
   no localized repair, no calibrated confidence, no evaluation as an inference-time architecture per se).
2. **SimulRAG (arXiv:2509.25459, 2025)** — claim-level generation with uncertainty estimation that selectively
   verifies and *updates* claims is a compressed version of HealGen's CRE + selective-healing story. Counter:
   uncertainty is uncalibrated, evidence is simulators, there is no graph or sufficiency-prediction module, and
   it is domain-scoped (scientific QA).
3. **FACTOR (arXiv:2606.22474, 2026)** — owns the exact motivational sentence "not all claims are equally
   risky" and demonstrates risk-adaptive verification at inference time. Counter: verification-effort
   allocation + re-ranking only; no retrieval control, no repair, no graph, no calibrated output.
4. **CURE (arXiv:2604.12046) + Claim-Level Confidence Calibration (arXiv:2608.22483)** — together they make
   "per-claim calibrated confidence" a claimed contribution on its own (one trained, one post-hoc inference-
   time). Counter: they calibrate; they do not retrieve, structure evidence, or heal. HealGen should present
   calibration as a *byproduct of the claim-evidence state*, and compare directly against both.

Secondary threats: **DSVD** (during-decoding rollback = in-generation localized healing, weakening "healing is
post-generation-only"), **DoctorRAG** and **OPT-Kraken** (granularity-selected local repair with
preserve-the-rest contracts), **Enoki** (claim→span localization), and **SURE-RAG** (formal claim-level
sufficiency). If any of these appear at the same venue cycle, HealGen's framing should shift from "first
architecture to do X" toward "first *unified, calibrated, claim-state-driven* loop, with evidence that
composition alone underperforms."

## 5. Bottom-line verdict on the novelty claim

- **Defensible as an integration/systems contribution** — the closed loop with a persistent claim–evidence
  state governing retrieval, repair, and calibrated output is not present in any single verified paper.
- **Not defensible as component novelty** — every one of the six "claimed differentiators" has direct prior art
  (cited above); the paper must be written as composition + closed-loop control + ablations.
- **Risk level: moderate-to-high.** The 2026 literature covers roughly 5 of 6 components across ~6 papers
  published within the last 18 months; a strict reviewer will call HealGen "RARR + DRAGIN + FActScore + graph
  memory, composed." The defense is empirical: show the integrated loop beats (a) every component-wise
  baseline and (b) a naive compose-together of RARR-style repair + DRAGIN-style retrieval + SURE-RAG-style
  gating, on claim-level factuality, cost, and calibration error.
- **Naming:** "HealGen" shows 0 collisions on arXiv (title + full-record search, 2026-10-26); "VeriGen" is
  taken (arXiv:2308.00708, Verilog LLM). Web-wide check still pending (search tooling unavailable) — see
  `01-prior-art.md` Part 3.
