# Prior Art Research for HealGen (claim-level self-healing inference architecture)

Compiled: 2026-10-26. All entries verified against the arXiv API / arXiv abstract pages unless marked `[unverified]`.
Venue notes: where the arXiv record itself does not state the venue, the venue is marked as "widely reported" —
re-check against DBLP/OpenReview before camera-ready.

**Corrections to assumptions in the original task brief (important):**
- **CoVe** first author is **Shehzaad Dhuliawala** et al. (Meta AI), *not* "Shashikumar".
- **RARR** is arXiv:2210.08726 (ACL 2023), *not* 2310.08461 (that ID is DistillSpec).
- **DRAGIN** is arXiv:2403.10081 (Su et al.), *not* 2403.06845 (that ID is DriveDreamer-2).
- **SAFE** is Wei et al. 2024 ("Long-form factuality in large language models", NeurIPS 2024) — it is **not**
  "Min et al. 2023". SAFE ships inside arXiv:2403.18802 together with the LongFact benchmark. (Min et al. 2023
  is FActScore, a different paper.)
- **FaithDial** is arXiv:2204.10757 (not 2204.09736, which is a quantum-physics paper).
- **CoV-RAG** (EMNLP 2024 Findings) = arXiv:2410.05801, "Retrieving, Rethinking and Revising: The
  Chain-of-Verification Can Improve Retrieval Augmented Generation" (He et al.).
- **CoVe-RAG+** = ASME IDETC/CIE 2025 paper by Ashly Joseph, DOI 10.1115/DETC2025-169730, arXiv:2609.26229.
- **VeriGen collision confirmed**: arXiv:2308.00708 is "VeriGen: A Large Language Model for Verilog Code
  Generation" (Thakur et al., 2023) — unrelated to factuality; renaming away from VeriGen is justified.

---

## Part 1 — Core prior art (verified)

### 1. RAG (foundational retrieval-before-generation)
- **Citation:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
  Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela.
  *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
  arXiv:2005.11401. DOI: 10.48550/arXiv.2005.11401. BibTeX key: `lewis2020rag`.
- **Mechanism:** Couples a parametric seq2seq generator (BART) with a non-parametric dense vector index of
  Wikipedia queried by a neural retriever (DPR). Two formulations: one conditions on the same retrieved passages
  for the whole sequence, the other can marginalize over different passages per token. Retrieval happens once,
  before/around generation; there is no feedback loop, no verification, and no repair.

### 2. Self-RAG
- **Citation:** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi.
  *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR 2024 (oral; widely
  reported — arXiv record does not state venue). arXiv:2310.11511. BibTeX key: `asai2024selfrag`.
- **Mechanism:** Trains one LM to adaptively decide *whether* to retrieve on-demand, and to emit special
  "reflection tokens" that critique its own generations (relevance of passages, support/contradiction of each
  generated segment, overall quality) at inference time. This is the earliest mainstream "adaptive retrieval +
  per-segment support checking" system. Limitations relevant to HealGen: support judgments are token-level
  critiques rather than a persistent claim-evidence structure, unsupported segments trigger regeneration/critique
  rather than localized repair, and no calibrated per-claim confidence is emitted.

### 3. CRAG (Corrective RAG)
- **Citation:** Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling. *Corrective Retrieval Augmented Generation.*
  ICLR 2024 (spotlight; widely reported — arXiv record does not state venue). arXiv:2401.15884.
  BibTeX key: `yan2024crag`.
- **Mechanism:** Adds a lightweight **retrieval evaluator** that scores the overall quality of retrieved documents
  for a query and returns a confidence degree that triggers one of three actions (correct / incorrect / ambiguous).
  Incorrect retrieval triggers large-scale web search as a corrective fallback; a decompose-then-recombinance
  (decompose-then-recompose) step filters key information from documents. Plug-and-play with RAG pipelines.
  Granularity is document/query-level, it acts *before* generation, and it never inspects or repairs generated
  claims.

### 4. CoVe (Chain-of-Verification)
- **Citation:** Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz,
  Jason Weston. *Chain-of-Verification Reduces Hallucination in Large Language Models.* Meta AI, 2023 preprint.
  arXiv:2309.11495. BibTeX key: `dhuliawala2023cove`.
- **Mechanism:** Four-stage post-generation protocol: (i) draft an initial response, (ii) plan verification
  questions to fact-check the draft, (iii) answer those questions *independently* (to avoid bias from the draft),
  (iv) generate a final verified response. Decomposition into checkable questions is the closest classical
  analog to claim decomposition, but it happens *after* generation and the final step is a **full-response
  regeneration**, not a localized edit.

### 5. RARR — closest classical prior work for localized repair
- **Citation:** Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan,
  Vincent Y. Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, Kelvin Guu. *RARR: Researching and Revising What Language
  Models Say, Using Language Models.* ACL 2023. arXiv:2210.08726. BibTeX key: `gao2023rarr`.
- **Mechanism:** **R**etrofit **A**ttribution using **R**esearch and **R**evision. A post-hoc system that (1) finds
  external attribution (web search) for the claims in *any* generation model's output, then (2) post-edits the
  output to fix unsupported content while **preserving the original wording as much as possible** (explicit
  preservation constraint). Requires only a handful of training examples, an LM, and web search. This is the
  closest prior work to HealGen's LSHE: it repairs specific unsupported spans against retrieved evidence rather
  than regenerating the whole answer. What it lacks: no pre-generation claim decomposition driving risk estimates,
  no evidence-sufficiency decisions *during* generation (repair is strictly post-hoc), no persistent structured
  claim-evidence graph with contradiction/freshness/confidence edges, and no calibrated per-claim confidence in
  the final output.

### 6. Self-Refine
- **Citation:** Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
  Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann,
  Sean Welleck, Amir Yazdanbakhsh, Peter Clark. *Self-Refine: Iterative Refinement with Self-Feedback.*
  NeurIPS 2023 (widely reported). arXiv:2303.17651. BibTeX key: `madaan2023selfrefine`.
- **Mechanism:** Same LLM acts as generator, feedback provider, and refiner in an iterative loop: generate →
  self-feedback → refine. No external tools, no retrieval, no training. Each iteration regenerates the whole
  output; there is no notion of which claims are unsupported, and no grounding evidence.

### 7. Reflexion
- **Citation:** Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao.
  *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS 2023 (widely reported).
  arXiv:2303.11366. BibTeX key: `shinn2023reflexion`.
- **Mechanism:** Agents verbally reflect on task feedback (scalar or free-form) and store reflections in an
  episodic memory buffer that conditions subsequent trials — reinforcement via linguistic feedback rather than
  weight updates. Aimed at agent tasks (coding, sequential decision-making); self-generated feedback, not
  external evidence; full re-attempts rather than localized edits.

### 8. CRITIC
- **Citation:** Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, Weizhu Chen.
  *CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing.* ICLR 2024 (arXiv comment
  confirms venue). arXiv:2305.11738. BibTeX key: `gou2024critic`.
- **Mechanism:** LLMs validate and progressively amend their outputs by interacting with external tools
  (search engine for factuality, code interpreter for code, toxicity classifier), then revise based on tool
  feedback. Key finding: external feedback is essential — *intrinsic* self-correction alone is unreliable.
  Revision is still largely whole-output; no claim-level bookkeeping or calibration.

### 9. DRAGIN — closest prior work for during-generation retrieval decisions
- **Citation:** Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, Yiqun Liu. *DRAGIN: Dynamic Retrieval Augmented
  Generation based on the Information Needs of Large Language Models.* ACL 2024 (widely reported — arXiv record
  does not state venue). arXiv:2403.10081. BibTeX key: `su2024dragin`.
- **Mechanism:** A **dynamic** RAG framework that decides *when* to retrieve (using a dedicated head that tracks
  the LLM's real-time information need via attention to the prefix during decoding) and *what* to retrieve
  (query constructed from the whole context span where the info need arose, not just the last sentence/tokens).
  Explicitly criticizes static when-to-retrieve rules in prior dynamic RAG. This is the closest analog to
  HealGen's AESP: a mid-generation, information-need-driven retrieval trigger. What it lacks: no hallucination
  risk estimation per claim, no evidence-sufficiency *prediction* (it triggers on uncertainty/information-need
  signals, not on sufficiency of already-retrieved evidence), no repair of generated text, no claim-evidence
  structure, no calibrated output.

### 10. Adaptive-RAG
- **Citation:** Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park. *Adaptive-RAG: Learning to
  Adapt Retrieval-Augmented Large Language Models through Question Complexity.* NAACL 2024 (arXiv comment
  confirms venue). arXiv:2403.14403. BibTeX key: `jeong2024adaptiverag`.
- **Mechanism:** A small classifier predicts incoming query complexity and routes it to one of: no retrieval,
  single-step retrieval, or iterative retrieval. Adaptivity is decided **once, pre-generation**, at the query
  level — no mid-generation decisions, no claim granularity, no verification or repair.

### 11. FActScore — atomic fact decomposition + verification
- **Citation:** Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer,
  Luke Zettlemoyer, Hannaneh Hajishirzi. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long
  Form Text Generation.* EMNLP 2023 (main; confirmed on arXiv). arXiv:2305.14251.
  BibTeX key: `min2023factscore`.
- **Mechanism:** Breaks a long-form generation into a series of **atomic facts** and computes the percentage of
  atomic facts supported by a reliable knowledge source (retrieval + LM judge; <2% error vs humans). Foundational
  for claim-level factuality, but it is a *metric*, not an inference architecture: no repair, no during-generation
  behavior, no calibration of the generator.

### 12. SAFE (Search-Augmented Factuality Evaluator)
- **Citation:** Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu, Nathan Hu, Jie Huang, Dustin Tran, Daiyi Peng,
  Ruibo Liu, Da Huang, Cosmo Du, Quoc V. Le. *Long-form factuality in large language models* (introduces LongFact
  and SAFE). NeurIPS 2024 (arXiv comment confirms). arXiv:2403.18802.
  BibTeX key: `wei2024longformfactuality`.
- **Mechanism:** SAFE is an LLM-agent evaluator that decomposes a long-form response into individual facts and
  evaluates each fact via multi-step reasoning: issue Google Search queries, then decide whether search results
  support the fact. Aggregates via extended F1 over supported facts vs. preferred response length. Outperforms
  crowdsourced annotators (72% agreement; wins 76% of sampled disagreements) at ~1/20th the cost. Again an
  *evaluation* method (post-hoc, per-fact support decisions), not a generation-time repair loop. **Note:** the
  task brief's attribution "Min et al. 2023, Google" is incorrect for SAFE.

### 13. HALoGEN
- **Citation:** Abhilasha Ravichander, Shrusti Ghela, David Wadden, Yejin Choi. *HALoGEN: Fantastic LLM
  Hallucinations and Where to Find Them.* 2025 preprint (arXiv comment: "Preprint"; venue beyond arXiv
  [unverified]). arXiv:2501.08292. BibTeX key: `ravichander2025halogen`.
- **Mechanism:** Benchmark of 10,923 prompts across nine domains plus **high-precision automatic verifiers that
  decompose generations into atomic units and verify each unit** against a high-quality knowledge source.
  Evaluates ~150k generations from 14 LLMs (best models still up to 86% hallucinated atomic facts in some
  domains). Introduces the Type A (incorrect recollection) / Type B (incorrect training knowledge) /
  Type C (fabrication) error taxonomy — useful as motivation for per-claim risk stratification (HealGen's CRE).

### 14. Rowen — adaptive retrieval triggered by detected hallucination risk
- **Citation:** Hanxing Ding, Liang Pang, Zihao Wei, Huawei Shen, Xueqi Cheng. *Rowen: Adaptive Retrieval-
  Augmented Generation for Hallucination Mitigation in LLMs.* Accepted at SIGIR-AP 2025 (arXiv comment
  confirms). arXiv:2402.10612. BibTeX key: `ding2025rowen`.
- **Mechanism:** A consistency-based hallucination detector measures the model's uncertainty about the input
  query by comparing **semantic inconsistencies across multiple responses generated in different languages or by
  different models**. When high uncertainty is detected, external retrieval is activated to rectify the outputs.
  Detection is response-level (not claim-level), retrieval is triggered by disagreement rather than by an
  evidence-sufficiency judgment, and there is no localized repair or calibration.

### 15. CoV-RAG
- **Citation:** Bolei He, Nuo Chen, Xinran He, Lingyong Yan, Zhenkai Wei, Jinchang Luo, Zhen-Hua Ling.
  *Retrieving, Rethinking and Revising: The Chain-of-Verification Can Improve Retrieval Augmented Generation.*
  Findings of EMNLP 2024 (confirmed on arXiv). arXiv:2410.05801. BibTeX key: `he2024covrag`.
- **Mechanism:** Integrates a verification module (scoring, judgment, **query rewriting**) into RAG to fix
  external retrieval errors — retrieval is retried with revised queries — and unifies QA with CoT-style
  verification during *training* to fix internal generation inconsistencies. Corrects at the query/document
  level and trains consistency in; it does not decompose outputs into claims, does not repair generated text
  locally, and does not calibrate confidence.

### 16. CoVe-RAG+
- **Citation:** Ashly Joseph. *Reducing Hallucinations in Large Language Models Through Integrated
  Self-Verification and Retrieval-Augmented Generation.* ASME IDETC/CIE 2025, Paper No. DETC2025-169730,
  DOI: 10.1115/DETC2025-169730. arXiv:2609.26229 (posted to arXiv 2026; journal_ref confirms 2025 venue).
  BibTeX key: `joseph2025coveRAGplus`.
- **Mechanism:** Domain-specific (engineering design/CAD) integration of Chain-of-Verification with RAG: an
  iterative self-verification loop validates key claims against authoritative external sources (standards, CAD
  data, simulation reports) and produces verification reports with source traceability; reports ~28% factual
  accuracy improvement over baseline CoVe/RAG. Post-hoc verification + response-level revision; small,
  application-focused paper, no claim graph or calibration.

### 17. FaithDial
- **Citation:** Nouha Dziri, Ehsan Kamalloo, Sivan Milton, Osmar Zaiane, Mo Yu, Edoardo M. Ponti, Siva Reddy.
  *FaithDial: A Faithful Benchmark for Information-Seeking Dialogue.* TACL 2022 (confirmed on arXiv).
  arXiv:2204.10757. DOI: 10.1162/tacl_a_00508 [DOI unverified]. BibTeX key: `dziri2022faithdial`.
- **Mechanism:** Data-centric mitigation: edits hallucinated responses in the Wizard of Wikipedia benchmark to
  create a faithful dialogue benchmark. Trains a **hallucination critic** (discriminating faithful vs.
  unfaithful utterances; +12.8 F1 on BEGIN) and high-quality generators via an auxiliary contrastive objective.
  Training-time supervision at utterance granularity; not an inference-time architecture.

---

## Part 2 — 2025–2026 landscape (searches: "claim-level hallucination repair", "localized revision
unsupported sentences LLM", "evidence sufficiency adaptive retrieval 2025 2026", "claim evidence graph
factuality"). Search performed via arXiv API (title/abstract), 2026-10-26.

These are the papers that most affect HealGen's novelty positioning. The space has moved fast: by 2026,
**each individual HealGen component has at least one close analog**, though (per Part 3 of the brief, see
`02-differentiation.md`) no single verified paper combines all six claimed differentiators.

### A. Claim-level risk / adaptive verification (threatens CRE)
1. **FACTOR** — Areeba Hassan, Arooj Kausar, Syeda Kisaa Fatima, Gibrail Islam, Mehwish Fatima.
   *Not All Claims Are Equally Risky: FACTOR for Adaptive Verification in Factual Long-Form Generation.*
   arXiv:2606.22474 (2026). **Directly overlaps HealGen's "per-claim hallucination risk estimation" claim:**
   an inference-time model that adapts *verification criteria according to claim-level uncertainty*, combining
   uncertainty estimation, adaptive verification, and candidate re-ranking to spend verification effort where
   risk is highest. Evaluated on FActScore. Key: "the same verification policy usually applies to all claims
   despite differences in hallucination risks" is their framing — identical motivation to HealGen's CRE.
2. **Claim-Level Confidence Calibration** — Toghrul Abbasli, Kentaroh Toyoda, Yuan Wang, Li Chen.
   *Claim-Level Confidence Calibration for Reliable Decision Making with Large Language Models.*
   arXiv:2608.22483 (KDD 2026 Uncertainty Workshop). Decomposes responses into atomic claims and assigns each a
   **calibrated confidence** via inference-time consistency + self-verification (closed-box, post-hoc),
   explicitly enabling selective intervention (evidence retrieval or human review) for low-confidence claims.
   Overlaps HealGen's reliability-calibrated per-claim output and the CRE→retrieve control path.
3. **CURE** — Xin Liu, Lu Wang. *Think Through Uncertainty: Improving Long-Form Generation Factuality via
   Reasoning Calibration.* arXiv:2604.12046 (2026). RL framework teaching models **claim-level confidence
   estimation** (Claim-Aware Reasoning Protocol: atomic claims + explicit confidence), aligned via multi-stage
   training; enables selective prediction/abstention. Strengthens the position that "per-claim calibrated
   confidence" is no longer novel by itself (they do it via training rather than an inference architecture).
4. **UQ heads** — Artem Shelmanov et al. *A Head to Predict and a Head to Question: Pre-trained Uncertainty
   Quantification Heads for Hallucination Detection in LLM Outputs.* arXiv:2505.08200 (2025). Supervised
   auxiliary heads over frozen LLMs achieving SOTA **claim-level hallucination detection** — relevant as a
   cheap CRE module alternative. See also Ghassabi et al., *Single-Pass Uncertainty Heads for Claim-Level
   Hallucination Detection*, arXiv:2610.03482 (2026).

### B. Evidence sufficiency / during-generation retrieval control (threatens AESP)
5. **SURE-RAG** — Jingxi Qiu, Zeyu Han, Cheng Huang. *SURE-RAG: Sufficiency and Uncertainty-Aware Evidence
   Verification for Selective Retrieval-Augmented Generation.* arXiv:2605.03534 (2026). "Relevance does not
   guarantee sufficiency" — a shared **claim-evidence verifier** produces per-(claim, passage) relation
   distributions, aggregated into four interpretable answer-level blocks (coverage, relation strength,
   uncertainty, retrieval) yielding a three-way support/refute/insufficient decision + auditable selective
   score. Overlaps both AESP (sufficiency judgment) and DCEG (structured claim-evidence relations).
6. **SimulRAG** — Haozhou Xu, Dongxia Wu, Matteo Chinazzi, Ruijia Niu, Rose Yu, Yi-An Ma. *SimulRAG:
   Simulator-based RAG for Grounding LLMs in Long-form Scientific QA.* arXiv:2509.25459 (2025).
   **Claim-level generation with uncertainty estimation + simulator-boundary assessment (UE+SBA) to selectively
   verify and update claims** — i.e., decompose → estimate per-claim uncertainty → only ground uncertain,
   verifiable claims. One of the closest 2025 systems to HealGen's overall pipeline shape (minus the
   graph/calibration, plus simulators-as-evidence).
7. **PAR²-RAG** — Xingyu Li et al. *Planned Active Retrieval and Reasoning for Multi-Hop Question Answering.*
   arXiv:2603.29085 (2026). Separates coverage from commitment; depth-first refinement with **evidence
   sufficiency control** in an iterative retrieval loop. Also: **REVEAL** (Caijun Yan et al., arXiv:2608.08612,
   2026, long-video QA) explicitly verifies evidence sufficiency against rubrics and triggers targeted
   re-retrieval; and **RECAST** (Yilun Hao et al., arXiv:2610.10507, 2026), where a router iteratively gathers
   evidence until it "judges the evidence sufficient" before answering. Sufficiency-gated retrieval is now a
   recognizable pattern across domains.

### C. Localized repair of unsupported content (threatens LSHE)
8. **DoctorRAG** — Shuguang Jiao et al. *Doctor-RAG: A Failure-Aware Repair Framework for Agentic Retrieval-
   Augmented Generation.* arXiv:2604.00865 (2026). Trajectory-level failure diagnosis (assesses **evidence
   sufficiency**, classifies failure type, localizes earliest failure point) followed by **tool-conditioned
   local repair that intervenes only at the diagnosed point while reusing valid prefixes** — explicitly avoids
   blind full reruns. Closest 2026 analog to LSHE, though it repairs agent *trajectories* in multi-hop QA, not
   factual claim spans in generated prose.
9. **OPT-Kraken** — Jiageng Wu et al. *Fix What's Wrong, Keep What's Right: Hallucination Repair for LLM-Based
   Optimization Modeling.* arXiv:2610.12529 (2026). Training-free **post-hoc localized hallucination repair**
   that localizes errors (problem-model fidelity, model-code consistency, data grounding) and **selects repair
   granularity by affected scope**, with contracts specifying permitted edits and protected content — the
   "preserve what's right" principle HealGen's LSHE should cite, but in the optimization-modeling domain.
10. **Enoki** — Elisei Rykov et al. *Enoki: Efficient Multi-Level Hallucination Detection.* arXiv:2609.00581
    (2026). OpenIE-based framework that verifies text-anchored relational facts and **projects unsupported
    facts back to hallucinated spans**, unifying claim-level verification with span-level localization
    (+EnokiQA dataset). Localization without repair.
11. **DSVD** — YiQiu Guo et al. *Dynamic Self-Verify Decoding for Faithful Generation in Large Language Models.*
    arXiv:2503.03149 (2025). **Real-time (during-decoding) self-verification with a dynamic rollback
    mechanism** for targeted error recovery — token-level "self-healing" that happens *inside* generation,
    the most direct competitor to the "healing during/after generation" story at the decoding level.
12. **ClaimWriter / CLAIMPROBE** — Hiroaki Hayashi, Pranav Narayanan Venkit, Prafulla Kumar Choubey,
    Chien-Sheng Wu. *Redesigning and Auditing Deep Research Writing for Faithful Reports.* arXiv:2608.28643
    (2026). Hierarchical **claim-based writer** that extracts source facts, maps them to an outline, drafts each
    section from a **source-linked claim representation**, and enables **localized revision when sources
    change** — pre-generation claim decomposition (HealGen step 1) is explicitly present here in a deep-research
    system.
13. **Causal Edit Serialization (CES)** — Louis Robinson. arXiv:2610.22820 (2026). Training-time framework
    giving autoregressive LMs a native **localized revision** interface (pointer tokens for insert/delete/
    replace on earlier spans) — relevant as an alternative implementation route for LSHE.
14. **Clinical verification-routing** — Nina Fatehi et al. *Verification Turns Capability Gaps into Routing
    Decisions: Small Language Models for Clinical Documentation.* arXiv:2610.19169 (2026). Four-evaluator
    verification stack with **section-scoped refinement that preserves unflagged sections by construction** —
    same "repair only what's flagged" principle in clinical note generation.

### D. Claim-evidence structure / graphs (threatens DCEG)
15. **ADORE** — Xincheng You et al. *Orchestrating Specialized Agents for Trustworthy Enterprise RAG.*
    arXiv:2601.18267 (2026). Central to it: a structured Memory Bank implemented as a **Claim-Evidence Graph**
    with explicit claim-evidence linkage and section-level admissible evidence; a retrieval-reflection loop
    audits **evidence coverage** to trigger targeted follow-up retrieval with an **evidence-driven stopping
    criterion**. This single system overlaps DCEG (claim-evidence graph) + AESP (coverage-driven retrieval
    stopping) in an enterprise/deep-research setting.
16. **SAFE-MR** — Shiwen Ni. arXiv:2610.06708 (2026). Multimodal rumor detection that decomposes posts into
    claims, **constructs a relation-aware claim-evidence graph** (provenance, contradictions), and trains
    separate **veracity and sufficiency heads** for selective prediction. Structured claim-evidence
    representation + sufficiency learning, but detection-only (no generation/repair).
17. **MuPlon** — Hanghui Guo et al. arXiv:2509.25715 (2025). Claim verification over a **Claim-Evidence Graph**
    with causal deconfounding (back-door/front-door paths) — shows "claim-evidence graph" as a named
    representation exists in the fact-verification literature. Older root: Causal Walk (Congzhi Zhang, Linhai
    Zhang, Deyu Zhou, AAAI 2024, arXiv:2403.02698) also models fact verification over a claim-evidence graph.
18. **Facet-level tracing** — Passant Elchafei, Monorama Swain, Shahed Masoudian, Markus Schedl. *Facet-Level
    Tracing of Evidence Uncertainty and Hallucination in RAG.* arXiv:2604.09174 (2026). Decomposes questions
    into atomic facets and builds a **Facet × Chunk matrix** combining retrieval relevance with NLI faithfulness
    per facet — a de-facto claim-evidence structure used diagnostically; finding: hallucinations stem more from
    evidence *misuse* than retrieval failure.

### E. Claim-level verification agents/benchmarks (context, less threatening)
- **MARCH** (Zhuo Li et al., arXiv:2603.24579, 2026): Solver/Proposer/Checker multi-agent RL; Proposer
  decomposes answers into claim-level atomic propositions, Checker verifies against evidence in isolation.
- **RT4CHART** (Boxi Yu et al., arXiv:2603.27752, 2026): claim decomposition + hierarchical verification with
  entailed/contradicted/baseless labels mapped back to answer spans (detection, no repair).
- **BALTO** (Ning Li et al., arXiv:2606.15893, 2026): projects claim-level verification judgments to token-level
  RL rewards — claim granularity used for training, not inference-time healing.
- **VISTA** (Ashley Lewis et al., arXiv:2510.27052, ACL 2026): claim-level + sequential-consistency
  factuality evaluation for dialogue, evaluated on FaithDial/BEGIN/AIS/FADE.
- **LLMPEDIA** (Muhammed Saeed, Simon Razniewski, arXiv:2609.01182, 2026): claim-level supported/refuted/
  **insufficient** trichotomy for long-form text — note "insufficient" is the same third state AESP needs.
- **Fine-grained UQ comparative study** (Dylan Bouchard et al., arXiv:2602.17431, TMLR 2026): taxonomy of
  claim-decomposition → unit-scoring → aggregation pipelines; finds claim-level > sentence-level scoring.
- **SERC** (Gyumin Kim et al., arXiv:2605.28837, ICPR 2026): LDPC-inspired sparse verification queries against
  external evidence to detect *and correct* errors in RAG (correction at claim granularity; training-free).

---

## Part 3 — Name-collision check

- **"HealGen"** — arXiv API searches on 2026-10-26: `ti:"HealGen"` → **0 results**; `all:"HealGen"` →
  **0 results**. No arXiv paper, in any field, uses the name.
- The general-web/GitHub/Google Scholar check could **not** be completed: the `websearch` tool was
  non-functional in this research environment (every query returned empty). **[unverified outside arXiv]** —
  run a Google/GitHub/HF-models sweep before committing to the name (possible collisions in biotech
  "healing/generative" tooling are plausible from naming style, though nothing surfaced in what we could check).
- **"VeriGen"** — collision **confirmed**: arXiv:2308.00708, *VeriGen: A Large Language Model for Verilog Code
  Generation* (Shailja Thakur, Baleegh Ahmad, Hammond Pearce, Benjamin Tan, Brendan Dolan-Gavitt, Ramesh Karri,
  Siddharth Garg; NYU, 2023). Do not use this name.
- **Recommendation:** "HealGen" appears free on arXiv as of 2026-10-26. Defensible. Final web-wide check still
  advised due to tool limitation noted above.
