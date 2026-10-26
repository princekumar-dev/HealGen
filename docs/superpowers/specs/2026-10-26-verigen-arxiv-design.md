# VeriGen → arXiv Publication: Design Document

Date: 2026-10-26
Author (partner): Prince R (243115243077), 5th semester, Dept. of AI & Data Science,
Meenakshi Sundararajan Engineering College — ECUBE 2026 paper presentation.

## 1. Intent and Why

Turn the VeriGen concept (currently a 9-slide symposium deck, `verigen.pdf`) into a
**real, citable arXiv preprint** backed by three things the deck does not have:

1. **Deep prior-art research** — a defensible map of what exists (RAG, Self-RAG, CRAG,
   CoVe, FActScore, etc.) and the exact sentence that says what VeriGen does that none
   of them do.
2. **A working prototype with real numbers** — an actual small-scale implementation of
   the 9-layer pipeline that runs on this machine and produces honest experimental
   results (no fabricated tables).
3. **A professionally styled paper** — LaTeX, arXiv-ready, real figures, real result
   tables, honest limitations section.

Target: **arXiv** (chosen by partner). Secondary venue: ECUBE 2026 symposium (already
prepared deck).

## 2. The System Being Described (from the deck's diagrams)

VeriGen: *Adaptive Evidence-Sufficiency-Guided Self-Healing Inference Architecture* —
a 9-layer inference pipeline:

| Layer | Name | Function |
|-------|------|----------|
| 1 | Semantic Query Understanding | intent / domain / complexity / reasoning type |
| 2 | Atomic Claim Planner | decompose query into independent atomic claims |
| 3 | Claim Risk Estimator (CRE) | per-claim hallucination risk: low / medium / high |
| 4 | Adaptive Evidence Sufficiency Predictor (AESP) | is current evidence sufficient? |
| 5 | Adaptive Evidence Acquisition | retrieval only when AESP says NO → Evidence Bundle |
| 6 | Dynamic Claim-Evidence Graph (DCEG) | claims ↔ evidence, contradictions, freshness, weights |
| 7 | Localized Self-Healing Engine (LSHE) | repair actions: retrieve more, rephrase, reduce confidence, express uncertainty, reconnect claim — only on affected segments |
| 8 | Adaptive Claim-Aware Generation (ACAG) | reliability-aware generation |
| 9 | Reliability Calibration & Response Composer | verified response + claim reliability score + uncertainty statement |

Stated novelty vs. existing work: claim-level evidence-sufficiency assessment,
localized (not full-response) repair, and adaptive (not pre/post-hoc) retrieval
*during* generation.

## 3. Goals

- **G1** A research report (`research/`) with a prior-art map (2023–2026), a
  feature-comparison table, the differentiation statement, and a target-venue/authorship
  analysis including the exact arXiv 2026 endorsement path available to a first-time
  undergraduate author.
- **G2** A runnable Python prototype (`prototype/`) implementing all 9 layers at small
  scale, plus 3 baselines (vanilla generation, standard RAG, CoVe-style post-hoc
  verification), evaluated on a public benchmark with claim-level metrics —
  **every number in the paper comes from a reproducible run on this machine.**
- **G3** A compiled arXiv-ready PDF (`paper/`) in LaTeX with: abstract, related work,
  method (with a redrawn professional architecture figure), experiments, results,
  limitations, conclusion, proper citations (BibTeX), honest scoping.

## 4. Non-Goals and Anti-Goals

- **No new LLM training/fine-tuning of large models.** No GPU exists here.
- **No fabricated or hand-written results.** Every table is generated from run outputs.
- **No multimodal / medical / legal extensions** (the deck's "future work" stays future work).
- **No fixing of the slide deck's errors** (note: its Methodology slide is Mendix
  low-code boilerplate, unrelated to VeriGen — flagged, not part of this project).
- **No peer-reviewed-venue submission** in this project — arXiv preprint only.
- Anti-goal: overclaiming. The paper must be honest that the prototype is a
  small-scale CPU demonstration, not a production system.

## 5. Constraints

- Windows 11, Python 3.14.8, **CPU-only (no NVIDIA GPU)**, no ML packages installed yet
  (torch 2.14 / transformers 5.20 / sentence-transformers 6.2 are installable from PyPI).
- **No API keys available** (checked: OpenAI/Gemini/Anthropic/HF/etc. all absent).
  Everything must run locally; model weights downloaded from Hugging Face (network OK).
- **No LaTeX** installed → use **Tectonic** (single static binary, auto-fetches TeX
  packages) as the compiler.
- arXiv 2026 policy: updated endorsement rules (Jan 2026). First-time submitters need
  either an endorsement from an established author in the category or the
  institutional-email path — exact requirements to be nailed down in G1 research.
- Name collision: **"VeriGen" is already a heavily-cited paper** (arXiv:2308.00708,
  Verilog LLM generation, ~724 citations). Title decision is an open question (§8).

## 6. How It Gets Made (builder's decisions — stated as calls)

These are defaults chosen by the builder; partner may override any of them:

- **6.1 Prototype stack** (all local, CPU-feasible):
  - Generator LM: `google/flan-t5-small` (fallback `flan-t5-base` if CPU speed allows).
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
  - Retrieval (Layer 5): BM25 (`rank_bm25`) over a Wikipedia-abstract / HotpotQA
    paragraph corpus, no vector DB needed at this scale.
  - Claim decomposition (Layer 2): rule/LLM-prompt based splitter + `spaCy`-less
    sentence segmentation.
  - CRE (Layer 3): small trained classifier (logistic regression / gradient boosting)
    over claim features (specificity, entity rarity, embedding uncertainty proxies) →
    risk score; calibration reported.
  - AESP (Layer 4): embedding-based entailment/sufficiency scoring of claim vs.
    evidence bundle; threshold decides retrieval trigger.
  - DCEG (Layer 6): `networkx` graph with claim/evidence nodes, contradiction and
    dependency edges, freshness + confidence attributes; exportable as DOT/PNG figure
    (used as a paper figure).
  - LSHE (Layer 7): localized re-generation of only unsupported claim sentences with
    retrieved evidence injected; five repair actions from the deck implemented.
  - ACAG + Calibration (Layers 8–9): assemble response with per-claim reliability
    scores and an uncertainty statement.
- **6.2 Benchmark + metrics**:
  - Primary dataset: **FaithDial** (hallucination in knowledge-grounded dialogue) or
    **HotpotQA** subset — final pick during implementation based on CPU runtime;
    ≤ 200 test instances.
  - Baselines: (a) vanilla generation, (b) standard RAG, (c) CoVe-style verify-then-fix,
    (d) full VeriGen — **same generator model for all four** for fairness.
  - Metrics: claim-level factuality / supported-claim precision (FActScore-style),
    hallucination rate, number of retrieval calls, end-to-end latency, and a
    calibration metric (ECE or Brier).
  - **Factuality judge (decided after builder check)**: a local pretrained NLI model
    (e.g. DeBERTa/BART MNLI variant) scoring each atomic claim against a **fixed gold
    evidence pool shared by all four systems** — never against each system's own
    retrievals (which would let VeriGen win by construction, since it retrieves more).
    Supplemented by judge-free metrics (retrieval calls, latency, calibration) so
    headline conclusions never rest on one judge. No API keys anywhere.
- **6.3 Paper format**: `IEEEtran` conference class (tectonic-compatible, standard for
  engineering-symposium-affiliated work), arXiv-compatible BibTeX bibliography,
  figures produced from the actual prototype (pipeline diagram redrawn cleanly +
  DCEG export + matplotlib result plots).
- **6.4 Repo layout**:
  ```
  verigen/
    research/          # 01-prior-art.md, 02-differentiation.md, arXiv-mechanics.md
    prototype/         # verigen/ package (9 layer modules), baselines/, experiments/, results/*.csv
    paper/             # LaTeX sources, figures/, refs.bib, verigen-paper.pdf
    docs/superpowers/specs/   # this document
  ```
- **6.5 Reproducibility**: fixed seeds, `results/*.csv` → LaTeX tables generated by a
  script, README with one-command reproduction.
- **6.6 Phasing**:
  - Phase A — deep research (G1): prior art, differentiation, arXiv mechanics.
  - Phase B — prototype (G2): 9 layers + baselines → run experiments → results.
  - Phase C — paper (G3): write LaTeX around the real results → compile → review pass.
  - Phase D — submission kit: arXiv checklist, endorsement plan, cover comment.

## 7. What's Left to the Builder

Everything not marked as partner decision below: exact benchmark pick, model choice
within the stated constraints, module APIs, experiment scripts, LaTeX wording, figure
design, citation selection, runtime tuning.

## 8. Decisions (resolved 2026-10-26 — partner approved with the defaults below)

1. **Null/mixed results policy (partner: accepted builder recommendation):**
   The paper is framed as an **architecture + measurement paper**. Headline claims are
   about the claim-level self-healing design, localized repair, adaptive retrieval, and
   the measured trade-offs/ablations — not about beating SOTA. Results are reported
   honestly whatever they show. No seed/benchmark shopping until a win appears.
2. **Endorsement/authorship (facts unknown — contingency planned):**
   Phase D plans both paths: (a) institutional (.ac.in) email route if available,
   (b) endorsement by a faculty co-author/supervisor or category endorser. Byline
   stays flexible until submission; default is solo with a faculty supervisor listed
   if one participates.
3. **Deadline (default):** ~4 weeks to a compiled arXiv-ready PDF, submission kit
   ready at the end; PDF compiles early (end of Phase B) so scope can shrink if a
   real date appears sooner.
4. **Keys (default accepted):** strictly local — no API keys in pipeline *or* judge.
5. **Name (default accepted):** rename the paper's system to **HealGen** (working
   title: *"HealGen: Claim-Level Evidence-Sufficiency-Guided Self-Healing Inference
   for Factual LLM Generation"*), with a name-collision verification step early in
   Phase A; "VeriGen" retained only as internal codename in the deck. Rename stays
   cheap until Phase C (figures/abstract).
