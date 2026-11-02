# HealGen — arXiv submission kit

Status legend: **[DONE]** (agent-completed), **[PARTNER]** (needs a human).

## What this folder contains

| File | Role |
|---|---|
| `main.tex` + `sections/*.tex` | paper source (headlines via `tables/headline.tex`) |
| `tables/*.tex`, `figures/*` | **generated** by `prototype/experiments/make_tables.py` — never hand-edited |
| `refs.bib` | bibliography (research/01-prior-art.md + verified additions) |
| `main.pdf` | locally built PDF (Tectonic 0.17.0, `.tools/tectonic.exe`) |
| `main.bbl` | pre-built bibliography for the arXiv zip (arXiv does not run BibTeX) |

## Rebuild from scratch

```powershell
# 1. regenerate every table/figure/headline from results/*.csv only
python prototype/experiments/make_tables.py
# 2. compile (run twice so cross-refs settle); --keep-intermediates is
#    required on the last pass so paper/main.bbl lands on disk for the zip
& .\.tools\tectonic.exe --keep-logs paper\main.tex
& .\.tools\tectonic.exe --keep-logs --keep-intermediates paper\main.tex
# 3. final artifact
Copy-Item paper\main.pdf paper\healgen-arxiv.pdf
```

Honesty invariants (design §4, enforced at Task 7):

- `grep -rn PLACEHOLDER paper/sections paper/tables paper/main.tex` →
  must return exactly **1** hit: `main.tex:22` `PLACEHOLDER-BYLINE`
  (the partner's byline decision — expected until Step 1 below lands).
  0 hits means the byline was filled in; any *other* hit is an interim
  stub that `make_tables.py` should have overwritten.
- every number in prose is a headline macro (`\SPHealGen`, …) or inside
  an `\input{tables/…}` — `grep -E '[0-9]+\.[0-9]{2,}' paper/sections`
  should only hit protocol constants (seeds, k1, bins) — audit manually.
- overclaim sweep: `grep -riE 'state-of-the-art|outperform|superior|SOTA|beats' paper/`
  → must be 0 (spec §8.1).

## arXiv zip contents (source route — required for LaTeX papers)

Zip the **files** (not the folder), at these paths:

```
main.tex
main.bbl
refs.bib
sections/intro.tex related.tex architecture.tex method.tex
      experiments.tex results.tex limitations.tex conclusion.tex
tables/headline.tex main_results.tex efficiency.tex pairwise_tests.tex
      ablation.tex tau_sensitivity.tex judge_validation.tex
figures/reliability_diagram.png latency_retrieval.png dceg_example.png
```

Build notes: preamble is pdfLaTeX-compatible (T1 fontenc + utf8 inputenc,
no fontspec), figures are all PNG (single format family), no `\pdfoutput`
override, no copyright block.  arXiv compiles with its own TeX Live;
`.bbl` supplies the bibliography.

## Submission steps **[PARTNER]**

1. arXiv account + **cs.CL endorsement** — policy updated 2026-01-21
   (details/sources: `research/04-arxiv-mechanics.md` §Endorsement):
   - **Plan A (automatic, only if both hold):** institutional email
     *and* a claimed prior arXiv-accepted paper in the cs.CL endorsement
     domain (claim via info.arxiv.org/help/authority.html).
   - **Plan B (personal endorsement, the likely route):** one positive
     endorsement from an established arXiv author who knows you
     personally and is knowledgeable in the area (thesis advisor /
     faculty).  **Arrange days ahead — not the night before**; do not
     mass-mail or repeat-contact potential endorsers.
2. New submission → license (default arXiv non-exclusive is fine) →
   upload zip → wait for the compile preview → verify the rendered PDF
   against the local build (fonts, figures, citations resolved).
3. Metadata: title (as in `main.tex`), abstract as **plain text** (expand
   macros by hand from the PDF — no `\SPHealGen`-style tokens in the
   metadata field), authors/byline **[decision pending]**, subjects:
   primary `cs.CL`, optional cross-list `cs.AI`.
4. Comments field — suggested draft (8 pp, 4 figures, 6 tables):
   > 8 pages, 4 figures, 6 tables. Draft-first claim-level self-healing
   > generation with a frozen shared-pool support protocol; code, frozen
   > evidence pools, and claim-level verdicts released.
5. Deadline math: received by 14:00 ET → announced 20:00 ET same day.
6. After moderation: share the paper ID; keep `main.bbl` in sync for any
   v2.

## Decisions still open (partner)

- **Byline / endorsement** (design spec §8.2): solo vs. faculty
  supervisor — affects `\author{}` in `main.tex` (currently
  `PLACEHOLDER-BYLINE`).
- Human spot-check rating (100 claims in `results/spotcheck_sample.csv`,
  `human_supported` column) → then Cohen's κ goes into the paper as the
  judge–human agreement number.
- Which arXiv license + whether to also publish the repo publicly (a
  code/data link must resolve publicly if stated in the paper).
