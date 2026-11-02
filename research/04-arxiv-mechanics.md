# arXiv submission mechanics (researched 2026-11-02)

Sources: info.arxiv.org official pages (format requirements, submission
overview, submit_tex, submit_pdf), plus community guides (Underleaf
2026-08, Carlini release checklist, Skidmore 2025-06, Cardenas-Avendano).
Official pages are authoritative; community guides marked as such.

## Endorsement policy (CONFIRMED, retrieved 2026-11-02)

Policy **updated 2026-01-21** for all sections (blog: "Attention
Authors: updated endorsement policy", blog.arxiv.org/2026/01/21/…;
help: info.arxiv.org/help/endorsement.html):

- Institutional email **alone no longer qualifies** for endorsement of
  new authors (this had been true in Math since 2025-12-10).
- **Path 1 (automatic):** requires BOTH (1) an institutional email
  meeting arXiv's criteria AND (2) claimed prior authorship of an
  existing arXiv-accepted paper in the endorsement domain of the target
  category (claim via info.arxiv.org/help/authority.html).
- **Path 2 (personal):** otherwise, one positive personal endorsement
  from an established arXiv author who knows you personally and is
  knowledgeable in the subject area (canonical: thesis advisor or a
  professor in your department).  At least one positive endorsement is
  required per endorsement category.  Mass-mailing or repeatedly
  pestering endorsers is explicitly prohibited.
- UNCERTAIN (confirm at submission time): the exact endorsement-domain
  boundary for `cs.CL` (cs.CL vs the wider `cs` domain), and whether a
  given institution's address meets the "institutional email criteria".
- Practical reading for us: if the partner has no prior arXiv paper to
  claim, **Path 1 is closed regardless of institutional email** — Path 2
  (ask an advisor/faculty member who has published on arXiv) is the
  realistic route.  Arrange days ahead; endorsers are humans with
  inboxes.

## The decision that shapes everything: source, not PDF

- arXiv **requires (La)TeX source when the paper was written in LaTeX**;
  PDFs produced from TeX are "typically rejected, case-by-case
  exceptions" (info.arxiv.org/help/submit_pdf).  Policy rationale:
  arXiv recompiles on its own TeX Live so the stored artifact is
  portable (info.arxiv.org/help/faq/whytex).
- Accepted order of preference: (La)TeX > PDF > HTML.  Figures: PS/EPS
  (legacy latex→dvips) or PDF/PNG/JPG (pdfLaTeX route).  **No mixed
  figure formats** — one family only.  Our figures are PNG (matplotlib)
  + PNG (dceg) → fine for the pdfLaTeX route.

## What the zip must contain

- main .tex + every file it `\input`s/\include`s (sections/*, tables/*,
  figures/*) **at the exact referenced paths**.
- **the `.bbl` file** — arXiv does not run BibTeX; the bibliography must
  arrive pre-built (community guides + official LaTeX best-practices
  page).  Our tectonic run produces `main.bbl` — include it.
- `refs.bib` can be included too (harmless), but `.bbl` is what gets used.
- IEEEtran: arXiv's TeX Live ships `IEEEtran.cls`; optionally include the
  class file in the zip for safety (only if redistribution terms allow —
  IEEEtran is LPPL, generally fine; keep it optional).
- Zip the FILES, not the containing folder (Skidmore guide).

## Compilation compatibility (our specific risk)

- Our preamble: T1 fontenc + utf8 inputenc + graphicx + booktabs +
  amsmath + tikz + hyperref — **no fontspec, no XeTeX-only features**,
  so the document is pdfLaTeX-compatible even though we build locally
  with Tectonic (XeTeX engine).  arXiv auto-detects; pdfLaTeX is the
  default route.
- Do NOT set `\pdfoutput=1` (old advice; current arXiv docs say don't
  override output detection — mixed signals cause failures).
- Pre-flight: compile once more and grep the log for
  `error|Undefined|undefined|Missing` — the five common arXiv failures
  are: mixed figure formats, missing \input files, missing .bbl,
  missing figures at referenced paths, macro-in-abstract.

## Format requirements (info.arxiv.org/help/policies/format_requirements)

- Title + authorship (no anonymous submissions).
- Complete references; links to code/data must resolve publicly.
- Machine readable; single spaced; 10–14 pt type; ≥ 1" margins.
- No copyright statements that impair arXiv's redistribution license
  (IEEE copyright blocks must be removed — we have none).
- Large non-text content (long code/image/table sections) → ancillary
  files rather than appendices.
- Moderation: scholarly-interest standard applies; new accounts in a
  category need an **endorsement** from someone who has published there
  (partner-side task).

## Metadata + timing

- Metadata fields: title, abstract (plain text — **no LaTeX macros** in
  the metadata abstract even if the PDF abstract uses them; macro-heavy
  abstracts are a checklist item from Carlini's guide), authors, subjects
  (cs.CL is the natural primary; cs.AI cross-list optional), comments
  field (page count etc. — optional).
- License choice at submission (default arXiv non-exclusive license is
  the norm).
- **Timing:** submissions received by 14:00 ET are generally announced
  20:00 ET same day (availability schedule page).
- Version replacement allowed later (v2 for corrections).

## Pre-submission checklist (merged from official + Carlini + practical)

1. All co-authors approved (partner task); author names spelled right.
2. Title correct; no TODO/PLACEHOLDER markers anywhere in source —
   **except** the known byline placeholder in `main.tex`
   (`PLACEHOLDER-BYLINE`, partner decision), which must be replaced with
   the final `\author{}` before the zip is built.
3. Build log clean: no `error`, `Undefined`, `Missing`, `undefined
   citations`; rerun until clean (bibtex pass included).
4. Abstract metadata = plain text (macros expand only in the PDF).
5. No `\thanks`/copyright block issues (IEEEtran `\thanks` is already
   warned as locked-out in conference mode — verify rendering).
6. Every figure resolves at its path; one format family (PNG).
7. `.bbl` present in zip; zip contains files not the folder.
8. PDF preview on arXiv matches local build (spot-check fonts/figures).
9. Comments field: page/figure count if desired.
10. Endorsement arranged ahead of time via one of the two current paths
    (see §Endorsement above — institutional email alone is no longer
    sufficient as of 2026-01-21).  Partner task; never the night
    before; endorsers are humans with inboxes.

## Out of agent scope (partner-only)

- Creating/owning the arXiv account, obtaining cs.CL endorsement.
- Choosing byline + affiliations, license pick, final metadata entry,
  clicking submit, handling moderation questions.
