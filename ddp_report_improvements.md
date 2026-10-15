---
name: ddp-report-improvements
description: "DDP final report improvements applied in June 2026: 18-item list of LaTeX changes — margins, headings, figures, tables, content"
metadata: 
  node_type: memory
  type: project
  originSessionId: 791ccf3f-9306-4772-9b4a-0ad248b62960
---

All 18 improvements from the user's checklist were applied to `Desktop\DDP\Final Report\` in session June 2026.

**Why:** User reviewed the generated report and found formatting, visual, and content issues.

**How to apply:** If user asks about report status, these are all done. Next step is inserting pipeline-generated PNG figures using `insert_results.py`.

## Changes made per file

### main.tex (complete rewrite)
- Margins: left=3.0cm, right=2.0cm (was 3.5/2.5)
- `\raggedbottom` added — eliminates white space gaps between pages
- `tocbibind` replaced with `tocloft`; LOF uses "Fig. X.Y", LOT uses "Tab. X.Y" (different from TOC)
- Chapter headings: `\raggedright` (was `\filcenter`) — now left-aligned
- IITB logo: `\IfFileExists{figures/iitb_logo.png}` guard added (place logo PNG in figures/)
- Nomenclature moved from front matter to just before `\printbibliography`
- `needspace` + `etoolbox` added; `\preto{\section}{\needspace{5\baselineskip}}` prevents section orphaning globally
- Float placement default: `ht` (less aggressive than `htbp`, avoids stacking)
- `tabularx`, `multirow`, `enumitem`, `etoolbox` packages added

### 01_introduction.tex
- Section 1.5 "Organization of the Report" removed entirely
- New TikZ flowchart `fig:framework_overview` added after Objectives section: shows data → features → two protocols → models → probabilistic layer → ensemble

### 02_literature.tex
- Figure 2.3 (`fig:patching`) fully redrawn: cleaner two-pass signal (drawn under then over patch boxes), alternating patch shading, arrows to tokens, encoder block, forecast block
- Section 2.8 title shortened: "Text, Sentiment, and Retrieval-Augmented Forecasting" → "Text and Retrieval-Based Features"
- Captions shortened: decomp pipeline and TSF pipeline captions now one line each
- Synthesis text updated: "protocols that are reproduced" → "architectures that are implemented for validation"

### 03_dataset.tex
- Table 3.1 (`tab:exog`) restructured: grouped into three sub-groups (market signals, fundamentals, behavioural) with `\multicolumn` sub-headers; better column widths
- New equation `eq:features`: momentum Mom_{t,h} and rolling volatility σ_{t,w} formulas
- New TikZ bar chart `fig:feat_counts`: horizontal bar chart of ~12 feature categories with approximate counts
- Figure captions shortened

### 04_methodology.tex
- Section 4.2 renamed: "Published Reproduction and Strict Walk-Forward" → "Global Fit and Walk-Forward"
- Section 4.3 renamed: "Reproducing the Published Decomposition Protocols" → "Published Decomposition Architectures"
- Section 4.3 text updated: "Two published protocols are reproduced" → "Two published architectures are implemented and evaluated"
- New equation `eq:crps`: CRPS formula added to Section 4.7 (Calibration diagnostics paragraph)
- Figure captions shortened

### 05_implementation.tex
- Figure 5.1 (`fig:workflow`): `minimum width=11.5cm` → `minimum width=0.86\linewidth`; annotation "synchronise results" moved above arrow (was `right=1mm`, causing overflow)
- Caption shortened

### 06_results.tex
- Section 6.11 Discussion expanded from ~10 lines to ~60 lines: four structured questions, expected pattern discussion, quantitative framing
- New TikZ diagram `fig:expected_pattern`: 4-column (horizon) × 2-row (skill/gap) matrix diagram
- New TikZ diagram `fig:ensemble_diagram`: shows model families → SLSQP weights → ensemble → DM test vs benchmarks
- All figure captions shortened to 1 line each

### 07_conclusions.tex
- Line 24: "protocols are reproduced for validation" → "architectures are implemented for validation"

## New figures summary (all TikZ, no placeholder)
- `fig:framework_overview` (Ch 1): overall methodology flowchart
- `fig:feat_counts` (Ch 3): horizontal bar chart of feature counts by category
- `fig:expected_pattern` (Ch 6): horizon × skill/gap matrix
- `fig:ensemble_diagram` (Ch 6): ensemble architecture with DM test

## Round 2 formatting fixes (same June 2026 session, 13 items)

User reviewed the rendered PDF and flagged 13 more layout issues. Decisions: **equal 2.0 cm margins** (text width 17 cm); **keep chapters on new pages** (chapter-end whitespace accepted as unavoidable). Plan file: `c-users-anand-desktop-important-papers-zany-abelson.md`.

### main.tex
- Geometry → `left=2.0cm,right=2.0cm,top=2.5cm,bottom=2.5cm`
- Added `\usepackage{parskip}` — block paragraphs (no first-line indent, line-space between paragraphs)
- `\captionsetup` → added `justification=centering` (centers all figure/table captions)
- Added `\clearpage` before `\tableofcontents`, `\listoftables`, `\listoffigures` (each on fresh page)
- Removed the "This is a placeholder approval sheet..." note (Declaration's similar note left intentionally)

### Tables — all data tables full-width + caption BELOW the table
- Numeric tables → `\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}...}`: `tab:wf`, `tab:archparams`, `tab:baselines`, `tab:protocolgap`, `tab:leaderboard_short`, `tab:leaderboard_long`, `tab:probmetrics`, `tab:evt`, `tab:garch`
- Text tables → `\begin{tabularx}{\textwidth}{...X...}`: `tab:prior_vs_current`, `tab:exog`, `tab:featuresets`, `tab:architectures`, `tab:apphyper`
- Every `\caption{}\label{}` moved to AFTER `\end{tabular...}`
- Fixed latent bug in `tab:archparams`: column spec was `lccccccc` (8) vs 7 columns → now `lcccccl`
- Exception: `tab:fulffeatures` (longtable) keeps caption at top; widened to `p{4.5cm}p{12.0cm}`
- Nomenclature longtables left as-is (symbol glossaries, not data tables)

### Figures
- **Fig 5.1 `fig:workflow`** (overflow fix): changed node style from `minimum width=...` (does NOT wrap) to `text width=0.80\linewidth, align=left, inner sep=6pt`; removed the sloped "synchronise results" arrow labels that overlapped boxes; changed "Phase 1 -- local" en-dash separators back to commas
- **Fig 1.1 `fig:framework_overview`**: was an unreferenced float (odd placement) → added `\Cref{fig:framework_overview}` sentence at end of Sec 1.4; placement `[ht]`
- **Fig 3.2 `fig:feat_counts`**: compacted (`\yskip` 0.52→0.46); moved axis tick labels + "Approximate feature count" title to BOTTOM (was colliding with top ticks); placement `[t]`; merged orphaned trailing sentence into the prose paragraph
- **Section 2.8 split**: added `\clearpage` before `\section{Text and Retrieval-Based Features}`
- Shortened captions: `fig:patching`, `fig:yu2025_framework`, `fig:huang2024_framework`, `fig:walkforward`, `fig:ensemble_diagram`, `fig:decomp_pipeline`, `fig:tsf_pipeline`, `fig:expected_pattern`, plus `tab:protocolgap`/`tab:leaderboard_short` captions

### Not locally verifiable
No TeX toolchain on the machine (pdflatex/biber/latexmk all absent) — user compiles on Overleaf. Verified syntax by hand: all tabular*/tabularx begin/end balanced.

## Round 3 — Two-horizon rewrite with real results (same June 2026 work)

Pipeline run was partial and plots could not be generated. Report cut from 4 horizons to **1-day and 7-day only**, and result placeholders filled with **real data tables** from `Desktop\Track A and B.txt`. User decisions: clean two-horizon study; keep EVT/GARCH + probabilistic (CRPS/PIT) tables as `[pending]` reduced to 1d/7d rows; convert result placeholders to data tables, keep TikZ, drop image-only placeholders.

**Key real numbers used (1d / 7d MAE):** honest walk-forward neural panel — Informer 1.23/1.96 (best 7d, beats ~$2.25 Huang benchmark), TimeMixer 1.23/2.14, TCN 1.23/2.14, TCNBiLSTM 1.23/2.16, PatchTST 1.22/2.30, NHiTS 1.22/2.35, TFT 1.23/2.93, iTransformer 1.23/3.02, Chronos zero-shot 2.97/4.01. **At 1d all models collapse to naive (~$1.22)** — honest finding, reported as such. Global-fit reproduction best 1d 0.60 (NLinear) / 7d 1.18 (DLinear). Protocol gap large (decomposition-linear 0.60→1.66 at 1d). Ensemble SLSQP 0.52/1.09, ridge 0.36/0.82 — **labelled reproduction-protocol only** (built over global-fit models; honest-WF ensemble pending). Regime day-counts 3117/2790/3227/1024. Crisis-stratified 7d MAE table from per-model Normal/GFC/2022/COVID blocks.

**06_results.tex** fully rewritten: EDA crisis-window + regime-distribution tables (replaced fig:price_overview/fig:regime); baselines 1d/7d (naive 1d=1.22, rest pending); tab:protocolgap filled; tab:leaderboard_short = honest WF leaderboard; **removed Long-Horizon section** (sec:long_results, tab:leaderboard_long, fig:perhorizon); probabilistic/EVT/GARCH tables kept `[pending]` at 1d/7d; reproduction + ensemble tables added (tab:reproduction, tab:ensemble); fig:regimemae → tab:regimemae; fig:expected_pattern TikZ trimmed 4→2 columns. Dropped figure placeholders: baselines, timeline, errortime, perhorizon, fan, pit, reliability, evt, dmheatmap, modelcompare.

**Reframing edits:** main.tex abstract (four→two horizons); 01 (long-horizon "most valuable" framing, prior-vs-current table, objectives); 03 (feature-set prose keep short set only, **dropped tab:featuresets**, reworded tab:exog rationale, dropped fig:price_overview/fig:regime); 04 (Four→two horizons, tab:wf + tab:archparams 90/180 rows removed, Student-t loss reframed as future work, DM benchmark→7-day, **redrew fig:yu2025_framework + fig:huang2024_framework as native TikZ** replacing the last two figplaceholders); 05 (lookback/loss prose); 07 (four→two, long-horizon limitation→future work); 08 (apphyper loss rows merged to one MSE row, **removed Appendix C** Supplementary Figures with all 4 image placeholders); 02 ("at long horizons"→"for longer horizons").

**Verified (grep):** zero `\figplaceholder` (only the `\newcommand` def in main.tex); no `90 day`/`180 day`/`four horizon` in body (remaining ninety/eighty are feature windows + confidence levels + 96-day lookback); zero dangling `\cref` to removed labels; no leak/tilde/em-dash; all table/figure/tikz begin-end balanced (11 tables in 06). User still compiles on Overleaf.
