# Research Roadmap: Multi-Horizon Crude Oil Price Forecasting with Transformers and Agentic AI

## Context

The goal is publication-level work that beats existing benchmarks for crude oil price forecasting across horizons.

**Your choices:**
- Venues: a Q1 forecasting/energy journal **and** an ML workshop.
- Horizons: daily (1–66 trading days) and monthly (1–24 months).
- Timeline and compute: about 6 months (week 1 = 19 Oct 2026, ending mid-Apr 2027) on a single GPU (Colab/Kaggle).
- Agentic AI plays two roles: (a) **a forecasting component**, the main focus of the workshop paper and one evaluated module in the journal paper, and (b) **research infrastructure**, which supports the work but is not claimed as a contribution.
- LLMs: open-weight models plus a small API budget.

The project folder is empty, so everything starts from scratch.

**Gaps in the field this work targets:**
1. **Weak benchmarks.** DL oil papers rarely compare against the no-change or futures forecasts, which the econometric literature shows are hard to beat (Alquist, Kilian & Vigfusson 2013; Baumeister & Kilian 2012, 2015).
2. **Look-ahead leakage.** Decomposition hybrids (EMD/VMD-LSTM) usually decompose the full series before splitting. General critiques exist (arXiv 2509.15394, 2512.06932), but there is **no audit of the oil literature**.
3. **Untested transformers.** No study compares modern transformers (PatchTST, iTransformer, TimeXer) and TSFMs (Chronos-2, TimesFM 2.5, Moirai 2, TiRex) on crude oil with exogenous drivers, against econometric benchmarks and with formal tests. The closest analogue covers agricultural commodities (arXiv 2601.06371).
4. **Agentic forecasting is new, and its leakage is unaddressed.** Recent frameworks (TimeCopilot 2509.00616, TimeSeriesScientist 2510.01538, KairosAgent 2605.30002, last-mile agents 2606.02497) have not been applied to oil. **LLM look-ahead bias** (memorized history) makes most historical LLM backtests unreliable (Look-Ahead-Bench 2601.13770; DatedGPT 2603.11838; Chronologically Consistent LLMs 2502.21206).
5. **Structural breaks** (2014–16, COVID with negative WTI, 2022, the 2025 shocks) break models, and point forecasts lack calibrated uncertainty.

**Being honest about "beating benchmarks":** daily 1–5 day MSPE gains over the random walk will be tiny. The realistic wins are at monthly 1–6 month horizons, in high-volatility regimes, in directional accuracy and in probabilistic accuracy (CRPS and calibration). Gains are claimed only where formal tests support them.

---

## 1. Research questions and contributions

| RQ | Question | Contribution |
|---|---|---|
| RQ1 | Under a leakage-free, rolling-origin protocol, do DL models, transformers and TSFMs beat the no-change and futures benchmarks, and at which horizons? | Rigorous multi-horizon benchmark (daily + monthly) |
| RQ2 | How much of the reported gain of decomposition hybrids comes from leakage? | Literature audit plus a leaky-vs-causal experiment |
| RQ3 | Does an anchor-residual, regime-routed transformer give significant statistical and economic gains? | **AR-MoE Transformer** (§2) |
| RQ4 | For oil, does attention (temporal and exogenous) add value over attention-free backbones of equal capacity? | Controlled transformer study that answers the DLinear critique (§3) |
| RQ5 | Does an agentic LLM layer that reads point-in-time news and reports add value beyond the quantitative model, and how much of its apparent value is LLM look-ahead bias? | **OilAgent** plus a leakage-controlled agent evaluation and a live forward test (§4) |
| RQ6 | Do the probabilistic forecasts stay calibrated through breaks, and which information drives gains at each horizon? | Online conformal calibration; horizon-resolved interpretability |

**Pre-registered success criteria:**
- MSPE ratio < 1 against the random walk with Clark–West p < 0.05 at ≥ 3 horizons.
- In the 90% Model Confidence Set at most horizons.
- Multi-horizon SPA rejection.
- CRPS better than a RW+GARCH-t density.
- Significant Pesaran–Timmermann directional accuracy.
- Positive certainty-equivalent return after costs.

**Priorities (MoSCoW), to protect the 6-month window:**
- **Must:** benchmark, leakage audit, AR-MoE Transformer, the transformer study, and LLM event features as covariates (agent mode M1).
- **Should:** agent adjudication (M2/M3), memorization probes, live forward test.
- **Could:** chronological-vintage LLM comparison, domain pretraining on an energy-commodity panel.
- **Won't, in this window:** RL fine-tuning of the agent.

---

## 2. Proposed model: AR-MoE Transformer (Anchor-Residual, Regime-routed Mixture-of-Experts)

The design principle is that the random walk is hard to beat, so the model **learns a state-dependent correction on top of a strong anchor**. Its backbone is an exogenous-aware patch transformer derived from TimeXer.

**Inputs at origin t:**
- Endogenous lookback: log price over L = 128–512 trading days.
- Exogenous series, each at its native frequency, forward-filled to its *release date*, with a "release age" channel.
- State vector s_t: realized volatility, OVX, futures slope, GPR, plus agent event intensity from M1.
- Horizon set H.

**Architecture:**
1. **Anchor:** ŷᵃ(h) = no-change forecast, or the futures-implied forecast where a matching maturity exists (EIA NYMEX contracts 1–4).
2. **Endogenous tokens:** RevIN (reversible instance normalization) → patches (length P ∈ {5, 10, 22}) → linear embedding plus RoPE or learned positions.
3. **Exogenous tokens:** each covariate's whole lookback becomes one variate token (iTransformer style), plus a type embedding (financial / fundamental / uncertainty / text-event). This handles mixed frequencies and missing values naturally.
4. **Global token** (TimeXer) bridges the exogenous information into the endogenous patches.
5. **Encoder (2–3 layers, d_model 64–128, 4 heads):** patch self-attention, then cross-attention from the global and endogenous tokens to the exogenous tokens, then an **MoE feed-forward layer**. The FFN has K = 4 experts with top-2 routing, the router sees [token, s_t], and a load-balancing loss keeps experts in use. The experts specialize by regime.
6. **Multi-horizon decoder:** learned horizon-query tokens q_h (an embedding of h plus the target-date calendar) cross-attend to the encoder. This produces all horizons jointly and keeps them consistent.
7. **Heads:**
   - Location residual μ_h, gated by the shrinkage gate α_h = σ(w·[q_h, s_t]) ∈ [0, 1].
   - A separate scale/shape head with non-crossing quantiles. Uncertainty does not collapse when α → 0.
   - Forecast: log ŷ_h^τ = log ŷᵃ(h) + α_h·μ_h + spread_h^τ.
8. **Loss:** pinball loss over the quantile grid + λ_α·mean(α) (shrinkage prior: deviate from the anchor only with evidence) + load-balancing loss + an optional directional auxiliary loss.
9. **Calibration:** Adaptive Conformal Inference (Gibbs & Candès 2021) runs online on the test stream.
10. **Training:** pooled Brent + WTI (a global model). Optionally, pretrain first on an energy-commodity panel (Brent, WTI, Dubai, RBOB, heating oil, Henry Hub), then fine-tune.
11. **Monthly variant:** monthly patches plus aggregated daily tokens (multi-resolution). Compare with temporally aggregated daily forecasts to test cross-frequency consistency.
12. **Variant:** frozen TSFM encoder (Chronos-2 or TiRex embeddings) with the AR-MoE head.

**Novelty check in Phase 0:** search for "residual over naive/futures + learned shrinkage" and "regime-routed MoE forecasting" so the contribution is framed correctly.

---

## 3. Transformer deep dive

### 3.1 Taxonomy and role in the study

| Family | Models | Key mechanism | Role |
|---|---|---|---|
| Legacy long-horizon | Informer, Autoformer, FEDformer | Sparse, auto-correlation and frequency attention | Baselines, since oil papers use them; expected to lose to DLinear (Zeng et al. 2023) |
| Token redesign | PatchTST, Crossformer, iTransformer | Patches as tokens, channel independence, variates as tokens | Core baselines; tokenization ablations |
| Exogenous-aware | **TimeXer**, **TFT** | Endogenous patches + exogenous cross-attention; variable selection + known-future inputs + quantiles | Strongest DL baselines; TimeXer is the origin of the AR-MoE backbone |
| Foundation transformers | Chronos-Bolt, **Chronos-2** (group attention, covariates), TimesFM 2.5, Moirai 2, Sundial, Toto, Time-MoE, Kronos (finance) | Pretrained, zero-shot, in-context covariates | Zero-shot + fine-tuned baselines; frozen-encoder variant |
| Attention-free controls | DLinear/NLinear, TSMixer(x), TiDE, N-HiTS, TiRex (xLSTM), S-Mamba (SSM) | Linear, MLP, recurrent, state-space | Controls that test whether attention adds value (RQ4) |

### 3.2 Transformer-specific experiments (RQ4)

- **Attention vs no attention:** at matched parameter count, swap the attention blocks for MLP-Mixer or linear blocks in AR-MoE.
- **Tokenization:** point vs patch vs variate tokens; patch length and stride; channel independence vs channel mixing.
- **Exogenous pathway:** cross-attention vs concatenation vs none; drop one covariate block at a time.
- **Lookback scaling:** L ∈ {64, 128, 256, 512}. Transformers claim to exploit long context; test whether that holds for oil.
- **Normalization under shift:** RevIN vs standard scaling vs none, compared specifically across break periods.
- **MoE:** K ∈ {1, 2, 4, 8}; routing on state vs on token only; expert usage over time vs known events.
- **TSFM adaptation:** zero-shot vs linear probe vs LoRA/adapter vs full fine-tune (small models); Chronos-2 native covariates vs a residual model on top.
- **Domain pretraining:** a small transformer pretrained on the energy panel vs a general TSFM, at equal fine-tuning budget.
- **Interpretability:** cross-attention maps over exogenous variables by horizon and regime, triangulated with integrated gradients (attention ≠ explanation); MoE routing timelines.
- **Efficiency:** parameters, training minutes per refit on a T4/L4, inference latency, reported for every model.

### 3.3 Training recipe for small data on Colab

- Mixed precision, small widths (d_model ≤ 128), dropout 0.1–0.3, weight decay, early stopping on a rolling validation block.
- 5-seed ensembles, with optional stochastic weight averaging.
- Optuna with a capped budget, run on the validation period only.
- Refit yearly over about 15 test years: roughly minutes per refit, which is feasible on a single GPU.

---

## 4. Agentic AI, part A: OilAgent forecasting system (RQ5; workshop focus, journal module)

### 4.1 Architecture: a multi-agent graph over a point-in-time tool layer

```
  Point-in-time store (documents and data with timestamp <= t, enforced server-side)
        │
  Curator ──► Event Extractor ──► structured events (pydantic JSON) ──► [M1] covariates into AR-MoE
        │                                   │
        └──► Quant Analyst ── MCP tools: get_prices / get_futures_curve / get_inventory_report /
                                         search_news / run_model / get_regime_state / get_past_performance
                                            │
             Adjudicator ([M2] model weights and shrinkage, or [M3] bounded adjustment) ◄── Memory (resolved past forecasts only)
                                            │
             Critic (temporal validity, bounds, cites real event IDs) ──► final quantile forecast + rationale
```

- **Event schema:**
  - Type: supply disruption / OPEC+ decision / sanctions / conflict / demand shock / inventory surprise / macro-policy / weather.
  - Also: region, direction, magnitude, expected duration, confidence, source document IDs, timestamp.
- **Leakage prevention lives in infrastructure, not prompts.** Every MCP tool takes `as_of` and filters server-side by *publication* timestamp. The agent cannot see the future even if it tries.
- **Bounded adjustment (M3):** δ is capped at |δ| ≤ κ·IQR of the model's forecast, so hallucinated extremes cannot do damage. Every adjustment must cite event IDs, and the Critic vetoes uncited or out-of-bounds outputs.
- **Reproducibility:** temperature 0, fixed seeds, and **every LLM call cached by prompt hash**, so historical backtests replay deterministically.

### 4.2 Integration modes compared (the core experiment)

| Mode | Agent role | Scale |
|---|---|---|
| M0 | None (AR-MoE alone) | Full history |
| M1 | Feature engineer: daily event features feed AR-MoE | Full 2015–2026 backtest with an open-weight model |
| M2 | Router/combiner: picks model weights and shrinkage per date | Clean windows + live test |
| M3 | Last-mile adjuster with bounded δ (cf. 2606.02497) | Clean windows + live test |
| M4 | LLM forecasts the number directly | Baseline; expected to be overconfident |

### 4.3 LLM look-ahead control (a contribution in itself)

1. **Chronological vintages:** event extraction with ChronoGPT-Instruct (vintages 2015/2020/2024) and DatedGPT (annual 2013–2024). Each forecast date uses a vintage whose cutoff precedes it. Compare against a current open-weight model to *measure* the bias.
2. **Memorization probes:**
   - Price recall on specific dates.
   - Entity and date anonymization ("a major Gulf producer", shifted dates), comparing outputs with and without masking.
   - Performance decay before vs after each model's cutoff (Look-Ahead-Bench style).
3. **Post-cutoff windows:** general LLMs are evaluated only after their documented cutoff.
4. **Live forward test:** a frozen v1 system runs from week 12 to week 24 (~60 trading days, h ∈ {1, 5, 10, 22}). A scheduled job logs the forecasts, and they are **hash-committed to git before the outcomes exist**. It is explicitly underpowered, so report descriptive statistics plus DM where feasible.
5. **Reporting rule:** historical M2/M3 results from general LLMs are labelled "upper bound, possibly contaminated". Claims rest on vintages, post-cutoff windows and the live test.

### 4.4 Models and costs

- **Extraction (M1):** a 7–9B open-weight instruct model (e.g., Qwen3-8B / Llama-3.1-8B), 4-bit, served with vLLM on Colab. GDELT 2.0 runs from 2015, giving about 4,000 daily prompts (top-N oil-relevant headlines per day plus EIA/OPEC documents), which is feasible.
- **Adjudicator/critic:** the open-weight model for clean historical windows; a small API budget (Claude) for the live test (~60 days × a few calls/day, low cost).
- **Comparators:** TimeCopilot (open source) run as an off-the-shelf agentic baseline; TemporalBench (2602.13272) as a reference evaluation design.

### 4.5 Agent-specific evaluation

- All §8 forecast metrics.
- Adjustment hit rate: the share of adjustments that reduced error.
- Event study: errors around OPEC+ meetings, WPSR releases, sanctions and conflict onsets.
- Rationale faithfulness: human audit of 100 sampled rationales. Does each cited event exist with timestamp ≤ t, and does its direction match the adjustment?
- Ablations: no retrieval, no memory, no critic; model size (vintage 1.3B / 8B / API).
- Cost per forecast, latency, and rerun variance.

---

## 5. Agentic AI, part B: research infrastructure (supports the work, not claimed)

| Agent role | Does | Guardrail |
|---|---|---|
| Data steward | Scheduled refresh; schema, gap, outlier and release-timestamp validation; data-card diffs | Raw data immutable and versioned |
| Experiment runner | Generates Hydra sweep configs, launches Colab/Kaggle runs, logs to MLflow | Runs only through `make`/CLI entry points, never ad hoc |
| Leakage auditor | Runs the pytest leakage suite and the statistical tests; **auto-flags any model beating the random walk by > 20% MSPE at h = 1 as likely leakage** | CI blocks merges on failure |
| Literature scout | Weekly arXiv/Semantic Scholar digests, screening against inclusion criteria into `lit_matrix.csv` | Every extracted field human-verified (hallucination risk) |
| Live-test operator | Daily scheduled job: pull data, run frozen v1, commit forecast hash before the close | Append-only log |
| Reporting | Builds tables and figures *only from result files*; no hand-typed numbers | Figures regenerated in CI |

- **Test-set lock:** test-split access is gated on a `protocol-frozen` git tag. A Claude Code PreToolUse hook blocks commands that touch test data before the tag exists. Protocol changes need human sign-off.
- **Leakage test suite (pytest):**
  - `test_no_future_rows`
  - `test_scaler_fit_window`
  - `test_release_alignment`
  - `test_decomposition_causal`
  - `test_tool_as_of_filter`
  - `test_sequence_after_split`
- **Disclosure:** include an AI-use statement per Elsevier/IJF policy.

---

## 6. Data and information sets

| Block | Series (frequency) | Source |
|---|---|---|
| Targets | Brent spot (daily, 1987–), WTI spot (daily, 1986–); monthly real Brent/WTI/RAC deflated by US CPI | EIA, FRED |
| Term structure | NYMEX WTI contracts 1–4 (daily); full curve via university Datastream/Bloomberg if available | EIA |
| Financial | OVX, VIX, broad dollar index, S&P 500, 10Y–2Y spread, gold, Henry Hub | FRED / CBOE |
| Fundamentals | WPSR stocks and production (weekly); world production, OECD stocks (monthly); RBOB and heating oil spreads | EIA, JODI |
| Global demand | Kilian real activity index, Baumeister–Korobilis–Lee indicators, OECD+6 industrial production | FRED / authors' sites |
| Uncertainty and positioning | Daily GPR, daily EPU, CFTC managed-money positions | authors' sites, CFTC |
| **Text corpus** | GDELT 2.0 events/GKG + headlines (2015–); EIA WPSR/STEO text; OPEC press releases and MOMR summaries; all timestamped by publication | GDELT, EIA, OPEC |

**Alignment rules:**
- Align every series to its publication timestamp.
- Use real-time vintages (ALFRED / Baumeister–Kilian) for revised series.
- Align Brent (London close) and WTI (New York close) by close time.
- WTI went negative on 2020-04-20, so Brent is the primary log target and WTI a robustness check.

**Small monthly sample:** use a panel of Brent, WTI and RAC, TSFMs, and cross-frequency aggregation.

---

## 7. Benchmark suite

- **Naive/statistical:** no-change (the denominator for every ratio), drift, historical mean, AutoARIMA, ETS, Theta; RW+GARCH-t (density).
- **Econometric:** futures-based forecasts (Alquist & Kilian 2010), product-spread models, monthly VAR/BVAR, Baumeister–Kilian forecast combination.
- **ML:** Ridge/LASSO/Elastic Net, LightGBM, Random Forest.
- **Deep learning and transformers:** the full §3.1 table, plus LSTM/GRU/TCN (common in oil papers).
- **Decomposition hybrids:** VMD-LSTM and CEEMDAN-LSTM, each leaky vs causal (RQ2).
- **Foundation models:** zero-shot and fine-tuned, with a **post-release clean holdout** (e.g., 2025-07 onward) to control pretraining contamination.
- **Agentic:** M1–M4 and TimeCopilot.

---

## 8. Evaluation protocol (frozen in week 6, including the agent protocol and the live-test design)

- **Design:** expanding-window, rolling-origin evaluation. Train from 1987, use a validation block for tuning, and run the out-of-sample test from about 2010 to the latest data. The text-based modes are tested 2015+. Refit deep models yearly. Never tune on the test set.
- **Horizons:** daily h ∈ {1, 5, 10, 22, 66} (cumulative log returns, also reported on price levels); monthly h ∈ {1, 3, 6, 12, 24} (log real monthly-average prices).
- **Sub-periods:** calm vs crisis (2014–16, 2020, 2022, 2025); volatility terciles.
- **Metrics:**
  - Point: MSPE and MAE ratios vs random walk, MASE.
  - Direction: success ratio.
  - Probabilistic: CRPS, pinball loss, coverage and width, Winkler score, PIT histograms.
- **Statistical tests:** DM with the HLN correction and HAC variance; Clark–West; Giacomini–White; MCS (`arch`); **multi-horizon SPA (Quaedvlieg 2021)**; Giacomini–Rossi fluctuation test; Pesaran–Timmermann.
- **Economic value:** sign-based trading with transaction costs (Sharpe ratio, maximum drawdown, certainty-equivalent return); a monthly hedging decision.
- **Reproducibility:** ≥ 5 seeds reporting mean ± std; LLM call cache; scripts that download the raw data.

---

## 9. Timeline (26 weeks; week 1 = 19 Oct 2026)

| Weeks | Phase | Key deliverables |
|---|---|---|
| 1–3 | **0. Literature and infrastructure bootstrap** | Lit matrix of ~100–140 papers (oil econometrics/DL, transformers, agentic forecasting, LLM look-ahead); gap statement; repo, CI leakage suite, MLflow, literature-scout agent |
| 2–6 | **1. Data, text corpus and tool layer** | Price and covariate pipeline; GDELT/EIA/OPEC corpus with publication timestamps; point-in-time store; MCP tool server v0 with `as_of` enforcement; EDA (unit roots, Bai–Perron breaks) |
| 5–8 | **2. Protocol and baselines** | Protocol (including agent and live test) **frozen and tagged in week 6**; rolling engine; tests module; statistical and econometric baseline tables |
| 7–10 | **3. Transformer, TSFM and leakage experiments** | §3.1 baselines, TSFM zero-shot and fine-tuning, leaky-vs-causal decomposition. **Fallback:** publishable as a benchmark/audit paper on its own |
| 9–17 | **4a. AR-MoE Transformer** | Development on validation only; §3.2 ablations; **frozen in week 17** |
| 10–17 | **4b. OilAgent** | Weeks 10–12: event extraction (M1) over the corpus. **Week 12: freeze v1, start the live test.** Weeks 13–16: adjudicator and critic (M2/M3), memorization probes, vintage comparison. Frozen in week 17 |
| 12–24 | **Live forward test** | Daily scheduled run of frozen v1 with hash-committed forecasts |
| 16–20 | **5. Final evaluation** | Single test run; all tests; regime and event studies; economic value; interpretability; robustness (Brent vs WTI, test start, lookback, refit frequency) |
| 19–26 | **6. Writing and submission** | **Journal** (IJF / Energy Economics): benchmark + leakage audit + AR-MoE Transformer + OilAgent M1 module, submitted around week 26. **Workshop** (non-archival, agentic focus: OilAgent + LLM look-ahead control + live test): target an ICML 2027 workshop (deadlines usually around May; verify). An optional early non-archival ICLR 2027 workshop submission could cover the transformer benchmark. arXiv/SSRN preprint; GitHub + Zenodo DOI |

**Journal outline:**
1. Introduction
2. Related work
3. Data and real-time information
4. AR-MoE Transformer
5. Agentic module (M1)
6. Evaluation design
7. Results: daily and monthly
8. Transformer analysis (RQ4)
9. Leakage audit
10. Regimes and economic value
11. Interpretability
12. Robustness
13. Conclusion

**Workshop outline:**
1. Problem: agentic forecasting under look-ahead bias
2. OilAgent architecture
3. M0–M4 comparison
4. Memorization probes and vintages
5. Live test
6. Limitations

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No gain at daily h = 1–5 | Expected and reported; the anchor bounds losses; claims rest on monthly and regime gains, probabilistic accuracy, the audit and the agent findings |
| Transformers don't beat attention-free controls | Still a publishable answer to RQ4; AR-MoE keeps whichever backbone wins |
| LLM look-ahead contaminates agent results | Vintages, post-cutoff windows, live test, plus the reporting rule in §4.3 |
| Hallucinated events | Pydantic schema, mandatory source IDs, critic veto, bounded δ, human audit |
| Live test too short | Framed as supplementary; daily horizons only; v1 frozen early (week 12) to maximize length |
| Colab limits / API cost | Small backbones, yearly refits, 4-bit vLLM, LLM cache, API used only for the live test |
| Scope overload (one person, 6 months) | MoSCoW in §1; Phase 3 is a standalone fallback paper; agent M2/M3 can slip to the workshop only |

---

## 11. Tooling and project structure

**Stack:**
- Python 3.11, PyTorch
- Forecasting libraries: Nixtla `neuralforecast` / `statsforecast` / `mlforecast` / `utilsforecast`
- Econometrics: `statsmodels`, `arch`
- Decomposition and conformal: `vmdpy`, `PyEMD`; `MAPIE` or a custom ACI
- TSFM packages: `chronos-forecasting`, `timesfm`, `uni2ts`, `tirex`
- Agents and LLM serving: `vllm`, `langgraph` (or `smolagents`), `pydantic`, MCP Python SDK, LanceDB/Chroma with timestamp-filtered metadata, `sentence-transformers`
- Experiment tooling: Optuna, Hydra, MLflow, pytest + GitHub Actions
- Claude Code (hooks + scheduled tasks) for research infrastructure

```
Forecasting Crude Oil Prices/
  ROADMAP.md  README.md  requirements.txt  .gitignore  Makefile
  configs/        data.yaml, protocol.yaml, models/*.yaml, agents/*.yaml
  data/           raw/ interim/ processed/            (gitignored)
  corpus/         text documents with publication timestamps (gitignored)
  literature/     lit_matrix.csv, notes/
  src/oilfc/
    data/         eia.py, fred.py, gpr_epu.py, cftc.py, gdelt.py, align.py
    features/
    eval/         rolling.py, metrics.py, tests.py (DM-HLN, CW, GW, MCS, MH-SPA, GR, PT), conformal.py
    models/       baselines/, econ/, ml/, dl/, transformers/, tsfm/, armoe/
    agents/       graph.py, schemas.py, prompts/, memory.py, llm_cache.py, probes.py
  mcp_server/     point-in-time tools with as_of enforcement
  tests/          leakage guard suite
  live/           forward-test logs + committed forecast hashes
  experiments/    one entry point per phase
  notebooks/      EDA
  results/        tables/, figures/
  paper/          journal/, workshop/
```

---

## 12. Execution on approval (this session)

1. Write this roadmap to `ROADMAP.md` in the project folder.
2. Create the §11 folder skeleton, `.gitignore`, a `requirements.txt` with pinned versions, and a `Makefile` with placeholder targets.
3. Create the `literature/lit_matrix.csv` template. Columns: citation, year, venue, target, frequency, horizons, model family, transformer?, agent/LLM?, RW benchmark?, futures benchmark?, stat tests, decomposition?, decomposition causal?, scaler/split leakage risk, LLM look-ahead control?, reported gain, notes.
4. Add stub leakage tests in `tests/` (marked xfail until implemented) and run `git init`, so protocol and model freezes become tagged commits.

Data download scripts, the evaluation harness, the MCP tool server and the agents follow as Phase 1–4 work in later turns.

## Verification

- **After steps 1–4:** the folder tree matches §11; `ROADMAP.md` renders; `pytest` collects the stub tests (as xfail); requirements resolve (dry run).
- **Project quality gates:**
  - Week 6: protocol tagged.
  - Week 8: the random walk beats most naive DL baselines at h = 1, confirming a leakage-free pipeline.
  - Week 10: the leaky VMD-LSTM's gains disappear when run causally.
  - Week 12: live-test forecasts are hash-committed daily.
  - Week 17: model and agent frozen and tagged before the single test run.

## Key references (verify each in Phase 0)

- **Oil econometrics:** Alquist, Kilian & Vigfusson (2013); Alquist & Kilian (2010, JAE); Baumeister & Kilian (2012, 2015, JBES); Kilian (2009, AER); Degiannakis & Filis (2018, Energy Econ.)
- **Oil ML/text:** Yu, Wang & Lai (2008, Energy Econ.); Li, Shang & Wang (2019, IJF); Bai et al. (2022, IJF); arXiv 2603.11408 (multi-dimensional LLM sentiment for WTI); arXiv 2410.12473 (GPT oil supply news)
- **Evaluation:** Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997); Clark & West (2007); Giacomini & White (2006); Hansen, Lunde & Nason (2011); Giacomini & Rossi (2010); Quaedvlieg (2021); Pesaran & Timmermann (1992); Gneiting & Raftery (2007); Gibbs & Candès (2021)
- **Transformers:**
  - Long-horizon and token redesign: Informer (AAAI'21); Autoformer (NeurIPS'21); FEDformer (ICML'22); PatchTST (ICLR'23); Crossformer (ICLR'23); iTransformer (ICLR'24); TimeXer (NeurIPS'24)
  - Covariates, critique and normalization: TFT (IJF'21); DLinear (AAAI'23); RevIN (ICLR'22)
  - Foundation models: Time-MoE (ICLR'25); Chronos (TMLR'24) and Chronos-2; TimesFM (ICML'24); Moirai (ICML'24); GIFT-Eval (2410.10393)
- **Agentic forecasting:** TimeCopilot (2509.00616); TimeSeriesScientist (2510.01538); TimeART (2601.13653); TemporalBench (2602.13272); Cast-R1 (2602.13802); KairosAgent (2605.30002); last-mile LLM agents (2606.02497)
- **LLM look-ahead:** Chronologically Consistent LLMs (2502.21206); Look-Ahead-Bench (2601.13770); DatedGPT (2603.11838)
- **Leakage in decomposition and LSTM pipelines:** VMDNet (2509.15394); 2512.06932
