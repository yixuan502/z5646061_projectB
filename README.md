# AtlasSignal — FINS5545 Part B

Student project folder: `z5646061_projectB`.

AtlasSignal is a prototype systematic multi-asset investment app. Part B builds
walk-forward equity, crypto, and combined funds, a sector news-sentiment analytic,
the structured/unstructured fusion extension, and the Streamlit investor journey.

## Current verified progress

- Stage 1 complete: the Part A equity, crypto, combined-return, and headline
  foundations are reproduced and validated from the provided data-access helper.
- Stage 2 complete: twelve OOS funds (three asset families by four methods) are
  built with monthly, no-look-ahead walk-forward backtests.
- Stage 3 complete: the standalone VADER sector index, explicit no-news handling,
  lag-safe signal, validation tables, and required time-series exhibit.
- Stage 4 complete: baseline sentiment fusion, the Part A HHI-based AtlasSignal
  confidence filter, transaction-cost/strength robustness, and before/after
  evidence.
- Stage 5 complete: all required fund exhibits, current holdings, and app-ready
  fact-sheet summaries are generated from the unchanged OOS backtests.
- Stage 6 complete: the AtlasSignal Streamlit app implements the four-step
  investor journey with precomputed results, responsive layout, and tested
  allocation controls.
- The student-authored report, final GitHub deployment, and final hand-in package
  remain pending.

## Environment and commands

Open this folder as the PyCharm project root and select its external virtual
environment. Install dependencies once:

    pip install -r requirements.txt -r requirements-dev.txt

Run and verify the completed stages separately:

    python scripts/prepare_part_b_data.py
    python -m pytest tests -q -p no:cacheprovider
    python scripts/build_funds.py
    python scripts/build_fund_exhibits.py
    python scripts/build_sentiment.py
    python scripts/plot_sentiment.py
    python scripts/build_fusion.py
    python scripts/plot_fusion.py
    python scripts/check_handin.py

Or rebuild all currently completed stages in sequence:

    python scripts/run_part_b.py

The Stage 2 success message reports 12 funds, 10,404 daily fund-return rows,
17,280 monthly target-weight rows, and PASS for timing, constraints, and solvers.

Launch the completed app locally:

    streamlit run streamlit_app.py

The default app journey is:

1. Compare all 12 funds by return, volatility, Sharpe, drawdown, and turnover.
2. Open any fund's fact sheet, OOS path, loss history, and latest targets.
3. Allocate an initial dollar across up to four funds using a calendar-aware
   static-sleeve scenario.
4. Explore sector sentiment and the coverage-adjusted fusion experiment.

## Stage 2 backtest design

- Funds: equity-only, crypto-only, and combined.
- Methods: Equal Weight, Minimum Variance, Maximum Sharpe, and Risk Parity.
- Estimation windows: 252 prior equity-calendar observations for equity/combined;
  365 prior native-calendar observations for crypto.
- Rebalancing: first available date of each month, using data strictly before it.
- Constraints: long-only, fully invested; 10% equity/combined asset cap, 25%
  crypto asset cap, and 20% aggregate crypto cap in combined funds.
- Annualisation: 252 for equity/combined and 365 for native crypto.
- Baseline assumptions: zero risk-free rate and zero transaction cost; actual
  monthly turnover and between-rebalance weight drift are retained.

Full machine-readable assumptions are saved in
`results/tables/backtest_design.csv`.

## Stage 3 sentiment design

- Every deduplicated original headline is scored with plain NLTK VADER while
  preserving casing, punctuation, negation, and intensifiers.
- Headline compound scores are first averaged within ticker-day, then observed
  tickers are equal-weighted within sector-day. This prevents high-news-volume
  companies from dominating the sector index.
- Ticker-days without headlines are omitted from the mean, not treated as neutral.
  A complete sector-date grid retains no-news observations as missing and exposes
  ticker/headline coverage.
- The tradeable field is exactly the preceding equity trading day's raw sector
  score. A headline aligned to Monday is therefore first usable on Tuesday.
- Plain VADER is a transparent baseline, not a claim that headline sentiment is
  economically complete. The later coverage-aware extension addresses sector
  representativeness while retaining this baseline for comparison.

## Stage 4 fusion and innovation

- Controlled experiment: Equity and Combined Risk Parity are each compared as
  Base, Baseline Sentiment, and AtlasSignal Coverage-Adjusted versions.
- The sentiment signal averages only the preceding 21 equity trading days, then
  standardises the ten sectors cross-sectionally and bounds the score to [-1, 1].
  This avoids fitting a full-sample time-series mean to plain VADER's positive
  level.
- The primary sector multiplier is `1 + 0.25 × signal`, a maximum relative weight
  adjustment of 25%, not a 25 percentage-point allocation.
- Part B independently reproduces Part A's monthly normalised news HHI. The
  coverage-adjusted signal multiplies relative sentiment by `(1 - trailing
  three-completed-month HHI) × sqrt(prior 21-day ticker coverage)`.
- Current-day sentiment and current-month HHI are excluded. Combined-fund crypto
  target weights remain exactly unchanged; only the equity sleeve is tilted.
- The primary comparison uses zero transaction cost as permitted by the brief.
  Robustness covers 10%, 25%, and 40% tilt strengths and 0/10 bps one-way-turnover
  costs without selecting the best setting after observing returns.
- OOS evidence is negative: neither sentiment version beats Base Risk Parity.
  Coverage adjustment consistently reduces the Sharpe loss and turnover relative
  to naive sentiment, but it does not create alpha. This is reported as a model
  limitation rather than hidden.

## Main derived outputs

- `results/data/fund_returns.csv` — OOS daily fund returns, growth of $1, and
  drawdown.
- `results/data/fund_weights.csv` — target weights and estimation/solver audit
  fields for every monthly rebalance.
- `results/tables/performance_metrics.csv` — required fact-sheet metrics plus
  CAGR, Sortino, historical VaR/ES, and turnover.
- `results/tables/portfolio_method_diagnostics.csv` — evidence that optimisation
  methods produce different weights.
- `results/data/latest_holdings.csv` — the most recent target holdings for all
  12 fund fact sheets, including rank and percentage weight.
- `results/data/combined_allocation_history.csv` — combined-fund targets
  aggregated to ten equity sectors plus the crypto sleeve for readable charts.
- `results/tables/fund_fact_sheets.csv` — one app-ready fact-sheet row per fund.
- `results/tables/performance_metrics_display.csv` — required metrics in explicit
  report-ready percentage units; the unrounded canonical calculations remain in
  `performance_metrics.csv`.
- `results/figures/fund_growth_comparison.png` — growth of $1 across all methods.
- `results/figures/fund_drawdown_comparison.png` — drawdowns across all methods.
- `results/figures/combined_weights_over_time.png` — monthly combined-fund
  allocation heatmaps across four methods.
- `results/figures/fund_sharpe_comparison.png` — OOS Sharpe comparison for all
  12 funds.
- `results/tables/backtest_design.csv` — calendars, windows, live dates,
  constraints, and assumptions.
- `results/data/sector_sentiment_index.csv` — complete daily sector index, coverage
  fields, and one-trading-day-lagged signal.
- `results/data/ticker_day_sentiment.csv` — auditable intermediate ticker-day
  scores before equal-ticker sector aggregation.
- `results/tables/sentiment_validation.csv` and
  `results/tables/sentiment_sector_summary.csv` — count, bounds, lag, and coverage
  evidence.
- `results/figures/sector_sentiment_timeseries.png` — self-contained required
  sector time-series exhibit.
- `results/data/monthly_news_concentration.csv` — self-contained reconstruction
  of the 480-row Part A monthly sector HHI panel.
- `results/data/fusion_signals.csv` — lag-safe relative sentiment, completed-month
  HHI, ticker coverage, and coverage-adjusted signals.
- `results/data/fusion_returns.csv` and `results/data/fusion_weights.csv` — six
  primary before/after funds and their auditable monthly targets.
- `results/tables/fusion_comparison.csv`, `fusion_robustness.csv`, and
  `fusion_validation.csv` — primary metrics/deltas, 28 robustness scenarios, and
  hard timing/constraint/reproduction checks.
- `results/figures/fusion_before_after.png` — growth and Sharpe-robustness exhibit.

Raw data is never committed. It loads only through `src/data_access.py`. Derived
`results/` artifacts are committed later because the deployed app reads them.

## Project structure

- `streamlit_app.py` — app entry point at repository root.
- `src/` — ETL, return features, portfolio optimisation, backtest, metrics,
  sentiment, fusion, exhibits, and lightweight app-scenario modules.
- `scripts/` — reproducible build and hand-in commands.
- `tests/` — network-free calculation, constraint, and timing tests.
- `results/` — precomputed app data, report tables, and later figures.
- `report/` — editable report source and final PDF.
- `ai/` — curated prompts, validation, errors, and corrections.
- `context/` — provided read-only course data guide.

## Deployment and hand-in

Stage 7A has produced `report/report.docx` as a visibly marked student-review
draft and `report/STUDENT_REVIEW_CHECKLIST.md` as the handoff checklist. The
student must rewrite and confirm the assessed narrative in their own words,
remove the draft notice, and export `report/report.pdf` before deployment and
hand-in.

Keep the repository private while building. After the report and final checks, follow
`docs/STUDENT_DEPLOY.md`: initialise this folder as its own GitHub repository,
push the precomputed results, deploy `streamlit_app.py`, and make the repository
public only at hand-in. Submit the full ZIP, public repository URL, and live app
URL.
