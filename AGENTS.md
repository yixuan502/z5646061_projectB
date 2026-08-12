# FINS5545 Project B — AI Working Instructions

## Student and project

This is my individual UNSW FINS5545 Financial Market Data Literacy Part B project.

Project folder:
`z5646061_projectB`

Product name carried forward from Part A:
`AtlasSignal`

Part B covers:
- Data Factory Floor Station 3: portfolio model design, out-of-sample backtesting, sentiment modelling, and structured/unstructured fusion
- Data Factory Floor Station 4: implementation as a deployed Streamlit investment app

The goal is an HD-quality submission, not merely the minimum pass requirements.

## Source of truth

Use sources in this order:

1. `PROJECT_BRIEF.md`
2. `SUBMISSION_CHECKLIST.md`
3. `context/DATA_GUIDE.md` and other provided `context/` files
4. my own Part A code and derived results
5. FINS5545 Week 1–10 course material and methods already studied

Do not invent assessment requirements. If a modelling choice is not prescribed by the brief, identify it as a project choice and justify it financially.

## Required working style

Do not rebuild the course from zero. Assume the core Week 1–10 concepts have already been studied.

For every project step:
1. State the current document/file, section, and project stage.
2. Explain the financial objective and why the step matters.
3. Explain the model or data logic before giving code.
4. When code is required, clearly state:
   - which file to create or edit in PyCharm
   - the exact command to run from the project root
   - expected output
   - how to judge success
   - common errors and how to diagnose them
5. Explain new functions, parameters, returned objects, side effects, and important finance interpretation. Do not repeatedly reteach basic pandas/Python that has already been covered.
6. Work in small verified stages. Do not dump a large unexplained final solution.
7. Record progress and do not skip required stages.

## Data rules

All raw data must be loaded through `src/data_access.py`.

Do not manually download, save, or commit raw project data.

Reuse the student's Part A data foundation where appropriate, but Part B must remain reproducible from the hosted raw data.

Price/return rules:
- use adjusted close (`adjClose`) for returns
- compute equity returns within each equity ticker calendar
- compute crypto returns within each crypto ticker calendar before any equity-calendar alignment
- cap crypto observations at 2023-12-31
- for a combined equity+crypto panel, left-align already-computed crypto returns to the equity trading calendar
- never merge equity and crypto price levels first and then difference

News rules:
- remove exact duplicates using ticker + date + title
- normalise the news UTC datetime before merging with equity dates
- align every headline to the same or next equity trading day
- preserve headline casing and punctuation for VADER/finVADER-style sentiment
- sentiment used for trading must be lagged by at least one trading day

## Portfolio and backtest rules

The high-band target is to build equity-only, crypto-only, and combined funds across several portfolio methods.

Baseline methods should include:
- Equal Weight
- Minimum Variance
- Maximum Sharpe
- Risk Parity

Backtests must be genuinely walk-forward and out-of-sample:
- weights at date t may use information available only before t
- no rebalance-day or future return may enter the estimation window
- use a documented estimation-window length and rebalance frequency
- distinguish equity/combined trading-day annualisation from native crypto annualisation
- sanity-check that optimisers actually produce different weights; solver stalling is a known project risk

Required fact-sheet metrics include:
- growth of $1 / cumulative performance
- annualised return
- annualised volatility
- Sharpe ratio
- maximum drawdown
- current holdings / latest target weights

Any extra metrics must be motivated and must not replace the required ones.

## Sentiment and fusion rules

Build a standalone equity-sector sentiment index from the headline data.

The baseline pipeline should be auditable:
headline -> sentiment score -> ticker-day aggregation -> equal-ticker sector aggregation -> lagged signal.

Do not treat no-news observations as automatically equivalent to neutral sentiment without explicit justification.

Fusion applies only to the equity side because crypto has no news data.

For combined funds, a sentiment extension must not accidentally change the crypto sleeve unless that is explicitly part of the tested design.

Any standardisation used as a trading signal must be look-ahead safe; do not estimate z-score parameters from the full future sample.

## Main innovation direction

Build on the student's Part A news-coverage concentration work rather than adding many shallow unrelated extensions.

Primary proposed extension:
`Coverage-Adjusted Sector Sentiment` / `AtlasSignal News Confidence Filter`.

Use recent sector news-coverage concentration (for example normalised HHI / effective ticker count) as a confidence measure for sector-level sentiment. When one ticker dominates a sector's news flow, reduce the strength of the sector sentiment tilt because the signal is less representative of the full sector.

The extension must be:
- defined clearly, preferably with equations
- implemented
- tested against a baseline sentiment tilt and a no-sentiment base fund
- evaluated using out-of-sample evidence
- interpreted even if it underperforms

A transaction-cost / turnover robustness analysis may be added, but the coverage-adjusted sentiment design should remain the main innovation unless evidence later supports a better direction.

## Required Part B output filenames

These exact files must be produced:
- `results/data/fund_returns.csv`
- `results/data/fund_weights.csv`
- `results/data/sector_sentiment_index.csv`
- `results/tables/performance_metrics.csv`

The app must read precomputed artifacts from `results/` rather than recomputing heavy backtests or VADER at runtime.

## Streamlit rules

The app entrypoint is `streamlit_app.py` at the project root.

The target investor journey should support:
- comparing funds
- opening a fund fact sheet
- seeing performance, risk, drawdown, and current holdings
- setting an allocation across offered funds
- reading sector sentiment analytics

Keep the deployed app light. Do not import or run nltk/VADER in `streamlit_app.py`.

The repository stays private while building and is made public only at hand-in. Deployment steps follow `docs/STUDENT_DEPLOY.md`.

## Reproducibility and checks

The main local reproduction command is:
`python scripts/run_part_b.py`

The local app command is:
`streamlit run streamlit_app.py`

Before hand-in run:
`python scripts/check_handin.py`

Fix every `[FAIL]`; review every `[WARN]`.

Do not commit:
- raw project data
- secrets
- virtual environments
- `__pycache__`
- `.pyc` files
- local absolute paths

## AI workflow and academic integrity

AI workflow is graded. Keep curated records under `ai/`.

For meaningful AI interactions record:
- the prompt or task
- what AI proposed or changed
- how it was checked
- errors, missing assumptions, or weaknesses found
- what the student changed or accepted
- why

Be candid when AI output is wrong or cannot be verified. Do not fabricate corrections merely to make the log look stronger.

The student must understand all submitted code. Written economic interpretation and critical reflection must be expressed in the student's own words and tied to actual project evidence.
