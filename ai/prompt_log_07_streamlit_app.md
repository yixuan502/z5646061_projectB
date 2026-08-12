# Prompt Log 07 — Streamlit Investor Journey

## Student request

Continue to the next Part B stage after the required fund exhibits and fact-sheet
files passed the student's PyCharm checks.

## App brief and model decisions

The AI treated the app as a financially literate but non-technical investor
journey rather than a model-development dashboard. It proposed one responsive
page with four ordered tabs:

1. compare all 12 funds;
2. open any fund's fact sheet and latest OOS target holdings;
3. allocate across up to four offered funds;
4. explore standalone sentiment and the coverage-adjusted fusion experiment.

The app reads nine bounded precomputed CSV artifacts under `results/`. It does
not import NLTK, download raw data, rescore headlines, or rerun an optimiser.

## Allocation logic and financial interpretation

Allocation inputs are non-negative and normalised to 100%. Each selected fund is
an initial sleeve whose wealth then compounds independently, so the fund mix
drifts rather than being silently reset every day. The scenario therefore answers
"what would an initial fund allocation have done in the common OOS sample?" It is
not a personalised optimiser.

When a native crypto fund is included, the app uses the seven-day calendar and
retains weekend crypto returns. Equity-calendar funds receive 0% return on those
non-trading days. When no native crypto fund is selected, the scenario retains
the equity calendar and 252-day annualisation. The app states that fees, tax,
slippage, and inter-fund rebalancing are excluded.

## Validation

- Five helper tests cover snapshot completeness, allocation normalisation,
  static-sleeve compounding, mixed-calendar weekends, invalid inputs, and
  sector-isolated sentiment rolling means.
- Two Streamlit app tests render the full page and change the selected fund.
- Complete project suite: 32 passed.
- Browser QA exercised all four tabs, selected Crypto Minimum Variance, changed
  an allocation input, and switched the fusion view from Equity to Combined.
- Desktop width 1,440 and narrow width 390 both had no page-level horizontal
  overflow.

## Errors found and corrected

1. The first scatter used a circle mark while also encoding optimisation method
   by shape. Browser logs showed that the shape channel was dropped. It was
   changed to a filled point mark so colour is not the only method distinction.
2. The initial chart theme followed the system dark setting and produced black
   charts inside a light app. `.streamlit/config.toml` now locks the original
   AtlasSignal light design system.
3. KPI fund names were initially passed as metric deltas, causing misleading
   green arrows. They were moved to neutral captions.
4. The sidebar was initially forced open and covered most of the 390-pixel test
   viewport. The initial state was changed to `auto`, retaining the full desktop
   navigation while collapsing it on narrow screens.
5. The app used a Streamlit chart-width option scheduled for removal. It was
   replaced with the current `width="stretch"` API before deployment.

These were presentation and compatibility corrections. No OOS return, portfolio
weight, sentiment score, fusion result, or model parameter changed.
