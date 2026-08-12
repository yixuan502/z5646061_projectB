# Prompt Log 03 — Walk-Forward Funds

## Student request

Continue to the next Part B stage after the Part A foundation and Stage 1 checks
were independently reproduced in PyCharm.

## AI proposal and implementation

The AI proposed an HD-band baseline of twelve investable funds: equity-only,
crypto-only, and combined universes, each using Equal Weight, Minimum Variance,
Maximum Sharpe, and Risk Parity.  It implemented separate portfolio, backtest,
and fact-sheet metric modules plus a reproducible fund-building script.

The documented project choices were:

- 252 prior equity-calendar observations for equity and combined funds;
- 365 prior native-calendar observations for crypto funds;
- monthly rebalancing on the first available date of each month;
- weights on date t use returns strictly before t;
- long-only and fully invested portfolios;
- 10% individual cap for equity/combined, 25% for crypto;
- a 20% total crypto-sleeve cap for combined funds;
- zero annual risk-free rate and zero baseline transaction costs;
- weights drift between monthly rebalances, while turnover is retained for a
  later transaction-cost robustness test.

## Financial rationale

The rolling windows provide roughly one year of information on each native
calendar.  Monthly rebalancing limits unnecessary trading and is permitted by
the brief.  Caps reduce single-name concentration, while the combined crypto cap
prevents the high-volatility sleeve from dominating the product.  A zero
risk-free rate and zero transaction cost are allowed baseline assumptions, but
they are stated rather than hidden.

## Validation requested and performed

The implementation includes tests for:

- all four methods summing to one and respecting long-only/group constraints;
- different methods producing economically different target weights;
- a synthetic future shock not changing the first target portfolio;
- every estimation end date being earlier than its rebalance date;
- monthly rather than accidental daily rebalancing;
- portfolio weights drifting between rebalances;
- explicit 252/365 annualisation and required drawdown/growth calculations.

The full-data build additionally checks all solver flags, target-weight sums,
asset caps, the combined crypto cap, unique return keys, finite values, and
pairwise method-weight distances.

## Critical review / correction record

During review, the initial daily-weight audit table did not explicitly assign the
live DatetimeIndex after constructing a DataFrame from a list of Series.  That
could have left an integer audit index even though portfolio returns were dated
correctly.  The AI corrected this before the tests and full build by assigning
the exact live dates to `daily_pre_return_weights`.

The first real-data build then exposed a second issue that the synthetic tests did
not: one constrained Maximum-Sharpe window stopped with SLSQP's "Positive
directional derivative for linesearch" message.  The AI did not accept or hide
that candidate.  It replaced the fragile single-start solve with deterministic
multi-start validation: equal weight, a converged minimum-variance portfolio, and
a feasible high-historical-return start.  Only converged candidates satisfying
all weight and group constraints are eligible, and the best Sharpe candidate is
retained.  The backtest error message was also amended to identify any failed
rebalance date.

After the repaired full build passed, a fact-sheet audit found that the initial
drawdown function used the first end-of-day wealth as its first high-water mark.
If the first OOS return were negative, that convention would omit the loss from
the investor's initial $1 and slightly understate maximum drawdown.  The function
was corrected to keep the high-water mark at no less than the initial $1, and a
specific negative-first-day regression test was added before rebuilding outputs.

The student should independently run the stated PyCharm commands and use the
actual output tables—not AI expectations—to write the final economic comparison.

## Audit limitation not triggered by the baseline

The optimiser's `_check_feasible_start()` currently tests whether the
equal-weight starting point satisfies the group cap. This does not affect the
verified Combined-fund baseline: its 10 crypto assets represent 10/60 = 16.67%
at equal weight, below the 20% crypto-sleeve cap.

If a future sensitivity used a crypto cap below 16.67%, such as 10% or 15%, the
existing check would reject the equal-weight start even though a feasible
portfolio could exist by reallocating weight to equities. Before running such a
low-cap sensitivity, the feasible-start generation should be changed and covered
by regression tests. This project does not run a low-cap sensitivity, so the
already validated baseline optimiser was not changed for an untriggered issue.
This limitation is recorded here; it is not a bug claimed as fixed.
