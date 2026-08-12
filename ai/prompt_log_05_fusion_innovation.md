# Prompt Log 05 — Sentiment Fusion and Coverage-Adjusted Innovation

## Student request

Start the next Part B stage after the standalone sector sentiment index passed the
student's PyCharm checks.

## AI model proposal

The AI proposed a controlled three-way comparison on the Equity and Combined Risk
Parity funds:

1. the unchanged base target weights;
2. a baseline relative-sector sentiment tilt;
3. the AtlasSignal Coverage-Adjusted Sector Sentiment tilt.

Risk Parity was selected because it is covariance-based and does not already use
estimated expected returns. This makes the incremental effect of a news signal
easier to attribute than in Maximum Sharpe, whose noisy historical mean estimate
could interact with sentiment.

For decision date t, the baseline uses the average raw sector sentiment from the
previous 21 equity trading days. It is standardised cross-sectionally across the
ten sectors, clipped at two standard deviations, and divided by two to create a
bounded score from -1 to +1. This removes plain VADER's positive full-sample level
without using future time-series means.

The base multiplier is:

`1 + 0.25 * relative sentiment score`.

The 25% coefficient is fixed before evaluating performance and means a sector's
relative target weight can be scaled by at most +/-25% before normalisation and
the existing asset cap. It is a relative multiplier, not a 25 percentage-point
allocation.

## Part A innovation inherited

Part B reproduces Part A's complete monthly within-sector HHI from original
headline dates, including zero-news ticker-months. It does not depend on the Part
A folder at runtime. During development, the recomputed 480 sector-month rows are
to be compared directly with the student's Part A table.

The coverage confidence is:

`(1 - trailing three-completed-month normalised HHI) * sqrt(prior 21-day ticker coverage)`.

The AtlasSignal multiplier is:

`1 + 0.25 * relative sentiment score * coverage confidence`.

Higher news concentration or thinner ticker coverage reduces the tilt magnitude
without reversing its sign. HHI from the current incomplete month is excluded.

## Portfolio and evaluation guardrails

- sentiment affects equities only;
- every combined-fund crypto target weight must remain exactly unchanged;
- long-only, fully invested, 10% asset caps remain in force;
- weights drift between monthly rebalances as in the Stage 2 base fund;
- the zero-cost base must reproduce Stage 2 Risk Parity daily returns;
- 10 bps one-way-turnover costs test implementation sensitivity;
- 10%, 25%, and 40% tilt strengths test parameter robustness rather than selecting
  the best coefficient after seeing results;
- conclusions will be descriptive OOS evidence, not causal claims or forecasts.

## Validation plan

Synthetic tests cover the complete HHI universe, exclusion of current-day
sentiment/current-month HHI, exact crypto preservation, weight constraints,
drifting weights, turnover, and transaction-cost arithmetic. Full-data validation
will additionally reconcile Part B HHI to Part A, reproduce Stage 2 base returns,
check signal source dates, compare variants on identical OOS dates, and review
whether results remain directionally stable across cost and strength assumptions.

## Critical review to update after execution

Record any implementation failure or evidence that changes the proposed model.
Do not describe the coverage adjustment as successful until the real-data
before/after metrics and robustness table have been inspected.

The first synthetic test run returned three passes and one failure. The failing
line compared the three-month HHI mean with `0.2` using exact floating-point
equality, even though the displayed output was 0.2 and the calculation was
correct. The assertion was changed to a tight `1e-12` numeric tolerance. This was
a test-quality correction; no model formula, timing rule, or result was changed.

Before the full experiment, Part B's independently rebuilt 480 sector-month HHI
rows were compared with the student's saved Part A table. All categorical/count
fields and top tickers matched. The maximum normalised-HHI difference was
9.7e-17, numerical rounding only. The new target-schedule engine also reproduced
all 753 Equity Risk Parity OOS returns with maximum error 1.18e-16 and turnover
with maximum error 4.51e-16. This established comparable baselines before fusion.

## Real-data result and interpretation check

At the predeclared 25% tilt and zero transaction cost, sentiment did not improve
the base Risk Parity funds. Equity Sharpe moved from 0.7232 to 0.6826 under the
baseline sentiment tilt and 0.6915 under coverage adjustment. Combined Sharpe
moved from 0.8881 to 0.8555 and 0.8625 respectively. The coverage-aware version
therefore reduced the damage relative to naive sentiment but did not beat the
price-only base. Maximum drawdowns were also slightly worse.

The same direction held for 10%, 25%, and 40% strengths under both zero and 10
bps transaction costs: all 24 non-base scenarios had a negative Sharpe delta.
However, every matched coverage-adjusted scenario had a smaller Sharpe loss and
lower turnover than its baseline-sentiment counterpart. This is evidence that
the confidence filter moderates an unreliable signal; it is not evidence that the
signal creates alpha.

Average coverage confidence was 0.801. Materials, Utilities, and Real Estate had
the lowest average confidence, consistent with Stage 3's thinner ticker coverage.
All combined-fund crypto target weights were preserved exactly (maximum change
0.0), and all timing, full-investment, long-only, and asset-cap checks passed.

The correct reporting conclusion is negative and descriptive: plain headline
VADER did not add OOS investment value in this sample, while the Part A-inspired
coverage filter improved robustness and reduced turnover relative to the naive
fusion. No causal or forward-looking claim should be made.
