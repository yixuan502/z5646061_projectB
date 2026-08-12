# Prompt Log 06 — Fund Exhibits and Fact-Sheet Evidence

## Student request

Continue to the next Part B stage after the sentiment-fusion outputs passed the
student's PyCharm checks.

## AI proposal and financial rationale

The AI identified the remaining explicit fund evidence in the brief: growth of
$1, drawdown, weights over time, and a Sharpe comparison. It proposed four
complementary figures rather than one overloaded dashboard:

1. three family panels of growth of $1 across all four methods;
2. three family panels of drawdown across all four methods;
3. four heatmaps of the Combined fund's monthly target allocation, aggregating
   50 equities to ten sectors and ten crypto assets to one sleeve;
4. three family panels of Sharpe bars covering all 12 funds.

The heatmap aggregation is presentation-only. It avoids an unreadable plot of 60
asset lines while preserving the monthly portfolio mass. Raw asset-level targets
remain in `fund_weights.csv`.

## App-ready transformations

The AI added deterministic functions that select only the latest rebalance per
fund, rank holdings, count active positions, identify the top holding, aggregate
the combined allocation, and create a one-row-per-fund fact-sheet table. A
separate report-ready metrics table converts decimals to explicit percentage
units while preserving `performance_metrics.csv` as the unrounded canonical
calculation.

No portfolio was re-estimated and no return, weight, sentiment, or fusion result
was modified in this stage.

## Student-checkable validation

- New transformation/metric tests: 5 passed.
- Complete test suite after network access to the supplied data: 25 passed.
- Fact-sheet rows: 12, exactly one per fund.
- Latest holdings: 480 rows, covering all 12 most recent target portfolios.
- Combined allocation history: 36 rebalance dates × four methods; every
  aggregated portfolio sums to one.
- Four figure files were generated and checked at their exported resolution.
- The required Stage 2 files were opened read-only by the exhibit script.

## Error found and corrected

The first weights heatmap exported correct data but its horizontal colour bar
overlapped the bottom panels' date labels. Visual inspection caught the problem.
The colour bar was moved to a dedicated figure row, the lower margin was
increased, and the PNG was regenerated and re-inspected. No numeric result or
model choice changed.

## Interpretation control

The evidence guide records both the supported conclusion and its boundary. For
example, Crypto Minimum Variance has the highest crypto Sharpe but still has
76.9% annualised volatility and a -72.8% maximum drawdown. Combined Risk Parity
has the highest combined-family Sharpe, but this is a sample-specific historical
result rather than a forecast. Maximum Sharpe's weaker OOS results are discussed
as expected-return estimation instability, not as a solver failure.

The AI prepared evidence and writing prompts but did not create the submitted
Word/PDF narrative. The student must interpret every exhibit in their own words.
