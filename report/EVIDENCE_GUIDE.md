# AtlasSignal Part B — Report Evidence Guide

This is a planning and audit document, not the submitted report. The submitted
`report/report.docx` and `report/report.pdf` must be written in the student's own
words for a financially literate, non-technical client.

## Reporting question and comparison basis

The report should answer: **which systematically managed funds offer the most
credible risk–return trade-off, what does sector news sentiment add, and how
should an investor use the app without treating a short backtest as a forecast?**

- Product set: 12 investable funds = three asset families × four methods.
- Evidence: daily walk-forward out-of-sample returns from January 2021 to
  December 2023 on each family's native calendar.
- Formation rule: monthly targets use only observations strictly before each
  rebalance date; weights drift with asset returns between rebalances.
- Annualisation: 252 for equity/combined and 365 for native-calendar crypto.
- Baseline: risk-free rate 0% and transaction cost 0 bps. Turnover is retained,
  and the fusion extension is also tested at 10 bps.
- Constraints: long-only and fully invested; 10% equity/combined asset cap, 25%
  crypto asset cap, and 20% combined-fund crypto sleeve cap.

The arithmetic annualised statistics shown in the required table are:

\[
R_{ann}=A\bar r_d,\qquad
\sigma_{ann}=\sqrt{A}\,s(r_d),\qquad
Sharpe=\frac{R_{ann}-r_f}{\sigma_{ann}},
\]

where `A` is 252 or 365. Drawdown is the percentage fall from the running
high-water mark of an initial $1 investment.

## Required-exhibit map

| Rubric evidence | Report/app artifact | Reader question | Main limitation to state |
|---|---|---|---|
| Metrics across funds | `results/tables/performance_metrics_display.csv` | What return, risk, Sharpe, and drawdown did each fund deliver OOS? | One three-year historical sample; zero-cost baseline. |
| Growth of $1 | `results/figures/fund_growth_comparison.png` | How did the investment path and ending wealth differ by family and method? | Ending wealth alone does not measure risk or repeatability. |
| Drawdown | `results/figures/fund_drawdown_comparison.png` | How deep and persistent were losses from prior peaks? | Drawdown depends on the observed path and can be worse in future. |
| Weights over time | `results/figures/combined_weights_over_time.png` | How stable or concentrated were the four combined-fund allocation rules? | Colours are sector/crypto target weights, not daily drifted holdings. |
| Sharpe comparison | `results/figures/fund_sharpe_comparison.png` | Which methods delivered more return per unit of realised volatility? | Sharpe is not a guarantee and understates non-normal/tail risk. |
| Fund fact sheets | `results/tables/fund_fact_sheets.csv` and `results/data/latest_holdings.csv` | What are the latest investable targets and core OOS statistics? | Latest targets date from 1 Dec 2023 and are a prototype, not live advice. |
| Sector sentiment | `results/figures/sector_sentiment_timeseries.png` | How did equal-ticker VADER sentiment vary across ten equity sectors? | VADER is a transparent lexical proxy; no-news is missing, not neutral. |
| Fusion before/after | `results/figures/fusion_before_after.png` and `results/tables/fusion_comparison.csv` | Did the lagged sentiment tilt improve price-only Risk Parity? | Descriptive experiment, not a causal test or tuned trading rule. |

## Verified fund findings to explain in plain English

1. **The best method depends on the family.** Equity Equal Weight has the highest
   equity Sharpe (0.819), Crypto Minimum Variance has the highest crypto Sharpe
   (1.075), and Combined Risk Parity has the highest combined Sharpe (0.888).
   There is no single optimiser that wins in every universe.

2. **Risk-adjusted strength is not the same as low absolute risk.** Crypto Minimum
   Variance finishes at about $4.90 per initial dollar and has the best crypto
   Sharpe, but its annualised volatility is 76.9% and maximum drawdown is -72.8%.
   Its label is relative to the crypto opportunity set, not to a conventional
   low-risk investment.

3. **Combined Risk Parity offers the clearest diversified core trade-off in this
   sample.** It records 14.39% annualised return, 16.20% volatility, 0.888 Sharpe,
   and -19.47% maximum drawdown. Combined Equal Weight earns a higher 16.43%
   annualised return but with 21.60% volatility, 0.761 Sharpe, and -27.87%
   drawdown. This supports a risk-adjusted comparison, not a claim that Risk
   Parity will always outperform.

4. **Minimum Variance reduces realised risk but gives up return outside crypto.**
   Equity and Combined Minimum Variance have roughly 12.5% annualised volatility
   and -15.7% to -15.8% maximum drawdown, but only about 6.5% annualised return.
   This is a defensiveness-versus-growth trade-off.

5. **Historical mean optimisation is unstable out of sample here.** Maximum
   Sharpe is not the highest realised OOS Sharpe in any family. Its annualised
   turnover is 293% for equity, 236% for crypto, and 300% for combined funds,
   much higher than the corresponding Equal Weight and Risk Parity funds. A
   financially defensible explanation is expected-return estimation error and
   concentrated cap-bound targets—not that the optimiser is mathematically
   incorrect.

6. **The allocation figure makes model behaviour visible.** Combined Equal
   Weight is mechanically stable, with a 16.7% crypto sleeve (10 crypto assets
   out of 60 equally weighted assets). Minimum Variance is most exposed on
   average to Healthcare (27.8%), Consumer (18.6%), and Communication (17.9%).
   Maximum Sharpe shows the largest sector regime shifts. Risk Parity is more
   stable and diversified, with its largest average sector, Healthcare, at
   14.0%. These figures are descriptive target-weight patterns.

## Sentiment and innovation finding

The sector index uses the sequence `headline → ticker-day → equal-ticker
sector-day`. This prevents a company with many headlines from dominating its
sector. Original punctuation, case, negation, and intensifiers are preserved for
VADER. A complete sector-date grid keeps 228 no-news sector-days as missing.

The AtlasSignal innovation adjusts the baseline relative-sector sentiment tilt
for how representative the news is:

\[
Confidence_{s,t}=(1-\overline{HHI}^{(3\ completed\ months)}_{s,t})
\sqrt{Coverage^{(21d)}_{s,t}}.
\]

At the predeclared 25% tilt, neither sentiment version beats matching Risk
Parity. Equity Sharpe falls from 0.723 to 0.683 under naive sentiment and to
0.692 under coverage adjustment. Combined Sharpe falls from 0.888 to 0.856 and
0.863 respectively. All 24 non-base strength/cost scenarios have a negative
Sharpe change, but every coverage-adjusted case loses less Sharpe and produces
less turnover than its matched naive case. The defensible conclusion is:
**coverage control improves the reliability of an otherwise weak signal, but it
does not create alpha in this sample.**

## Suggested answer-first report path

1. **The funds and OOS design.** Define the 12 products, calendars, window,
   monthly formation, constraints, and no-look-ahead rule. Keep implementation
   details brief; direct reproducibility detail to the appendix.
2. **Risk-adjusted results differ by universe.** Place the metrics table, Sharpe
   bars, growth figure, and drawdown figure. Explain return, volatility, and tail
   experience together rather than naming a winner from return alone.
3. **Portfolio construction changes economic exposure.** Place the combined
   weight heatmap and discuss stability, concentration, turnover, and the crypto
   cap. Use `latest_holdings.csv` for selected fact-sheet examples or an appendix.
4. **Sentiment is observable but not automatically investable.** Place the sector
   index and distinguish descriptive news tone from a return forecast.
5. **Coverage adjustment is useful risk control, not demonstrated alpha.** Place
   the fusion before/after exhibit and report the negative result honestly.
6. **The app turns evidence into a guarded investor journey.** Explain the four
   steps now implemented: compare the shelf, inspect a fact sheet, build a
   fund-level allocation scenario, and test the news-confidence evidence.
7. **Critical reflection and three concrete recommendations.** Rewrite the
   evidence-based prompts below in the student's own voice.

## Evidence-based recommendation prompts

- Consider Combined Risk Parity as the prototype diversified core option because
  it has the strongest combined-family risk-adjusted OOS result—not because its
  future outperformance is guaranteed.
- Keep the sentiment tilt labelled experimental and do not market it as alpha
  until it survives a longer live/forward sample, realistic costs, alternative
  text models, and pre-registered parameter choices.
- Add investor risk profiling and allocation guardrails in a real product. Show
  crypto drawdown scenarios prominently, retain the combined crypto cap, and
  prevent a high historical return from being interpreted as low risk.

## Caveats that must remain visible

- The sample is approximately three OOS years and includes unusual crypto market
  conditions; cross-family results are not directly comparable without risk.
- The same historical period is used to describe multiple funds, so rankings are
  sample-dependent and not statistically independent discoveries.
- Zero baseline transaction costs favour high-turnover methods. Turnover is
  reported; the fusion experiment's 10 bps test does not replace a full
  implementation-cost model for all funds.
- Sharpe uses a zero risk-free rate and volatility as its risk denominator. The
  report must also discuss maximum drawdown and, where useful, VaR/Expected
  Shortfall already stored in the fact-sheet table.
- The historical results are prototype evidence, not personal financial advice,
  a forecast, or proof of causal news-to-return predictability.

## Chart map and QA record

| Section | Question | Visual family | Data fields | QA decision |
|---|---|---|---|---|
| Fund paths | How did wealth evolve? | Three-panel multi-series line | date, family, method, growth_1 | Separate family scales; start $1 shown; 753/1,095 time points per line. |
| Loss experience | How deep were peak losses? | Three-panel multi-series line | date, family, method, drawdown | Percentage axes; zero line; initial $1 included in peak. |
| Allocation | How did method change exposure? | Four-panel heatmap | rebalance date, method, sector/crypto bucket, target weight | 36 dates, 11 buckets, common scale to the observed maximum (~40.1%); colour bar placed outside labels. |
| Risk-adjusted rank | Which method had higher Sharpe? | Three-panel categorical bars | family, method, Sharpe | All bars start at zero; exact two-decimal labels; common y-scale. |
| Sentiment | How did sector tone vary? | Ten-panel smoothed line | date, sector, 21-day mean | Common full observed y-scale; no-news caveat retained. |
| Fusion | Did the tilt add value and survive assumptions? | Growth lines + robustness lines | date/growth and tilt/cost/Sharpe delta | Matching base and common scales; negative evidence retained. |

## App journey evidence

The default Streamlit page answers the comparison question before interaction.
It shows all 12 funds, the highest observed Sharpe, the lowest observed
volatility, and the shallowest drawdown, followed by a return–risk map, Sharpe
ranking, and exact table. The remaining tabs progressively support due diligence:

1. **Fund Fact Sheet:** any of the 12 funds, six required headline statistics,
   growth, drawdown, latest target holdings, sector/crypto mix, and supplementary
   risk/turnover measures.
2. **Allocation Lab:** up to four fund sleeves, automatic input normalisation,
   scenario growth/drawdown, and contribution detail. Sleeves are initially
   allocated and then allowed to drift; the tool does not imply daily
   rebalancing. When native crypto is selected, weekend crypto returns are
   retained and equity-fund weekend returns are 0%.
3. **News & Innovation:** selectable sector sentiment, news coverage, aggregation
   explanation, fusion family control, before/after performance, and 0/10 bps ×
   10/25/40% robustness evidence.

The visual system uses AtlasSignal navy, blue, gold, and orange with non-colour
method shapes/line patterns. Browser QA covered the default page, a crypto fact
sheet, allocation changes, both fusion families, a 1,440-pixel desktop viewport,
and a 390-pixel narrow viewport with no page-level horizontal overflow.

## Current report status

The student has reviewed and revised the assessed narrative, interpretation,
reflection, limitations, and recommendations. The final report is available as:

- `report/report.docx` — editable source document
- `report/report.pdf` — final submission version

The numerical statements and exhibits in the report have been reconciled
against the validated files in `results/`. Historical AI drafting evidence
remains recorded separately in `ai/prompt_log_08_report_draft.md`.
