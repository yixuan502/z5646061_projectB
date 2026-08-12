# AI Workflow Notes

## How I used AI

I used AI throughout Part B, mainly for code structure, debugging, testing ideas
and checking whether the implementation matched the project requirements. It
also helped with chart and Streamlit design and with discussing the financial
meaning of the results.

I did not assume that AI output was correct. Suggested code was run in my own
PyCharm environment, and I checked important numerical results against the saved
CSV outputs.
## Errors and risks identified during review

1. Headline Ordering
One issue appeared in the Stage 1 headline aggregation. The first implementation did not fully specify the sorting order before concatenating headlines, so some ticker-day rows had the same headlines but in a different text order. I compared the outputs and confirmed that the information itself had not changed. The sorting rule was then made deterministic and a regression test was added.

2. Maximum Sharpe solver
A second issue occurred in the Maximum Sharpe optimisation. One real-data rebalance produced an SLSQP convergence failure. Instead of ignoring the warning, I checked the optimisation diagnostics. The implementation was changed to use deterministic multiple starting points and to accept only converged and feasible solutions. I then reran the tests and checked that the portfolio constraints still passed.

3. Drawdown Bug
I also found that the first drawdown implementation could miss a loss occurring immediately after the initial investment because the first end-of-day wealth value was treated as the first high-water mark. This was corrected so that the initial $1 is included in the high-water mark. I verified the change with a regression test and later checked that the same definition was used in the app allocation scenario.

4. Sentiment Duplicate Validation
During the sentiment stage, an initial duplicate check used aligned trading dates and incorrectly flagged syndicated headlines published on different original dates. I reviewed the rows and found that the correct duplicate key should use the original publication date. The validation rule was changed, rather than deleting valid observations.

5. Streamlit Presentation
The Streamlit stage also produced several presentation issues, such as a chart shape encoding being dropped, misleading green metric arrows, and the sidebar covering most of the mobile screen. These did not affect portfolio results, but I checked them in the browser and corrected them before finalising the app.

## Suggestions I accepted or rejected

I accepted AI suggestions when they fixed a clear implementation or validation problem and the change could be tested. For example, I accepted the deterministic headline sorting, the Maximum Sharpe multi-start solution and the corrected drawdown definition because each issue had a specific failure that could be reproduced.

I did not keep changing the sentiment model simply to obtain a positive result. The coverage-adjusted signal still underperformed the price-only Risk Parity benchmark. After checking the robustness results, I kept the negative result rather than searching for another parameter that happened to perform better in the same sample. I think changing the rule after seeing the result would create a larger overfitting problem.

## How I verified the work

I ran the project tests locally in PyCharm throughout the development process rather than relying only on AI-generated checks. The final local test suite passed 35 tests, and scripts/check_handin.py passed 23 checks with no warnings.

I also compared saved outputs when changes were made. During the final non-report cleanup, all 31 Stage 1–6 files under results/ had identical SHA-256 hashes before and after the cleanup, confirming that wording and documentation changes did not alter the numerical results.

For the Streamlit app, I tested fund selection, allocation changes and both fusion families in a browser. I also checked desktop and narrow mobile layouts. Numerical statements in the report were compared with the saved performance, holdings and fusion CSV files.

## Final responsibility

AI was useful for speeding up coding, debugging and review, but I made the final decisions about which changes to keep. I reviewed the project results, ran the validation locally and rewrote the assessed economic interpretation in my own words. Any remaining errors in the submitted project are my responsibility.
