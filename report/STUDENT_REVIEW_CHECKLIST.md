# Stage 7A — Student review checklist

`report/report.docx` is an editable evidence draft, not the final submitted report.
Complete this checklist in Microsoft Word before creating `report/report.pdf`.

## Required student work

- Rewrite the executive summary, economic interpretation, critical reflection,
  and three recommendations in your own voice. Keep the verified numbers and
  do not change a claim unless you also re-run and check the underlying output.
- Confirm that you can explain the four portfolio methods, the walk-forward
  timing rule, 252 versus 365 annualisation, the sentiment lag, and the
  coverage-confidence equation without relying on the AI draft.
- Check every reference to Exhibits A1–A7 against the appendix and the matching
  file in `results/`.
- Add your final report word count if your tutor requires it.
- Remove every `STUDENT REVIEW DRAFT` notice only after the text is genuinely
  yours and the report is ready for submission.
- Export the reviewed Word file to `report/report.pdf`; do not print a browser
  page or convert a Markdown planning file.

## Word checks

1. Open `report/report.docx` in Microsoft Word from the PyCharm project tree.
2. Review spelling, page breaks, captions, table wrapping, and all six figures.
3. Confirm the narrative is no more than 10 pages, excluding the cover,
   references, and appendices as permitted by the brief.
4. Save, then export a PDF and compare the Word and PDF page by page.
5. Run `python scripts/check_handin.py` from the project root.

## Evidence that must not be softened or removed

- No optimiser is the realised Sharpe winner in all three asset families.
- Crypto Minimum Variance still has 76.9% annualised volatility and a -72.8%
  maximum drawdown.
- Maximum Sharpe has high turnover and is not the top realised Sharpe method in
  any family.
- All 24 non-base sentiment robustness scenarios have negative Sharpe changes.
- Coverage adjustment reduces the damage relative to naive sentiment but does
  not demonstrate alpha.
- The three-year backtest, zero-cost baseline, zero risk-free rate, and
  prototype/not-advice limitation remain visible.

