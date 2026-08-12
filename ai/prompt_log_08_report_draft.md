# Prompt log 08 — Part B Word report evidence draft

## Student request

Continue to the next project stage after the fund, sentiment, fusion, exhibit,
and Streamlit stages. Preserve the requirement that the final assessed writing
is in the student's own words.

## AI proposal and changes

- Read the Part B report requirements and HD rubric, the verified evidence
  guide, and the student's own Part A report for product continuity.
- Created `report/report.docx` as a visibly marked student-review draft.
- Used the saved Part B CSV outputs for all reported numbers and embedded the
  six precomputed required figures. Added the full metrics table and the fusion
  before/after table.
- Structured the draft as an answer-first report covering the fund shelf and
  OOS design, risk/return results, portfolio behaviour, sentiment, the
  coverage-adjusted innovation, app journey, reflection, recommendations,
  caveats, references, exhibits, and AI transparency.
- Created `report/STUDENT_REVIEW_CHECKLIST.md` so the draft is not mistaken for
  a final student-authored submission.

## Checks performed

- Cross-checked the main reported values against
  `results/tables/performance_metrics.csv`,
  `results/tables/fusion_comparison.csv`,
  `results/tables/sentiment_sector_summary.csv`, and
  `results/tables/backtest_design.csv`.
- Preserved the negative sentiment result and the stated timing, calendar,
  constraint, turnover, cost, and risk-free-rate assumptions.
- Rendered the Word document to page images and inspected every page for layout
  defects. The 17-page render contains one cover page, seven narrative pages,
  one references page, seven required-exhibit pages, and one reproducibility/AI
  appendix page. No clipping, overlap, broken table, unreadable figure, or
  misplaced header/footer was found; the temporary QA PDF and page images were
  kept outside the submission folder.
- Re-ran the automated test suite and `scripts/check_handin.py` after removing
  generated cache and operating-system files.

## Student correction and responsibility

The AI draft is evidence-backed but is not the final assessed writing. The
student must rewrite and confirm the interpretation, reflection, and
recommendations in their own words, remove the draft label, export the final
PDF from Word, and add any material changes or rejected AI suggestions to the
final AI reflection.
