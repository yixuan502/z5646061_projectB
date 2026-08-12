# Prompt Log 04 — Standalone Sector Sentiment Index

## Student request

Continue to the next Part B stage after independently confirming the 13 tests and
the twelve walk-forward funds in PyCharm.

## Metric definition proposed by AI

The AI proposed a deliberately auditable three-level VADER pipeline:

1. score each original deduplicated headline using VADER compound sentiment;
2. average all headline scores for a ticker on a trading day;
3. equal-weight the observed ticker-day scores in each equity sector.

The third step follows the brief's required equal-ticker aggregation.  It avoids
allowing a heavily reported company to dominate a sector merely because it has
more headlines.

No-news ticker-days are omitted from the sector mean, not assigned a compound
score of zero.  Sector-days with no news are kept in the complete calendar grid
with a missing raw score, zero coverage, and a `no_news` label.  This distinguishes
absence of information from a headline that VADER actually classified as neutral.

The tradeable field is the prior equity trading day's sector score.  Thus weekend
and Monday headlines mapped to Monday cannot affect Monday's decision; they are
first available for Tuesday.

## Text and model choices

The code sends the exact headline text to VADER. It does not lowercase, strip
punctuation, remove stopwords, or stem because VADER's rules use capitalisation,
punctuation, intensifiers, and negation.  Standard VADER compound thresholds of
+0.05 and -0.05 classify positive, neutral, and negative observations.

Plain VADER is retained as the transparent baseline.  The main project innovation
remains the later Coverage-Adjusted Sector Sentiment design based on the student's
Part A news concentration work, rather than mixing several untested enhancements
into this baseline index.

## Validation plan

Network-free unit tests check that:

- casing and punctuation reach the analyzer unchanged;
- headline scores are averaged within ticker before sector aggregation;
- two tickers receive equal sector weight even when their headline counts differ;
- no-news stays missing rather than being coded neutral;
- a Monday-aligned score becomes the Tuesday signal;
- duplicate ticker-day-title inputs are rejected.

The full-data build reconciles headline counts across every aggregation level,
constructs the complete 1,006-equity-date by 10-sector grid, verifies compound
bounds and unique keys, and proves every available signal source date is earlier
than its use date.

## Critical review to update after execution

The synthetic tests and real-data distribution must be reviewed before accepting
the index.  In particular, the neutral share, thin-sector coverage, extreme-score
headlines, and lag audit should be checked rather than assuming VADER is an
accurate measure of economic news.

The first full-data run stopped because the AI had initially checked duplicates
using `ticker + aligned trading_date + title`.  Investigation found 286 rows in
143 paired groups where the same syndicated title appeared on different original
publication dates but mapped onto one trading day (for example a weekend and the
following Monday).  There were zero duplicates under the brief's correct key,
`ticker + original date + title`.  The validation was corrected to use the
original date whenever it is present, and a regression test now proves that a
same-title/different-publication-date pair is retained.  This avoids deleting
legitimate observations created by calendar alignment.

The corrected real-data run scored all 146,830 mapped headlines and reconciled
them to 37,962 ticker-days and a complete 10,060-row sector-date grid.  Plain
VADER labelled 49.57% of headlines neutral, supporting the brief's warning about
false-neutral finance headlines.  There were 228 no-news sector-days. Materials
and Real Estate each had 66 while Tech had none, confirming unequal coverage.
All sectors had a positive full-sample average compound score, so the later
fusion should use a look-ahead-safe relative or standardised signal rather than
interpreting any raw positive value as bullish.

The equal-ticker sector score differed from the headline-count-weighted diagnostic
by 0.0339 compound units on average and as much as 0.4079, demonstrating why the
brief's equal-ticker aggregation materially matters.  Extreme ticker-days were
reviewed and corresponded to recognisably negative risk/weakness headlines or
positive higher/optimism/rebound headlines; this is a face-validity check, not
proof that VADER captures economic value.

A ten-panel, shared-scale 21-trading-day mean figure was added for the required
sector time-series exhibit.  It states the data grain, dates, score unit, rolling
window, no-news policy, neutral reference, and source/caveat.

Visual QA found that the first figure used the 0.5th and 99.5th percentiles to set
the shared y-axis.  This made the panels compact but slightly clipped the highest
Utilities rolling values, which is not acceptable for evidence.  The export was
revised to use the full observed rolling-score range plus padding, preserving a
shared and honest scale across all ten panels.
