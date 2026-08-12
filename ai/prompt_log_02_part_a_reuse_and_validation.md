# Prompt Log 02 - Part A Reuse and Stage 1 Validation

## Task / prompt

I asked Codex to treat my `z5646061_projectB_step0.zip` as the updated checkpoint,
verify it before changing the working folder, and restart the staged Part B build.
The first development stage was to reuse and validate my own Part A ETL, return,
calendar-alignment, and headline-assembly foundation before writing any portfolio
or sentiment model.

## Financial and data objective

The optimiser must receive returns that were calculated within each asset's own
calendar. Equities use an equity-trading-day calendar, while cryptocurrencies
trade every day. Calculating cryptocurrency returns only after restricting prices
to equity dates would incorrectly turn several daily moves into one return.

The sentiment model also needs a reproducible information date. A headline on a
weekend or market holiday is mapped to the next equity trading day, but it is not
yet a tradable signal; the Stage 3 sentiment pipeline will apply the required
additional trading-day lag.

## AI implementation

Codex replaced the Part B ETL and feature stubs with reusable functions that:

- load only through the provided `src/data_access.py` helper;
- validate the hosted schemas and unique price keys;
- cap cryptocurrency observations at 2023-12-31;
- remove only exact ticker-date-title news duplicates;
- calculate adjusted-close simple returns within ticker;
- preserve native equity and cryptocurrency return panels;
- left-align already-calculated cryptocurrency returns to equity dates;
- map headlines to the same or next equity trading day;
- assemble one deterministic ticker-day headline panel.

It added `scripts/prepare_part_b_data.py`, four network-free timing and alignment
tests, and `pytest` as a development-only dependency.

## Commands and evidence

The official helper loaded:

- equities: 50,300 rows and 9 columns;
- cryptocurrencies: 14,620 raw rows and 8 columns;
- news: 149,683 raw rows and 6 columns.

After cleaning and feature construction, the Stage 1 script produced:

- clean equities: 50,300 rows;
- clean cryptocurrencies: 14,610 rows;
- clean headlines: 146,836 rows;
- equity return panel: 1,006 dates by 50 equities;
- native cryptocurrency panel: 1,461 dates by 10 coins;
- combined panel: 1,006 equity dates by 60 assets;
- daily headline panel: 37,962 ticker-day rows;
- mapped headlines: 146,830;
- headlines moved to a later trading day: 12,551;
- headlines after the final usable trading date: 6.

All seven rows in `results/tables/part_b_data_validation.csv` passed. The four
network-free tests also passed.

## AI weakness found and correction

The first migrated assembly retained an under-specified order for multiple
headlines on the same ticker-day. A comparison with my Part A output found 182
rows containing the same headlines in a different concatenation order. Counts
and information content were unchanged, but the saved CSV was not guaranteed to
be identical across pandas implementations.

I accepted a correction that sorts by trading date, ticker, sector, and title
immediately before aggregation. A new regression test requires deterministic
alphabetical title assembly. After rebuilding, a multiset comparison confirmed
that every Part B ticker-day contains exactly the same headlines as Part A,
although the deliberately standardised ordering differs from the historical
Part A file.

## Interpretation and next decision

The combined Part A return panel was reproduced exactly. The native 365-day
cryptocurrency panel is retained separately because it is required for a valid
crypto-only fund and contains 418 weekend dates. Stage 2 can now build portfolio
functions on validated inputs without silently changing the return definition.
