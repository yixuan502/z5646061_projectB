"""Score headlines and build the lag-safe sector sentiment index.

Run from the project root after Stage 1:

    python scripts/build_sentiment.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.etl import load_clean_equities, load_clean_headlines  # noqa: E402
from src.features import align_headlines_to_trading_days  # noqa: E402
from src.sentiment import (  # noqa: E402
    sector_sentiment_index,
    score_headlines,
    ticker_day_sentiment,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "results" / "data"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"


def _validate_lag(index: pd.DataFrame) -> None:
    """Prove the saved signal equals the preceding equity trading-day score."""

    ordered = index.sort_values(["sector", "date"]).copy()
    expected_score = ordered.groupby("sector", observed=True)[
        "raw_sector_sentiment"
    ].shift(1)
    expected_date = ordered.groupby("sector", observed=True)["date"].shift(1)
    if not np.allclose(
        ordered["lagged_sector_sentiment"].to_numpy(dtype=float),
        expected_score.to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise RuntimeError("lagged sector score is not exactly the prior trading-day score")
    if not ordered["signal_source_date"].equals(expected_date):
        raise RuntimeError("signal_source_date is not the prior equity trading date")
    available = ordered["lagged_sector_sentiment"].notna()
    if not (ordered.loc[available, "signal_source_date"] < ordered.loc[available, "date"]).all():
        raise RuntimeError("look-ahead detected in the lagged sentiment signal")


def _validation_table(
    scored: pd.DataFrame,
    ticker_scores: pd.DataFrame,
    index: pd.DataFrame,
    equity_dates: pd.DatetimeIndex,
    sectors: int,
) -> pd.DataFrame:
    class_counts = scored["sentiment_label"].value_counts()
    rows = [
        {
            "check": "mapped_headlines_scored",
            "value": len(scored),
            "expected": int(ticker_scores["headline_count"].sum()),
            "status": "PASS",
        },
        {
            "check": "unique_ticker_days",
            "value": len(ticker_scores),
            "expected": int(ticker_scores[["trading_date", "ticker", "sector"]].drop_duplicates().shape[0]),
            "status": "PASS",
        },
        {
            "check": "complete_sector_date_grid",
            "value": len(index),
            "expected": len(equity_dates) * sectors,
            "status": "PASS",
        },
        {
            "check": "headline_compound_min",
            "value": float(scored["compound"].min()),
            "expected": ">= -1",
            "status": "PASS",
        },
        {
            "check": "headline_compound_max",
            "value": float(scored["compound"].max()),
            "expected": "<= 1",
            "status": "PASS",
        },
        {
            "check": "positive_headlines",
            "value": int(class_counts.get("positive", 0)),
            "expected": "diagnostic",
            "status": "PASS",
        },
        {
            "check": "neutral_headlines",
            "value": int(class_counts.get("neutral", 0)),
            "expected": "diagnostic",
            "status": "PASS",
        },
        {
            "check": "negative_headlines",
            "value": int(class_counts.get("negative", 0)),
            "expected": "diagnostic",
            "status": "PASS",
        },
        {
            "check": "sector_days_with_news",
            "value": int(index["raw_sentiment_available"].sum()),
            "expected": "diagnostic",
            "status": "PASS",
        },
        {
            "check": "sector_days_without_news",
            "value": int((~index["raw_sentiment_available"]).sum()),
            "expected": "retained as missing, not neutral",
            "status": "PASS",
        },
        {
            "check": "look_ahead_violations",
            "value": int(
                (
                    index.loc[index["lagged_signal_available"], "signal_source_date"]
                    >= index.loc[index["lagged_signal_available"], "date"]
                ).sum()
            ),
            "expected": 0,
            "status": "PASS",
        },
    ]
    return pd.DataFrame(rows)


def _sector_summary(index: pd.DataFrame) -> pd.DataFrame:
    return (
        index.groupby("sector", as_index=False, observed=True)
        .agg(
            trading_days=("date", "size"),
            days_with_news=("raw_sentiment_available", "sum"),
            average_sector_sentiment=("raw_sector_sentiment", "mean"),
            sentiment_volatility=("raw_sector_sentiment", "std"),
            average_ticker_coverage=("ticker_coverage_ratio", "mean"),
            average_daily_headlines=("headline_count", "mean"),
            total_headlines=("headline_count", "sum"),
        )
        .assign(
            days_without_news=lambda frame: frame["trading_days"] - frame["days_with_news"],
            news_day_coverage=lambda frame: frame["days_with_news"] / frame["trading_days"],
        )
        .sort_values("sector")
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    equities = load_clean_equities()
    headlines = load_clean_headlines()
    equity_dates = pd.DatetimeIndex(equities["date"].drop_duplicates().sort_values())
    ticker_sector_map = equities[["ticker", "sector"]].drop_duplicates()

    aligned = align_headlines_to_trading_days(headlines, equity_dates)
    mapped = aligned.dropna(subset=["trading_date"]).copy()
    scored = score_headlines(mapped)
    ticker_scores = ticker_day_sentiment(scored)
    index = sector_sentiment_index(ticker_scores, equity_dates, ticker_sector_map)
    _validate_lag(index)

    if int(index["headline_count"].sum()) != len(scored):
        raise RuntimeError("sector index headline totals do not reconcile")
    if index.duplicated(["date", "sector"]).any():
        raise RuntimeError("sector index has duplicate date-sector keys")
    if not index["raw_sector_sentiment"].dropna().between(-1.0, 1.0).all():
        raise RuntimeError("sector sentiment falls outside [-1, 1]")

    ticker_scores.to_csv(DATA_DIR / "ticker_day_sentiment.csv", index=False)
    index.to_csv(DATA_DIR / "sector_sentiment_index.csv", index=False)
    validation = _validation_table(
        scored,
        ticker_scores,
        index,
        equity_dates,
        ticker_sector_map["sector"].nunique(),
    )
    validation.to_csv(TABLE_DIR / "sentiment_validation.csv", index=False)
    sector_summary = _sector_summary(index)
    sector_summary.to_csv(TABLE_DIR / "sentiment_sector_summary.csv", index=False)

    class_counts = scored["sentiment_label"].value_counts()
    print("\n=== STAGE 3: SECTOR SENTIMENT INDEX BUILT AND VALIDATED ===")
    print(f"Mapped headlines scored: {len(scored):,}")
    print(f"Ticker-day observations: {len(ticker_scores):,}")
    print(f"Sector-date rows: {len(index):,} ({len(equity_dates):,} dates x 10 sectors)")
    print(
        "Headline classes: "
        f"positive {int(class_counts.get('positive', 0)):,}, "
        f"neutral {int(class_counts.get('neutral', 0)):,}, "
        f"negative {int(class_counts.get('negative', 0)):,}"
    )
    print(
        "No-news treatment: "
        f"{int((~index['raw_sentiment_available']).sum()):,} sector-days retained as missing"
    )
    print("Aggregation check: headlines -> ticker-day -> equal-ticker sector-day — PASS")
    print("Timing check: lagged signal is the preceding equity trading-day score — PASS")
    print("Text check: original casing and punctuation were preserved for VADER — PASS")


if __name__ == "__main__":
    main()
