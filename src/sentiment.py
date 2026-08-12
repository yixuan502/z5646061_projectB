"""Auditable VADER sector-sentiment pipeline for Station 3.

The pipeline deliberately separates three grains:

1. score each original headline without stripping casing or punctuation;
2. average headline scores within ticker-day;
3. equal-weight the observed ticker-day scores within each sector-day.

Missing news stays missing.  A zero compound score is a model observation, not a
substitute for no information.  The trading signal is lagged on the complete
equity trading calendar so a score aligned to day t is first usable on day t+1.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np
import pandas as pd


VADER_POSITIVE_THRESHOLD = 0.05
VADER_NEGATIVE_THRESHOLD = -0.05
REQUIRED_HEADLINE_COLUMNS = {"trading_date", "ticker", "sector", "title"}


class SentimentAnalyzer(Protocol):
    """Small interface implemented by NLTK's VADER analyzer and test doubles."""

    def polarity_scores(self, text: str) -> Mapping[str, float]: ...


def make_vader_analyzer() -> SentimentAnalyzer:
    """Create NLTK VADER, with a clear one-time setup error if data is absent."""

    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        return SentimentIntensityAnalyzer()
    except LookupError as error:
        raise LookupError(
            "NLTK VADER lexicon is missing. From the project environment run: "
            "python -m nltk.downloader vader_lexicon"
        ) from error


def _sentiment_label(compound: pd.Series) -> pd.Series:
    """Apply the standard VADER +/-0.05 compound-score classes."""

    labels = np.select(
        [
            compound >= VADER_POSITIVE_THRESHOLD,
            compound <= VADER_NEGATIVE_THRESHOLD,
        ],
        ["positive", "negative"],
        default="neutral",
    )
    return pd.Series(labels, index=compound.index, dtype="string")


def score_headlines(
    headlines: pd.DataFrame,
    analyzer: SentimentAnalyzer | None = None,
) -> pd.DataFrame:
    """Score every mapped headline while preserving its exact input text.

    Parameters
    ----------
    headlines:
        One row per deduplicated headline, already aligned to an equity trading
        day. Unmapped rows after the final trading day must be removed first.
    analyzer:
        Optional analyzer implementing ``polarity_scores``.  Dependency injection
        keeps unit tests independent of downloaded NLTK resources.
    """

    missing = REQUIRED_HEADLINE_COLUMNS.difference(headlines.columns)
    if missing:
        raise ValueError(f"headlines missing required columns: {sorted(missing)}")
    if headlines.empty:
        raise ValueError("headlines cannot be empty")
    if headlines["trading_date"].isna().any():
        raise ValueError("headlines contain unmapped trading dates")
    if headlines["title"].isna().any():
        raise ValueError("headline title cannot be missing")

    scored = headlines.copy()
    scored["trading_date"] = pd.to_datetime(scored["trading_date"])
    # The brief defines an exact duplicate on the original publication date.
    # After weekend-to-next-trading-day alignment, two legitimately separate
    # publication dates can share one trading date and the same syndicated title.
    duplicate_key = (
        ["ticker", "date", "title"]
        if "date" in scored.columns
        else ["ticker", "trading_date", "title"]
    )
    if scored.duplicated(duplicate_key).any():
        raise ValueError(f"duplicate headline keys detected on {duplicate_key}")

    model = analyzer if analyzer is not None else make_vader_analyzer()
    results = [model.polarity_scores(str(title)) for title in scored["title"]]
    required_scores = {"neg", "neu", "pos", "compound"}
    if any(required_scores.difference(result) for result in results):
        raise ValueError("sentiment analyzer did not return VADER score fields")

    for field in ("neg", "neu", "pos", "compound"):
        scored[field] = [float(result[field]) for result in results]
    if not np.isfinite(scored[["neg", "neu", "pos", "compound"]].to_numpy()).all():
        raise ValueError("sentiment analyzer returned missing or infinite scores")
    if not scored["compound"].between(-1.0, 1.0).all():
        raise ValueError("VADER compound scores must lie within [-1, 1]")

    scored["sentiment_label"] = _sentiment_label(scored["compound"])
    return scored


def ticker_day_sentiment(scored_headlines: pd.DataFrame) -> pd.DataFrame:
    """Average headline-level VADER scores within each ticker and trading day."""

    required = {
        "trading_date",
        "ticker",
        "sector",
        "title",
        "neg",
        "neu",
        "pos",
        "compound",
        "sentiment_label",
    }
    missing = required.difference(scored_headlines.columns)
    if missing:
        raise ValueError(f"scored_headlines missing columns: {sorted(missing)}")
    if scored_headlines.empty:
        raise ValueError("scored_headlines cannot be empty")

    grouped = (
        scored_headlines.groupby(
            ["trading_date", "ticker", "sector"],
            as_index=False,
            observed=True,
        )
        .agg(
            ticker_sentiment=("compound", "mean"),
            ticker_positive=("pos", "mean"),
            ticker_neutral=("neu", "mean"),
            ticker_negative=("neg", "mean"),
            headline_count=("title", "size"),
            positive_headlines=("sentiment_label", lambda values: int((values == "positive").sum())),
            neutral_headlines=("sentiment_label", lambda values: int((values == "neutral").sum())),
            negative_headlines=("sentiment_label", lambda values: int((values == "negative").sum())),
            sentiment_dispersion=("compound", lambda values: float(np.std(values, ddof=0))),
        )
        .sort_values(["trading_date", "sector", "ticker"])
        .reset_index(drop=True)
    )
    grouped["ticker_sentiment_label"] = _sentiment_label(grouped["ticker_sentiment"])

    classified = grouped[
        ["positive_headlines", "neutral_headlines", "negative_headlines"]
    ].sum(axis=1)
    if not classified.eq(grouped["headline_count"]).all():
        raise RuntimeError("headline sentiment classes do not reconcile to counts")
    return grouped


def sector_sentiment_index(
    ticker_scores: pd.DataFrame,
    trading_dates: pd.Series | pd.DatetimeIndex,
    ticker_sector_map: pd.DataFrame,
) -> pd.DataFrame:
    """Build a complete sector-by-trading-day index and one-day-lagged signal.

    ``raw_sector_sentiment`` equal-weights observed ticker-day scores.  Tickers
    without headlines are omitted rather than coded as neutral, while coverage
    fields make that reduced information set visible.  Sector-days with no news
    retain a missing score and ``raw_sentiment_available=False``.
    """

    required_scores = {
        "trading_date",
        "ticker",
        "sector",
        "ticker_sentiment",
        "headline_count",
        "ticker_sentiment_label",
    }
    missing_scores = required_scores.difference(ticker_scores.columns)
    if missing_scores:
        raise ValueError(f"ticker_scores missing columns: {sorted(missing_scores)}")
    required_map = {"ticker", "sector"}
    missing_map = required_map.difference(ticker_sector_map.columns)
    if missing_map:
        raise ValueError(f"ticker_sector_map missing columns: {sorted(missing_map)}")

    mapping = ticker_sector_map[["ticker", "sector"]].drop_duplicates()
    if mapping["ticker"].duplicated().any():
        raise ValueError("a ticker maps to more than one sector")
    universe_sizes = mapping.groupby("sector", observed=True)["ticker"].nunique()
    sectors = pd.Index(sorted(universe_sizes.index.astype(str)), name="sector")
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(trading_dates)).unique()).sort_values()
    if dates.empty or dates.has_duplicates:
        raise ValueError("trading_dates must contain unique dates")

    scored = ticker_scores.copy()
    scored["trading_date"] = pd.to_datetime(scored["trading_date"])
    if scored.duplicated(["trading_date", "ticker", "sector"]).any():
        raise ValueError("duplicate ticker-day-sector scores detected")
    unknown_pairs = scored[["ticker", "sector"]].drop_duplicates().merge(
        mapping,
        on=["ticker", "sector"],
        how="left",
        indicator=True,
    )
    if unknown_pairs["_merge"].ne("both").any():
        raise ValueError("ticker scores contain ticker-sector pairs outside the universe")

    scored["headline_weighted_component"] = (
        scored["ticker_sentiment"] * scored["headline_count"]
    )
    scored["positive_ticker"] = scored["ticker_sentiment_label"].eq("positive").astype(int)
    scored["neutral_ticker"] = scored["ticker_sentiment_label"].eq("neutral").astype(int)
    scored["negative_ticker"] = scored["ticker_sentiment_label"].eq("negative").astype(int)

    observed = (
        scored.groupby(["trading_date", "sector"], as_index=False, observed=True)
        .agg(
            raw_sector_sentiment=("ticker_sentiment", "mean"),
            active_tickers=("ticker", "nunique"),
            headline_count=("headline_count", "sum"),
            headline_weighted_numerator=("headline_weighted_component", "sum"),
            positive_tickers=("positive_ticker", "sum"),
            neutral_tickers=("neutral_ticker", "sum"),
            negative_tickers=("negative_ticker", "sum"),
            cross_ticker_dispersion=("ticker_sentiment", lambda values: float(np.std(values, ddof=0))),
        )
    )
    observed["headline_weighted_sentiment"] = (
        observed["headline_weighted_numerator"] / observed["headline_count"]
    )
    observed = observed.drop(columns="headline_weighted_numerator")

    complete = pd.MultiIndex.from_product(
        [dates, sectors],
        names=["date", "sector"],
    ).to_frame(index=False)
    index = complete.merge(
        observed,
        left_on=["date", "sector"],
        right_on=["trading_date", "sector"],
        how="left",
        validate="one_to_one",
    ).drop(columns="trading_date")

    count_columns = [
        "active_tickers",
        "headline_count",
        "positive_tickers",
        "neutral_tickers",
        "negative_tickers",
    ]
    index[count_columns] = index[count_columns].fillna(0).astype(int)
    index["sector_universe_size"] = index["sector"].map(universe_sizes).astype(int)
    index["ticker_coverage_ratio"] = (
        index["active_tickers"] / index["sector_universe_size"]
    )
    index["raw_sentiment_available"] = index["raw_sector_sentiment"].notna()
    index["raw_sentiment_label"] = _sentiment_label(index["raw_sector_sentiment"])
    index.loc[~index["raw_sentiment_available"], "raw_sentiment_label"] = "no_news"

    index["signal_source_date"] = index.groupby("sector", observed=True)["date"].shift(1)
    index["lagged_sector_sentiment"] = index.groupby(
        "sector", observed=True
    )["raw_sector_sentiment"].shift(1)
    index["lagged_signal_available"] = index["lagged_sector_sentiment"].notna()

    if not index["ticker_coverage_ratio"].between(0.0, 1.0).all():
        raise RuntimeError("ticker coverage falls outside [0, 1]")
    no_news = index["headline_count"].eq(0)
    if not index.loc[no_news, "raw_sector_sentiment"].isna().all():
        raise RuntimeError("no-news sector-days were incorrectly assigned sentiment")
    if not index.loc[~no_news, "raw_sector_sentiment"].notna().all():
        raise RuntimeError("observed sector-days are missing sentiment")

    output_columns = [
        "date",
        "sector",
        "raw_sector_sentiment",
        "lagged_sector_sentiment",
        "signal_source_date",
        "raw_sentiment_available",
        "lagged_signal_available",
        "raw_sentiment_label",
        "active_tickers",
        "sector_universe_size",
        "ticker_coverage_ratio",
        "headline_count",
        "headline_weighted_sentiment",
        "positive_tickers",
        "neutral_tickers",
        "negative_tickers",
        "cross_ticker_dispersion",
    ]
    return index[output_columns].sort_values(["date", "sector"]).reset_index(drop=True)
