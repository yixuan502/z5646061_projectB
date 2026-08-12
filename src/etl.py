"""Reusable data cleaning for the Part B modelling pipeline.

All raw inputs are loaded through the provided ``src.data_access`` helper.
The functions return copies so the cached source frames are never modified.
"""

from __future__ import annotations

import pandas as pd

from src import data_access


EQUITY_REQUIRED_COLUMNS = {
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adjClose",
    "volume",
    "sector",
}

CRYPTO_REQUIRED_COLUMNS = EQUITY_REQUIRED_COLUMNS - {"sector"}

HEADLINE_REQUIRED_COLUMNS = {
    "date",
    "ticker",
    "sector",
    "title",
    "url",
    "publisher",
}


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    dataset_name: str,
) -> None:
    """Raise a clear error when a hosted dataset schema has changed."""

    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {sorted(missing)}"
        )


def _validate_price_keys(prices: pd.DataFrame, dataset_name: str) -> None:
    """Require one price observation per ticker and date."""

    duplicate_count = int(prices.duplicated(["ticker", "date"]).sum())
    if duplicate_count:
        raise ValueError(
            f"{dataset_name} contains {duplicate_count} duplicate ticker-date rows"
        )


def load_clean_equities() -> pd.DataFrame:
    """Load, validate, and sort the 50-equity price panel."""

    equities = data_access.load_equity_prices().copy()
    _require_columns(equities, EQUITY_REQUIRED_COLUMNS, "equity_prices")

    equities["date"] = pd.to_datetime(equities["date"]).dt.normalize()
    equities = equities.sort_values(["ticker", "date"]).reset_index(drop=True)
    _validate_price_keys(equities, "equity_prices")

    return equities


def load_clean_crypto() -> pd.DataFrame:
    """Load crypto prices, cap the sample at 2023-12-31, and sort."""

    crypto = data_access.load_crypto_prices().copy()
    _require_columns(crypto, CRYPTO_REQUIRED_COLUMNS, "crypto_prices")

    crypto["date"] = pd.to_datetime(crypto["date"]).dt.normalize()
    crypto = crypto.loc[
        crypto["date"] <= pd.Timestamp("2023-12-31")
    ].copy()
    crypto = crypto.sort_values(["ticker", "date"]).reset_index(drop=True)
    _validate_price_keys(crypto, "crypto_prices")

    return crypto


def load_clean_headlines() -> pd.DataFrame:
    """Normalise dates and remove only exact ticker-date-title duplicates."""

    headlines = data_access.load_news_headlines().copy()
    _require_columns(headlines, HEADLINE_REQUIRED_COLUMNS, "news_headlines")

    headlines["date"] = (
        pd.to_datetime(headlines["date"], utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
        .astype("datetime64[ns]")
    )

    headlines = headlines.drop_duplicates(
        subset=["ticker", "date", "title"],
        keep="first",
    )
    headlines = headlines.sort_values(
        ["ticker", "date", "title"]
    ).reset_index(drop=True)

    return headlines
