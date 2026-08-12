"""Part A features reused and validated for Part B.

Returns are calculated inside each asset's native calendar before panels are
aligned. Headline alignment is text assembly only; sentiment scoring and signal
lagging remain Station 3 tasks.
"""

from __future__ import annotations

import pandas as pd


def daily_returns(
    prices: pd.DataFrame,
    price_col: str = "adjClose",
) -> pd.DataFrame:
    """Add simple returns calculated separately within each ticker."""

    required = {"ticker", "date", price_col}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {sorted(missing)}")

    returns = prices.copy()
    returns = returns.sort_values(["ticker", "date"]).reset_index(drop=True)
    returns["return"] = (
        returns.groupby("ticker", sort=False)[price_col]
        .pct_change(fill_method=None)
    )

    return returns


def wide_return_panel(returns: pd.DataFrame) -> pd.DataFrame:
    """Convert long ticker returns to one date row and one column per asset."""

    required = {"ticker", "date", "return"}
    missing = required.difference(returns.columns)
    if missing:
        raise ValueError(f"returns is missing required columns: {sorted(missing)}")

    duplicate_count = int(returns.duplicated(["ticker", "date"]).sum())
    if duplicate_count:
        raise ValueError(
            f"returns contains {duplicate_count} duplicate ticker-date rows"
        )

    panel = returns.pivot(index="date", columns="ticker", values="return")
    panel = panel.sort_index()
    panel.columns.name = None

    return panel


def build_combined_returns_panel(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Left-align already-calculated crypto returns to equity trading dates."""

    equity_panel = wide_return_panel(equity_returns)
    crypto_panel = wide_return_panel(crypto_returns)

    combined = equity_panel.join(crypto_panel, how="left")
    combined = combined.sort_index()
    combined.columns.name = None

    return combined


def align_headlines_to_trading_days(
    headlines: pd.DataFrame,
    equity_dates: pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Map each headline to the same or next available equity trading day."""

    required = {"date", "ticker", "sector", "title", "publisher"}
    missing = required.difference(headlines.columns)
    if missing:
        raise ValueError(
            f"headlines is missing required columns: {sorted(missing)}"
        )

    aligned = headlines.copy()
    aligned["date"] = (
        pd.to_datetime(aligned["date"], utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
        .astype("datetime64[ns]")
    )

    trading_calendar = pd.DataFrame(
        {
            "trading_date": (
                pd.to_datetime(pd.Series(equity_dates))
                .dt.normalize()
                .astype("datetime64[ns]")
            )
        }
    )
    trading_calendar = (
        trading_calendar.drop_duplicates()
        .dropna()
        .sort_values("trading_date")
        .reset_index(drop=True)
    )

    aligned = aligned.sort_values("date").reset_index(drop=True)
    aligned = pd.merge_asof(
        aligned,
        trading_calendar,
        left_on="date",
        right_on="trading_date",
        direction="forward",
        allow_exact_matches=True,
    )

    aligned["moved_to_next_trading_day"] = (
        aligned["trading_date"].notna()
        & aligned["date"].ne(aligned["trading_date"])
    )

    return aligned.reset_index(drop=True)


def assemble_headline_panel(
    headlines: pd.DataFrame,
    equity_dates: pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build one row per trading date, ticker, and sector with raw text intact."""

    aligned = align_headlines_to_trading_days(headlines, equity_dates)
    mapped = aligned.dropna(subset=["trading_date"]).copy()
    mapped["headline_word_count"] = (
        mapped["title"].fillna("").str.split().str.len()
    )
    # A deterministic within-day title order makes the saved text panel stable
    # across pandas sorting implementations and repeated clean builds.
    mapped = mapped.sort_values(
        ["trading_date", "ticker", "sector", "title"]
    ).reset_index(drop=True)

    panel = (
        mapped.groupby(
            ["trading_date", "ticker", "sector"],
            as_index=False,
        )
        .agg(
            headline_count=("title", "size"),
            word_count=("headline_word_count", "sum"),
            unique_publishers=(
                "publisher",
                lambda values: values.dropna().nunique(),
            ),
            combined_headlines=(
                "title",
                lambda values: " || ".join(values.dropna().astype(str)),
            ),
        )
        .sort_values(["trading_date", "ticker"])
        .reset_index(drop=True)
    )

    return panel
