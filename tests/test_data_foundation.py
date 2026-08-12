"""Fast, network-free tests for the reused Part A data foundation."""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.features import (  # noqa: E402
    align_headlines_to_trading_days,
    assemble_headline_panel,
    build_combined_returns_panel,
    daily_returns,
)


def test_daily_returns_are_calculated_within_ticker() -> None:
    prices = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "date": pd.to_datetime(
                ["2023-01-02", "2023-01-03", "2023-01-02", "2023-01-03"]
            ),
            "adjClose": [100.0, 110.0, 50.0, 45.0],
        }
    )

    result = daily_returns(prices)

    assert result.groupby("ticker")["return"].first().notna().all()
    assert result.groupby("ticker")["return"].apply(lambda x: x.iloc[0]).isna().all()
    assert np.isclose(result.loc[result["ticker"].eq("AAA"), "return"].iloc[1], 0.10)
    assert np.isclose(result.loc[result["ticker"].eq("BBB"), "return"].iloc[1], -0.10)


def test_combined_panel_uses_equity_dates_without_recomputing_crypto() -> None:
    equity_returns = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": pd.to_datetime(["2023-01-06", "2023-01-09"]),
            "return": [0.01, 0.02],
        }
    )
    crypto_returns = pd.DataFrame(
        {
            "ticker": ["BTC-USD"] * 4,
            "date": pd.to_datetime(
                ["2023-01-06", "2023-01-07", "2023-01-08", "2023-01-09"]
            ),
            "return": [0.03, 0.04, -0.02, 0.05],
        }
    )

    result = build_combined_returns_panel(equity_returns, crypto_returns)

    assert result.index.tolist() == list(pd.to_datetime(["2023-01-06", "2023-01-09"]))
    assert np.isclose(result.loc[pd.Timestamp("2023-01-09"), "BTC-USD"], 0.05)


def test_headlines_map_to_same_or_next_trading_day() -> None:
    headlines = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-06", "2023-01-07", "2023-01-09"], utc=True),
            "ticker": ["AAA", "AAA", "AAA"],
            "sector": ["Tech", "Tech", "Tech"],
            "title": ["Friday", "Saturday", "Monday"],
            "publisher": ["X", "X", "X"],
        }
    )
    trading_dates = pd.to_datetime(["2023-01-06", "2023-01-09"])

    result = align_headlines_to_trading_days(headlines, trading_dates)

    assert result["trading_date"].tolist() == list(
        pd.to_datetime(["2023-01-06", "2023-01-09", "2023-01-09"])
    )
    assert result["moved_to_next_trading_day"].tolist() == [False, True, False]


def test_assembled_headline_order_is_deterministic() -> None:
    headlines = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-09", "2023-01-09"], utc=True),
            "ticker": ["AAA", "AAA"],
            "sector": ["Tech", "Tech"],
            "title": ["Zulu headline", "Alpha headline"],
            "publisher": ["X", "Y"],
        }
    )

    panel = assemble_headline_panel(
        headlines,
        pd.to_datetime(["2023-01-09"]),
    )

    assert panel.loc[0, "combined_headlines"] == (
        "Alpha headline || Zulu headline"
    )
