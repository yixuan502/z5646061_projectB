"""Tests for the Streamlit app's precomputed-data and allocation logic."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.app_logic import (
    build_allocation_scenario,
    load_app_data,
    rolling_sector_sentiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_real_app_snapshot_is_complete_and_reconciled():
    data = load_app_data(PROJECT_ROOT)

    assert data["fact_sheets"]["fund_id"].nunique() == 12
    assert data["latest_holdings"]["fund_id"].nunique() == 12
    assert data["sentiment"]["sector"].nunique() == 10
    assert set(data["fusion_comparison"]["fusion_variant"]) == {
        "base",
        "sentiment",
        "coverage_adjusted",
    }


def _facts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fund_id": ["equity", "crypto"],
            "fund_name": ["Equity Fund", "Crypto Fund"],
            "asset_family": ["equity", "crypto"],
            "method_label": ["Risk Parity", "Minimum Variance"],
            "periods_per_year": [252, 365],
        }
    )


def test_equity_only_scenario_uses_252_calendar_and_static_sleeves():
    dates = pd.bdate_range("2023-01-02", periods=4)
    returns = pd.concat(
        [
            pd.DataFrame({"date": dates, "fund_id": "equity", "net_return": 0.01}),
            pd.DataFrame({"date": dates, "fund_id": "equity_2", "net_return": 0.00}),
        ],
        ignore_index=True,
    )
    facts = pd.concat(
        [
            _facts().iloc[[0]],
            pd.DataFrame(
                {
                    "fund_id": ["equity_2"],
                    "fund_name": ["Second Equity Fund"],
                    "asset_family": ["equity"],
                    "method_label": ["Equal Weight"],
                    "periods_per_year": [252],
                }
            ),
        ],
        ignore_index=True,
    )

    daily, metrics, allocation = build_allocation_scenario(
        returns,
        facts,
        {"equity": 60.0, "equity_2": 40.0},
    )

    expected_end = 0.60 * (1.01**4) + 0.40
    assert metrics["periods_per_year"] == 252
    assert daily["growth_1"].iloc[-1] == pytest.approx(expected_end)
    assert allocation["normalized_weight"].sum() == pytest.approx(1.0)


def test_mixed_scenario_retains_crypto_weekend_returns():
    crypto_dates = pd.date_range("2023-01-06", "2023-01-09", freq="D")
    equity_dates = pd.to_datetime(["2023-01-06", "2023-01-09"])
    returns = pd.concat(
        [
            pd.DataFrame(
                {"date": crypto_dates, "fund_id": "crypto", "net_return": 0.01}
            ),
            pd.DataFrame(
                {"date": equity_dates, "fund_id": "equity", "net_return": 0.00}
            ),
        ],
        ignore_index=True,
    )

    daily, metrics, _ = build_allocation_scenario(
        returns,
        _facts(),
        {"equity": 50.0, "crypto": 50.0},
    )

    assert metrics["periods_per_year"] == 365
    assert list(daily["date"]) == list(crypto_dates)
    assert daily["growth_1"].iloc[-1] == pytest.approx(0.50 + 0.50 * (1.01**4))


def test_allocation_rejects_negative_or_unknown_inputs():
    dates = pd.date_range("2023-01-01", periods=3)
    returns = pd.DataFrame(
        {"date": dates, "fund_id": "crypto", "net_return": [0.01, 0.0, -0.01]}
    )

    with pytest.raises(ValueError, match="non-negative"):
        build_allocation_scenario(returns, _facts(), {"crypto": -1.0})
    with pytest.raises(ValueError, match="unknown fund"):
        build_allocation_scenario(returns, _facts(), {"unknown": 1.0})


def test_sentiment_rolling_mean_stays_within_sector():
    dates = pd.date_range("2023-01-01", periods=3)
    sentiment = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "sector": ["Tech"] * 3 + ["Energy"] * 3,
            "raw_sector_sentiment": [1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
        }
    )

    result = rolling_sector_sentiment(
        sentiment,
        ["Tech", "Energy"],
        window=2,
        minimum_observations=1,
    )

    assert np.allclose(result.loc[result["sector"].eq("Tech"), "sentiment_21d"], 1.0)
    assert np.allclose(result.loc[result["sector"].eq("Energy"), "sentiment_21d"], -1.0)
