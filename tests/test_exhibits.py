"""Tests for investor-facing fund exhibit transformations."""

import numpy as np
import pandas as pd
import pytest

from src.exhibits import (
    combined_allocation_history,
    latest_holdings,
    performance_display_table,
)


def _weight_rows() -> pd.DataFrame:
    rows = []
    for date, first in (("2023-01-03", 0.60), ("2023-02-01", 0.55)):
        rows.extend(
            [
                {
                    "rebalance_date": date,
                    "fund_id": "combined_equal_weight",
                    "fund_name": "Combined Equal Weight",
                    "asset_family": "combined",
                    "method": "equal_weight",
                    "method_label": "Equal Weight",
                    "ticker": "AAA",
                    "target_weight": first,
                    "asset_class": "Equity",
                    "sector": "Tech",
                },
                {
                    "rebalance_date": date,
                    "fund_id": "combined_equal_weight",
                    "fund_name": "Combined Equal Weight",
                    "asset_family": "combined",
                    "method": "equal_weight",
                    "method_label": "Equal Weight",
                    "ticker": "BTC-USD",
                    "target_weight": 1.0 - first,
                    "asset_class": "Crypto",
                    "sector": "Crypto",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_latest_holdings_uses_only_most_recent_target_and_ranks():
    holdings = latest_holdings(_weight_rows())

    assert holdings["rebalance_date"].nunique() == 1
    assert holdings["rebalance_date"].iloc[0] == pd.Timestamp("2023-02-01")
    assert holdings["target_weight"].sum() == pytest.approx(1.0)
    assert holdings.iloc[0]["ticker"] == "AAA"
    assert holdings.iloc[0]["holding_rank"] == 1
    assert holdings.iloc[0]["target_weight_pct"] == pytest.approx(55.0)


def test_combined_allocation_history_preserves_portfolio_mass():
    history = combined_allocation_history(_weight_rows())
    totals = history.groupby(["fund_id", "rebalance_date"])["target_weight"].sum()

    assert np.allclose(totals.to_numpy(), 1.0)
    assert set(history["allocation_bucket"]) == {"Tech", "Crypto"}


def test_display_table_uses_explicit_percentage_units():
    metrics = pd.DataFrame(
        {
            "fund_name": ["Equity Equal Weight"],
            "asset_family": ["equity"],
            "method": ["equal_weight"],
            "method_label": ["Equal Weight"],
            "start_date": ["2021-01-04"],
            "end_date": ["2023-12-29"],
            "annualized_return": [0.12],
            "annualized_volatility": [0.20],
            "sharpe_ratio": [0.60],
            "maximum_drawdown": [-0.25],
        }
    )

    display = performance_display_table(metrics)

    assert display.loc[0, "Annualised Return (%)"] == pytest.approx(12.0)
    assert display.loc[0, "Annualised Volatility (%)"] == pytest.approx(20.0)
    assert display.loc[0, "Maximum Drawdown (%)"] == pytest.approx(-25.0)
