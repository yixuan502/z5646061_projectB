"""Tests for no-look-ahead timing, monthly rebalancing, and weight drift."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, walk_forward_backtest


def _synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    dates = pd.bdate_range("2022-03-01", periods=70)
    return pd.DataFrame(
        rng.normal([0.0002, 0.0004, 0.0001], [0.008, 0.012, 0.005], size=(70, 3)),
        index=dates,
        columns=["A", "B", "C"],
    )


def test_first_rebalance_cannot_see_live_or_future_returns():
    original = _synthetic_panel()
    changed_future = original.copy()
    window = 20
    first_live = original.index[window]
    changed_future.loc[first_live:, "A"] = 0.50

    config = BacktestConfig(
        estimation_window=window,
        periods_per_year=252,
        max_asset_weight=0.80,
    )
    first = walk_forward_backtest(original, "minimum_variance", config)
    second = walk_forward_backtest(changed_future, "minimum_variance", config)

    first_target = first.rebalance_weights.query(
        "rebalance_date == @first_live"
    ).set_index("ticker")["target_weight"]
    second_target = second.rebalance_weights.query(
        "rebalance_date == @first_live"
    ).set_index("ticker")["target_weight"]
    pd.testing.assert_series_equal(first_target, second_target)
    assert first.daily.index.min() == first_live
    assert (first.rebalance_weights["estimation_end"] < first.rebalance_weights["rebalance_date"]).all()
    assert first.rebalance_weights["window_observations"].eq(window).all()


def test_weights_drift_between_monthly_rebalances():
    dates = pd.bdate_range("2022-03-01", periods=40)
    returns = pd.DataFrame(0.0, index=dates, columns=["Winner", "Flat"])
    window = 20
    first_live = dates[window]
    second_live = dates[window + 1]
    returns.loc[first_live, "Winner"] = 0.10

    result = walk_forward_backtest(
        returns,
        "equal_weight",
        BacktestConfig(
            estimation_window=window,
            periods_per_year=252,
            max_asset_weight=0.75,
        ),
    )

    assert result.daily.loc[first_live, "is_rebalance"]
    assert not result.daily.loc[second_live, "is_rebalance"]
    assert result.daily_pre_return_weights.loc[first_live, "Winner"] == pytest.approx(0.50)
    assert result.daily_pre_return_weights.loc[second_live, "Winner"] > 0.50


def test_first_live_day_has_zero_seed_turnover():
    returns = _synthetic_panel()
    result = walk_forward_backtest(
        returns,
        "equal_weight",
        BacktestConfig(20, 252, 0.80, transaction_cost_bps=10.0),
    )
    assert result.daily.iloc[0]["turnover"] == pytest.approx(0.0)
    assert result.daily.iloc[0]["gross_return"] == pytest.approx(
        result.daily.iloc[0]["net_return"]
    )
