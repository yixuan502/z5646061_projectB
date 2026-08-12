"""Tests for required fact-sheet metrics."""

import pandas as pd
import pytest

from src.metrics import drawdown_series, performance_metrics, wealth_index


def test_growth_drawdown_and_annualisation_are_explicit():
    dates = pd.date_range("2023-01-01", periods=4)
    returns = pd.Series([0.10, -0.20, 0.05, 0.02], index=dates)

    growth = wealth_index(returns)
    drawdown = drawdown_series(returns)
    metrics = performance_metrics(returns, periods_per_year=4)

    assert growth.iloc[-1] == pytest.approx(1.10 * 0.80 * 1.05 * 1.02)
    assert drawdown.iloc[1] == pytest.approx(-0.20)
    assert metrics["annualized_return"] == pytest.approx(returns.mean() * 4)
    assert metrics["maximum_drawdown"] == pytest.approx(drawdown.min())
    assert metrics["observations"] == 4


def test_drawdown_includes_initial_one_dollar_high_water_mark():
    dates = pd.date_range("2023-01-01", periods=2)
    returns = pd.Series([-0.10, 0.0], index=dates)

    drawdown = drawdown_series(returns)

    assert drawdown.iloc[0] == pytest.approx(-0.10)
    assert drawdown.iloc[1] == pytest.approx(-0.10)
