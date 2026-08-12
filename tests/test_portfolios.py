"""Unit tests for the four portfolio-construction rules."""

import numpy as np
import pandas as pd
import pytest

from src.portfolios import SUPPORTED_METHODS, optimise_weights


@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    rng = np.random.default_rng(5545)
    dates = pd.bdate_range("2020-01-01", periods=500)
    common = rng.normal(0.0002, 0.006, size=(len(dates), 1))
    idiosyncratic = rng.normal(
        loc=[0.0001, 0.0006, -0.0001, 0.0003],
        scale=[0.004, 0.012, 0.020, 0.007],
        size=(len(dates), 4),
    )
    return pd.DataFrame(
        common + idiosyncratic,
        index=dates,
        columns=["A", "B", "C", "D"],
    )


def test_all_methods_respect_long_only_fully_invested_caps(synthetic_returns):
    solutions = {}
    for method in SUPPORTED_METHODS:
        solution = optimise_weights(
            synthetic_returns,
            method=method,
            periods_per_year=252,
            max_asset_weight=0.60,
            group_tickers=("D",),
            group_cap=0.30,
        )
        weights = solution.weights
        assert solution.solver_success
        assert weights.sum() == pytest.approx(1.0, abs=1e-7)
        assert weights.min() >= -1e-8
        assert weights.max() <= 0.60 + 1e-7
        assert weights["D"] <= 0.30 + 1e-7
        solutions[method] = weights

    pairwise_distances = [
        float((solutions[a] - solutions[b]).abs().sum())
        for index, a in enumerate(SUPPORTED_METHODS)
        for b in SUPPORTED_METHODS[index + 1 :]
    ]
    assert min(pairwise_distances) > 1e-4


def test_unknown_method_is_rejected(synthetic_returns):
    with pytest.raises(ValueError, match="Unsupported method"):
        optimise_weights(
            synthetic_returns,
            method="future_oracle",
            periods_per_year=252,
            max_asset_weight=1.0,
        )
