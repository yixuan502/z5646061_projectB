"""Fund fact-sheet calculations used by scripts and the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd


def wealth_index(daily_returns: pd.Series) -> pd.Series:
    """Growth of one dollar with returns reinvested."""

    returns = daily_returns.astype(float)
    if returns.isna().any() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("daily returns must be complete and finite")
    return (1.0 + returns).cumprod().rename("growth_1")


def drawdown_series(daily_returns: pd.Series) -> pd.Series:
    """Percentage fall in wealth from the running historical peak."""

    growth = wealth_index(daily_returns)
    # The investor starts with $1 immediately before the first live return, so
    # the high-water mark must never begin below one after a first-day loss.
    running_peak = growth.cummax().clip(lower=1.0)
    return (growth / running_peak - 1.0).rename("drawdown")


def performance_metrics(
    daily_returns: pd.Series,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> dict[str, float | int | pd.Timestamp]:
    """Calculate required and supplementary fund fact-sheet metrics.

    ``annualized_return`` is the arithmetic daily mean times the appropriate
    calendar factor. ``cagr`` is reported separately so the two conventions are
    transparent rather than silently mixed.
    """

    returns = daily_returns.astype(float)
    if len(returns) < 2:
        raise ValueError("at least two daily returns are required")
    if returns.isna().any() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("daily returns must be complete and finite")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    growth = wealth_index(returns)
    drawdown = drawdown_series(returns)
    annualized_return = float(returns.mean() * periods_per_year)
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = (
        (annualized_return - risk_free_rate) / annualized_volatility
        if annualized_volatility > 0
        else np.nan
    )
    years = len(returns) / periods_per_year
    cagr = float(growth.iloc[-1] ** (1.0 / years) - 1.0)

    downside = np.minimum(returns.to_numpy(), 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year))
    sortino = (
        (annualized_return - risk_free_rate) / downside_deviation
        if downside_deviation > 0
        else np.nan
    )
    fifth_percentile = float(returns.quantile(0.05))
    tail = returns[returns <= fifth_percentile]

    return {
        "start_date": returns.index.min(),
        "end_date": returns.index.max(),
        "observations": int(len(returns)),
        "periods_per_year": int(periods_per_year),
        "average_period_return": float(returns.mean()),
        "annualized_return": annualized_return,
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "total_return": float(growth.iloc[-1] - 1.0),
        "maximum_drawdown": float(drawdown.min()),
        "historical_var_95": float(-fifth_percentile),
        "historical_expected_shortfall_95": float(-tail.mean()),
    }
