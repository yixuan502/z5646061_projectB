"""Look-ahead-safe walk-forward backtesting for the Part B funds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.portfolios import optimise_weights


@dataclass(frozen=True)
class BacktestConfig:
    """Documented project choices for one asset-family backtest."""

    estimation_window: int
    periods_per_year: int
    max_asset_weight: float
    risk_free_rate: float = 0.0
    transaction_cost_bps: float = 0.0
    group_tickers: tuple[str, ...] = ()
    group_cap: float | None = None


@dataclass(frozen=True)
class WalkForwardResult:
    """Daily fund evidence plus monthly targets and audit-friendly weights."""

    daily: pd.DataFrame
    rebalance_weights: pd.DataFrame
    daily_pre_return_weights: pd.DataFrame


def monthly_rebalance_dates(live_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Select the first available observation in every live calendar month."""

    if live_index.empty:
        return pd.DatetimeIndex([])
    month = live_index.to_period("M")
    first_positions = np.r_[True, month[1:] != month[:-1]]
    return live_index[first_positions]


def walk_forward_backtest(
    returns: pd.DataFrame,
    method: str,
    config: BacktestConfig,
) -> WalkForwardResult:
    """Run a monthly fixed-window walk-forward OOS backtest.

    At each rebalance date ``t``, the target uses exactly the preceding
    ``estimation_window`` observations and never includes the return at ``t``.
    Between monthly rebalances, weights drift with asset performance instead of
    incorrectly assuming free daily rebalancing.
    """

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must use a DatetimeIndex")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("returns index must be unique and increasing")
    if config.estimation_window < 2:
        raise ValueError("estimation_window must be at least two")
    if config.transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative")

    panel = returns.astype(float)
    if panel.isna().any().any() or not np.isfinite(panel.to_numpy()).all():
        raise ValueError("backtest returns must be complete and finite")
    if len(panel) <= config.estimation_window:
        raise ValueError("not enough returns for the estimation window and a live period")

    live_index = panel.index[config.estimation_window:]
    rebalance_dates = set(monthly_rebalance_dates(live_index))
    current_weights: pd.Series | None = None
    daily_records: list[dict] = []
    rebalance_records: list[dict] = []
    daily_weight_records: list[pd.Series] = []

    for date in live_index:
        turnover = 0.0
        if date in rebalance_dates:
            history = panel.loc[panel.index < date].tail(config.estimation_window)
            if len(history) != config.estimation_window or history.index.max() >= date:
                raise RuntimeError("estimation-window timing check failed")

            try:
                solution = optimise_weights(
                    history,
                    method=method,
                    periods_per_year=config.periods_per_year,
                    max_asset_weight=config.max_asset_weight,
                    risk_free_rate=config.risk_free_rate,
                    group_tickers=config.group_tickers,
                    group_cap=config.group_cap,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"{method} failed for rebalance date {date.date()}: {error}"
                ) from error
            target = solution.weights
            if current_weights is not None:
                # One-way turnover: 0.5 times the absolute weight change.  The
                # initial portfolio purchase is excluded from recurring turnover.
                turnover = float(0.5 * np.abs(target - current_weights).sum())
            current_weights = target.copy()

            for ticker, target_weight in target.items():
                rebalance_records.append(
                    {
                        "rebalance_date": date,
                        "estimation_start": history.index.min(),
                        "estimation_end": history.index.max(),
                        "window_observations": len(history),
                        "ticker": ticker,
                        "target_weight": float(target_weight),
                        "expected_annual_return": solution.expected_annual_return,
                        "expected_annual_volatility": solution.expected_annual_volatility,
                        "expected_sharpe": solution.expected_sharpe,
                        "solver_success": solution.solver_success,
                        "solver_message": solution.solver_message,
                    }
                )

        if current_weights is None:
            raise RuntimeError("live backtest began without target weights")

        pre_return_weights = current_weights.copy()
        asset_returns = panel.loc[date]
        gross_return = float(pre_return_weights @ asset_returns)
        transaction_cost = turnover * config.transaction_cost_bps / 10_000.0
        net_return = (1.0 - transaction_cost) * (1.0 + gross_return) - 1.0

        daily_records.append(
            {
                "date": date,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "is_rebalance": date in rebalance_dates,
            }
        )
        pre_return_weights.name = date
        daily_weight_records.append(pre_return_weights)

        denominator = 1.0 + gross_return
        if denominator <= 0:
            raise RuntimeError("portfolio lost 100% or more in one observation")
        current_weights = pre_return_weights * (1.0 + asset_returns) / denominator

    daily = pd.DataFrame(daily_records).set_index("date")
    daily_weights = pd.DataFrame(daily_weight_records)
    daily_weights.index = live_index
    daily_weights.index.name = "date"
    rebalances = pd.DataFrame(rebalance_records)
    return WalkForwardResult(daily, rebalances, daily_weights)
