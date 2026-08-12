"""Portfolio construction methods for Station 3.

The functions in this module form target weights from an estimation window that
has already been cut off before the live return date.  Timing is enforced by the
walk-forward engine in :mod:`src.backtest`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


SUPPORTED_METHODS = (
    "equal_weight",
    "minimum_variance",
    "maximum_sharpe",
    "risk_parity",
)

METHOD_LABELS = {
    "equal_weight": "Equal Weight",
    "minimum_variance": "Minimum Variance",
    "maximum_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity",
}

_EPSILON = 1e-12


@dataclass(frozen=True)
class PortfolioSolution:
    """A target portfolio and diagnostics based on the estimation window."""

    weights: pd.Series
    expected_annual_return: float
    expected_annual_volatility: float
    expected_sharpe: float
    solver_success: bool
    solver_message: str


def _validate_history(returns: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric, complete, ordered estimation panel."""

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must use a DatetimeIndex")
    if returns.empty or returns.shape[1] < 2:
        raise ValueError("returns must contain at least two assets")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("returns index must be unique and increasing")

    numeric = returns.astype(float)
    values = numeric.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("estimation returns contain missing or infinite values")
    return numeric


def _annual_moments(
    returns: pd.DataFrame,
    periods_per_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate annual arithmetic means and the annual sample covariance."""

    mean = returns.mean().to_numpy(dtype=float) * periods_per_year
    covariance = returns.cov().to_numpy(dtype=float) * periods_per_year

    # A numerically negligible ridge prevents divide-by-zero or singular-matrix
    # failures without acting as an economic covariance-shrinkage model.
    average_variance = float(np.trace(covariance) / len(covariance))
    ridge = max(average_variance * 1e-10, _EPSILON)
    covariance = covariance + np.eye(len(covariance)) * ridge
    return mean, covariance


def risk_contribution_fractions(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Return each asset's fraction of total portfolio variance."""

    marginal = covariance @ weights
    total_variance = float(weights @ marginal)
    if total_variance <= _EPSILON:
        return np.zeros_like(weights)
    return weights * marginal / total_variance


def _check_feasible_start(
    n_assets: int,
    max_asset_weight: float,
    group_indices: np.ndarray,
    group_cap: float | None,
) -> np.ndarray:
    """Create and validate the equal-weight starting portfolio."""

    start = np.full(n_assets, 1.0 / n_assets)
    if start.max() > max_asset_weight + 1e-12:
        raise ValueError("max_asset_weight makes a fully invested portfolio infeasible")
    if group_cap is not None and start[group_indices].sum() > group_cap + 1e-12:
        raise ValueError("group_cap is below the equal-weight feasible starting point")
    return start


def _is_feasible(
    weights: np.ndarray,
    max_asset_weight: float,
    group_indices: np.ndarray,
    group_cap: float | None,
) -> bool:
    """Check numerical feasibility before accepting an optimiser candidate."""

    if not np.isfinite(weights).all():
        return False
    if abs(weights.sum() - 1.0) > 1e-7:
        return False
    if weights.min() < -1e-8 or weights.max() > max_asset_weight + 1e-7:
        return False
    if group_cap is not None and weights[group_indices].sum() > group_cap + 1e-7:
        return False
    return True


def _greedy_return_start(
    mean: np.ndarray,
    max_asset_weight: float,
    group_indices: np.ndarray,
    group_cap: float | None,
) -> np.ndarray:
    """Build a deterministic feasible start favouring high historical means."""

    n_assets = len(mean)
    group_mask = np.zeros(n_assets, dtype=bool)
    group_mask[group_indices] = True
    weights = np.zeros(n_assets)
    remaining_total = 1.0
    remaining_group = group_cap if group_cap is not None else np.inf

    for position in np.argsort(mean)[::-1]:
        allowance = min(max_asset_weight, remaining_total)
        if group_mask[position]:
            allowance = min(allowance, remaining_group)
        allocation = max(float(allowance), 0.0)
        weights[position] = allocation
        remaining_total -= allocation
        if group_mask[position]:
            remaining_group -= allocation
        if remaining_total <= 1e-12:
            break

    if remaining_total > 1e-8:
        raise ValueError("constraints do not permit a fully invested greedy start")
    weights = weights / weights.sum()
    return weights


def optimise_weights(
    returns: pd.DataFrame,
    method: str,
    periods_per_year: int,
    max_asset_weight: float,
    risk_free_rate: float = 0.0,
    group_tickers: tuple[str, ...] = (),
    group_cap: float | None = None,
) -> PortfolioSolution:
    """Form a long-only, fully invested target portfolio.

    Parameters
    ----------
    returns:
        Complete historical return observations available before the live date.
    method:
        One of equal weight, minimum variance, maximum Sharpe, or risk parity.
    periods_per_year:
        252 for equity-calendar panels and 365 for native crypto returns.
    max_asset_weight:
        Long-only upper bound applied to every asset.
    risk_free_rate:
        Annual rate used in the maximum-Sharpe objective.  The project baseline
        explicitly assumes zero.
    group_tickers / group_cap:
        Optional aggregate cap, used to limit the combined fund's crypto sleeve.
    """

    history = _validate_history(returns)
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if not 0 < max_asset_weight <= 1:
        raise ValueError("max_asset_weight must be in (0, 1]")
    if group_cap is not None and not 0 < group_cap <= 1:
        raise ValueError("group_cap must be in (0, 1]")

    columns = history.columns
    n_assets = len(columns)
    group_indices = np.flatnonzero(columns.isin(group_tickers))
    if group_cap is not None and len(group_indices) == 0:
        raise ValueError("group_cap was supplied but no group_tickers were found")

    initial = _check_feasible_start(
        n_assets,
        max_asset_weight,
        group_indices,
        group_cap,
    )
    mean, covariance = _annual_moments(history, periods_per_year)

    constraints: list[dict] = [
        {"type": "eq", "fun": lambda weights: weights.sum() - 1.0}
    ]
    if group_cap is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights: group_cap - weights[group_indices].sum(),
            }
        )

    if method == "equal_weight":
        weights = initial
        solver_success = True
        solver_message = "Deterministic equal-weight rule"
    else:
        if method == "minimum_variance":
            objective = lambda weights: float(weights @ covariance @ weights)
        elif method == "maximum_sharpe":
            def objective(weights: np.ndarray) -> float:
                volatility = np.sqrt(max(float(weights @ covariance @ weights), _EPSILON))
                return -float((weights @ mean - risk_free_rate) / volatility)
        else:
            target = np.full(n_assets, 1.0 / n_assets)

            def objective(weights: np.ndarray) -> float:
                contributions = risk_contribution_fractions(weights, covariance)
                return float(n_assets * np.square(contributions - target).sum())

        starting_points = [initial]
        if method == "maximum_sharpe":
            # Maximum Sharpe is non-linear and can stall on a binding cap from a
            # single start.  Two additional deterministic feasible starts make
            # convergence auditable without random, non-reproducible retries.
            minimum_variance_start = minimize(
                lambda candidate: float(candidate @ covariance @ candidate),
                initial,
                method="SLSQP",
                bounds=[(0.0, max_asset_weight)] * n_assets,
                constraints=constraints,
                options={"maxiter": 2_000, "ftol": 1e-10, "disp": False},
            )
            if minimum_variance_start.success and _is_feasible(
                np.asarray(minimum_variance_start.x),
                max_asset_weight,
                group_indices,
                group_cap,
            ):
                starting_points.append(np.asarray(minimum_variance_start.x))
            starting_points.append(
                _greedy_return_start(
                    mean,
                    max_asset_weight,
                    group_indices,
                    group_cap,
                )
            )

        candidates = []
        failure_messages = []
        for start in starting_points:
            result = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=[(0.0, max_asset_weight)] * n_assets,
                constraints=constraints,
                options={"maxiter": 2_000, "ftol": 1e-10, "disp": False},
            )
            candidate = np.asarray(result.x, dtype=float)
            if result.success and _is_feasible(
                candidate,
                max_asset_weight,
                group_indices,
                group_cap,
            ):
                candidates.append((float(objective(candidate)), candidate, str(result.message)))
            else:
                failure_messages.append(str(result.message))

        if not candidates:
            details = "; ".join(failure_messages)
            raise RuntimeError(f"{METHOD_LABELS[method]} optimisation failed: {details}")
        _, weights, best_message = min(candidates, key=lambda item: item[0])
        solver_success = True
        solver_message = (
            f"{best_message}; accepted best of {len(candidates)} converged "
            f"candidate(s) from {len(starting_points)} deterministic start(s)"
        )

    if abs(weights.sum() - 1.0) > 1e-7:
        raise RuntimeError("optimised weights do not sum to one")
    if weights.min() < -1e-8 or weights.max() > max_asset_weight + 1e-7:
        raise RuntimeError("optimised weights violate the asset bounds")
    if group_cap is not None and weights[group_indices].sum() > group_cap + 1e-7:
        raise RuntimeError("optimised weights violate the aggregate group cap")

    # Remove numerical dust and renormalise only when it is safe to do so.
    weights[np.abs(weights) < 1e-12] = 0.0
    weights = weights / weights.sum()
    annual_return = float(weights @ mean)
    annual_volatility = float(np.sqrt(max(weights @ covariance @ weights, 0.0)))
    annual_sharpe = (
        (annual_return - risk_free_rate) / annual_volatility
        if annual_volatility > _EPSILON
        else np.nan
    )

    return PortfolioSolution(
        weights=pd.Series(weights, index=columns, name="target_weight"),
        expected_annual_return=annual_return,
        expected_annual_volatility=annual_volatility,
        expected_sharpe=float(annual_sharpe),
        solver_success=solver_success,
        solver_message=solver_message,
    )
