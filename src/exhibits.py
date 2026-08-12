"""Report- and app-ready transformations for the twelve investable funds.

These functions do not estimate portfolios or change backtest results.  They
only reshape the validated Stage 2 outputs into investor-facing exhibits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


METHOD_ORDER = (
    "equal_weight",
    "minimum_variance",
    "maximum_sharpe",
    "risk_parity",
)

FAMILY_ORDER = ("equity", "crypto", "combined")


def latest_holdings(
    fund_weights: pd.DataFrame,
    active_tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Return each fund's most recent target portfolio with stable ranks."""

    required = {
        "rebalance_date",
        "fund_id",
        "fund_name",
        "asset_family",
        "method",
        "method_label",
        "ticker",
        "target_weight",
        "asset_class",
        "sector",
    }
    missing = required.difference(fund_weights.columns)
    if missing:
        raise ValueError(f"fund_weights missing columns: {sorted(missing)}")
    if active_tolerance < 0:
        raise ValueError("active_tolerance must be non-negative")

    weights = fund_weights.copy()
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])
    latest_dates = weights.groupby("fund_id")["rebalance_date"].transform("max")
    latest = weights.loc[weights["rebalance_date"].eq(latest_dates)].copy()
    latest["target_weight"] = latest["target_weight"].astype(float)

    sums = latest.groupby("fund_id")["target_weight"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, atol=1e-7):
        raise ValueError("a latest target portfolio does not sum to one")
    if (latest["target_weight"] < -active_tolerance).any():
        raise ValueError("a latest target portfolio contains a short position")

    latest = latest.sort_values(
        ["fund_id", "target_weight", "ticker"],
        ascending=[True, False, True],
    )
    latest["holding_rank"] = latest.groupby("fund_id").cumcount() + 1
    latest["is_active_holding"] = latest["target_weight"].gt(active_tolerance)
    latest["target_weight_pct"] = latest["target_weight"] * 100.0

    columns = [
        "fund_id",
        "fund_name",
        "asset_family",
        "method",
        "method_label",
        "rebalance_date",
        "holding_rank",
        "ticker",
        "asset_class",
        "sector",
        "target_weight",
        "target_weight_pct",
        "is_active_holding",
    ]
    return latest[columns].reset_index(drop=True)


def combined_allocation_history(fund_weights: pd.DataFrame) -> pd.DataFrame:
    """Aggregate combined-fund targets to ten equity sectors plus crypto."""

    required = {
        "rebalance_date",
        "fund_id",
        "fund_name",
        "asset_family",
        "method",
        "method_label",
        "target_weight",
        "asset_class",
        "sector",
    }
    missing = required.difference(fund_weights.columns)
    if missing:
        raise ValueError(f"fund_weights missing columns: {sorted(missing)}")

    combined = fund_weights.loc[fund_weights["asset_family"].eq("combined")].copy()
    if combined.empty:
        raise ValueError("fund_weights does not contain combined funds")
    combined["rebalance_date"] = pd.to_datetime(combined["rebalance_date"])
    combined["allocation_bucket"] = np.where(
        combined["asset_class"].eq("Crypto"), "Crypto", combined["sector"]
    )
    group_columns = [
        "rebalance_date",
        "fund_id",
        "fund_name",
        "asset_family",
        "method",
        "method_label",
        "allocation_bucket",
    ]
    history = (
        combined.groupby(group_columns, as_index=False, observed=True)["target_weight"]
        .sum()
        .sort_values(["method", "rebalance_date", "allocation_bucket"])
        .reset_index(drop=True)
    )
    history["target_weight_pct"] = history["target_weight"] * 100.0

    sums = history.groupby(["fund_id", "rebalance_date"])["target_weight"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, atol=1e-7):
        raise ValueError("aggregated combined targets do not sum to one")
    return history


def fact_sheet_summary(
    performance: pd.DataFrame,
    holdings: pd.DataFrame,
) -> pd.DataFrame:
    """Create one app-readable fact-sheet row per investable fund."""

    required_metrics = {
        "fund_id",
        "fund_name",
        "asset_family",
        "method",
        "method_label",
        "start_date",
        "end_date",
        "observations",
        "periods_per_year",
        "annualized_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "total_return",
        "maximum_drawdown",
        "historical_var_95",
        "historical_expected_shortfall_95",
        "annualized_turnover",
        "latest_rebalance_date",
        "risk_free_rate",
        "transaction_cost_bps",
    }
    missing = required_metrics.difference(performance.columns)
    if missing:
        raise ValueError(f"performance metrics missing columns: {sorted(missing)}")
    required_holdings = {
        "fund_id",
        "holding_rank",
        "ticker",
        "target_weight",
        "is_active_holding",
    }
    missing_holdings = required_holdings.difference(holdings.columns)
    if missing_holdings:
        raise ValueError(f"latest holdings missing columns: {sorted(missing_holdings)}")

    active_counts = (
        holdings.loc[holdings["is_active_holding"]]
        .groupby("fund_id")
        .size()
        .rename("active_holdings")
    )
    top = (
        holdings.loc[holdings["holding_rank"].eq(1), ["fund_id", "ticker", "target_weight"]]
        .rename(columns={"ticker": "top_holding", "target_weight": "top_holding_weight"})
    )
    summary = performance.copy().merge(active_counts, on="fund_id", how="left")
    summary = summary.merge(top, on="fund_id", how="left", validate="one_to_one")
    summary["active_holdings"] = summary["active_holdings"].fillna(0).astype(int)
    summary["top_holding_weight_pct"] = summary["top_holding_weight"] * 100.0
    summary["calendar_basis"] = np.where(
        summary["periods_per_year"].eq(365),
        "Native seven-day crypto calendar",
        "US equity trading-day calendar",
    )
    summary["sharpe_rank_within_family"] = (
        summary.groupby("asset_family")["sharpe_ratio"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    summary["risk_rank_within_family"] = (
        summary.groupby("asset_family")["annualized_volatility"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    family_order = pd.Categorical(summary["asset_family"], FAMILY_ORDER, ordered=True)
    method_order = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    summary = (
        summary.assign(_family_order=family_order, _method_order=method_order)
        .sort_values(["_family_order", "_method_order"])
        .drop(columns=["_family_order", "_method_order"])
        .reset_index(drop=True)
    )
    return summary


def performance_display_table(performance: pd.DataFrame) -> pd.DataFrame:
    """Return the required metrics in report-ready percentage units."""

    required = {
        "fund_name",
        "asset_family",
        "method",
        "method_label",
        "start_date",
        "end_date",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
    }
    missing = required.difference(performance.columns)
    if missing:
        raise ValueError(f"performance metrics missing columns: {sorted(missing)}")

    display = performance[list(required)].copy()
    display["Annualised Return (%)"] = display["annualized_return"] * 100.0
    display["Annualised Volatility (%)"] = display["annualized_volatility"] * 100.0
    display["Sharpe Ratio"] = display["sharpe_ratio"]
    display["Maximum Drawdown (%)"] = display["maximum_drawdown"] * 100.0
    display = display.rename(
        columns={
            "fund_name": "Fund",
            "asset_family": "Asset Family",
            "method_label": "Method",
            "start_date": "OOS Start",
            "end_date": "OOS End",
        }
    )
    display = display[
        [
            "Fund",
            "Asset Family",
            "Method",
            "OOS Start",
            "OOS End",
            "Annualised Return (%)",
            "Annualised Volatility (%)",
            "Sharpe Ratio",
            "Maximum Drawdown (%)",
        ]
    ]
    numeric = [
        "Annualised Return (%)",
        "Annualised Volatility (%)",
        "Sharpe Ratio",
        "Maximum Drawdown (%)",
    ]
    display[numeric] = display[numeric].round(3)
    return display
