"""Lightweight, testable data and scenario logic for the Streamlit app.

The deployed app reads only precomputed Part B artifacts.  It never rebuilds a
portfolio backtest or rescoring headlines at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.metrics import performance_metrics


APP_FILE_SPECS = {
    "fact_sheets": ("results/tables/fund_fact_sheets.csv", ["start_date", "end_date", "latest_rebalance_date"]),
    "fund_returns": ("results/data/fund_returns.csv", ["date"]),
    "latest_holdings": ("results/data/latest_holdings.csv", ["rebalance_date"]),
    "allocation_history": ("results/data/combined_allocation_history.csv", ["rebalance_date"]),
    "sentiment": ("results/data/sector_sentiment_index.csv", ["date", "signal_source_date"]),
    "sentiment_summary": ("results/tables/sentiment_sector_summary.csv", []),
    "fusion_comparison": ("results/tables/fusion_comparison.csv", ["start_date", "end_date"]),
    "fusion_robustness": ("results/tables/fusion_robustness.csv", ["start_date", "end_date"]),
    "fusion_returns": ("results/data/fusion_returns.csv", ["date"]),
}


def load_app_data(project_root: str | Path) -> dict[str, pd.DataFrame]:
    """Load and validate the bounded CSV snapshot used by Streamlit."""

    root = Path(project_root)
    data: dict[str, pd.DataFrame] = {}
    missing_files = []
    for key, (relative_path, date_columns) in APP_FILE_SPECS.items():
        path = root / relative_path
        if not path.exists():
            missing_files.append(relative_path)
            continue
        data[key] = pd.read_csv(path, parse_dates=date_columns or None)
    if missing_files:
        raise FileNotFoundError(
            "Missing precomputed app artifacts: "
            f"{missing_files}. Run python scripts/run_part_b.py first."
        )

    fact_sheets = data["fact_sheets"]
    returns = data["fund_returns"]
    holdings = data["latest_holdings"]
    sentiment = data["sentiment"]
    fusion = data["fusion_comparison"]

    if len(fact_sheets) != 12 or fact_sheets["fund_id"].nunique() != 12:
        raise ValueError("fact-sheet snapshot must contain exactly twelve funds")
    if returns["fund_id"].nunique() != 12:
        raise ValueError("fund-return snapshot must contain twelve funds")
    if returns.duplicated(["fund_id", "date"]).any():
        raise ValueError("fund-return snapshot contains duplicate fund-date rows")
    if holdings["fund_id"].nunique() != 12:
        raise ValueError("latest-holdings snapshot does not cover twelve funds")
    holding_sums = holdings.groupby("fund_id")["target_weight"].sum()
    if not np.allclose(holding_sums.to_numpy(), 1.0, atol=1e-7):
        raise ValueError("a latest target portfolio does not sum to one")
    if sentiment["sector"].nunique() != 10:
        raise ValueError("sentiment snapshot must contain ten equity sectors")
    expected_fusion = {"base", "sentiment", "coverage_adjusted"}
    if set(fusion["fusion_variant"]) != expected_fusion:
        raise ValueError("fusion snapshot is missing a required comparison variant")
    if fusion["asset_family"].nunique() != 2:
        raise ValueError("fusion snapshot must cover equity and combined funds")

    return data


def rolling_sector_sentiment(
    sentiment: pd.DataFrame,
    sectors: list[str] | tuple[str, ...],
    window: int = 21,
    minimum_observations: int = 10,
) -> pd.DataFrame:
    """Create the chart-only trailing mean used in the sentiment explorer."""

    if window <= 0 or minimum_observations <= 0 or minimum_observations > window:
        raise ValueError("rolling-window settings are invalid")
    required = {"date", "sector", "raw_sector_sentiment"}
    missing = required.difference(sentiment.columns)
    if missing:
        raise ValueError(f"sentiment missing columns: {sorted(missing)}")

    selected = sentiment.loc[sentiment["sector"].isin(sectors)].copy()
    if selected.empty:
        raise ValueError("select at least one available sector")
    selected["date"] = pd.to_datetime(selected["date"])
    selected = selected.sort_values(["sector", "date"])
    selected["sentiment_21d"] = selected.groupby("sector", observed=True)[
        "raw_sector_sentiment"
    ].transform(
        lambda values: values.rolling(
            window,
            min_periods=minimum_observations,
        ).mean()
    )
    return selected


def build_allocation_scenario(
    fund_returns: pd.DataFrame,
    fact_sheets: pd.DataFrame,
    allocations: Mapping[str, float],
) -> tuple[pd.DataFrame, dict[str, float | int | pd.Timestamp | str], pd.DataFrame]:
    """Build a static fund-sleeve scenario on a calendar-aware common sample.

    Allocations are initial dollars and are normalised to one.  Each fund sleeve
    then compounds independently, so weights drift rather than being reset every
    day.  When a native crypto fund is included, the scenario uses a seven-day
    calendar and assigns zero return to equity-calendar funds on non-trading days.
    """

    required_returns = {"date", "fund_id", "net_return"}
    missing_returns = required_returns.difference(fund_returns.columns)
    if missing_returns:
        raise ValueError(f"fund_returns missing columns: {sorted(missing_returns)}")
    required_facts = {
        "fund_id",
        "fund_name",
        "asset_family",
        "method_label",
        "periods_per_year",
    }
    missing_facts = required_facts.difference(fact_sheets.columns)
    if missing_facts:
        raise ValueError(f"fact_sheets missing columns: {sorted(missing_facts)}")
    if not allocations:
        raise ValueError("select at least one fund")

    allocation = pd.Series(allocations, dtype=float)
    if not np.isfinite(allocation.to_numpy()).all() or (allocation < 0).any():
        raise ValueError("allocation inputs must be finite and non-negative")
    allocation = allocation.loc[allocation.gt(0)]
    if allocation.empty:
        raise ValueError("allocation must contain a positive amount")
    allocation = allocation / allocation.sum()

    facts = fact_sheets.set_index("fund_id")
    unknown = allocation.index.difference(facts.index)
    if len(unknown):
        raise ValueError(f"unknown fund ids: {unknown.tolist()}")

    selected_returns = fund_returns.loc[
        fund_returns["fund_id"].isin(allocation.index),
        ["date", "fund_id", "net_return"],
    ].copy()
    selected_returns["date"] = pd.to_datetime(selected_returns["date"])
    available = set(selected_returns["fund_id"])
    missing_series = set(allocation.index).difference(available)
    if missing_series:
        raise ValueError(f"missing return series: {sorted(missing_series)}")

    date_bounds = selected_returns.groupby("fund_id")["date"].agg(["min", "max"])
    start_date = date_bounds["min"].max()
    end_date = date_bounds["max"].min()
    if start_date >= end_date:
        raise ValueError("selected funds do not have a usable common sample")

    periods = facts.loc[allocation.index, "periods_per_year"].astype(int)
    uses_daily_calendar = bool(periods.eq(365).any())
    if uses_daily_calendar:
        calendar = pd.date_range(start_date, end_date, freq="D")
        scenario_periods = 365
        calendar_basis = "Seven-day calendar; non-trading equity days carry 0% return"
    else:
        calendar = pd.DatetimeIndex(
            sorted(
                selected_returns.loc[
                    selected_returns["date"].between(start_date, end_date), "date"
                ].unique()
            )
        )
        scenario_periods = 252
        calendar_basis = "US equity trading-day calendar"

    return_matrix = pd.DataFrame(index=calendar)
    for fund_id in allocation.index:
        series = (
            selected_returns.loc[selected_returns["fund_id"].eq(fund_id)]
            .set_index("date")["net_return"]
            .sort_index()
            .reindex(calendar)
        )
        fund_periods = int(facts.loc[fund_id, "periods_per_year"])
        if fund_periods == 365 and series.isna().any():
            raise ValueError(f"native-calendar returns contain gaps for {fund_id}")
        if not uses_daily_calendar and series.isna().any():
            raise ValueError(f"equity-calendar returns do not align for {fund_id}")
        return_matrix[fund_id] = series.fillna(0.0)

    sleeve_growth = (1.0 + return_matrix).cumprod()
    portfolio_growth = sleeve_growth.mul(allocation, axis=1).sum(axis=1)
    portfolio_return = portfolio_growth.pct_change()
    portfolio_return.iloc[0] = portfolio_growth.iloc[0] - 1.0
    peak = portfolio_growth.cummax().clip(lower=1.0)
    drawdown = portfolio_growth / peak - 1.0

    daily = pd.DataFrame(
        {
            "date": calendar,
            "portfolio_return": portfolio_return.to_numpy(),
            "growth_1": portfolio_growth.to_numpy(),
            "drawdown": drawdown.to_numpy(),
        }
    )
    metrics = performance_metrics(
        pd.Series(portfolio_return.to_numpy(), index=calendar),
        periods_per_year=scenario_periods,
        risk_free_rate=0.0,
    )
    metrics.update(
        {
            "calendar_basis": calendar_basis,
            "allocation_model": "Static initial fund sleeves; no inter-fund rebalancing",
            "ending_value": float(portfolio_growth.iloc[-1]),
        }
    )

    allocation_table = (
        facts.loc[
            allocation.index,
            ["fund_name", "asset_family", "method_label"],
        ]
        .reset_index()
        .rename(columns={"index": "fund_id"})
    )
    allocation_table["normalized_weight"] = allocation_table["fund_id"].map(allocation)
    allocation_table["sleeve_ending_growth"] = allocation_table["fund_id"].map(
        sleeve_growth.iloc[-1]
    )
    allocation_table["ending_value_contribution"] = (
        allocation_table["normalized_weight"] * allocation_table["sleeve_ending_growth"]
    )
    allocation_table["gain_contribution"] = allocation_table["normalized_weight"] * (
        allocation_table["sleeve_ending_growth"] - 1.0
    )
    allocation_table = allocation_table.sort_values(
        "normalized_weight", ascending=False
    ).reset_index(drop=True)
    return daily, metrics, allocation_table
