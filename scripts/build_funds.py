"""Build the twelve baseline funds and their OOS fact-sheet artifacts.

Run from the project root after Stage 1:

    python scripts/build_funds.py
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.backtest import BacktestConfig, walk_forward_backtest  # noqa: E402
from src.metrics import drawdown_series, performance_metrics, wealth_index  # noqa: E402
from src.portfolios import METHOD_LABELS, SUPPORTED_METHODS  # noqa: E402


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "results" / "data"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"


@dataclass(frozen=True)
class FundFamily:
    """One investable universe and its calendar/constraint choices."""

    family: str
    family_label: str
    returns: pd.DataFrame
    source_file: str
    estimation_window: int
    periods_per_year: int
    max_asset_weight: float
    calendar_label: str
    group_tickers: tuple[str, ...] = ()
    group_cap: float | None = None


def _load_panel(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python scripts/prepare_part_b_data.py first."
        )
    panel = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    panel = panel.sort_index().astype(float)
    missing_rows = panel.isna().any(axis=1)
    if int(missing_rows.sum()) != 1 or not missing_rows.iloc[0]:
        raise ValueError(f"{filename} does not have the validated first-return NaN pattern")
    panel = panel.loc[~missing_rows]
    if panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError(f"{filename} dates are not unique and increasing")
    if not np.isfinite(panel.to_numpy()).all():
        raise ValueError(f"{filename} contains missing or infinite live inputs")
    return panel


def _sector_map() -> dict[str, str]:
    path = DATA_DIR / "daily_headline_panel.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python scripts/prepare_part_b_data.py first."
        )
    news_keys = pd.read_csv(path, usecols=["ticker", "sector"])
    unique = news_keys.drop_duplicates()
    if unique["ticker"].duplicated().any():
        raise ValueError("A ticker maps to more than one sector")
    mapping = unique.set_index("ticker")["sector"].to_dict()
    return {str(ticker): str(sector) for ticker, sector in mapping.items()}


def _fund_id(family: str, method: str) -> str:
    return f"{family}_{method}"


def _fund_name(family_label: str, method: str) -> str:
    return f"{family_label} {METHOD_LABELS[method]}"


def _method_distance_table(weights: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for family, family_weights in weights.groupby("asset_family"):
        dates = family_weights["rebalance_date"].drop_duplicates().sort_values()
        for method_a, method_b in combinations(SUPPORTED_METHODS, 2):
            distances = []
            for date in dates:
                at_date = family_weights[family_weights["rebalance_date"].eq(date)]
                a = at_date[at_date["method"].eq(method_a)].set_index("ticker")["target_weight"]
                b = at_date[at_date["method"].eq(method_b)].set_index("ticker")["target_weight"]
                distances.append(float((a - b).abs().sum()))
            records.append(
                {
                    "asset_family": family,
                    "method_a": method_a,
                    "method_b": method_b,
                    "average_l1_weight_distance": float(np.mean(distances)),
                    "minimum_l1_weight_distance": float(np.min(distances)),
                    "maximum_l1_weight_distance": float(np.max(distances)),
                    "rebalance_dates_compared": len(distances),
                }
            )
    return pd.DataFrame(records)


def _validate_outputs(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    if fund_returns.duplicated(["date", "fund_id"]).any():
        raise RuntimeError("fund_returns has duplicate date-fund keys")
    return_values = fund_returns[["gross_return", "net_return", "growth_1", "drawdown"]]
    if not np.isfinite(return_values.to_numpy()).all():
        raise RuntimeError("fund_returns contains missing or infinite values")

    sums = fund_weights.groupby(["fund_id", "rebalance_date"])["target_weight"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, atol=1e-7):
        raise RuntimeError("target weights do not sum to one")
    if (fund_weights["target_weight"] < -1e-8).any():
        raise RuntimeError("a target portfolio contains a short position")
    if not fund_weights["solver_success"].all():
        raise RuntimeError("at least one optimisation did not converge")
    if not (fund_weights["estimation_end"] < fund_weights["rebalance_date"]).all():
        raise RuntimeError("look-ahead detected: estimation_end is not before rebalance_date")
    if not fund_weights["window_observations"].eq(
        fund_weights["estimation_window"]
    ).all():
        raise RuntimeError("an estimation window has the wrong number of observations")
    if (
        fund_weights["target_weight"]
        > fund_weights["max_asset_weight"] + 1e-7
    ).any():
        raise RuntimeError("an asset-weight cap was violated")

    combined = fund_weights[fund_weights["asset_family"].eq("combined")]
    crypto_sleeves = (
        combined[combined["asset_class"].eq("Crypto")]
        .groupby(["fund_id", "rebalance_date"])["target_weight"]
        .sum()
    )
    if (crypto_sleeves > 0.20 + 1e-7).any():
        raise RuntimeError("the combined-fund crypto sleeve exceeds 20%")

    if (diagnostics["average_l1_weight_distance"] <= 1e-6).any():
        identical = diagnostics.loc[
            diagnostics["average_l1_weight_distance"] <= 1e-6,
            ["asset_family", "method_a", "method_b"],
        ]
        raise RuntimeError(f"portfolio methods produced indistinguishable weights: {identical}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    equity = _load_panel("equity_returns_panel.csv")
    crypto = _load_panel("crypto_returns_panel.csv")
    combined = _load_panel("combined_returns_panel.csv")
    crypto_tickers = tuple(ticker for ticker in combined.columns if ticker.endswith("-USD"))
    sectors = _sector_map()

    families = [
        FundFamily(
            family="equity",
            family_label="Equity",
            returns=equity,
            source_file="equity_returns_panel.csv",
            estimation_window=252,
            periods_per_year=252,
            max_asset_weight=0.10,
            calendar_label="US equity trading days",
        ),
        FundFamily(
            family="crypto",
            family_label="Crypto",
            returns=crypto,
            source_file="crypto_returns_panel.csv",
            estimation_window=365,
            periods_per_year=365,
            max_asset_weight=0.25,
            calendar_label="Native seven-day crypto calendar",
        ),
        FundFamily(
            family="combined",
            family_label="Combined",
            returns=combined,
            source_file="combined_returns_panel.csv",
            estimation_window=252,
            periods_per_year=252,
            max_asset_weight=0.10,
            calendar_label="US equity trading days; precomputed crypto returns aligned",
            group_tickers=crypto_tickers,
            group_cap=0.20,
        ),
    ]

    return_outputs: list[pd.DataFrame] = []
    weight_outputs: list[pd.DataFrame] = []
    metric_records: list[dict] = []
    design_records: list[dict] = []

    for family in families:
        config = BacktestConfig(
            estimation_window=family.estimation_window,
            periods_per_year=family.periods_per_year,
            max_asset_weight=family.max_asset_weight,
            risk_free_rate=0.0,
            transaction_cost_bps=0.0,
            group_tickers=family.group_tickers,
            group_cap=family.group_cap,
        )
        first_live_date = family.returns.index[family.estimation_window]

        design_records.append(
            {
                "asset_family": family.family,
                "universe_size": family.returns.shape[1],
                "source_file": family.source_file,
                "calendar": family.calendar_label,
                "estimation_window": family.estimation_window,
                "periods_per_year": family.periods_per_year,
                "first_live_date": first_live_date,
                "final_live_date": family.returns.index.max(),
                "rebalance_rule": "First available date of each calendar month",
                "information_cutoff": "Strictly before each rebalance date",
                "long_only": True,
                "fully_invested": True,
                "max_asset_weight": family.max_asset_weight,
                "crypto_sleeve_cap": family.group_cap,
                "risk_free_rate": 0.0,
                "transaction_cost_bps": 0.0,
                "between_rebalances": "Weights drift with asset returns",
            }
        )

        for method in SUPPORTED_METHODS:
            fund_id = _fund_id(family.family, method)
            fund_name = _fund_name(family.family_label, method)
            result = walk_forward_backtest(family.returns, method, config)

            daily = result.daily.copy()
            daily["growth_1"] = wealth_index(daily["net_return"])
            daily["drawdown"] = drawdown_series(daily["net_return"])
            daily = daily.reset_index()
            daily.insert(1, "fund_id", fund_id)
            daily.insert(2, "fund_name", fund_name)
            daily.insert(3, "asset_family", family.family)
            daily.insert(4, "method", method)
            daily.insert(5, "method_label", METHOD_LABELS[method])
            daily.insert(6, "periods_per_year", family.periods_per_year)
            return_outputs.append(daily)

            weights = result.rebalance_weights.copy()
            weights.insert(1, "fund_id", fund_id)
            weights.insert(2, "fund_name", fund_name)
            weights.insert(3, "asset_family", family.family)
            weights.insert(4, "method", method)
            weights.insert(5, "method_label", METHOD_LABELS[method])
            weights["asset_class"] = np.where(
                weights["ticker"].str.endswith("-USD"), "Crypto", "Equity"
            )
            weights["sector"] = weights["ticker"].map(sectors).fillna("Crypto")
            weights["estimation_window"] = family.estimation_window
            weights["periods_per_year"] = family.periods_per_year
            weights["max_asset_weight"] = family.max_asset_weight
            weights["crypto_sleeve_cap"] = family.group_cap
            weight_outputs.append(weights)

            metrics = performance_metrics(
                result.daily["net_return"],
                periods_per_year=family.periods_per_year,
                risk_free_rate=0.0,
            )
            years = len(result.daily) / family.periods_per_year
            metric_records.append(
                {
                    "fund_id": fund_id,
                    "fund_name": fund_name,
                    "asset_family": family.family,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    **metrics,
                    "rebalance_count": int(result.daily["is_rebalance"].sum()),
                    "average_rebalance_turnover": float(
                        result.daily.loc[result.daily["is_rebalance"], "turnover"].mean()
                    ),
                    "annualized_turnover": float(result.daily["turnover"].sum() / years),
                    "latest_rebalance_date": result.rebalance_weights["rebalance_date"].max(),
                    "estimation_window": family.estimation_window,
                    "max_asset_weight": family.max_asset_weight,
                    "crypto_sleeve_cap": family.group_cap,
                    "risk_free_rate": 0.0,
                    "transaction_cost_bps": 0.0,
                }
            )

    fund_returns = pd.concat(return_outputs, ignore_index=True)
    fund_weights = pd.concat(weight_outputs, ignore_index=True)
    metrics_table = pd.DataFrame(metric_records)
    design_table = pd.DataFrame(design_records)
    diagnostics = _method_distance_table(fund_weights)

    _validate_outputs(fund_returns, fund_weights, diagnostics)

    fund_returns.to_csv(DATA_DIR / "fund_returns.csv", index=False)
    fund_weights.to_csv(DATA_DIR / "fund_weights.csv", index=False)
    metrics_table.to_csv(TABLE_DIR / "performance_metrics.csv", index=False)
    design_table.to_csv(TABLE_DIR / "backtest_design.csv", index=False)
    diagnostics.to_csv(TABLE_DIR / "portfolio_method_diagnostics.csv", index=False)

    print("\n=== STAGE 2: WALK-FORWARD FUNDS BUILT AND VALIDATED ===")
    print(f"Funds: {metrics_table['fund_id'].nunique()} (3 families x 4 methods)")
    for row in design_table.itertuples(index=False):
        print(
            f"{row.asset_family.title()}: {row.universe_size} assets, "
            f"{row.estimation_window}-observation window, "
            f"live {row.first_live_date.date()} to {row.final_live_date.date()}"
        )
    print(f"Fund return rows: {len(fund_returns):,}")
    print(f"Monthly target-weight rows: {len(fund_weights):,}")
    print("Timing check: every estimation_end is strictly before rebalance_date — PASS")
    print("Constraint check: long-only, fully invested, asset/group caps — PASS")
    print("Solver check: all optimisations converged and methods differ — PASS")
    print("Baseline cost assumption: 0 bps; turnover retained for later robustness")


if __name__ == "__main__":
    main()
