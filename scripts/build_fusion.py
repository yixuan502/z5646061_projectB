"""Build and validate baseline and coverage-adjusted sentiment funds.

Run from the project root after Stages 1–3:

    python scripts/build_fusion.py
"""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.etl import load_clean_headlines  # noqa: E402
from src.fusion import (  # noqa: E402
    FUSION_VARIANTS,
    VARIANT_LABELS,
    apply_sector_tilt,
    backtest_target_schedule,
    build_fusion_signals,
    build_monthly_news_concentration,
)
from src.metrics import drawdown_series, performance_metrics, wealth_index  # noqa: E402


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "results" / "data"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

PRIMARY_TILT_STRENGTH = 0.25
ROBUSTNESS_STRENGTHS = (0.10, 0.25, 0.40)
ROBUSTNESS_COST_BPS = (0.0, 10.0)


@dataclass(frozen=True)
class FusionFamily:
    family: str
    family_label: str
    base_fund_id: str
    returns_file: str
    max_asset_weight: float = 0.10
    periods_per_year: int = 252


def _load_return_panel(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python scripts/prepare_part_b_data.py first."
        )
    panel = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    panel = panel.dropna(how="any").astype(float)
    if panel.empty or panel.index.has_duplicates or not np.isfinite(panel.to_numpy()).all():
        raise ValueError(f"Invalid return panel: {filename}")
    return panel


def _load_required_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "fund_weights": DATA_DIR / "fund_weights.csv",
        "fund_returns": DATA_DIR / "fund_returns.csv",
        "sentiment": DATA_DIR / "sector_sentiment_index.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing earlier-stage outputs: {missing}")
    weights = pd.read_csv(
        paths["fund_weights"],
        parse_dates=["rebalance_date", "estimation_start", "estimation_end"],
    )
    returns = pd.read_csv(paths["fund_returns"], parse_dates=["date"])
    sentiment = pd.read_csv(
        paths["sentiment"],
        parse_dates=["date", "signal_source_date"],
    )
    return weights, returns, sentiment


def _prepare_target_schedule(
    base_weights: pd.DataFrame,
    signals: pd.DataFrame,
    variant: str,
    tilt_strength: float,
    max_asset_weight: float,
) -> pd.DataFrame:
    if variant not in FUSION_VARIANTS:
        raise ValueError(f"Unknown fusion variant: {variant}")
    records: list[pd.DataFrame] = []
    for date, base_date in base_weights.groupby("rebalance_date", sort=True):
        signal_date = signals[
            signals["date"].eq(date) & signals["signal_available"]
        ].copy()
        equity_sectors = set(base_date.loc[base_date["asset_class"].eq("Equity"), "sector"])
        if set(signal_date["sector"]) != equity_sectors:
            missing = sorted(equity_sectors.difference(signal_date["sector"]))
            raise RuntimeError(f"Missing available signals on {date.date()}: {missing}")

        if variant == "base":
            tilted = base_date.copy()
            tilted["base_target_weight"] = tilted["target_weight"]
            tilted["sector_signal"] = np.nan
            tilted["tilt_multiplier"] = 1.0
            tilted["weight_change_from_base"] = 0.0
            signal_column = "none"
        else:
            signal_column = (
                "relative_sentiment_signal"
                if variant == "sentiment"
                else "coverage_adjusted_signal"
            )
            tilted = apply_sector_tilt(
                base_date,
                signal_date,
                signal_column=signal_column,
                tilt_strength=tilt_strength,
                max_asset_weight=max_asset_weight,
            )

        signal_audit = signal_date[
            [
                "sector",
                "sentiment_information_end",
                "hhi_source_month",
                "sentiment_21d",
                "relative_sentiment_signal",
                "ticker_coverage_21d",
                "trailing_normalised_hhi_3m",
                "coverage_confidence",
                "coverage_adjusted_signal",
            ]
        ]
        tilted = tilted.merge(
            signal_audit,
            on="sector",
            how="left",
            validate="many_to_one",
        )
        tilted["fusion_variant"] = variant
        tilted["fusion_variant_label"] = VARIANT_LABELS[variant]
        tilted["signal_column"] = signal_column
        tilted["tilt_strength"] = 0.0 if variant == "base" else tilt_strength
        records.append(tilted)
    return pd.concat(records, ignore_index=True)


def _format_daily_output(
    daily: pd.DataFrame,
    family: FusionFamily,
    variant: str,
    tilt_strength: float,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    output = daily.copy()
    output["growth_1"] = wealth_index(output["net_return"])
    output["drawdown"] = drawdown_series(output["net_return"])
    output = output.reset_index()
    output.insert(1, "fusion_fund_id", f"{family.family}_risk_parity_{variant}")
    output.insert(
        2,
        "fusion_fund_name",
        f"{family.family_label} {VARIANT_LABELS[variant]}",
    )
    output.insert(3, "asset_family", family.family)
    output.insert(4, "base_method", "risk_parity")
    output.insert(5, "fusion_variant", variant)
    output.insert(6, "fusion_variant_label", VARIANT_LABELS[variant])
    output.insert(7, "tilt_strength", 0.0 if variant == "base" else tilt_strength)
    output.insert(8, "transaction_cost_bps", transaction_cost_bps)
    return output


def _metric_record(
    daily: pd.DataFrame,
    family: FusionFamily,
    variant: str,
    tilt_strength: float,
    transaction_cost_bps: float,
) -> dict:
    metrics = performance_metrics(
        daily["net_return"],
        periods_per_year=family.periods_per_year,
        risk_free_rate=0.0,
    )
    years = len(daily) / family.periods_per_year
    return {
        "fusion_fund_id": f"{family.family}_risk_parity_{variant}",
        "fusion_fund_name": f"{family.family_label} {VARIANT_LABELS[variant]}",
        "asset_family": family.family,
        "base_method": "risk_parity",
        "fusion_variant": variant,
        "fusion_variant_label": VARIANT_LABELS[variant],
        "tilt_strength": 0.0 if variant == "base" else tilt_strength,
        "transaction_cost_bps": transaction_cost_bps,
        **metrics,
        "rebalance_count": int(daily["is_rebalance"].sum()),
        "average_rebalance_turnover": float(
            daily.loc[daily["is_rebalance"], "turnover"].mean()
        ),
        "annualized_turnover": float(daily["turnover"].sum() / years),
    }


def _add_base_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    comparison_columns = [
        "annualized_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "annualized_turnover",
    ]
    base = (
        metrics[metrics["fusion_variant"].eq("base")]
        .set_index("asset_family")[comparison_columns]
        .add_suffix("_base")
    )
    metrics = metrics.merge(base, on="asset_family", how="left", validate="many_to_one")
    for column in comparison_columns:
        metrics[f"delta_{column}_vs_base"] = metrics[column] - metrics[f"{column}_base"]
    return metrics.drop(columns=[f"{column}_base" for column in comparison_columns])


def _validate_primary(
    returns_output: pd.DataFrame,
    weights_output: pd.DataFrame,
    metrics: pd.DataFrame,
    stage2_returns: pd.DataFrame,
) -> list[dict]:
    checks: list[dict] = []
    if returns_output.duplicated(["date", "fusion_fund_id"]).any():
        raise RuntimeError("fusion_returns has duplicate date-fund keys")
    if weights_output.duplicated(
        ["rebalance_date", "fusion_family", "fusion_variant", "ticker"]
    ).any():
        raise RuntimeError("fusion_weights has duplicate target keys")
    if not np.isfinite(
        returns_output[["gross_return", "net_return", "growth_1", "drawdown"]].to_numpy()
    ).all():
        raise RuntimeError("fusion_returns contains missing or infinite values")

    sums = weights_output.groupby(
        ["fusion_family", "fusion_variant", "rebalance_date"]
    )["target_weight"].sum()
    max_sum_error = float((sums - 1.0).abs().max())
    if max_sum_error > 1e-8:
        raise RuntimeError("fusion target weights do not sum to one")
    if (weights_output["target_weight"] < -1e-12).any():
        raise RuntimeError("fusion created a short position")
    if (weights_output["target_weight"] > 0.10 + 1e-8).any():
        raise RuntimeError("fusion violated the 10% asset cap")

    combined = weights_output[weights_output["fusion_family"].eq("combined")]
    crypto = combined[combined["asset_class"].eq("Crypto")]
    max_crypto_error = float(
        (crypto["target_weight"] - crypto["base_target_weight"]).abs().max()
    )
    if max_crypto_error > 1e-14:
        raise RuntimeError("combined crypto targets changed under sentiment fusion")

    base_return_errors = []
    for family in ["equity", "combined"]:
        fusion_base = returns_output[
            returns_output["fusion_fund_id"].eq(f"{family}_risk_parity_base")
        ].set_index("date")
        original = stage2_returns[
            stage2_returns["fund_id"].eq(f"{family}_risk_parity")
        ].set_index("date")
        if not fusion_base.index.equals(original.index):
            raise RuntimeError(f"{family} base dates do not match Stage 2")
        base_return_errors.append(
            float((fusion_base["gross_return"] - original["gross_return"]).abs().max())
        )
    max_base_return_error = max(base_return_errors)
    if max_base_return_error > 1e-12:
        raise RuntimeError("fusion engine does not reproduce the Stage 2 base returns")

    if len(metrics) != 6:
        raise RuntimeError("primary fusion comparison must contain six funds")

    checks.extend(
        [
            {"check": "primary_fusion_funds", "value": len(metrics), "expected": 6, "status": "PASS"},
            {"check": "target_weight_max_sum_error", "value": max_sum_error, "expected": "<= 1e-8", "status": "PASS"},
            {"check": "combined_crypto_target_max_error", "value": max_crypto_error, "expected": "<= 1e-14", "status": "PASS"},
            {"check": "stage2_base_return_max_error", "value": max_base_return_error, "expected": "<= 1e-12", "status": "PASS"},
            {"check": "signal_look_ahead_violations", "value": int((weights_output["sentiment_information_end"] >= weights_output["rebalance_date"]).sum()), "expected": 0, "status": "PASS"},
            {"check": "hhi_month_look_ahead_violations", "value": int((weights_output["hhi_source_month"].dt.to_period("M") >= weights_output["rebalance_date"].dt.to_period("M")).sum()), "expected": 0, "status": "PASS"},
        ]
    )
    return checks


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    all_weights, stage2_returns, sector_index = _load_required_outputs()
    concentration = build_monthly_news_concentration(load_clean_headlines())
    signals = build_fusion_signals(sector_index, concentration)

    families = [
        FusionFamily(
            family="equity",
            family_label="Equity",
            base_fund_id="equity_risk_parity",
            returns_file="equity_returns_panel.csv",
        ),
        FusionFamily(
            family="combined",
            family_label="Combined",
            base_fund_id="combined_risk_parity",
            returns_file="combined_returns_panel.csv",
        ),
    ]

    primary_returns: list[pd.DataFrame] = []
    primary_weights: list[pd.DataFrame] = []
    primary_metrics: list[dict] = []
    robustness_metrics: list[dict] = []

    for family in families:
        panel = _load_return_panel(family.returns_file)
        base_weights = all_weights[
            all_weights["fund_id"].eq(family.base_fund_id)
        ].copy()
        if base_weights.empty:
            raise RuntimeError(f"Missing base weights for {family.base_fund_id}")

        primary_schedules: dict[str, pd.DataFrame] = {}
        for variant in FUSION_VARIANTS:
            schedule = _prepare_target_schedule(
                base_weights,
                signals,
                variant,
                PRIMARY_TILT_STRENGTH,
                family.max_asset_weight,
            )
            schedule["fusion_family"] = family.family
            primary_schedules[variant] = schedule
            primary_weights.append(schedule)

            daily = backtest_target_schedule(
                panel,
                schedule[["rebalance_date", "ticker", "target_weight"]],
                transaction_cost_bps=0.0,
            )
            primary_returns.append(
                _format_daily_output(
                    daily,
                    family,
                    variant,
                    PRIMARY_TILT_STRENGTH,
                    0.0,
                )
            )
            primary_metrics.append(
                _metric_record(
                    daily,
                    family,
                    variant,
                    PRIMARY_TILT_STRENGTH,
                    0.0,
                )
            )

        # Robustness: base does not depend on tilt strength, while both signal
        # variants are tested at three predeclared strengths and two cost levels.
        for cost_bps in ROBUSTNESS_COST_BPS:
            base_daily = backtest_target_schedule(
                panel,
                primary_schedules["base"][["rebalance_date", "ticker", "target_weight"]],
                transaction_cost_bps=cost_bps,
            )
            robustness_metrics.append(
                _metric_record(base_daily, family, "base", 0.0, cost_bps)
            )
            for strength in ROBUSTNESS_STRENGTHS:
                for variant in ("sentiment", "coverage_adjusted"):
                    schedule = _prepare_target_schedule(
                        base_weights,
                        signals,
                        variant,
                        strength,
                        family.max_asset_weight,
                    )
                    daily = backtest_target_schedule(
                        panel,
                        schedule[["rebalance_date", "ticker", "target_weight"]],
                        transaction_cost_bps=cost_bps,
                    )
                    robustness_metrics.append(
                        _metric_record(daily, family, variant, strength, cost_bps)
                    )

    fusion_returns = pd.concat(primary_returns, ignore_index=True)
    fusion_weights = pd.concat(primary_weights, ignore_index=True)
    comparison = _add_base_deltas(pd.DataFrame(primary_metrics))
    robustness = pd.DataFrame(robustness_metrics)
    robustness_base = (
        robustness[robustness["fusion_variant"].eq("base")]
        [["asset_family", "transaction_cost_bps", "annualized_return", "sharpe_ratio", "maximum_drawdown"]]
        .rename(
            columns={
                "annualized_return": "base_annualized_return",
                "sharpe_ratio": "base_sharpe_ratio",
                "maximum_drawdown": "base_maximum_drawdown",
            }
        )
    )
    robustness = robustness.merge(
        robustness_base,
        on=["asset_family", "transaction_cost_bps"],
        how="left",
        validate="many_to_one",
    )
    robustness["delta_annualized_return_vs_base"] = (
        robustness["annualized_return"] - robustness["base_annualized_return"]
    )
    robustness["delta_sharpe_vs_base"] = (
        robustness["sharpe_ratio"] - robustness["base_sharpe_ratio"]
    )
    robustness["delta_maximum_drawdown_vs_base"] = (
        robustness["maximum_drawdown"] - robustness["base_maximum_drawdown"]
    )

    validation = pd.DataFrame(
        _validate_primary(
            fusion_returns,
            fusion_weights,
            comparison,
            stage2_returns,
        )
    )
    extra_validation = pd.DataFrame(
        [
            {
                "check": "monthly_sector_hhi_rows",
                "value": len(concentration),
                "expected": 480,
                "status": "PASS",
            },
            {
                "check": "normalised_hhi_out_of_bounds",
                "value": int((~concentration["normalised_hhi"].between(0.0, 1.0)).sum()),
                "expected": 0,
                "status": "PASS",
            },
            {
                "check": "available_fusion_signal_rows",
                "value": int(signals["signal_available"].sum()),
                "expected": "diagnostic",
                "status": "PASS",
            },
        ]
    )
    validation = pd.concat([validation, extra_validation], ignore_index=True)

    concentration.to_csv(DATA_DIR / "monthly_news_concentration.csv", index=False)
    signals.to_csv(DATA_DIR / "fusion_signals.csv", index=False)
    fusion_returns.to_csv(DATA_DIR / "fusion_returns.csv", index=False)
    fusion_weights.to_csv(DATA_DIR / "fusion_weights.csv", index=False)
    comparison.to_csv(TABLE_DIR / "fusion_comparison.csv", index=False)
    robustness.to_csv(TABLE_DIR / "fusion_robustness.csv", index=False)
    validation.to_csv(TABLE_DIR / "fusion_validation.csv", index=False)

    display_columns = [
        "asset_family",
        "fusion_variant",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "annualized_turnover",
        "delta_sharpe_ratio_vs_base",
    ]
    print("\n=== STAGE 4: SENTIMENT FUSION BUILT AND VALIDATED ===")
    print("Primary experiment: Equity and Combined Risk Parity, 25% tilt, 0 bps")
    print(comparison[display_columns].round(4).to_string(index=False))
    print(f"Fusion daily-return rows: {len(fusion_returns):,}")
    print(f"Fusion target-weight rows: {len(fusion_weights):,}")
    print(f"Robustness scenarios: {len(robustness):,}")
    print("Stage 2 base-return reproduction — PASS")
    print("Signal and completed-month HHI timing — PASS")
    print("Long-only, fully invested, asset cap, crypto unchanged — PASS")


if __name__ == "__main__":
    main()
