"""Look-ahead-safe fusion of sector news sentiment with fund target weights.

The AtlasSignal extension carries Part A's monthly news-concentration HHI into
Part B as a confidence filter.  It never creates crypto sentiment: combined-fund
crypto target weights are preserved exactly while only the equity sleeve tilts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FUSION_VARIANTS = ("base", "sentiment", "coverage_adjusted")
VARIANT_LABELS = {
    "base": "Base Risk Parity",
    "sentiment": "Baseline Sentiment",
    "coverage_adjusted": "AtlasSignal Coverage-Adjusted",
}


def build_monthly_news_concentration(headlines: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Part A's complete monthly sector HHI from original news dates."""

    required = {"date", "sector", "ticker", "title"}
    missing = required.difference(headlines.columns)
    if missing:
        raise ValueError(f"headlines missing required columns: {sorted(missing)}")
    if headlines.empty:
        raise ValueError("headlines cannot be empty")
    if headlines.duplicated(["ticker", "date", "title"]).any():
        raise ValueError("headlines must be deduplicated before HHI calculation")

    data = headlines.copy()
    data["month"] = (
        pd.to_datetime(data["date"], utc=True)
        .dt.tz_convert(None)
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    universe = data[["sector", "ticker"]].drop_duplicates()
    if universe["ticker"].duplicated().any():
        raise ValueError("a ticker maps to more than one sector")
    months = pd.DataFrame({"month": sorted(data["month"].unique())})
    complete = universe.merge(months, how="cross")
    observed = (
        data.groupby(["month", "sector", "ticker"], as_index=False, observed=True)
        .agg(article_count=("title", "size"))
    )
    panel = complete.merge(
        observed,
        on=["month", "sector", "ticker"],
        how="left",
        validate="one_to_one",
    )
    panel["article_count"] = panel["article_count"].fillna(0).astype(int)
    panel["sector_month_total"] = panel.groupby(
        ["month", "sector"], observed=True
    )["article_count"].transform("sum")
    panel["article_share"] = np.where(
        panel["sector_month_total"] > 0,
        panel["article_count"] / panel["sector_month_total"],
        0.0,
    )
    panel["squared_article_share"] = panel["article_share"] ** 2

    concentration = (
        panel.groupby(["month", "sector"], as_index=False, observed=True)
        .agg(
            total_articles=("article_count", "sum"),
            active_tickers=("article_count", lambda values: int((values > 0).sum())),
            ticker_universe=("ticker", "nunique"),
            hhi=("squared_article_share", "sum"),
        )
    )
    minimum_hhi = 1.0 / concentration["ticker_universe"]
    concentration["normalised_hhi"] = (
        (concentration["hhi"] - minimum_hhi) / (1.0 - minimum_hhi)
    ).where(concentration["total_articles"] > 0).clip(0.0, 1.0)
    concentration["effective_number_of_tickers"] = np.where(
        concentration["hhi"] > 0,
        1.0 / concentration["hhi"],
        np.nan,
    )

    top = (
        panel.sort_values(
            ["month", "sector", "article_count", "ticker"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["month", "sector"])
        [["month", "sector", "ticker", "article_count", "article_share"]]
        .rename(
            columns={
                "ticker": "top_ticker",
                "article_count": "top_ticker_article_count",
                "article_share": "top_ticker_share",
            }
        )
    )
    top["top_ticker_share_pct"] = top["top_ticker_share"] * 100.0
    concentration = concentration.merge(
        top,
        on=["month", "sector"],
        how="left",
        validate="one_to_one",
    )
    return concentration.sort_values(["month", "sector"]).reset_index(drop=True)


def _cross_sectional_score(values: pd.Series) -> pd.Series:
    """Demean and scale one date's sectors, then bound the signal to [-1, 1]."""

    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    if valid.empty:
        return output
    dispersion = float(valid.std(ddof=0))
    if dispersion <= 1e-12:
        output.loc[valid.index] = 0.0
    else:
        z_score = (valid - valid.mean()) / dispersion
        output.loc[valid.index] = z_score.clip(-2.0, 2.0) / 2.0
    return output


def build_fusion_signals(
    sector_index: pd.DataFrame,
    monthly_concentration: pd.DataFrame,
    sentiment_window: int = 21,
    minimum_sentiment_observations: int = 10,
    hhi_window_months: int = 3,
) -> pd.DataFrame:
    """Build relative sentiment and AtlasSignal confidence using only past data.

    On decision date t, sentiment and ticker coverage are shifted one equity
    trading day before the 21-observation rolling calculation.  HHI uses the last
    three fully completed calendar months; the current partial month is excluded.
    """

    required_index = {
        "date",
        "sector",
        "raw_sector_sentiment",
        "ticker_coverage_ratio",
    }
    missing_index = required_index.difference(sector_index.columns)
    if missing_index:
        raise ValueError(f"sector_index missing columns: {sorted(missing_index)}")
    required_hhi = {"month", "sector", "normalised_hhi"}
    missing_hhi = required_hhi.difference(monthly_concentration.columns)
    if missing_hhi:
        raise ValueError(f"monthly_concentration missing columns: {sorted(missing_hhi)}")
    if sentiment_window < 2 or minimum_sentiment_observations < 1:
        raise ValueError("sentiment window settings are invalid")
    if minimum_sentiment_observations > sentiment_window:
        raise ValueError("minimum observations cannot exceed sentiment_window")
    if hhi_window_months < 1:
        raise ValueError("hhi_window_months must be positive")

    signal = sector_index.copy()
    signal["date"] = pd.to_datetime(signal["date"])
    signal = signal.sort_values(["sector", "date"]).reset_index(drop=True)
    if signal.duplicated(["date", "sector"]).any():
        raise ValueError("sector_index has duplicate date-sector keys")

    by_sector = signal.groupby("sector", observed=True)
    signal["sentiment_information_end"] = by_sector["date"].shift(1)
    signal["past_sector_sentiment"] = by_sector["raw_sector_sentiment"].shift(1)
    signal["past_ticker_coverage"] = by_sector["ticker_coverage_ratio"].shift(1)
    signal["sentiment_21d"] = signal.groupby("sector", observed=True)[
        "past_sector_sentiment"
    ].transform(
        lambda values: values.rolling(
            sentiment_window,
            min_periods=minimum_sentiment_observations,
        ).mean()
    )
    signal["ticker_coverage_21d"] = signal.groupby("sector", observed=True)[
        "past_ticker_coverage"
    ].transform(
        lambda values: values.rolling(
            sentiment_window,
            min_periods=minimum_sentiment_observations,
        ).mean()
    )
    signal["relative_sentiment_signal"] = signal.groupby(
        "date", observed=True
    )["sentiment_21d"].transform(_cross_sectional_score)

    hhi = monthly_concentration[["month", "sector", "normalised_hhi"]].copy()
    hhi["month"] = pd.to_datetime(hhi["month"])
    hhi = hhi.sort_values(["sector", "month"])
    if hhi.duplicated(["month", "sector"]).any():
        raise ValueError("monthly_concentration has duplicate month-sector keys")
    hhi["trailing_normalised_hhi_3m"] = hhi.groupby(
        "sector", observed=True
    )["normalised_hhi"].transform(
        lambda values: values.rolling(
            hhi_window_months,
            min_periods=hhi_window_months,
        ).mean()
    )
    hhi = hhi.rename(columns={"month": "hhi_source_month"})

    # Exact key for the most recently completed month. Example: every January
    # decision maps to December, so no partial January news can enter January.
    signal["hhi_source_month"] = (
        signal["date"].dt.to_period("M") - 1
    ).dt.to_timestamp()
    signal = signal.merge(
        hhi[
            [
                "hhi_source_month",
                "sector",
                "normalised_hhi",
                "trailing_normalised_hhi_3m",
            ]
        ],
        on=["hhi_source_month", "sector"],
        how="left",
        validate="many_to_one",
    )
    signal["hhi_confidence"] = 1.0 - signal["trailing_normalised_hhi_3m"]
    signal["ticker_coverage_confidence"] = np.sqrt(
        signal["ticker_coverage_21d"].clip(0.0, 1.0)
    )
    signal["coverage_confidence"] = (
        signal["hhi_confidence"] * signal["ticker_coverage_confidence"]
    )
    signal["coverage_adjusted_signal"] = (
        signal["relative_sentiment_signal"] * signal["coverage_confidence"]
    )
    signal["signal_available"] = signal[
        [
            "relative_sentiment_signal",
            "coverage_adjusted_signal",
            "sentiment_information_end",
            "hhi_source_month",
        ]
    ].notna().all(axis=1)

    available = signal["signal_available"]
    if not (
        signal.loc[available, "sentiment_information_end"]
        < signal.loc[available, "date"]
    ).all():
        raise RuntimeError("sentiment look-ahead detected")
    decision_month = signal.loc[available, "date"].dt.to_period("M")
    hhi_month = signal.loc[available, "hhi_source_month"].dt.to_period("M")
    if not (hhi_month < decision_month).all():
        raise RuntimeError("current or future month HHI entered the signal")
    if not signal.loc[available, "coverage_confidence"].between(0.0, 1.0).all():
        raise RuntimeError("coverage confidence falls outside [0, 1]")
    if not signal.loc[available, "relative_sentiment_signal"].between(-1.0, 1.0).all():
        raise RuntimeError("relative sentiment signal falls outside [-1, 1]")

    return signal.sort_values(["date", "sector"]).reset_index(drop=True)


def _capped_proportional_weights(
    raw_weights: pd.Series,
    target_total: float,
    max_weight: float,
) -> pd.Series:
    """Scale positive relative weights to a target sum under an upper bound."""

    if target_total < -1e-12 or max_weight <= 0:
        raise ValueError("invalid target_total or max_weight")
    if len(raw_weights) * max_weight < target_total - 1e-10:
        raise ValueError("asset cap makes the requested sleeve infeasible")
    raw = raw_weights.astype(float).clip(lower=0.0)
    if raw.sum() <= 0:
        raw[:] = 1.0

    result = pd.Series(0.0, index=raw.index)
    remaining = pd.Series(True, index=raw.index)
    remaining_total = float(target_total)
    for _ in range(len(raw) + 1):
        if not remaining.any():
            break
        candidates = raw[remaining]
        if candidates.sum() <= 0:
            proposed = pd.Series(
                remaining_total / int(remaining.sum()), index=candidates.index
            )
        else:
            proposed = candidates / candidates.sum() * remaining_total
        capped = proposed > max_weight + 1e-12
        if not capped.any():
            result.loc[proposed.index] = proposed
            remaining_total = 0.0
            break
        capped_names = proposed.index[capped]
        result.loc[capped_names] = max_weight
        remaining.loc[capped_names] = False
        remaining_total = target_total - float(result.sum())

    if abs(result.sum() - target_total) > 1e-8:
        raise RuntimeError("capped weight projection did not reach target sleeve")
    if (result > max_weight + 1e-8).any() or (result < -1e-12).any():
        raise RuntimeError("capped weight projection violated bounds")
    return result


def apply_sector_tilt(
    base_weights: pd.DataFrame,
    sector_signals: pd.DataFrame,
    signal_column: str,
    tilt_strength: float = 0.25,
    max_asset_weight: float = 0.10,
) -> pd.DataFrame:
    """Tilt only equity weights and preserve any crypto sleeve exactly."""

    required_weights = {"ticker", "asset_class", "sector", "target_weight"}
    missing_weights = required_weights.difference(base_weights.columns)
    if missing_weights:
        raise ValueError(f"base_weights missing columns: {sorted(missing_weights)}")
    if signal_column not in sector_signals.columns or "sector" not in sector_signals.columns:
        raise ValueError("sector_signals does not contain the requested signal")
    if not 0 <= tilt_strength < 1:
        raise ValueError("tilt_strength must be in [0, 1)")
    if base_weights["ticker"].duplicated().any():
        raise ValueError("base_weights must contain one row per ticker")

    output = base_weights.copy()
    output["base_target_weight"] = output["target_weight"].astype(float)
    signal_map = sector_signals.set_index("sector")[signal_column]
    equity = output["asset_class"].eq("Equity")
    output["sector_signal"] = np.nan
    output.loc[equity, "sector_signal"] = output.loc[equity, "sector"].map(signal_map)
    if output.loc[equity, "sector_signal"].isna().any():
        missing_sectors = sorted(
            output.loc[equity & output["sector_signal"].isna(), "sector"].unique()
        )
        raise ValueError(f"missing fusion signals for equity sectors: {missing_sectors}")
    output["tilt_multiplier"] = 1.0
    output.loc[equity, "tilt_multiplier"] = (
        1.0 + tilt_strength * output.loc[equity, "sector_signal"]
    )
    if (output.loc[equity, "tilt_multiplier"] <= 0).any():
        raise RuntimeError("sentiment tilt created a non-positive multiplier")

    equity_total = float(output.loc[equity, "base_target_weight"].sum())
    raw_equity = (
        output.loc[equity, "base_target_weight"]
        * output.loc[equity, "tilt_multiplier"]
    )
    tilted_equity = _capped_proportional_weights(
        raw_equity,
        target_total=equity_total,
        max_weight=max_asset_weight,
    )
    output.loc[equity, "target_weight"] = tilted_equity
    output["weight_change_from_base"] = (
        output["target_weight"] - output["base_target_weight"]
    )

    crypto = ~equity
    if not np.allclose(
        output.loc[crypto, "target_weight"],
        output.loc[crypto, "base_target_weight"],
        atol=1e-14,
    ):
        raise RuntimeError("crypto target weights changed during equity sentiment fusion")
    if abs(output["target_weight"].sum() - 1.0) > 1e-8:
        raise RuntimeError("tilted portfolio is not fully invested")
    if (output["target_weight"] < -1e-12).any():
        raise RuntimeError("tilted portfolio created a short position")
    if (output["target_weight"] > max_asset_weight + 1e-8).any():
        raise RuntimeError("tilted portfolio violates the asset cap")
    return output


def backtest_target_schedule(
    returns: pd.DataFrame,
    target_schedule: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Backtest externally supplied monthly targets with drifting daily weights."""

    required = {"rebalance_date", "ticker", "target_weight"}
    missing = required.difference(target_schedule.columns)
    if missing:
        raise ValueError(f"target_schedule missing columns: {sorted(missing)}")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative")
    panel = returns.astype(float).sort_index()
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TypeError("returns must use a DatetimeIndex")
    if panel.isna().any().any() or not np.isfinite(panel.to_numpy()).all():
        raise ValueError("returns must be complete and finite")

    schedule = target_schedule.copy()
    schedule["rebalance_date"] = pd.to_datetime(schedule["rebalance_date"])
    if schedule.duplicated(["rebalance_date", "ticker"]).any():
        raise ValueError("target_schedule has duplicate date-ticker keys")
    rebalance_dates = pd.DatetimeIndex(schedule["rebalance_date"].unique()).sort_values()
    if rebalance_dates.empty or not rebalance_dates.isin(panel.index).all():
        raise ValueError("rebalance dates must be present in the return panel")

    targets: dict[pd.Timestamp, pd.Series] = {}
    for date, group in schedule.groupby("rebalance_date", observed=True):
        target = group.set_index("ticker")["target_weight"].reindex(panel.columns)
        if target.isna().any():
            raise ValueError(f"target schedule is incomplete on {date}")
        if abs(target.sum() - 1.0) > 1e-8:
            raise ValueError(f"target weights do not sum to one on {date}")
        targets[pd.Timestamp(date)] = target.astype(float)

    live_index = panel.index[panel.index >= rebalance_dates.min()]
    current: pd.Series | None = None
    records: list[dict] = []
    for date in live_index:
        turnover = 0.0
        is_rebalance = date in targets
        if is_rebalance:
            target = targets[date]
            if current is not None:
                turnover = float(0.5 * np.abs(target - current).sum())
            current = target.copy()
        if current is None:
            raise RuntimeError("backtest began without target weights")

        asset_returns = panel.loc[date]
        gross_return = float(current @ asset_returns)
        transaction_cost = turnover * transaction_cost_bps / 10_000.0
        net_return = (1.0 - transaction_cost) * (1.0 + gross_return) - 1.0
        records.append(
            {
                "date": date,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "is_rebalance": is_rebalance,
            }
        )
        denominator = 1.0 + gross_return
        if denominator <= 0:
            raise RuntimeError("portfolio lost 100% or more in one observation")
        current = current * (1.0 + asset_returns) / denominator

    return pd.DataFrame(records).set_index("date")
