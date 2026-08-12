"""Tests for HHI timing, sector tilts, crypto preservation, and schedule returns."""

import numpy as np
import pandas as pd
import pytest

from src.fusion import (
    apply_sector_tilt,
    backtest_target_schedule,
    build_fusion_signals,
    build_monthly_news_concentration,
)


def test_monthly_hhi_keeps_zero_news_ticker_in_the_universe():
    headlines = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-03"]),
            "sector": ["Tech", "Tech", "Tech"],
            "ticker": ["A", "B", "A"],
            "title": ["a", "b", "c"],
        }
    )
    concentration = build_monthly_news_concentration(headlines)
    february = concentration.loc[concentration["month"].eq(pd.Timestamp("2020-02-01"))].iloc[0]

    assert february["ticker_universe"] == 2
    assert february["active_tickers"] == 1
    assert february["hhi"] == pytest.approx(1.0)
    assert february["normalised_hhi"] == pytest.approx(1.0)


def test_fusion_signal_excludes_current_month_hhi_and_current_day_sentiment():
    dates = pd.bdate_range("2020-01-01", "2020-04-03")
    rows = []
    for sector, offset in [("A", 0.0), ("B", 0.2)]:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "sector": sector,
                    "raw_sector_sentiment": index / 100.0 + offset,
                    "ticker_coverage_ratio": 1.0,
                }
            )
    sector_index = pd.DataFrame(rows)
    months = pd.date_range("2020-01-01", "2020-04-01", freq="MS")
    hhi = pd.DataFrame(
        [
            {"month": month, "sector": sector, "normalised_hhi": value}
            for sector in ["A", "B"]
            for month, value in zip(months, [0.1, 0.2, 0.3, 0.99])
        ]
    )

    first = build_fusion_signals(sector_index, hhi, 5, 2, 3)
    changed = sector_index.copy()
    changed.loc[changed["date"].eq(pd.Timestamp("2020-04-01")), "raw_sector_sentiment"] = 99.0
    second = build_fusion_signals(changed, hhi, 5, 2, 3)
    decision = pd.Timestamp("2020-04-01")
    first_day = first[first["date"].eq(decision)].sort_values("sector")
    second_day = second[second["date"].eq(decision)].sort_values("sector")

    np.testing.assert_allclose(
        first_day["relative_sentiment_signal"],
        second_day["relative_sentiment_signal"],
    )
    assert first_day["hhi_source_month"].eq(pd.Timestamp("2020-03-01")).all()
    np.testing.assert_allclose(
        first_day["trailing_normalised_hhi_3m"].to_numpy(),
        0.2,
        atol=1e-12,
    )
    assert (first_day["sentiment_information_end"] < first_day["date"]).all()


def test_combined_tilt_preserves_crypto_and_constraints():
    base = pd.DataFrame(
        {
            "ticker": ["A", "B", "BTC-USD"],
            "asset_class": ["Equity", "Equity", "Crypto"],
            "sector": ["Tech", "Energy", "Crypto"],
            "target_weight": [0.45, 0.35, 0.20],
        }
    )
    signals = pd.DataFrame(
        {"sector": ["Tech", "Energy"], "signal": [1.0, -1.0]}
    )
    tilted = apply_sector_tilt(
        base,
        signals,
        signal_column="signal",
        tilt_strength=0.25,
        max_asset_weight=0.60,
    )

    crypto = tilted.loc[tilted["ticker"].eq("BTC-USD")].iloc[0]
    assert crypto["target_weight"] == pytest.approx(0.20)
    assert crypto["base_target_weight"] == pytest.approx(0.20)
    assert tilted["target_weight"].sum() == pytest.approx(1.0)
    assert tilted.loc[tilted["ticker"].eq("A"), "target_weight"].iloc[0] > 0.45


def test_target_schedule_drifts_and_cost_reduces_net_return():
    dates = pd.bdate_range("2022-01-03", periods=25)
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    returns.loc[dates[0], "A"] = 0.10
    schedule = pd.DataFrame(
        {
            "rebalance_date": [dates[0], dates[0], dates[21], dates[21]],
            "ticker": ["A", "B", "A", "B"],
            "target_weight": [0.5, 0.5, 0.5, 0.5],
        }
    )
    zero_cost = backtest_target_schedule(returns, schedule, 0.0)
    with_cost = backtest_target_schedule(returns, schedule, 10.0)

    assert zero_cost.iloc[0]["gross_return"] == pytest.approx(0.05)
    assert zero_cost.iloc[0]["turnover"] == pytest.approx(0.0)
    assert zero_cost.loc[dates[21], "turnover"] > 0.0
    assert with_cost.loc[dates[21], "net_return"] < zero_cost.loc[dates[21], "net_return"]
