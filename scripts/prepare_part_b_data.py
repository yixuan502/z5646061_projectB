"""Build and validate the reusable Part B data foundation.

Run from the project root:

    python scripts/prepare_part_b_data.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.etl import (  # noqa: E402
    load_clean_crypto,
    load_clean_equities,
    load_clean_headlines,
)
from src.features import (  # noqa: E402
    align_headlines_to_trading_days,
    assemble_headline_panel,
    build_combined_returns_panel,
    daily_returns,
    wide_return_panel,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "results" / "data"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"


def _output_panel(panel: pd.DataFrame, path: pathlib.Path) -> None:
    """Save a date-indexed panel with an explicit date column."""

    output = panel.reset_index()
    output.to_csv(path, index=False)


def _validation_row(
    dataset: str,
    rows: int,
    columns: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    duplicate_keys: int,
    missing_values: int,
    status: str,
) -> dict:
    return {
        "dataset": dataset,
        "rows": rows,
        "columns": columns,
        "start_date": start_date.date(),
        "end_date": end_date.date(),
        "duplicate_keys": duplicate_keys,
        "missing_values": missing_values,
        "status": status,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    equities = load_clean_equities()
    crypto = load_clean_crypto()
    headlines = load_clean_headlines()

    equity_returns = daily_returns(equities)
    crypto_returns = daily_returns(crypto)

    equity_panel = wide_return_panel(equity_returns)
    crypto_panel = wide_return_panel(crypto_returns)
    combined_panel = build_combined_returns_panel(
        equity_returns,
        crypto_returns,
    )

    equity_dates = equities["date"].drop_duplicates()
    aligned_headlines = align_headlines_to_trading_days(
        headlines,
        equity_dates,
    )
    headline_panel = assemble_headline_panel(headlines, equity_dates)

    _output_panel(equity_panel, DATA_DIR / "equity_returns_panel.csv")
    _output_panel(crypto_panel, DATA_DIR / "crypto_returns_panel.csv")
    _output_panel(combined_panel, DATA_DIR / "combined_returns_panel.csv")
    headline_panel.to_csv(DATA_DIR / "daily_headline_panel.csv", index=False)

    equity_missing_expected = equities["ticker"].nunique()
    crypto_missing_expected = crypto["ticker"].nunique()
    if int(equity_panel.isna().sum().sum()) != equity_missing_expected:
        raise ValueError("Unexpected missing-value pattern in equity returns")
    if int(crypto_panel.isna().sum().sum()) != crypto_missing_expected:
        raise ValueError("Unexpected missing-value pattern in crypto returns")
    if not combined_panel.index.equals(equity_panel.index):
        raise ValueError("Combined panel is not aligned to the equity calendar")
    if np.isinf(combined_panel.to_numpy(dtype=float)).any():
        raise ValueError("Combined return panel contains infinite values")

    unmapped_headlines = int(aligned_headlines["trading_date"].isna().sum())
    moved_headlines = int(
        aligned_headlines["moved_to_next_trading_day"].sum()
    )
    mapped_headlines = int(headline_panel["headline_count"].sum())

    validation = pd.DataFrame(
        [
            _validation_row(
                "clean_equities",
                len(equities),
                equities.shape[1],
                equities["date"].min(),
                equities["date"].max(),
                int(equities.duplicated(["ticker", "date"]).sum()),
                int(equities.isna().sum().sum()),
                "PASS",
            ),
            _validation_row(
                "clean_crypto",
                len(crypto),
                crypto.shape[1],
                crypto["date"].min(),
                crypto["date"].max(),
                int(crypto.duplicated(["ticker", "date"]).sum()),
                int(crypto.isna().sum().sum()),
                "PASS",
            ),
            _validation_row(
                "clean_headlines",
                len(headlines),
                headlines.shape[1],
                headlines["date"].min(),
                headlines["date"].max(),
                int(headlines.duplicated(["ticker", "date", "title"]).sum()),
                int(headlines[["date", "ticker", "sector", "title"]].isna().sum().sum()),
                "PASS",
            ),
            _validation_row(
                "equity_returns_panel",
                len(equity_panel),
                equity_panel.shape[1],
                equity_panel.index.min(),
                equity_panel.index.max(),
                int(equity_panel.index.duplicated().sum()),
                int(equity_panel.isna().sum().sum()),
                "PASS",
            ),
            _validation_row(
                "crypto_returns_panel",
                len(crypto_panel),
                crypto_panel.shape[1],
                crypto_panel.index.min(),
                crypto_panel.index.max(),
                int(crypto_panel.index.duplicated().sum()),
                int(crypto_panel.isna().sum().sum()),
                "PASS",
            ),
            _validation_row(
                "combined_returns_panel",
                len(combined_panel),
                combined_panel.shape[1],
                combined_panel.index.min(),
                combined_panel.index.max(),
                int(combined_panel.index.duplicated().sum()),
                int(combined_panel.isna().sum().sum()),
                "PASS",
            ),
            _validation_row(
                "daily_headline_panel",
                len(headline_panel),
                headline_panel.shape[1],
                headline_panel["trading_date"].min(),
                headline_panel["trading_date"].max(),
                int(
                    headline_panel.duplicated(
                        ["trading_date", "ticker", "sector"]
                    ).sum()
                ),
                int(headline_panel["combined_headlines"].isna().sum()),
                "PASS",
            ),
        ]
    )
    validation.to_csv(
        TABLE_DIR / "part_b_data_validation.csv",
        index=False,
    )

    print("\n=== STAGE 1: PART A FOUNDATION REUSED AND VALIDATED ===")
    print(f"Clean equities: {equities.shape}")
    print(f"Clean crypto: {crypto.shape}")
    print(f"Clean headlines: {headlines.shape}")
    print(f"Equity return panel: {equity_panel.shape}")
    print(f"Crypto return panel: {crypto_panel.shape}")
    print(f"Combined return panel: {combined_panel.shape}")
    print(f"Daily headline panel: {headline_panel.shape}")
    print(f"Mapped headline observations: {mapped_headlines:,}")
    print(f"Moved to a later trading day: {moved_headlines:,}")
    print(f"Unmapped after final trading date: {unmapped_headlines}")
    print("Validation status: all seven datasets PASS")


if __name__ == "__main__":
    main()
