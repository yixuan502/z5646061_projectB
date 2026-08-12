"""Create the required self-contained sector-sentiment time-series exhibit."""

from __future__ import annotations

import pathlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "results" / "data" / "sector_sentiment_index.csv"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_PATH = FIGURE_DIR / "sector_sentiment_timeseries.png"

INK = "#1F2933"
BLUE = "#1F5A85"
GRID = "#D9E0E6"
ZERO = "#687783"
BACKGROUND = "#FAFBFC"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run python scripts/build_sentiment.py first."
        )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    sentiment = pd.read_csv(DATA_PATH, parse_dates=["date"])
    required = {
        "date",
        "sector",
        "raw_sector_sentiment",
        "raw_sentiment_available",
    }
    missing = required.difference(sentiment.columns)
    if missing:
        raise ValueError(f"sentiment index missing columns: {sorted(missing)}")

    sentiment = sentiment.sort_values(["sector", "date"])
    sentiment["sentiment_21d"] = sentiment.groupby("sector", observed=True)[
        "raw_sector_sentiment"
    ].transform(lambda values: values.rolling(21, min_periods=10).mean())
    sectors = sorted(sentiment["sector"].unique())
    if len(sectors) != 10:
        raise ValueError(f"expected 10 sectors, found {len(sectors)}")

    plotted = sentiment["sentiment_21d"].dropna()
    lower = min(float(plotted.min()), 0.0)
    upper = max(float(plotted.max()), 0.0)
    padding = max((upper - lower) * 0.06, 0.02)
    limits = (lower - padding, upper + padding)

    fig, axes = plt.subplots(
        5,
        2,
        figsize=(13.5, 15.2),
        sharex=True,
        sharey=True,
        facecolor=BACKGROUND,
    )
    axes = axes.ravel()

    for axis, sector in zip(axes, sectors):
        subset = sentiment[sentiment["sector"].eq(sector)]
        coverage = float(subset["raw_sentiment_available"].mean())
        axis.set_facecolor(BACKGROUND)
        axis.plot(
            subset["date"],
            subset["sentiment_21d"],
            color=BLUE,
            linewidth=1.7,
        )
        axis.axhline(0.0, color=ZERO, linewidth=0.9, linestyle=(0, (4, 3)))
        axis.set_title(
            f"{sector}  |  news-day coverage {coverage:.1%}",
            loc="left",
            color=INK,
            fontsize=10.5,
            fontweight="bold",
            pad=7,
        )
        axis.set_ylim(*limits)
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(GRID)
        axis.tick_params(colors=INK, labelsize=8.5)
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(
        "Equity Sector News Sentiment (21-Trading-Day Mean)",
        x=0.075,
        y=0.987,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.075,
        0.965,
        "VADER compound score; headline → ticker-day → equal-ticker sector-day | "
        "2 Jan 2020–29 Dec 2023",
        ha="left",
        fontsize=10.5,
        color=ZERO,
    )
    fig.text(
        0.075,
        0.946,
        "No-news observations remain missing (not neutral). Rolling windows require "
        "at least 10 observed sector-days; dashed line = neutral score (0).",
        ha="left",
        fontsize=9.5,
        color=ZERO,
    )
    fig.text(
        0.018,
        0.52,
        "VADER compound sentiment",
        rotation=90,
        va="center",
        fontsize=10.5,
        color=INK,
    )
    fig.text(
        0.075,
        0.018,
        "Source: FINS5545 provided news headlines; AtlasSignal calculations. "
        "Headline sentiment is a noisy proxy and is not an investment recommendation.",
        ha="left",
        fontsize=8.8,
        color=ZERO,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.925,
        bottom=0.055,
        hspace=0.34,
        wspace=0.12,
    )
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)

    if not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size < 50_000:
        raise RuntimeError("sentiment figure was not written correctly")
    print(f"Sentiment figure saved: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figure panels: {len(sectors)} sectors; shared y-axis {limits[0]:.3f} to {limits[1]:.3f}")


if __name__ == "__main__":
    main()
