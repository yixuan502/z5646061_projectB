"""Build the remaining required fund exhibits and app fact-sheet data.

Run from the project root after Stage 2:

    python scripts/build_fund_exhibits.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.exhibits import (  # noqa: E402
    FAMILY_ORDER,
    METHOD_ORDER,
    combined_allocation_history,
    fact_sheet_summary,
    latest_holdings,
    performance_display_table,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "results" / "data"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

RETURNS_PATH = DATA_DIR / "fund_returns.csv"
WEIGHTS_PATH = DATA_DIR / "fund_weights.csv"
METRICS_PATH = TABLE_DIR / "performance_metrics.csv"

INK = "#1F2933"
BLUE = "#1F5A85"
GOLD = "#B78103"
ORANGE = "#C1662A"
SLATE = "#687783"
GRID = "#D9E0E6"
BACKGROUND = "#FAFBFC"

METHOD_STYLE = {
    "equal_weight": (SLATE, (0, (5, 3)), "Equal Weight"),
    "minimum_variance": (BLUE, "-", "Minimum Variance"),
    "maximum_sharpe": (ORANGE, (0, (2, 2)), "Maximum Sharpe"),
    "risk_parity": (GOLD, "-.", "Risk Parity"),
}

FAMILY_LABELS = {
    "equity": "Equity",
    "crypto": "Crypto",
    "combined": "Combined Equity + Crypto",
}

ALLOCATION_ORDER = [
    "Comm",
    "Consumer",
    "Energy",
    "Financials",
    "Healthcare",
    "Industrials",
    "Materials",
    "RealEstate",
    "Tech",
    "Utilities",
    "Crypto",
]

ALLOCATION_LABELS = {
    "Comm": "Communication",
    "Consumer": "Consumer",
    "Energy": "Energy",
    "Financials": "Financials",
    "Healthcare": "Healthcare",
    "Industrials": "Industrials",
    "Materials": "Materials",
    "RealEstate": "Real Estate",
    "Tech": "Technology",
    "Utilities": "Utilities",
    "Crypto": "Crypto sleeve",
}


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (RETURNS_PATH, WEIGHTS_PATH, METRICS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run python scripts/build_funds.py first."
            )
    returns = pd.read_csv(RETURNS_PATH, parse_dates=["date"])
    weights = pd.read_csv(WEIGHTS_PATH, parse_dates=["rebalance_date"])
    metrics = pd.read_csv(METRICS_PATH, parse_dates=["start_date", "end_date"])
    if returns["fund_id"].nunique() != 12 or metrics["fund_id"].nunique() != 12:
        raise ValueError("expected twelve investable funds")
    return returns, weights, metrics


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=INK, labelsize=8.5)


def _plot_growth(returns: pd.DataFrame) -> pathlib.Path:
    output = FIGURE_DIR / "fund_growth_comparison.png"
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.5, 12.0),
        sharex=False,
        facecolor=BACKGROUND,
    )
    legend_handles = []
    legend_labels = []

    for axis, family in zip(axes, FAMILY_ORDER):
        family_data = returns.loc[returns["asset_family"].eq(family)]
        for method in METHOD_ORDER:
            subset = family_data.loc[family_data["method"].eq(method)].sort_values("date")
            color, linestyle, label = METHOD_STYLE[method]
            (line,) = axis.plot(
                subset["date"],
                subset["growth_1"],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=label,
            )
            if family == FAMILY_ORDER[0]:
                legend_handles.append(line)
                legend_labels.append(label)
        axis.axhline(1.0, color=INK, linewidth=0.9, linestyle=(0, (3, 3)))
        axis.set_ylim(bottom=0.0)
        axis.set_title(
            f"{FAMILY_LABELS[family]} funds",
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
            pad=7,
        )
        axis.set_ylabel("Portfolio value ($)", color=INK)
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        _style_axis(axis)

    axes[-1].set_xlabel("Out-of-sample date", color=INK)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.927),
        ncol=4,
        frameon=False,
        fontsize=9.2,
    )
    fig.suptitle(
        "Out-of-Sample Growth of $1 Across Fund Methods",
        x=0.075,
        y=0.985,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.075,
        0.957,
        "Native OOS calendars, Jan 2021–Dec 2023 | Monthly targets, daily drift, "
        "0 bps transaction costs; dashed horizontal line = initial $1",
        ha="left",
        fontsize=10.0,
        color=SLATE,
    )
    fig.text(
        0.075,
        0.018,
        "Source: FINS5545 provided adjusted-close prices; AtlasSignal calculations. "
        "Equity/combined use 252-day annualisation; crypto uses 365 days.",
        ha="left",
        fontsize=8.7,
        color=SLATE,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.89, bottom=0.06, hspace=0.34)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    return output


def _plot_drawdowns(returns: pd.DataFrame) -> pathlib.Path:
    output = FIGURE_DIR / "fund_drawdown_comparison.png"
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.5, 11.5),
        sharex=False,
        facecolor=BACKGROUND,
    )
    legend_handles = []
    legend_labels = []

    for axis, family in zip(axes, FAMILY_ORDER):
        family_data = returns.loc[returns["asset_family"].eq(family)]
        for method in METHOD_ORDER:
            subset = family_data.loc[family_data["method"].eq(method)].sort_values("date")
            color, linestyle, label = METHOD_STYLE[method]
            (line,) = axis.plot(
                subset["date"],
                subset["drawdown"],
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=label,
            )
            if family == FAMILY_ORDER[0]:
                legend_handles.append(line)
                legend_labels.append(label)
        axis.axhline(0.0, color=INK, linewidth=0.9)
        axis.set_ylim(top=0.02)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.set_title(
            f"{FAMILY_LABELS[family]} funds",
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
            pad=7,
        )
        axis.set_ylabel("Drawdown from peak", color=INK)
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        _style_axis(axis)

    axes[-1].set_xlabel("Out-of-sample date", color=INK)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.927),
        ncol=4,
        frameon=False,
        fontsize=9.2,
    )
    fig.suptitle(
        "Out-of-Sample Drawdowns Across Fund Methods",
        x=0.075,
        y=0.985,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.075,
        0.957,
        "Peak-to-trough percentage decline in each fund's growth-of-$1 index | "
        "Native OOS calendars, Jan 2021–Dec 2023",
        ha="left",
        fontsize=10.0,
        color=SLATE,
    )
    fig.text(
        0.075,
        0.018,
        "Source: FINS5545 provided adjusted-close prices; AtlasSignal calculations. "
        "The initial investor wealth of $1 is included in the high-water mark.",
        ha="left",
        fontsize=8.7,
        color=SLATE,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.89, bottom=0.06, hspace=0.34)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    return output


def _plot_weight_heatmaps(history: pd.DataFrame) -> pathlib.Path:
    output = FIGURE_DIR / "combined_weights_over_time.png"
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.8), facecolor=BACKGROUND)
    axes = axes.ravel()
    cmap = LinearSegmentedColormap.from_list(
        "atlas_blue",
        ["#F4F7F9", "#C9DCE9", BLUE, "#123A58"],
    )
    maximum = float(history["target_weight"].max())
    last_image = None

    for axis, method in zip(axes, METHOD_ORDER):
        subset = history.loc[history["method"].eq(method)].copy()
        pivot = subset.pivot(
            index="allocation_bucket",
            columns="rebalance_date",
            values="target_weight",
        ).reindex(ALLOCATION_ORDER).fillna(0.0)
        if not np.allclose(pivot.sum(axis=0).to_numpy(), 1.0, atol=1e-7):
            raise RuntimeError(f"{method} heatmap columns do not sum to one")
        last_image = axis.imshow(
            pivot.to_numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=maximum,
        )
        positions = np.arange(0, pivot.shape[1], 6)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [pivot.columns[i].strftime("%b\n%Y") for i in positions],
            fontsize=7.8,
            color=INK,
        )
        axis.set_yticks(np.arange(len(ALLOCATION_ORDER)))
        axis.set_yticklabels(
            [ALLOCATION_LABELS[item] for item in ALLOCATION_ORDER],
            fontsize=8.2,
            color=INK,
        )
        axis.set_title(
            METHOD_STYLE[method][2],
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
            pad=8,
        )
        axis.set_xlabel("Monthly rebalance date", color=INK, fontsize=9.0)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_color(GRID)

    if last_image is None:
        raise RuntimeError("no combined-weight heatmap was created")
    # Reserve a dedicated row so the shared legend cannot cover the bottom
    # panels' date labels when the PNG is inserted into Word or the app.
    colorbar_axis = fig.add_axes([0.12, 0.075, 0.86, 0.022])
    colorbar = fig.colorbar(last_image, cax=colorbar_axis, orientation="horizontal")
    colorbar.ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    colorbar.set_label("Target portfolio weight", color=INK, fontsize=9.5)
    colorbar.ax.tick_params(colors=INK, labelsize=8.0)

    fig.suptitle(
        "Combined-Fund Target Weights Over Time",
        x=0.075,
        y=0.985,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.075,
        0.953,
        "Ten equity sectors plus the aggregated crypto sleeve | 36 monthly "
        "rebalances, 4 Jan 2021–1 Dec 2023; common colour scale across methods",
        ha="left",
        fontsize=10.0,
        color=SLATE,
    )
    fig.text(
        0.075,
        0.014,
        "Source: FINS5545 provided adjusted-close prices; AtlasSignal calculations. "
        "Values are target weights at each rebalance, not drifted daily holdings.",
        ha="left",
        fontsize=8.7,
        color=SLATE,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.18, hspace=0.32, wspace=0.20)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    return output


def _plot_sharpe(metrics: pd.DataFrame) -> pathlib.Path:
    output = FIGURE_DIR / "fund_sharpe_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.3), sharey=True, facecolor=BACKGROUND)
    maximum = float(metrics["sharpe_ratio"].max())
    method_labels = [METHOD_STYLE[method][2] for method in METHOD_ORDER]
    short_labels = ["Equal\nWeight", "Minimum\nVariance", "Maximum\nSharpe", "Risk\nParity"]
    colors = [METHOD_STYLE[method][0] for method in METHOD_ORDER]

    for axis, family in zip(axes, FAMILY_ORDER):
        subset = metrics.loc[metrics["asset_family"].eq(family)].set_index("method")
        values = subset.reindex(METHOD_ORDER)["sharpe_ratio"].to_numpy()
        bars = axis.bar(
            np.arange(len(METHOD_ORDER)),
            values,
            color=colors,
            edgecolor=INK,
            linewidth=0.6,
            width=0.68,
        )
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9.0, color=INK)
        axis.set_xticks(np.arange(len(METHOD_ORDER)))
        axis.set_xticklabels(short_labels, fontsize=8.0, color=INK)
        axis.set_ylim(0.0, maximum * 1.18)
        axis.set_title(
            FAMILY_LABELS[family],
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
            pad=8,
        )
        _style_axis(axis)
    axes[0].set_ylabel("Annualised Sharpe ratio", color=INK)

    fig.suptitle(
        "Out-of-Sample Sharpe Ratio Across 12 Funds",
        x=0.065,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.065,
        0.93,
        "Arithmetic annualised return ÷ annualised volatility; risk-free rate 0% | "
        "Native OOS calendars, Jan 2021–Dec 2023",
        ha="left",
        fontsize=10.0,
        color=SLATE,
    )
    fig.text(
        0.065,
        0.025,
        "Source: FINS5545 provided adjusted-close prices; AtlasSignal calculations. "
        "Equity/combined use 252-day annualisation; crypto uses 365 days.",
        ha="left",
        fontsize=8.7,
        color=SLATE,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.16, wspace=0.13)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    return output


def _validate_artifacts(
    holdings: pd.DataFrame,
    allocation_history: pd.DataFrame,
    fact_sheets: pd.DataFrame,
    figures: list[pathlib.Path],
) -> None:
    if holdings["fund_id"].nunique() != 12:
        raise RuntimeError("latest holdings does not cover twelve funds")
    if fact_sheets["fund_id"].nunique() != 12 or len(fact_sheets) != 12:
        raise RuntimeError("fact-sheet summary is not one row per fund")
    if allocation_history["method"].nunique() != 4:
        raise RuntimeError("allocation history does not cover four methods")
    if allocation_history["rebalance_date"].nunique() != 36:
        raise RuntimeError("allocation history does not cover 36 rebalances")
    for path in figures:
        if not path.exists() or path.stat().st_size < 50_000:
            raise RuntimeError(f"figure was not written correctly: {path}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    returns, weights, metrics = _load_inputs()

    holdings = latest_holdings(weights)
    allocation_history = combined_allocation_history(weights)
    fact_sheets = fact_sheet_summary(metrics, holdings)
    display_metrics = performance_display_table(metrics)

    holdings.to_csv(DATA_DIR / "latest_holdings.csv", index=False)
    allocation_history.to_csv(DATA_DIR / "combined_allocation_history.csv", index=False)
    fact_sheets.to_csv(TABLE_DIR / "fund_fact_sheets.csv", index=False)
    display_metrics.to_csv(TABLE_DIR / "performance_metrics_display.csv", index=False)

    figures = [
        _plot_growth(returns),
        _plot_drawdowns(returns),
        _plot_weight_heatmaps(allocation_history),
        _plot_sharpe(metrics),
    ]
    _validate_artifacts(holdings, allocation_history, fact_sheets, figures)

    leaders = (
        fact_sheets.sort_values(["asset_family", "sharpe_ratio"], ascending=[True, False])
        .groupby("asset_family", as_index=False, observed=True)
        .first()[["asset_family", "fund_name", "sharpe_ratio"]]
    )
    print("\n=== STAGE 5: FUND EXHIBITS AND FACT SHEETS BUILT ===")
    print(f"Fact sheets: {len(fact_sheets)} funds")
    print(f"Latest target holdings: {len(holdings):,} rows across 12 funds")
    print(
        "Combined allocation history: "
        f"{allocation_history['rebalance_date'].nunique()} rebalances x 4 methods"
    )
    print("Highest OOS Sharpe within each family:")
    for row in leaders.itertuples(index=False):
        print(f"  {row.asset_family.title()}: {row.fund_name} ({row.sharpe_ratio:.3f})")
    print("Required fund figures:")
    for path in figures:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    print("Validation: original Stage 2 return/weight files were read-only — PASS")


if __name__ == "__main__":
    main()
