"""Create the required fusion before/after and robustness exhibit."""

from __future__ import annotations

import pathlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "results" / "data" / "fusion_returns.csv"
ROBUSTNESS_PATH = PROJECT_ROOT / "results" / "tables" / "fusion_robustness.csv"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_PATH = FIGURE_DIR / "fusion_before_after.png"

INK = "#1F2933"
BLUE = "#1F5A85"
GOLD = "#B78103"
GRID = "#D9E0E6"
ZERO = "#687783"
BACKGROUND = "#FAFBFC"

VARIANT_STYLE = {
    "base": (INK, "-", "Base Risk Parity"),
    "sentiment": (BLUE, "--", "Baseline Sentiment"),
    "coverage_adjusted": (GOLD, "-.", "AtlasSignal Coverage-Adjusted"),
}


def main() -> None:
    for path in (DATA_PATH, ROBUSTNESS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run python scripts/build_fusion.py first."
            )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(DATA_PATH, parse_dates=["date"])
    robustness = pd.read_csv(ROBUSTNESS_PATH)
    if daily["fusion_fund_id"].nunique() != 6:
        raise ValueError("fusion_returns must contain six primary funds")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.2, 10.3),
        facecolor=BACKGROUND,
        gridspec_kw={"height_ratios": [1.12, 1.0]},
    )

    for column, family in enumerate(("equity", "combined")):
        top = axes[0, column]
        family_daily = daily[daily["asset_family"].eq(family)]
        for variant in ("base", "sentiment", "coverage_adjusted"):
            subset = family_daily[family_daily["fusion_variant"].eq(variant)]
            color, linestyle, label = VARIANT_STYLE[variant]
            top.plot(
                subset["date"],
                subset["growth_1"],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=label,
            )
        top.set_title(
            f"{family.title()} Risk Parity: OOS Growth of $1",
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
            pad=8,
        )
        top.set_ylabel("Portfolio value ($)" if column == 0 else "")
        top.xaxis.set_major_locator(mdates.YearLocator())
        top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        top.legend(frameon=False, fontsize=8.5, loc="upper left")

        bottom = axes[1, column]
        family_robustness = robustness[
            robustness["asset_family"].eq(family)
            & robustness["fusion_variant"].ne("base")
        ]
        for variant in ("sentiment", "coverage_adjusted"):
            color, _, label = VARIANT_STYLE[variant]
            for cost, linestyle, marker in ((0.0, "-", "o"), (10.0, "--", "s")):
                subset = family_robustness[
                    family_robustness["fusion_variant"].eq(variant)
                    & family_robustness["transaction_cost_bps"].eq(cost)
                ].sort_values("tilt_strength")
                bottom.plot(
                    subset["tilt_strength"] * 100.0,
                    subset["delta_sharpe_vs_base"],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=5.2,
                    linewidth=1.7,
                    label=f"{label}, {cost:.0f} bps",
                )
        bottom.axhline(0.0, color=ZERO, linewidth=1.0, linestyle=(0, (4, 3)))
        bottom.set_title(
            f"{family.title()}: Sharpe Change vs Matching Base",
            loc="left",
            fontsize=11.5,
            fontweight="bold",
            color=INK,
            pad=8,
        )
        bottom.set_xlabel("Maximum relative sector tilt (%)")
        bottom.set_ylabel("Sharpe ratio change" if column == 0 else "")
        bottom.set_xticks([10, 25, 40])
        bottom.legend(frameon=False, fontsize=7.8, loc="lower left")

    # Use honest shared scales within each analytical comparison.
    growth_min = min(axis.get_ylim()[0] for axis in axes[0])
    growth_max = max(axis.get_ylim()[1] for axis in axes[0])
    sharpe_min = min(axis.get_ylim()[0] for axis in axes[1])
    sharpe_max = max(axis.get_ylim()[1] for axis in axes[1])
    for axis in axes[0]:
        axis.set_ylim(growth_min, growth_max)
    for axis in axes[1]:
        axis.set_ylim(sharpe_min, sharpe_max)

    for axis in axes.ravel():
        axis.set_facecolor(BACKGROUND)
        axis.grid(axis="y", color=GRID, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(GRID)
        axis.tick_params(colors=INK, labelsize=8.5)

    fig.suptitle(
        "Sentiment Fusion: Base, Baseline, and Coverage-Adjusted",
        x=0.07,
        y=0.984,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.07,
        0.955,
        "Out-of-sample monthly targets, 4 Jan 2021–29 Dec 2023 | "
        "Risk-free rate 0%; primary growth uses 25% tilt and 0 bps",
        ha="left",
        fontsize=10.3,
        color=ZERO,
    )
    fig.text(
        0.07,
        0.928,
        "AtlasSignal attenuates the lagged relative-sector VADER signal using "
        "three completed months of news HHI and prior ticker coverage. "
        "Combined-fund crypto targets are unchanged.",
        ha="left",
        fontsize=9.3,
        color=ZERO,
    )
    fig.text(
        0.07,
        0.018,
        "Source: FINS5545 provided prices and news; AtlasSignal calculations. "
        "Negative OOS results are reported without parameter selection. "
        "Transaction cost = bps × one-way turnover.",
        ha="left",
        fontsize=8.7,
        color=ZERO,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.895,
        bottom=0.075,
        hspace=0.33,
        wspace=0.16,
    )
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)

    if not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size < 50_000:
        raise RuntimeError("fusion figure was not written correctly")
    print(f"Fusion figure saved: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print("Panels: 2 growth comparisons + 2 Sharpe-robustness comparisons")


if __name__ == "__main__":
    main()
