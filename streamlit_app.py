"""AtlasSignal — the investor-facing FINS5545 Part B Streamlit app.

Run locally from the repository root:

    streamlit run streamlit_app.py

The app reads bounded, precomputed CSV artifacts under ``results/``.  Portfolio
optimisation and headline scoring remain offline build steps.
"""

from __future__ import annotations

import pathlib

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.app_logic import (
    build_allocation_scenario,
    load_app_data,
    rolling_sector_sentiment,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

INK = "#1F2933"
BLUE = "#1F5A85"
GOLD = "#B78103"
ORANGE = "#C1662A"
SLATE = "#687783"
PALE = "#EEF3F6"

FAMILY_LABELS = {
    "equity": "Equity",
    "crypto": "Crypto",
    "combined": "Combined Equity + Crypto",
}
FAMILY_COLORS = {
    "Equity": BLUE,
    "Crypto": ORANGE,
    "Combined Equity + Crypto": GOLD,
}
METHOD_ORDER = ["Equal Weight", "Minimum Variance", "Maximum Sharpe", "Risk Parity"]
METHOD_COLORS = {
    "Equal Weight": SLATE,
    "Minimum Variance": BLUE,
    "Maximum Sharpe": ORANGE,
    "Risk Parity": GOLD,
}
VARIANT_ORDER = ["Base Risk Parity", "Baseline Sentiment", "AtlasSignal Coverage-Adjusted"]
VARIANT_COLORS = {
    "Base Risk Parity": INK,
    "Baseline Sentiment": BLUE,
    "AtlasSignal Coverage-Adjusted": GOLD,
}


st.set_page_config(
    page_title="AtlasSignal | Systematic Funds",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    f"""
    <style>
    :root {{ --atlas-ink: {INK}; --atlas-blue: {BLUE}; --atlas-gold: {GOLD}; }}
    .stApp {{ background: #F7F9FA; color: {INK}; }}
    [data-testid="stHeader"] {{ background: rgba(247,249,250,.92); }}
    [data-testid="stSidebar"] {{ background: #102C40; }}
    [data-testid="stSidebar"] * {{ color: #F6F8F9; }}
    [data-testid="stSidebar"] .stCaption {{ color: #C7D4DC !important; }}
    .block-container {{ max-width: 1420px; padding-top: 1.6rem; padding-bottom: 3rem; }}
    .atlas-hero {{
        background: linear-gradient(120deg, #123A58 0%, #1F5A85 62%, #2B6E99 100%);
        border-radius: 18px; padding: 2.0rem 2.2rem 1.8rem; color: white;
        box-shadow: 0 14px 34px rgba(18,58,88,.16); margin-bottom: 1.1rem;
    }}
    .atlas-kicker {{ color: #F0C35B; font-size: .76rem; font-weight: 800;
        letter-spacing: .15em; text-transform: uppercase; margin-bottom: .45rem; }}
    .atlas-hero h1 {{ color: white; font-size: 2.45rem; line-height: 1.05;
        margin: 0 0 .55rem; letter-spacing: -.035em; }}
    .atlas-hero p {{ color: #DBE7EE; max-width: 850px; font-size: 1.02rem;
        margin: 0; line-height: 1.55; }}
    .journey {{ display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1.15rem; }}
    .journey span {{ border:1px solid rgba(255,255,255,.25); border-radius:999px;
        padding:.30rem .68rem; color:#F7FAFC; font-size:.78rem; }}
    .section-lede {{ color: {SLATE}; font-size: .94rem; margin-top: -.35rem;
        margin-bottom: .85rem; max-width: 980px; }}
    .insight {{ background:#FFF8E7; border-left:4px solid {GOLD};
        border-radius:8px; padding:.8rem 1rem; color:{INK}; margin:.6rem 0 1rem; }}
    .risk-note {{ background:#FFF3ED; border-left:4px solid {ORANGE};
        border-radius:8px; padding:.8rem 1rem; color:{INK}; margin:.6rem 0 1rem; }}
    [data-testid="stMetric"] {{ background:white; border:1px solid #E2E8EC;
        border-radius:12px; padding:.72rem .88rem; box-shadow:0 4px 14px rgba(31,41,51,.04); }}
    [data-testid="stMetricLabel"] {{ color:{SLATE}; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:.35rem; }}
    .stTabs [data-baseweb="tab"] {{ border-radius:9px 9px 0 0; padding:.55rem .9rem; }}
    .stTabs [aria-selected="true"] {{ background:#E8F0F5; color:{BLUE}; }}
    .sidebar-mark {{ color:#F0C35B; font-size:.74rem; letter-spacing:.16em;
        font-weight:800; text-transform:uppercase; }}
    .sidebar-title {{ color:white; font-size:1.48rem; font-weight:800;
        letter-spacing:-.03em; margin:.15rem 0 .4rem; }}
    .footer {{ color:{SLATE}; font-size:.78rem; padding-top:1.5rem;
        border-top:1px solid #DDE4E8; margin-top:2rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_snapshot() -> dict[str, pd.DataFrame]:
    return load_app_data(PROJECT_ROOT)


def _finish_chart(chart: alt.Chart, height: int = 330) -> alt.Chart:
    return (
        chart.properties(height=height)
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor=INK,
            titleColor=INK,
            gridColor="#DCE4E9",
            domainColor="#AAB8C2",
            tickColor="#AAB8C2",
            labelFontSize=11,
            titleFontSize=12,
        )
        .configure_legend(
            labelColor=INK,
            titleColor=INK,
            orient="top",
            direction="horizontal",
        )
    )


def _family_color() -> alt.Color:
    domain = list(FAMILY_COLORS)
    return alt.Color(
        "family_label:N",
        scale=alt.Scale(domain=domain, range=[FAMILY_COLORS[item] for item in domain]),
        title="Asset family",
    )


def _method_color(field: str = "method_label:N") -> alt.Color:
    return alt.Color(
        field,
        scale=alt.Scale(
            domain=METHOD_ORDER,
            range=[METHOD_COLORS[item] for item in METHOD_ORDER],
        ),
        title="Method",
    )


def _variant_color() -> alt.Color:
    return alt.Color(
        "fusion_variant_label:N",
        scale=alt.Scale(
            domain=VARIANT_ORDER,
            range=[VARIANT_COLORS[item] for item in VARIANT_ORDER],
        ),
        title="Variant",
    )


def _render_compare(data: dict[str, pd.DataFrame]) -> None:
    facts = data["fact_sheets"].copy()
    facts["family_label"] = facts["asset_family"].map(FAMILY_LABELS)

    st.subheader("Compare the fund shelf")
    st.markdown(
        '<p class="section-lede">Start with risk and diversification, then use return. '
        "All figures are walk-forward out-of-sample results, not fitted in-sample scores.</p>",
        unsafe_allow_html=True,
    )
    control_a, control_b = st.columns([1, 1.25])
    with control_a:
        selected_families = st.multiselect(
            "Asset families",
            options=list(FAMILY_COLORS),
            default=list(FAMILY_COLORS),
            key="compare_families",
        )
    with control_b:
        selected_methods = st.multiselect(
            "Portfolio methods",
            options=METHOD_ORDER,
            default=METHOD_ORDER,
            key="compare_methods",
        )
    filtered = facts.loc[
        facts["family_label"].isin(selected_families)
        & facts["method_label"].isin(selected_methods)
    ].copy()
    if filtered.empty:
        st.warning("Select at least one family and one method to compare funds.")
        return

    best = filtered.loc[filtered["sharpe_ratio"].idxmax()]
    lowest_risk = filtered.loc[filtered["annualized_volatility"].idxmin()]
    shallowest = filtered.loc[filtered["maximum_drawdown"].idxmax()]
    cards = st.columns(4)
    cards[0].metric("Funds in view", f"{len(filtered)}")
    cards[1].metric("Highest OOS Sharpe", f"{best['sharpe_ratio']:.2f}")
    cards[1].caption(best["fund_name"])
    cards[2].metric(
        "Lowest annualised volatility",
        f"{lowest_risk['annualized_volatility']:.1%}",
    )
    cards[2].caption(lowest_risk["fund_name"])
    cards[3].metric(
        "Shallowest maximum drawdown",
        f"{shallowest['maximum_drawdown']:.1%}",
    )
    cards[3].caption(shallowest["fund_name"])

    left, right = st.columns([1.08, 0.92])
    with left:
        st.markdown("#### Return–risk map")
        scatter = (
            alt.Chart(filtered)
            .mark_point(
                filled=True,
                size=170,
                opacity=0.88,
                stroke="white",
                strokeWidth=1.2,
            )
            .encode(
                x=alt.X(
                    "annualized_volatility:Q",
                    title="Annualised volatility",
                    axis=alt.Axis(format=".0%"),
                    scale=alt.Scale(zero=True),
                ),
                y=alt.Y(
                    "annualized_return:Q",
                    title="Annualised return",
                    axis=alt.Axis(format=".0%"),
                    scale=alt.Scale(zero=True),
                ),
                color=_family_color(),
                shape=alt.Shape("method_label:N", sort=METHOD_ORDER, title="Method"),
                tooltip=[
                    alt.Tooltip("fund_name:N", title="Fund"),
                    alt.Tooltip("annualized_return:Q", title="Annualised return", format=".2%"),
                    alt.Tooltip("annualized_volatility:Q", title="Annualised volatility", format=".2%"),
                    alt.Tooltip("sharpe_ratio:Q", title="Sharpe", format=".3f"),
                    alt.Tooltip("maximum_drawdown:Q", title="Maximum drawdown", format=".2%"),
                ],
            )
        )
        st.altair_chart(_finish_chart(scatter, 370), width="stretch")
        st.caption("Higher return and lower volatility is preferable, but drawdown and turnover still matter.")
    with right:
        st.markdown("#### Risk-adjusted ranking")
        sharpe_bars = (
            alt.Chart(filtered)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                y=alt.Y("fund_name:N", sort="-x", title=None),
                x=alt.X(
                    "sharpe_ratio:Q",
                    title="Annualised Sharpe ratio",
                    scale=alt.Scale(zero=True),
                    stack=None,
                ),
                color=_family_color(),
                tooltip=[
                    alt.Tooltip("fund_name:N", title="Fund"),
                    alt.Tooltip("sharpe_ratio:Q", title="Sharpe", format=".3f"),
                ],
            )
        )
        st.altair_chart(_finish_chart(sharpe_bars, 370), width="stretch")
        st.caption("Sharpe uses a 0% risk-free rate and calendar-appropriate annualisation.")

    st.markdown(
        '<div class="insight"><b>Evidence, not a universal winner.</b> Equity Equal Weight, '
        "Crypto Minimum Variance, and Combined Risk Parity lead their own families on OOS Sharpe. "
        "Crypto remains high-risk even when its risk-adjusted ratio is strongest.</div>",
        unsafe_allow_html=True,
    )

    display = filtered[
        [
            "fund_name",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "annualized_turnover",
        ]
    ].copy()
    display.columns = [
        "Fund",
        "Annualised return",
        "Annualised volatility",
        "Sharpe",
        "Maximum drawdown",
        "Annualised turnover",
    ]
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "Annualised return": st.column_config.NumberColumn(format="percent"),
            "Annualised volatility": st.column_config.NumberColumn(format="percent"),
            "Sharpe": st.column_config.NumberColumn(format="%.3f"),
            "Maximum drawdown": st.column_config.NumberColumn(format="percent"),
            "Annualised turnover": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.download_button(
        "Download comparison CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="atlassignal_fund_comparison.csv",
        mime="text/csv",
    )


def _render_fact_sheet(data: dict[str, pd.DataFrame]) -> None:
    facts = data["fact_sheets"].copy()
    returns = data["fund_returns"]
    holdings = data["latest_holdings"]
    names = facts.sort_values(["asset_family", "method_label"])["fund_name"].tolist()
    default_name = "Combined Risk Parity"

    st.subheader("Open an investable fund fact sheet")
    st.markdown(
        '<p class="section-lede">Inspect one fund’s return path, loss experience, '
        "latest OOS target holdings, and implementation characteristics before allocating capital.</p>",
        unsafe_allow_html=True,
    )
    selected_name = st.selectbox(
        "Choose a fund",
        names,
        index=names.index(default_name) if default_name in names else 0,
        key="fact_sheet_fund",
    )
    fact = facts.loc[facts["fund_name"].eq(selected_name)].iloc[0]
    fund_returns = returns.loc[returns["fund_id"].eq(fact["fund_id"])].sort_values("date")
    fund_holdings = holdings.loc[
        holdings["fund_id"].eq(fact["fund_id"]) & holdings["is_active_holding"]
    ].copy()

    st.markdown(f"### {selected_name}")
    st.caption(
        f"OOS {fact['start_date']:%-d %b %Y}–{fact['end_date']:%-d %b %Y} · "
        f"{int(fact['estimation_window'])}-observation rolling window · monthly rebalance · "
        f"{fact['calendar_basis']} · 0 bps baseline"
    )
    cards = st.columns(6)
    cards[0].metric("Annualised return", f"{fact['annualized_return']:.1%}")
    cards[1].metric("Annualised volatility", f"{fact['annualized_volatility']:.1%}")
    cards[2].metric("Sharpe ratio", f"{fact['sharpe_ratio']:.2f}")
    cards[3].metric("Maximum drawdown", f"{fact['maximum_drawdown']:.1%}")
    cards[4].metric("Growth of $1", f"${1.0 + fact['total_return']:.2f}")
    cards[5].metric("Active holdings", f"{int(fact['active_holdings'])}")

    growth_col, drawdown_col = st.columns([1.05, 0.95])
    with growth_col:
        st.markdown("#### Growth of $1")
        growth = (
            alt.Chart(fund_returns)
            .mark_line(color=BLUE, strokeWidth=2.4)
            .encode(
                x=alt.X("date:T", title="OOS date"),
                y=alt.Y("growth_1:Q", title="Portfolio value ($)", scale=alt.Scale(zero=True)),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("growth_1:Q", title="Value", format="$.3f"),
                ],
            )
        )
        baseline = alt.Chart(pd.DataFrame({"value": [1.0]})).mark_rule(
            color=SLATE, strokeDash=[4, 4]
        ).encode(y="value:Q")
        st.altair_chart(_finish_chart(growth + baseline, 310), width="stretch")
    with drawdown_col:
        st.markdown("#### Drawdown from prior peak")
        drawdown = (
            alt.Chart(fund_returns)
            .mark_area(color=ORANGE, opacity=0.42, line={"color": ORANGE, "strokeWidth": 1.6})
            .encode(
                x=alt.X("date:T", title="OOS date"),
                y=alt.Y("drawdown:Q", title="Drawdown", axis=alt.Axis(format=".0%")),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("drawdown:Q", title="Drawdown", format=".2%"),
                ],
            )
        )
        st.altair_chart(_finish_chart(drawdown, 310), width="stretch")

    if fact["asset_family"] == "crypto":
        st.markdown(
            '<div class="risk-note"><b>High absolute risk.</b> “Minimum Variance” or a '
            "high Sharpe within crypto does not make a crypto fund conventionally low risk. "
            "Use maximum drawdown alongside return.</div>",
            unsafe_allow_html=True,
        )
    elif fact["method"] == "maximum_sharpe":
        st.markdown(
            '<div class="risk-note"><b>Mean-estimation sensitivity.</b> Maximum Sharpe '
            "uses estimated historical expected returns and produced concentrated, high-turnover "
            "targets in this sample.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Latest OOS target holdings")
    st.caption(
        f"As of {fact['latest_rebalance_date']:%-d %b %Y} (latest OOS rebalance). "
        "These are backtest model targets, not live holdings, live prices, or personal recommendations."
    )
    max_holdings = len(fund_holdings)
    top_n = st.slider(
        "Holdings shown",
        min_value=5,
        max_value=max(5, max_holdings),
        value=min(12, max(5, max_holdings)),
        step=1,
        key=f"holdings_{fact['fund_id']}",
        disabled=max_holdings <= 5,
    )
    shown = fund_holdings.nlargest(top_n, "target_weight").copy()
    holding_col, mix_col = st.columns([1.08, 0.92])
    with holding_col:
        bars = (
            alt.Chart(shown)
            .mark_bar(color=BLUE, cornerRadiusEnd=4)
            .encode(
                y=alt.Y("ticker:N", sort="-x", title=None),
                x=alt.X(
                    "target_weight:Q",
                    title="Target portfolio weight",
                    axis=alt.Axis(format=".0%"),
                    scale=alt.Scale(zero=True),
                ),
                tooltip=[
                    alt.Tooltip("ticker:N", title="Ticker"),
                    alt.Tooltip("sector:N", title="Sector"),
                    alt.Tooltip("target_weight:Q", title="Weight", format=".2%"),
                ],
            )
        )
        st.altair_chart(_finish_chart(bars, 350), width="stretch")
    with mix_col:
        mix = fund_holdings.copy()
        mix["allocation_bucket"] = np.where(
            mix["asset_class"].eq("Crypto"), "Crypto sleeve", mix["sector"]
        )
        mix = mix.groupby("allocation_bucket", as_index=False)["target_weight"].sum()
        mix_chart = (
            alt.Chart(mix)
            .mark_bar(color=GOLD, cornerRadiusEnd=4)
            .encode(
                y=alt.Y("allocation_bucket:N", sort="-x", title=None),
                x=alt.X(
                    "target_weight:Q",
                    title="Target portfolio weight",
                    axis=alt.Axis(format=".0%"),
                    scale=alt.Scale(zero=True),
                ),
                tooltip=[
                    alt.Tooltip("allocation_bucket:N", title="Exposure"),
                    alt.Tooltip("target_weight:Q", title="Weight", format=".2%"),
                ],
            )
        )
        st.altair_chart(_finish_chart(mix_chart, 350), width="stretch")

    table = shown[["holding_rank", "ticker", "asset_class", "sector", "target_weight"]].copy()
    table.columns = ["Rank", "Ticker", "Asset class", "Sector", "Target weight"]
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={"Target weight": st.column_config.NumberColumn(format="percent")},
    )

    with st.expander("Additional risk and implementation statistics"):
        extra = st.columns(5)
        extra[0].metric("CAGR", f"{fact['cagr']:.1%}")
        extra[1].metric("Sortino ratio", f"{fact['sortino_ratio']:.2f}")
        extra[2].metric("Historical VaR 95%", f"{fact['historical_var_95']:.2%}")
        extra[3].metric("Expected shortfall 95%", f"{fact['historical_expected_shortfall_95']:.2%}")
        extra[4].metric("Annualised turnover", f"{fact['annualized_turnover']:.1%}")


def _render_builder(data: dict[str, pd.DataFrame]) -> None:
    facts = data["fact_sheets"].copy()
    returns = data["fund_returns"]
    name_to_id = facts.set_index("fund_name")["fund_id"].to_dict()
    options = facts["fund_name"].tolist()
    defaults = ["Combined Risk Parity", "Equity Equal Weight", "Crypto Minimum Variance"]
    default_weights = {
        "Combined Risk Parity": 50.0,
        "Equity Equal Weight": 30.0,
        "Crypto Minimum Variance": 20.0,
    }

    st.subheader("Build a fund-level allocation scenario")
    st.markdown(
        '<p class="section-lede">Allocate an initial $1 across up to four funds and '
        "inspect the historical path. This is a transparent scenario tool—not a personalised optimiser.</p>",
        unsafe_allow_html=True,
    )
    selected_names = st.multiselect(
        "Select one to four funds",
        options=options,
        default=[item for item in defaults if item in options],
        max_selections=4,
        key="builder_funds",
    )
    if not selected_names:
        st.warning("Select at least one fund to build an allocation.")
        return

    input_columns = st.columns(len(selected_names))
    raw_allocations: dict[str, float] = {}
    equal_default = 100.0 / len(selected_names)
    for column, name in zip(input_columns, selected_names):
        with column:
            raw_allocations[name_to_id[name]] = st.number_input(
                f"{name} (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(default_weights.get(name, equal_default)),
                step=5.0,
                key=f"allocation_{name_to_id[name]}",
            )
    input_total = float(sum(raw_allocations.values()))
    if input_total <= 0:
        st.error("Allocation inputs must contain a positive amount.")
        return
    if not np.isclose(input_total, 100.0):
        st.info(f"Inputs total {input_total:.1f}%; the scenario normalises them to 100.0%.")

    try:
        daily, metrics, allocation_table = build_allocation_scenario(
            returns,
            facts,
            raw_allocations,
        )
    except ValueError as exc:
        st.error(f"The allocation cannot be evaluated: {exc}")
        return

    cards = st.columns(5)
    cards[0].metric("Ending value of $1", f"${metrics['ending_value']:.2f}")
    cards[1].metric("Annualised return", f"{metrics['annualized_return']:.1%}")
    cards[2].metric("Annualised volatility", f"{metrics['annualized_volatility']:.1%}")
    cards[3].metric("Sharpe ratio", f"{metrics['sharpe_ratio']:.2f}")
    cards[4].metric("Maximum drawdown", f"{metrics['maximum_drawdown']:.1%}")
    st.caption(
        f"{metrics['start_date']:%-d %b %Y}–{metrics['end_date']:%-d %b %Y} · "
        f"{metrics['calendar_basis']} · {metrics['allocation_model']} · 0% risk-free rate"
    )

    allocation_col, growth_col = st.columns([0.75, 1.25])
    with allocation_col:
        st.markdown("#### Normalised initial allocation")
        allocation_chart = (
            alt.Chart(allocation_table)
            .mark_bar(color=GOLD, cornerRadiusEnd=4)
            .encode(
                y=alt.Y("fund_name:N", sort="-x", title=None),
                x=alt.X(
                    "normalized_weight:Q",
                    title="Initial allocation",
                    axis=alt.Axis(format=".0%"),
                    scale=alt.Scale(zero=True),
                ),
                tooltip=[
                    alt.Tooltip("fund_name:N", title="Fund"),
                    alt.Tooltip("normalized_weight:Q", title="Allocation", format=".1%"),
                ],
            )
        )
        st.altair_chart(_finish_chart(allocation_chart, 330), width="stretch")
    with growth_col:
        st.markdown("#### Historical scenario growth")
        growth = (
            alt.Chart(daily)
            .mark_line(color=BLUE, strokeWidth=2.5)
            .encode(
                x=alt.X("date:T", title="Scenario date"),
                y=alt.Y("growth_1:Q", title="Portfolio value ($)", scale=alt.Scale(zero=True)),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("growth_1:Q", title="Value", format="$.3f"),
                    alt.Tooltip("drawdown:Q", title="Drawdown", format=".2%"),
                ],
            )
        )
        st.altair_chart(_finish_chart(growth, 330), width="stretch")

    st.markdown("#### Sleeve contribution and loss path")
    contribution_col, drawdown_col = st.columns([0.85, 1.15])
    with contribution_col:
        contribution = allocation_table[
            ["fund_name", "normalized_weight", "sleeve_ending_growth", "gain_contribution"]
        ].copy()
        contribution.columns = [
            "Fund",
            "Initial allocation",
            "Sleeve ending growth",
            "Contribution to portfolio gain",
        ]
        st.dataframe(
            contribution,
            hide_index=True,
            width="stretch",
            column_config={
                "Initial allocation": st.column_config.NumberColumn(format="percent"),
                "Sleeve ending growth": st.column_config.NumberColumn(format="$%.3f"),
                "Contribution to portfolio gain": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    with drawdown_col:
        drawdown = (
            alt.Chart(daily)
            .mark_area(color=ORANGE, opacity=0.42, line={"color": ORANGE, "strokeWidth": 1.5})
            .encode(
                x=alt.X("date:T", title="Scenario date"),
                y=alt.Y("drawdown:Q", title="Drawdown", axis=alt.Axis(format=".0%")),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("drawdown:Q", title="Drawdown", format=".2%"),
                ],
            )
        )
        st.altair_chart(_finish_chart(drawdown, 260), width="stretch")

    st.markdown(
        '<div class="risk-note"><b>Scenario boundary.</b> Fund choices and weights are '
        "evaluated on the same historical sample used elsewhere in the app. The result includes "
        "no management fee, tax, slippage, or inter-fund rebalancing and is not a forecast.</div>",
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download scenario path",
        data=daily.to_csv(index=False).encode("utf-8"),
        file_name="atlassignal_allocation_scenario.csv",
        mime="text/csv",
    )


def _render_news(data: dict[str, pd.DataFrame]) -> None:
    sentiment = data["sentiment"]
    summary = data["sentiment_summary"].copy()
    fusion = data["fusion_comparison"]
    fusion_returns = data["fusion_returns"]
    robustness = data["fusion_robustness"]
    sectors = sorted(sentiment["sector"].unique())

    st.subheader("Read sector news tone—and test whether it was investable")
    st.markdown(
        '<p class="section-lede">VADER is used as a transparent descriptive text model. '
        "The investable experiment is lagged, coverage-aware, and reported even when it underperforms.</p>",
        unsafe_allow_html=True,
    )
    total_headlines = int(summary["total_headlines"].sum())
    no_news_days = int(summary["days_without_news"].sum())
    average_coverage = float(sentiment["ticker_coverage_ratio"].mean())
    cards = st.columns(4)
    cards[0].metric("Scored headlines", f"{total_headlines:,}")
    cards[1].metric("Equity sectors", f"{len(sectors)}")
    cards[2].metric("Mean ticker coverage", f"{average_coverage:.1%}")
    cards[3].metric("No-news sector-days", f"{no_news_days:,}")

    selected_sectors = st.multiselect(
        "Sectors shown",
        options=sectors,
        default=[item for item in ["Tech", "Healthcare", "Materials"] if item in sectors],
        key="sentiment_sectors",
    )
    if selected_sectors:
        rolling = rolling_sector_sentiment(sentiment, selected_sectors).dropna(
            subset=["sentiment_21d"]
        )
        line = (
            alt.Chart(rolling)
            .mark_line(strokeWidth=2.0)
            .encode(
                x=alt.X("date:T", title="Equity trading date"),
                y=alt.Y("sentiment_21d:Q", title="21-trading-day mean VADER compound"),
                color=alt.Color("sector:N", title="Sector"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("sector:N", title="Sector"),
                    alt.Tooltip("sentiment_21d:Q", title="21-day mean", format=".3f"),
                ],
            )
        )
        zero = alt.Chart(pd.DataFrame({"zero": [0.0]})).mark_rule(
            color=SLATE, strokeDash=[4, 4]
        ).encode(y="zero:Q")
        st.altair_chart(_finish_chart(line + zero, 380), width="stretch")
    else:
        st.warning("Select at least one sector to display the sentiment index.")

    coverage_col, method_col = st.columns([1.0, 1.0])
    with coverage_col:
        st.markdown("#### News coverage by sector")
        coverage_chart = (
            alt.Chart(summary)
            .mark_bar(color=BLUE, cornerRadiusEnd=4)
            .encode(
                y=alt.Y("sector:N", sort="-x", title=None),
                x=alt.X(
                    "average_ticker_coverage:Q",
                    title="Mean share of sector tickers with news",
                    axis=alt.Axis(format=".0%"),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=[
                    alt.Tooltip("sector:N", title="Sector"),
                    alt.Tooltip("average_ticker_coverage:Q", title="Ticker coverage", format=".1%"),
                    alt.Tooltip("total_headlines:Q", title="Headlines", format=","),
                    alt.Tooltip("days_without_news:Q", title="No-news days", format=","),
                ],
            )
        )
        st.altair_chart(_finish_chart(coverage_chart, 330), width="stretch")
    with method_col:
        st.markdown("#### Why equal-ticker aggregation matters")
        st.markdown(
            "Each headline is scored first, then averaged within a **ticker-day**, and only "
            "then averaged equally across observed tickers in a sector. This prevents a "
            "high-volume company from mechanically becoming the sector signal."
        )
        st.markdown(
            "No-news observations remain **missing rather than neutral**. The tradeable "
            "signal uses the preceding equity trading day, so same-day news cannot leak "
            "into a rebalance decision."
        )
        st.caption("Text casing, punctuation, negation, and intensifiers are preserved for VADER.")

    st.divider()
    st.markdown("### AtlasSignal coverage-adjusted fusion")
    st.markdown(
        "The innovation reduces a sector tilt when its recent news is concentrated in a few "
        "tickers or when ticker coverage is thin. It modifies the confidence in sentiment, "
        "not the direction of the sentiment score."
    )
    st.latex(
        r"Confidence_{s,t}=(1-\overline{HHI}^{\,(3\ completed\ months)}_{s,t})"
        r"\sqrt{Coverage^{\,(21d)}_{s,t}}"
    )
    family_label = st.radio(
        "Fusion fund family",
        options=["Equity", "Combined Equity + Crypto"],
        horizontal=True,
        key="fusion_family",
    )
    family = "equity" if family_label == "Equity" else "combined"
    comparison = fusion.loc[fusion["asset_family"].eq(family)].copy()
    base = comparison.loc[comparison["fusion_variant"].eq("base")].iloc[0]
    naive = comparison.loc[comparison["fusion_variant"].eq("sentiment")].iloc[0]
    adjusted = comparison.loc[comparison["fusion_variant"].eq("coverage_adjusted")].iloc[0]
    fusion_cards = st.columns(4)
    fusion_cards[0].metric("Base Sharpe", f"{base['sharpe_ratio']:.3f}")
    fusion_cards[1].metric(
        "Coverage-adjusted Sharpe",
        f"{adjusted['sharpe_ratio']:.3f}",
        f"{adjusted['delta_sharpe_ratio_vs_base']:+.3f} vs base",
        delta_color="normal",
    )
    fusion_cards[2].metric(
        "Naive sentiment Sharpe",
        f"{naive['sharpe_ratio']:.3f}",
        f"{naive['delta_sharpe_ratio_vs_base']:+.3f} vs base",
        delta_color="normal",
    )
    turnover_reduction = naive["annualized_turnover"] - adjusted["annualized_turnover"]
    fusion_cards[3].metric(
        "Turnover saved vs naive",
        f"{turnover_reduction:.1%}",
        "annualised one-way turnover",
        delta_color="off",
    )

    selected_fusion_returns = fusion_returns.loc[
        fusion_returns["asset_family"].eq(family)
    ].copy()
    growth_col, robust_col = st.columns([1.08, 0.92])
    with growth_col:
        st.markdown("#### Base versus sentiment variants")
        growth = (
            alt.Chart(selected_fusion_returns)
            .mark_line(strokeWidth=2.2)
            .encode(
                x=alt.X("date:T", title="OOS date"),
                y=alt.Y("growth_1:Q", title="Portfolio value ($)", scale=alt.Scale(zero=True)),
                color=_variant_color(),
                strokeDash=alt.StrokeDash("fusion_variant_label:N", sort=VARIANT_ORDER, title=None),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("fusion_variant_label:N", title="Variant"),
                    alt.Tooltip("growth_1:Q", title="Value", format="$.3f"),
                ],
            )
        )
        st.altair_chart(_finish_chart(growth, 340), width="stretch")
    with robust_col:
        st.markdown("#### Sharpe robustness")
        robust = robustness.loc[
            robustness["asset_family"].eq(family)
            & robustness["fusion_variant"].ne("base")
        ].copy()
        robust["tilt_percent"] = robust["tilt_strength"] * 100.0
        robust["cost_label"] = robust["transaction_cost_bps"].map(
            lambda value: f"{value:.0f} bps"
        )
        robust_chart = (
            alt.Chart(robust)
            .mark_line(point=True, strokeWidth=2.0)
            .encode(
                x=alt.X("tilt_percent:Q", title="Maximum relative sector tilt (%)"),
                y=alt.Y("delta_sharpe_vs_base:Q", title="Sharpe change vs matching base"),
                color=_variant_color(),
                strokeDash=alt.StrokeDash("cost_label:N", title="Transaction cost"),
                tooltip=[
                    alt.Tooltip("fusion_variant_label:N", title="Variant"),
                    alt.Tooltip("tilt_percent:Q", title="Tilt", format=".0f"),
                    alt.Tooltip("cost_label:N", title="Cost"),
                    alt.Tooltip("delta_sharpe_vs_base:Q", title="Sharpe change", format="+.4f"),
                ],
            )
        )
        zero = alt.Chart(pd.DataFrame({"zero": [0.0]})).mark_rule(
            color=SLATE, strokeDash=[4, 4]
        ).encode(y="zero:Q")
        st.altair_chart(_finish_chart(robust_chart + zero, 340), width="stretch")

    st.markdown(
        '<div class="risk-note"><b>Negative result retained.</b> Neither sentiment variant '
        "beats the matching price-only Risk Parity base. Coverage adjustment consistently "
        "reduces the Sharpe loss and turnover relative to naive sentiment, so it acts as a "
        "confidence control—not demonstrated alpha.</div>",
        unsafe_allow_html=True,
    )
    fusion_table = comparison[
        [
            "fusion_variant_label",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "annualized_turnover",
        ]
    ].copy()
    fusion_table.columns = [
        "Variant",
        "Annualised return",
        "Annualised volatility",
        "Sharpe",
        "Maximum drawdown",
        "Annualised turnover",
    ]
    st.dataframe(
        fusion_table,
        hide_index=True,
        width="stretch",
        column_config={
            "Annualised return": st.column_config.NumberColumn(format="percent"),
            "Annualised volatility": st.column_config.NumberColumn(format="percent"),
            "Sharpe": st.column_config.NumberColumn(format="%.3f"),
            "Maximum drawdown": st.column_config.NumberColumn(format="percent"),
            "Annualised turnover": st.column_config.NumberColumn(format="percent"),
        },
    )


try:
    app_data = _load_snapshot()
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error("AtlasSignal could not open its validated result snapshot.")
    st.code(str(error))
    st.info("From the repository root, run: python scripts/run_part_b.py")
    st.stop()

with st.sidebar:
    st.markdown('<div class="sidebar-mark">FINS5545 · Part B</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">ATLAS / SIGNAL</div>', unsafe_allow_html=True)
    st.caption("Systematic multi-asset funds with news-confidence analytics.")
    st.divider()
    st.markdown("**Investor journey**")
    st.markdown("1. Compare the fund shelf")
    st.markdown("2. Read one fact sheet")
    st.markdown("3. Set a fund allocation")
    st.markdown("4. Test the news signal")
    st.divider()
    st.success("Validated snapshot loaded")
    st.caption(
        "Prices and OOS fund results through Dec 2023. Backtest model targets as of "
        "1 Dec 2023; not live holdings."
    )
    with st.expander("Method guardrails"):
        st.markdown(
            "- Walk-forward OOS backtests\n"
            "- Information strictly before rebalance\n"
            "- Monthly target formation\n"
            "- Long-only and fully invested\n"
            "- 252 equity / 365 crypto annualisation\n"
            "- 0% risk-free rate; 0 bps baseline"
        )

st.markdown(
    """
    <div class="atlas-hero">
      <div class="atlas-kicker">Systematic funds · transparent evidence</div>
      <h1>Invest by risk profile, not by headline return.</h1>
      <p>Compare twelve walk-forward funds, inspect the latest OOS model targets, build a
      fund-level allocation, and see why a coverage-aware news signal remains
      experimental rather than marketed as alpha.</p>
      <div class="journey">
        <span>01 Compare</span><span>02 Fact sheet</span><span>03 Allocate</span><span>04 News confidence</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_compare, tab_fact_sheet, tab_builder, tab_news = st.tabs(
    ["Compare Funds", "Fund Fact Sheet", "Allocation Lab", "News & Innovation"]
)
with tab_compare:
    _render_compare(app_data)
with tab_fact_sheet:
    _render_fact_sheet(app_data)
with tab_builder:
    _render_builder(app_data)
with tab_news:
    _render_news(app_data)

st.markdown(
    '<div class="footer"><b>AtlasSignal research prototype.</b> Historical OOS '
    "results use the supplied FINS5545 data and are not personal financial advice, "
    "a live offer, or a guarantee of future performance.</div>",
    unsafe_allow_html=True,
)
