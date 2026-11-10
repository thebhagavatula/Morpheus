"""
Matrix Dashboard — Cross-Sectional Expected Return Model

A Streamlit dashboard with a Matrix (1999) inspired interface.
Visualises factor premia, portfolio performance, IC analysis,
and regime diagnostics — all rendered in glowing green on black.

Launch:
    streamlit run src/dashboard/app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime

# Path setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dashboard.matrix_theme import (
    STREAMLIT_CSS, MATRIX_GREEN, MATRIX_BLACK, MATRIX_BG, MATRIX_CARD,
    MATRIX_DIM, MATRIX_DARK, MATRIX_AMBER, MATRIX_RED, MATRIX_TEXT,
    MATRIX_MUTED, CHART_COLORS, get_plotly_template, _rain_html,
    live_clock_html,
)
from src.data.universe import SUPPORTED_INDEXES

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="MORPHEUS // Factor Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Matrix CSS + rain
st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)
st.markdown(_rain_html(25), unsafe_allow_html=True)
st.markdown('<div class="scanlines"></div>', unsafe_allow_html=True)

TEMPLATE = get_plotly_template()
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def matrix_fig(**extra_layout) -> go.Figure:
    """Create a Plotly figure with explicit Matrix dark backgrounds and styling."""
    layout = dict(
        paper_bgcolor=MATRIX_BLACK,
        plot_bgcolor=MATRIX_BG,
        font=dict(family="Share Tech Mono, Courier New, monospace", color=MATRIX_TEXT, size=12),
        title=dict(font=dict(family="Orbitron, sans-serif", color=MATRIX_GREEN, size=16)),
        xaxis=dict(gridcolor="#1a1a1a", zerolinecolor=MATRIX_DARK, tickfont=dict(color=MATRIX_DIM)),
        yaxis=dict(gridcolor="#1a1a1a", zerolinecolor=MATRIX_DARK, tickfont=dict(color=MATRIX_DIM)),
        legend=dict(bgcolor="rgba(0,0,0,0.6)", bordercolor=MATRIX_DARK, font=dict(color=MATRIX_TEXT)),
        colorway=CHART_COLORS,
    )
    layout.update(extra_layout)
    return go.Figure(layout=layout)


# ══════════════════════════════════════════════════════════════════════
# INDEX-SPECIFIC PROFILES  (realistic per-market characteristics)
# ══════════════════════════════════════════════════════════════════════

_INDEX_PROFILES = {
    "S&P 500": {
        "seed": 42,
        "factors": ["momentum", "low_volatility", "liquidity", "value",
                    "quality", "earnings_surprise"],
        "short_factors": ["momentum", "low_vol", "value", "quality",
                          "liquidity", "esm"],
        "mean_ret": 0.008, "vol": 0.035,
        "date_start": "2015-01-01", "n_periods": 96,
        "currency": "USD",
    },
    "NIFTY 50": {
        "seed": 100,
        "factors": ["momentum", "low_volatility", "liquidity", "value",
                    "quality", "promoter_holding"],
        "short_factors": ["momentum", "low_vol", "value", "quality",
                          "liquidity", "promoter"],
        "mean_ret": 0.011, "vol": 0.042,
        "date_start": "2015-01-01", "n_periods": 96,
        "currency": "INR",
    },
    "NIFTY NEXT 50": {
        "seed": 200,
        "factors": ["momentum", "low_volatility", "liquidity", "value",
                    "quality", "promoter_holding"],
        "short_factors": ["momentum", "low_vol", "value", "quality",
                          "liquidity", "promoter"],
        "mean_ret": 0.013, "vol": 0.052,
        "date_start": "2015-01-01", "n_periods": 96,
        "currency": "INR",
    },
}


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING  — parameterised by index so cache invalidates on switch
# ══════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(index_name: str = "S&P 500"):
    """Load real pipeline output or generate index-specific synthetic demo data."""
    profile = _INDEX_PROFILES.get(index_name, _INDEX_PROFILES["S&P 500"])
    data = {"index_name": index_name, "profile": profile}

    # Try loading real pipeline outputs first
    premia_path = REPORTS_DIR / "factor_premia_summary.csv"
    if premia_path.exists():
        data["premia"] = pd.read_csv(premia_path, index_col=0)
    else:
        data["premia"] = _synthetic_premia(profile)

    ic_path = REPORTS_DIR / "ic_series.csv"
    if ic_path.exists():
        data["ic"] = pd.read_csv(ic_path, index_col=0, parse_dates=True)
    else:
        data["ic"] = _synthetic_ic(profile)

    for name in ["gross_performance", "net_performance"]:
        path = REPORTS_DIR / f"{name}.csv"
        if path.exists():
            data[name] = pd.read_csv(path)
        else:
            data[name] = _synthetic_performance(profile, name)

    data["returns"] = _synthetic_returns(profile)
    data["lambda_series"] = _synthetic_lambda_series(profile)

    return data


def _synthetic_premia(p):
    rng = np.random.RandomState(p["seed"])
    factors = p["factors"]
    return pd.DataFrame({
        "mean_lambda": rng.uniform(0.001, 0.025, len(factors)),
        "t_stat": rng.uniform(0.5, 3.5, len(factors)),
        "nw_t_stat": rng.uniform(0.4, 3.2, len(factors)),
        "std_error": rng.uniform(0.002, 0.008, len(factors)),
        "nw_std_error": rng.uniform(0.003, 0.010, len(factors)),
        "n_periods": [p["n_periods"]] * len(factors),
    }, index=factors)


def _synthetic_ic(p):
    rng = np.random.RandomState(p["seed"] + 1)
    dates = pd.date_range(p["date_start"], periods=p["n_periods"], freq="ME")
    return pd.DataFrame({"IC": rng.normal(0.04, 0.08, len(dates))}, index=dates)


def _synthetic_performance(p, name):
    rng = np.random.RandomState(p["seed"] + (2 if "gross" in name else 3))
    return pd.DataFrame([{
        "total_return": rng.uniform(0.3, 1.2),
        "annualized_return": rng.uniform(0.04, 0.15),
        "annualized_volatility": rng.uniform(0.08, 0.20),
        "sharpe_ratio": rng.uniform(0.3, 1.5),
        "sortino_ratio": rng.uniform(0.5, 2.0),
        "max_drawdown": rng.uniform(-0.40, -0.05),
        "n_periods": p["n_periods"],
    }])


def _synthetic_returns(p):
    rng = np.random.RandomState(p["seed"] + 4)
    dates = pd.date_range(p["date_start"], periods=p["n_periods"], freq="ME")
    gross = pd.Series(
        rng.normal(p["mean_ret"], p["vol"], p["n_periods"]),
        index=dates, name="gross",
    )
    net = gross - 0.001
    return pd.DataFrame({"gross": gross, "net": net})


def _synthetic_lambda_series(p):
    rng = np.random.RandomState(p["seed"] + 5)
    dates = pd.date_range(p["date_start"], periods=p["n_periods"], freq="ME")
    data = {}
    for f in p["short_factors"]:
        data[f] = rng.normal(0.01, 0.015, p["n_periods"])
    return pd.DataFrame(data, index=dates)



# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown(
            f'<h2 style="font-family: Orbitron; color: {MATRIX_GREEN}; '
            f'text-align: center; font-size: 1.1rem; letter-spacing: 3px;'
            f'margin-bottom: 0;">'
            f'MORPHEUS</h2>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="text-align: center; color: {MATRIX_DIM}; '
            f'font-size: 0.8rem; letter-spacing: 2px; margin-top: 2px;">'
            f'FACTOR MODEL v2.0</p>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Live JS Clock (syncs seconds) ─────────────────────────────
        components.html(live_clock_html(), height=55)

        # ── System Status ─────────────────────────────────────────────
        has_data = (REPORTS_DIR / "factor_premia_summary.csv").exists()
        data_status = "LIVE DATA" if has_data else "DEMO MODE"
        status_icon = "⬤" if has_data else "◯"
        data_color = MATRIX_GREEN if has_data else MATRIX_AMBER
        st.markdown(
            f'<p style="color: {data_color}; font-size: 0.85rem; '
            f'text-align: center; letter-spacing: 1px; margin: 4px 0;">'
            f'{status_icon} {data_status}</p>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Index Selector ────────────────────────────────────────────
        index_options = list(SUPPORTED_INDEXES.keys())
        selected_index = st.selectbox(
            "▸ UNIVERSE",
            options=index_options,
            index=0,
            help="Select the stock index universe for the model",
        )

        # Show index description
        idx_info = SUPPORTED_INDEXES.get(selected_index, {})
        st.markdown(
            f'<p style="color: {MATRIX_DIM}; font-size: 0.8rem; '
            f'text-align: center; margin-top: -8px;">'
            f'{idx_info.get("description", "")}</p>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Navigation ────────────────────────────────────────────────
        page = st.radio(
            "▸ NAVIGATION",
            ["Overview", "Factor Premia", "IC Analysis",
             "Portfolio", "Regime"],
            label_visibility="visible",
        )

        st.markdown("---")
        st.markdown(
            f'<p style="color: {MATRIX_MUTED}; font-size: 0.75rem; '
            f'text-align: center; letter-spacing: 1px;">'
            f'THE MATRIX HAS YOU</p>',
            unsafe_allow_html=True,
        )

        return page, selected_index


# ══════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════

def page_overview(data, selected_index="S&P 500"):
    st.markdown(
        f'<h1 style="text-align: center;">⬡ CROSS-SECTIONAL EXPECTED RETURN MODEL</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="text-align: center; color: {MATRIX_DIM}; letter-spacing: 2px; '
        f'font-size: 0.85rem;">FAMA-MACBETH FACTOR PREMIA ESTIMATION // {selected_index}</p>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row
    gross = data["gross_performance"].iloc[0] if not data["gross_performance"].empty else {}
    net = data["net_performance"].iloc[0] if not data["net_performance"].empty else {}

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Sharpe (Net)", f"{net.get('sharpe_ratio', 0):.2f}")
    with c2:
        st.metric("Ann. Return", f"{net.get('annualized_return', 0):.1%}")
    with c3:
        st.metric("Ann. Vol", f"{net.get('annualized_volatility', 0):.1%}")
    with c4:
        st.metric("Max DD", f"{net.get('max_drawdown', 0):.1%}")
    with c5:
        st.metric("Sortino", f"{net.get('sortino_ratio', 0):.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Cumulative return chart
    ret = data["returns"]
    cum_gross = (1 + ret["gross"]).cumprod()
    cum_net = (1 + ret["net"]).cumprod()

    fig = matrix_fig()
    fig.add_trace(go.Scatter(
        x=cum_gross.index, y=cum_gross.values,
        name="Gross", line=dict(color=MATRIX_GREEN, width=2),
        fill='tozeroy', fillcolor=f'rgba(0, 255, 65, 0.05)',
    ))
    fig.add_trace(go.Scatter(
        x=cum_net.index, y=cum_net.values,
        name="Net", line=dict(color=MATRIX_AMBER, width=2, dash='dot'),
    ))
    fig.update_layout(
        title="CUMULATIVE PORTFOLIO RETURN",
        height=400, margin=dict(l=40, r=20, t=50, b=30),
        yaxis_title="Growth of $1",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Factor premia table
    st.markdown(f'### λ FACTOR PREMIA SUMMARY')
    premia = data["premia"].copy()
    premia.index.name = "FACTOR"

    # Style the DataFrame
    def color_t(val):
        if abs(val) >= 1.96:
            return f"color: {MATRIX_GREEN}; font-weight: bold;"
        elif abs(val) >= 1.64:
            return f"color: {MATRIX_AMBER};"
        return f"color: {MATRIX_RED};"

    styled = premia.style.map(color_t, subset=["t_stat", "nw_t_stat"]).format({
        "mean_lambda": "{:.4f}",
        "t_stat": "{:.2f}",
        "nw_t_stat": "{:.2f}",
        "std_error": "{:.4f}",
        "nw_std_error": "{:.4f}",
        "n_periods": "{:.0f}",
    })
    st.dataframe(styled, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: FACTOR PREMIA
# ══════════════════════════════════════════════════════════════════════

def page_factor_premia(data, selected_index="S&P 500"):
    st.markdown(f'<h1>λ FACTOR PREMIA</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color: {MATRIX_DIM}; letter-spacing: 2px; font-size: 0.8rem; '
        f'text-align: center; margin-top: -10px;">{selected_index} // CROSS-SECTIONAL LAMBDAS</p>',
        unsafe_allow_html=True,
    )


    lam = data["lambda_series"]

    # Rolling premia
    rolled = lam.rolling(12, min_periods=3).mean()

    fig = matrix_fig()
    for i, col in enumerate(rolled.columns):
        fig.add_trace(go.Scatter(
            x=rolled.index, y=rolled[col],
            name=col.upper(), line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.8),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color=MATRIX_DARK)
    fig.update_layout(
        title="12-MONTH ROLLING FACTOR PREMIA (λ)",
        height=450, margin=dict(l=40, r=20, t=50, b=30),
        yaxis_title="λ (return per unit exposure)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bar chart: mean premia with significance
    premia = data["premia"]
    fig2 = matrix_fig()
    colors = [MATRIX_GREEN if abs(t) >= 1.96 else MATRIX_AMBER if abs(t) >= 1.64 else MATRIX_RED
              for t in premia["nw_t_stat"]]

    fig2.add_trace(go.Bar(
        x=premia.index, y=premia["mean_lambda"],
        marker_color=colors,
        text=[f't={t:.2f}' for t in premia["nw_t_stat"]],
        textposition="outside",
        textfont=dict(color=MATRIX_DIM, size=10),
    ))
    fig2.update_layout(
        title="MEAN FACTOR PREMIA (NW t-stat annotated)",
        height=400, margin=dict(l=40, r=20, t=50, b=30),
        yaxis_title="Mean λ",
        xaxis_title="Factor",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Significance legend
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<p style="color: {MATRIX_GREEN};">⬤ |t| ≥ 1.96 — SIGNIFICANT</p>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<p style="color: {MATRIX_AMBER};">⬤ |t| ≥ 1.64 — MARGINAL</p>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<p style="color: {MATRIX_RED};">⬤ |t| < 1.64 — INSIGNIFICANT</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: IC ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def page_ic_analysis(data, selected_index="S&P 500"):
    st.markdown(f'<h1>◎ INFORMATION COEFFICIENT</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color: {MATRIX_DIM}; letter-spacing: 2px; font-size: 0.8rem; '
        f'text-align: center; margin-top: -10px;">{selected_index} // RANK IC ANALYSIS</p>',
        unsafe_allow_html=True,
    )


    ic = data["ic"]
    if "IC" in ic.columns:
        ic_vals = ic["IC"]
    else:
        ic_vals = ic.iloc[:, 0]

    # IC stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Mean IC", f"{ic_vals.mean():.4f}")
    with c2:
        ic_ir = ic_vals.mean() / ic_vals.std() if ic_vals.std() > 0 else 0
        st.metric("IC IR", f"{ic_ir:.2f}")
    with c3:
        st.metric("% Positive", f"{(ic_vals > 0).mean():.1%}")
    with c4:
        st.metric("Periods", f"{len(ic_vals)}")

    # IC time series
    fig = matrix_fig()
    colors = [MATRIX_GREEN if v > 0 else MATRIX_RED for v in ic_vals]
    fig.add_trace(go.Bar(
        x=ic_vals.index, y=ic_vals.values,
        marker_color=colors, opacity=0.8,
    ))
    # Rolling mean
    ic_roll = ic_vals.rolling(12, min_periods=3).mean()
    fig.add_trace(go.Scatter(
        x=ic_roll.index, y=ic_roll.values,
        name="12M Rolling IC", line=dict(color=MATRIX_AMBER, width=2.5),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=MATRIX_DARK)
    fig.update_layout(
        title="MONTHLY INFORMATION COEFFICIENT",
        height=400, margin=dict(l=40, r=20, t=50, b=30),
        yaxis_title="IC (rank correlation)",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # IC distribution
    fig2 = matrix_fig()
    fig2.add_trace(go.Histogram(
        x=ic_vals.values, nbinsx=25,
        marker_color=MATRIX_GREEN, opacity=0.7,
        marker_line=dict(color=MATRIX_DIM, width=1),
    ))
    fig2.add_vline(x=0, line_dash="dash", line_color=MATRIX_RED)
    fig2.add_vline(x=ic_vals.mean(), line_dash="solid", line_color=MATRIX_AMBER,
                   annotation_text=f"Mean: {ic_vals.mean():.4f}")
    fig2.update_layout(
        title="IC DISTRIBUTION",
        height=350, margin=dict(l=40, r=20, t=50, b=30),
        xaxis_title="IC", yaxis_title="Frequency",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO
# ══════════════════════════════════════════════════════════════════════

def page_portfolio(data, selected_index="S&P 500"):
    st.markdown(f'<h1>◈ PORTFOLIO ANALYTICS</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color: {MATRIX_DIM}; letter-spacing: 2px; font-size: 0.8rem; '
        f'text-align: center; margin-top: -10px;">{selected_index} // BACKTEST METRICS</p>',
        unsafe_allow_html=True,
    )


    ret = data["returns"]

    col1, col2 = st.columns(2)

    # Drawdown
    with col1:
        cum = (1 + ret["net"]).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak

        fig = matrix_fig()
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            fill='tozeroy', fillcolor=f'rgba(255, 32, 64, 0.15)',
            line=dict(color=MATRIX_RED, width=1.5), name="Drawdown",
        ))
        fig.update_layout(
            title="DRAWDOWN", height=350,
            margin=dict(l=40, r=20, t=50, b=30),
            yaxis_title="Drawdown", yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Rolling Sharpe
    with col2:
        rolling_sr = ret["net"].rolling(24, min_periods=12).apply(
            lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0
        )
        fig = matrix_fig()
        fig.add_trace(go.Scatter(
            x=rolling_sr.index, y=rolling_sr.values,
            line=dict(color=MATRIX_GREEN, width=2), name="Rolling Sharpe",
            fill='tozeroy', fillcolor=f'rgba(0, 255, 65, 0.05)',
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color=MATRIX_AMBER,
                      annotation_text="SR = 1.0")
        fig.update_layout(
            title="24M ROLLING SHARPE RATIO", height=350,
            margin=dict(l=40, r=20, t=50, b=30),
            yaxis_title="Sharpe Ratio",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Monthly returns heatmap
    st.markdown(f'### MONTHLY RETURN MATRIX')
    monthly = ret["net"].copy()
    monthly.index = pd.to_datetime(monthly.index)
    years = monthly.index.year
    months = monthly.index.month

    pivot = pd.DataFrame({
        "Year": years, "Month": months, "Return": monthly.values
    }).pivot_table(index="Year", columns="Month", values="Return", aggfunc="mean")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

    fig = matrix_fig()
    fig.add_trace(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[
            [0, '#b3142c'], [0.3, '#2a0d12'],
            [0.5, MATRIX_BLACK], [0.7, '#0b3d12'],
            [1.0, '#007a1f'],
        ],
        zmin=-0.08, zmax=0.08,
        text=np.vectorize(lambda x: f"{x:.1%}" if not np.isnan(x) else "")(pivot.values),
        texttemplate="%{text}",
        textfont=dict(size=12, color='#ffffff'),
    ))
    fig.update_layout(
        title="MONTHLY RETURNS HEATMAP",
        height=300, margin=dict(l=40, r=20, t=50, b=30),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: REGIME
# ══════════════════════════════════════════════════════════════════════

def page_regime(data, selected_index="S&P 500"):
    st.markdown(f'<h1>⚡ REGIME ANALYSIS</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color: {MATRIX_DIM}; letter-spacing: 2px; font-size: 0.8rem; '
        f'text-align: center; margin-top: -10px;">{selected_index} // MARKET REGIME DECOMPOSITION</p>',
        unsafe_allow_html=True,
    )


    ret = data["returns"]["net"]

    # Define crisis periods
    crises = [
        ("EU Debt Crisis", "2011-08-01", "2011-12-31"),
        ("China Devaluation", "2015-08-01", "2016-02-28"),
        ("Vol-mageddon", "2018-10-01", "2018-12-31"),
        ("COVID Crash", "2020-02-01", "2020-04-30"),
        ("Rate Hiking", "2022-01-01", "2022-06-30"),
    ]

    crisis_mask = pd.Series(False, index=ret.index)
    for _, start, end in crises:
        crisis_mask |= (ret.index >= start) & (ret.index <= end)

    crisis_ret = ret[crisis_mask]
    normal_ret = ret[~crisis_mask]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Normal Sharpe",
                   f"{normal_ret.mean() / normal_ret.std() * np.sqrt(12):.2f}" if normal_ret.std() > 0 else "N/A")
    with col2:
        st.metric("Crisis Sharpe",
                   f"{crisis_ret.mean() / crisis_ret.std() * np.sqrt(12):.2f}" if len(crisis_ret) > 1 and crisis_ret.std() > 0 else "N/A")
    with col3:
        st.metric("Normal Mean", f"{normal_ret.mean() * 12:.1%}")
    with col4:
        st.metric("Crisis Mean", f"{crisis_ret.mean() * 12:.1%}" if len(crisis_ret) > 0 else "N/A")

    # Distribution comparison
    fig = matrix_fig()
    if not normal_ret.empty:
        fig.add_trace(go.Histogram(
            x=normal_ret.values, name="Normal", nbinsx=20,
            marker_color=MATRIX_GREEN, opacity=0.6,
        ))
    if not crisis_ret.empty:
        fig.add_trace(go.Histogram(
            x=crisis_ret.values, name="Crisis", nbinsx=15,
            marker_color=MATRIX_RED, opacity=0.6,
        ))
    fig.update_layout(
        title="RETURN DISTRIBUTION: CRISIS vs NORMAL",
        height=400, margin=dict(l=40, r=20, t=50, b=30),
        barmode="overlay", xaxis_title="Monthly Return",
        yaxis_title="Frequency",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rolling volatility with crisis shading
    rolling_vol = ret.rolling(12, min_periods=6).std() * np.sqrt(12)

    fig2 = matrix_fig()
    fig2.add_trace(go.Scatter(
        x=rolling_vol.index, y=rolling_vol.values,
        line=dict(color=MATRIX_GREEN, width=2), name="12M Rolling Vol",
        fill='tozeroy', fillcolor=f'rgba(0, 255, 65, 0.05)',
    ))

    for name, start, end in crises:
        fig2.add_vrect(
            x0=start, x1=end,
            fillcolor=f"rgba(255, 32, 64, 0.1)",
            line=dict(color=MATRIX_RED, width=0.5),
            annotation_text=name, annotation_position="top left",
            annotation_font=dict(color=MATRIX_RED, size=9),
        )

    fig2.update_layout(
        title="ROLLING VOLATILITY WITH CRISIS PERIODS",
        height=400, margin=dict(l=40, r=20, t=50, b=30),
        yaxis_title="Annualised Volatility", yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Crisis detail table
    st.markdown("### CRISIS PERIOD BREAKDOWN")
    crisis_data = []
    for name, start, end in crises:
        mask = (ret.index >= start) & (ret.index <= end)
        period_ret = ret[mask]
        if len(period_ret) > 0:
            crisis_data.append({
                "Period": name,
                "Start": start,
                "End": end,
                "Cum. Return": f"{(1 + period_ret).prod() - 1:.1%}",
                "Worst Month": f"{period_ret.min():.1%}",
                "N Months": len(period_ret),
            })

    if crisis_data:
        st.dataframe(pd.DataFrame(crisis_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    page, selected_index = render_sidebar()

    # Load data keyed by selected index — cache invalidates on switch
    data = load_data(selected_index)
    st.session_state["selected_index"] = selected_index

    if page == "Overview":
        page_overview(data, selected_index)
    elif page == "Factor Premia":
        page_factor_premia(data, selected_index)
    elif page == "IC Analysis":
        page_ic_analysis(data, selected_index)
    elif page == "Portfolio":
        page_portfolio(data, selected_index)
    elif page == "Regime":
        page_regime(data, selected_index)


if __name__ == "__main__":
    main()

