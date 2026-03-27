"""
Research-Grade Visualizations

All plots use the IB_PALETTE (old-money investment banking aesthetic)
defined in config.py.

Generates:
  - Rolling factor premia plots
  - IC heatmap
  - Factor correlation heatmap (before/after orthogonalization)
  - Decile spread performance
  - Drawdown distribution
  - Crisis vs. normal regime comparison
  - Rolling t-stat chart
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CI/server
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import logging
import sys
from pathlib import Path

try:
    from src.config import IB_PALETTE, CACHE_DIR, BASE_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config import IB_PALETTE, CACHE_DIR, BASE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global style
REPORTS_DIR = BASE_DIR / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

_COLORS = list(IB_PALETTE.values())
_BG = "#FAF8F5"  # warm off-white
_GRID = "#D4CFC7"

def _apply_style(ax, title=""):
    """Apply the old-money IB aesthetic to an axis."""
    ax.set_facecolor(_BG)
    ax.figure.set_facecolor(_BG)
    ax.set_title(title, fontsize=14, fontweight="bold", color=IB_PALETTE["navy_blue"], pad=12)
    ax.tick_params(colors=IB_PALETTE["dark_grey"], labelsize=9)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color(_GRID)


def plot_rolling_factor_premia(
    lambda_series: pd.DataFrame,
    rolling_window: int = 12,
    save: bool = True,
) -> plt.Figure:
    """Rolling mean of estimated factor premia over time."""
    cols = [c for c in lambda_series.columns if c != "intercept"]
    rolled = lambda_series[cols].rolling(rolling_window, min_periods=3).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, col in enumerate(cols):
        ax.plot(rolled.index, rolled[col], label=col, color=_COLORS[i % len(_COLORS)], linewidth=1.5)

    ax.axhline(0, color=IB_PALETTE["dark_grey"], linewidth=0.8, linestyle="--")
    ax.legend(fontsize=9, framealpha=0.8)
    _apply_style(ax, "Rolling Factor Premia (λ)")
    ax.set_ylabel("λ (mean return per unit exposure)", fontsize=10, color=IB_PALETTE["dark_grey"])

    plt.tight_layout()
    if save:
        fig.savefig(REPORTS_DIR / "rolling_factor_premia.png", dpi=150, bbox_inches="tight")
    return fig


def plot_ic_heatmap(
    ic_dict: dict[str, pd.Series],
    save: bool = True,
) -> plt.Figure:
    """Heatmap of monthly IC for each factor."""
    ic_df = pd.DataFrame(ic_dict)

    fig, ax = plt.subplots(figsize=(14, max(4, len(ic_df) * 0.15)))
    sns.heatmap(
        ic_df.T, cmap="RdYlGn", center=0, linewidths=0.3,
        annot=False, cbar_kws={"shrink": 0.6, "label": "IC"}, ax=ax,
    )
    _apply_style(ax, "Information Coefficient Heatmap")

    plt.tight_layout()
    if save:
        fig.savefig(REPORTS_DIR / "ic_heatmap.png", dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Factor Correlation Matrix",
    filename: str = "factor_correlation.png",
    save: bool = True,
) -> plt.Figure:
    """Factor correlation heatmap with annotations."""
    fig, ax = plt.subplots(figsize=(8, 6))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, linewidths=0.5,
        cbar_kws={"shrink": 0.7}, ax=ax,
        vmin=-1, vmax=1,
    )
    _apply_style(ax, title)

    plt.tight_layout()
    if save:
        fig.savefig(REPORTS_DIR / filename, dpi=150, bbox_inches="tight")
    return fig


def plot_decile_performance(
    decile_returns: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Cumulative return by decile + long-short spread."""
    cum_ret = (1 + decile_returns).cumprod()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Cumulative returns by decile
    ax = axes[0]
    for col in cum_ret.columns:
        color_idx = int(col) - 1 if isinstance(col, (int, float)) else 0
        alpha = 0.3 + 0.7 * (color_idx / max(len(cum_ret.columns) - 1, 1))
        ax.plot(cum_ret.index, cum_ret[col], label=f"D{int(col)}",
                color=IB_PALETTE["navy_blue"], alpha=alpha, linewidth=1.2)
    ax.legend(fontsize=8, ncol=2, framealpha=0.8)
    _apply_style(ax, "Cumulative Return by Decile")
    ax.set_ylabel("Cumulative Return", fontsize=10)

    # Panel 2: Long-short spread
    ax2 = axes[1]
    top = cum_ret.columns.max()
    bottom = cum_ret.columns.min()
    spread = cum_ret[top] - cum_ret[bottom]
    ax2.plot(spread.index, spread, color=IB_PALETTE["dark_green"], linewidth=2)
    ax2.fill_between(spread.index, 0, spread, alpha=0.15, color=IB_PALETTE["dark_green"])
    _apply_style(ax2, f"Long-Short Spread (D{int(top)} − D{int(bottom)})")

    plt.tight_layout()
    if save:
        fig.savefig(REPORTS_DIR / "decile_performance.png", dpi=150, bbox_inches="tight")
    return fig


def plot_drawdown(
    returns: pd.Series,
    title: str = "Portfolio Drawdown",
    save: bool = True,
) -> plt.Figure:
    """Drawdown chart over time."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(dd.index, dd.values, 0, color=IB_PALETTE["burgundy"], alpha=0.4)
    ax.plot(dd.index, dd.values, color=IB_PALETTE["burgundy"], linewidth=1)
    _apply_style(ax, title)
    ax.set_ylabel("Drawdown", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    if save:
        fig.savefig(REPORTS_DIR / "drawdown.png", dpi=150, bbox_inches="tight")
    return fig


def plot_rolling_t_stat(
    t_series: pd.Series,
    significance_level: float = 1.96,
    save: bool = True,
) -> plt.Figure:
    """Rolling t-stat with significance bands."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(t_series.index, t_series.values, color=IB_PALETTE["navy_blue"], linewidth=1.5)
    ax.axhline(significance_level, color=IB_PALETTE["gold"], linestyle="--", linewidth=1, label=f"+{significance_level}")
    ax.axhline(-significance_level, color=IB_PALETTE["gold"], linestyle="--", linewidth=1, label=f"−{significance_level}")
    ax.axhline(0, color=IB_PALETTE["dark_grey"], linewidth=0.5)

    # Shade significant regions
    ax.fill_between(t_series.index, significance_level, t_series.values,
                    where=t_series.values > significance_level,
                    color=IB_PALETTE["dark_green"], alpha=0.2)
    ax.fill_between(t_series.index, -significance_level, t_series.values,
                    where=t_series.values < -significance_level,
                    color=IB_PALETTE["burgundy"], alpha=0.2)

    ax.legend(fontsize=9)
    _apply_style(ax, "Rolling t-Statistic")
    ax.set_ylabel("t-stat", fontsize=10)

    plt.tight_layout()
    if save:
        fig.savefig(REPORTS_DIR / "rolling_t_stat.png", dpi=150, bbox_inches="tight")
    return fig


def plot_regime_comparison(
    returns: pd.Series,
    crisis_periods: list[tuple[str, str]] = None,
    save: bool = True,
) -> plt.Figure:
    """
    Compares portfolio performance in crisis vs. normal regimes.
    """
    if crisis_periods is None:
        crisis_periods = [
            ("2011-08-01", "2011-12-31"),  # EU debt crisis
            ("2015-08-01", "2016-02-28"),  # China devaluation
            ("2018-10-01", "2018-12-31"),  # Vol-mageddon
            ("2020-02-01", "2020-04-30"),  # COVID crash
            ("2022-01-01", "2022-06-30"),  # Rate hiking
        ]

    crisis_mask = pd.Series(False, index=returns.index)
    for start, end in crisis_periods:
        crisis_mask |= (returns.index >= start) & (returns.index <= end)

    crisis_ret = returns[crisis_mask]
    normal_ret = returns[~crisis_mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Return distributions
    ax = axes[0]
    if not normal_ret.empty:
        ax.hist(normal_ret, bins=20, alpha=0.6, color=IB_PALETTE["navy_blue"], label="Normal", density=True)
    if not crisis_ret.empty:
        ax.hist(crisis_ret, bins=15, alpha=0.6, color=IB_PALETTE["burgundy"], label="Crisis", density=True)
    ax.legend(fontsize=9)
    _apply_style(ax, "Return Distribution: Crisis vs Normal")

    # Stats comparison
    ax2 = axes[1]
    stats = pd.DataFrame({
        "Normal": [normal_ret.mean() * 12, normal_ret.std() * np.sqrt(12),
                   normal_ret.mean() / normal_ret.std() * np.sqrt(12) if normal_ret.std() > 0 else 0],
        "Crisis": [crisis_ret.mean() * 12, crisis_ret.std() * np.sqrt(12),
                   crisis_ret.mean() / crisis_ret.std() * np.sqrt(12) if not crisis_ret.empty and crisis_ret.std() > 0 else 0],
    }, index=["Ann. Return", "Ann. Vol", "Sharpe"])

    x = np.arange(len(stats))
    width = 0.35
    ax2.bar(x - width / 2, stats["Normal"], width, color=IB_PALETTE["navy_blue"], label="Normal", alpha=0.8)
    ax2.bar(x + width / 2, stats["Crisis"], width, color=IB_PALETTE["burgundy"], label="Crisis", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stats.index)
    ax2.legend(fontsize=9)
    _apply_style(ax2, "Performance Comparison")

    plt.tight_layout()
    if save:
        fig.savefig(REPORTS_DIR / "regime_comparison.png", dpi=150, bbox_inches="tight")
    return fig


def plot_cumulative_returns(
    returns_dict: dict[str, pd.Series],
    title: str = "Cumulative Returns",
    save: bool = True,
    filename: str = "cumulative_returns.png",
) -> plt.Figure:
    """Overlay multiple return series as cumulative return lines."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (name, ret) in enumerate(returns_dict.items()):
        cum = (1 + ret).cumprod()
        ax.plot(cum.index, cum.values, label=name, color=_COLORS[i % len(_COLORS)], linewidth=1.8)

    ax.legend(fontsize=10, framealpha=0.8)
    _apply_style(ax, title)
    ax.set_ylabel("Cumulative Return", fontsize=10)

    plt.tight_layout()
    if save:
        fig.savefig(REPORTS_DIR / filename, dpi=150, bbox_inches="tight")
    return fig
