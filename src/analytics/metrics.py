"""
Performance Metrics

Factor-level and portfolio-level metrics for evaluating
the cross-sectional expected return model.

Factor Level:
  - Information Coefficient (IC) — rank correlation between predicted and actual
  - IC decay — how IC degrades over holding periods
  - Hit ratio — % of correct directional predictions

Portfolio Level:
  - Sharpe ratio
  - Sortino ratio
  - Maximum drawdown
  - Alpha / Beta vs benchmark
  - Rolling t-statistics
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

try:
    from src.config import CACHE_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config import CACHE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================================================================
# Factor-Level Metrics
# ======================================================================

def information_coefficient(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """
    Cross-sectional rank IC at each date.
    IC_t = corr(rank(predicted_t), rank(actual_t))
    """
    common_dates = predicted.index.intersection(actual.index)
    ics = []

    for date in common_dates:
        p = predicted.loc[date].dropna()
        a = actual.loc[date].dropna()
        common_tickers = p.index.intersection(a.index)

        if len(common_tickers) < 5:
            ics.append(np.nan)
            continue

        if method == "spearman":
            ic = p[common_tickers].rank().corr(a[common_tickers].rank())
        else:
            ic = p[common_tickers].corr(a[common_tickers])

        ics.append(ic)

    return pd.Series(ics, index=common_dates, name="IC")


def ic_summary(ic_series: pd.Series) -> dict:
    """Mean IC, IC IR (mean/std), and proportion positive."""
    ic = ic_series.dropna()
    if ic.empty:
        return {"mean_ic": np.nan, "ic_ir": np.nan, "pct_positive": np.nan}

    mean_ic = ic.mean()
    ic_std = ic.std()
    return {
        "mean_ic": mean_ic,
        "ic_ir": mean_ic / ic_std if ic_std > 0 else np.nan,
        "pct_positive": (ic > 0).mean(),
        "n_periods": len(ic),
    }


def ic_decay(
    predicted: pd.DataFrame,
    prices: pd.DataFrame,
    max_lag: int = 10,
) -> pd.Series:
    """
    IC decay curve: IC at holding periods 1 through max_lag.
    """
    decay = {}
    for lag in range(1, max_lag + 1):
        fwd_ret = prices.pct_change(lag).shift(-lag)
        ic = information_coefficient(predicted, fwd_ret)
        decay[lag] = ic.mean()

    return pd.Series(decay, name="IC_decay")


def hit_ratio(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
) -> pd.Series:
    """
    Percentage of stocks where sign(predicted) == sign(actual).
    """
    common_dates = predicted.index.intersection(actual.index)
    hits = []

    for date in common_dates:
        p = predicted.loc[date].dropna()
        a = actual.loc[date].dropna()
        common = p.index.intersection(a.index)
        if len(common) < 2:
            hits.append(np.nan)
            continue

        correct = (np.sign(p[common]) == np.sign(a[common])).mean()
        hits.append(correct)

    return pd.Series(hits, index=common_dates, name="hit_ratio")


# ======================================================================
# Portfolio-Level Metrics
# ======================================================================

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualised Sharpe ratio (assumes risk-free = 0)."""
    if returns.empty or returns.std() == 0:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    downside = returns[returns < 0]
    if downside.empty or downside.std() == 0:
        return np.nan
    return returns.mean() / downside.std() * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (as a negative number)."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def alpha_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    """
    CAPM alpha and beta via OLS: r_p = α + β·r_b + ε
    """
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common) < 3:
        return {"alpha": np.nan, "beta": np.nan}

    y = portfolio_returns.loc[common].values
    x = benchmark_returns.loc[common].values
    X = np.column_stack([np.ones_like(x), x])

    try:
        beta_vec = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"alpha": np.nan, "beta": np.nan}

    return {"alpha": beta_vec[0], "beta": beta_vec[1]}


def rolling_t_stat(
    returns: pd.Series,
    window: int = 36,
) -> pd.Series:
    """
    Rolling t-statistic of the mean return.
    t = mean / (std / √n)
    """
    rolling_mean = returns.rolling(window, min_periods=window // 2).mean()
    rolling_std = returns.rolling(window, min_periods=window // 2).std()
    t = rolling_mean / (rolling_std / np.sqrt(window))
    t.name = "rolling_t_stat"
    return t


def full_performance_report(
    returns: pd.Series,
    benchmark: pd.Series = None,
    periods_per_year: int = 12,
) -> dict:
    """
    Consolidated performance summary.
    """
    report = {
        "total_return": (1 + returns).prod() - 1,
        "annualized_return": (1 + returns).prod() ** (periods_per_year / len(returns)) - 1 if len(returns) > 0 else np.nan,
        "annualized_volatility": returns.std() * np.sqrt(periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "n_periods": len(returns),
    }

    if benchmark is not None:
        ab = alpha_beta(returns, benchmark)
        report["alpha"] = ab["alpha"]
        report["beta"] = ab["beta"]

    return report
