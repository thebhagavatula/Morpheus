"""
Tests for portfolio construction, cost model, and performance metrics.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.portfolio.construction import PortfolioConstructor, PortfolioReturns
from src.portfolio.costs import CostModel
from src.analytics.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    alpha_beta,
    information_coefficient,
    hit_ratio,
    full_performance_report,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_deciles():
    dates = pd.date_range("2020-01-01", periods=12, freq="ME")
    tickers = [f"S{i}" for i in range(20)]
    np.random.seed(42)
    data = np.random.choice(range(1, 11), size=(12, 20))
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture
def mock_returns():
    dates = pd.date_range("2020-01-01", periods=12, freq="ME")
    tickers = [f"S{i}" for i in range(20)]
    np.random.seed(42)
    data = np.random.randn(12, 20) * 0.05
    return pd.DataFrame(data, index=dates, columns=tickers)


# ── Portfolio Construction ────────────────────────────────────────────

def test_long_short_weights_sum_to_zero(mock_deciles):
    pc = PortfolioConstructor()
    portfolio = pc.long_short_decile(mock_deciles)
    net = portfolio["net"]
    # Weights should approximately sum to zero (dollar neutral)
    assert (net.sum(axis=1).abs() < 1e-10).all()


def test_market_neutral():
    weights = pd.DataFrame({"A": [0.5, 0.3], "B": [0.3, 0.4], "C": [0.2, 0.3]})
    pc = PortfolioConstructor()
    neutral = pc.market_neutral(weights)
    assert (neutral.sum(axis=1).abs() < 1e-10).all()


def test_portfolio_returns(mock_deciles, mock_returns):
    pc = PortfolioConstructor()
    portfolio = pc.long_short_decile(mock_deciles)
    ret = PortfolioReturns.compute(portfolio["net"], mock_returns)
    assert len(ret) > 0
    assert isinstance(ret, pd.Series)


# ── Cost Model ────────────────────────────────────────────────────────

def test_turnover_calculation():
    weights = pd.DataFrame({
        "A": [0.5, 0.3, 0.4],
        "B": [0.5, 0.7, 0.6],
    }, index=pd.date_range("2020-01-01", periods=3, freq="ME"))

    cm = CostModel(cost_bps=10)
    turnover = cm.compute_turnover(weights)
    # First row is NaN (no prior), second has changes
    assert pd.isna(turnover.iloc[0])
    assert turnover.iloc[1] > 0


def test_cost_adjustment():
    np.random.seed(1)
    n = 12
    dates = pd.date_range("2020-01-01", periods=n, freq="ME")
    weights = pd.DataFrame(np.random.dirichlet([1]*5, n), index=dates, columns=[f"S{i}" for i in range(5)])
    gross_returns = pd.Series(np.random.randn(n) * 0.03, index=dates)

    cm = CostModel(cost_bps=10)
    net = cm.adjust_returns(gross_returns, weights)

    # Net returns should be <= gross returns (costs are always positive)
    common = gross_returns.index.intersection(net.index)
    assert (net.loc[common] <= gross_returns.loc[common] + 1e-10).all()


def test_turnover_summary():
    weights = pd.DataFrame({
        "A": [0.5, 0.2, 0.6, 0.1],
        "B": [0.5, 0.8, 0.4, 0.9],
    }, index=pd.date_range("2020-01-01", periods=4, freq="ME"))

    cm = CostModel(cost_bps=10)
    summary = cm.turnover_summary(weights)
    assert "mean_turnover" in summary
    assert summary["mean_turnover"] > 0


# ── Performance Metrics ───────────────────────────────────────────────

def test_sharpe_ratio():
    returns = pd.Series([0.01] * 12)  # Constant positive returns
    sr = sharpe_ratio(returns, periods_per_year=12)
    # With zero vol, Sharpe is undefined → NaN or inf
    # But std is 0, so we handle it
    assert np.isinf(sr) or np.isnan(sr) or sr > 0


def test_sortino_ratio():
    returns = pd.Series([-0.01, 0.02, -0.005, 0.03, 0.01, -0.02])
    sr = sortino_ratio(returns)
    assert isinstance(sr, float)


def test_max_drawdown():
    returns = pd.Series([0.1, -0.2, 0.05, -0.1, 0.15])
    mdd = max_drawdown(returns)
    assert mdd < 0  # Drawdown is negative


def test_alpha_beta():
    np.random.seed(42)
    n = 60
    benchmark = pd.Series(np.random.randn(n) * 0.03, index=pd.date_range("2020-01-01", periods=n, freq="ME"))
    portfolio = 0.001 + 1.2 * benchmark + pd.Series(np.random.randn(n) * 0.01, index=benchmark.index)

    ab = alpha_beta(portfolio, benchmark)
    assert abs(ab["beta"] - 1.2) < 0.3
    assert "alpha" in ab


def test_information_coefficient():
    dates = pd.date_range("2020-01-01", periods=5, freq="ME")
    tickers = [f"S{i}" for i in range(20)]
    np.random.seed(42)
    predicted = pd.DataFrame(np.random.randn(5, 20), index=dates, columns=tickers)
    actual = predicted + pd.DataFrame(np.random.randn(5, 20) * 0.5, index=dates, columns=tickers)

    ic = information_coefficient(predicted, actual)
    # Since actual ≈ predicted + small noise, IC should be positive
    assert ic.mean() > 0


def test_full_performance_report():
    returns = pd.Series(np.random.randn(36) * 0.03, index=pd.date_range("2020-01-01", periods=36, freq="ME"))
    report = full_performance_report(returns)
    assert "sharpe_ratio" in report
    assert "max_drawdown" in report
    assert "annualized_return" in report
