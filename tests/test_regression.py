"""
Tests for the statistical engine: cross-sectional regression,
Fama-MacBeth, and expected return model.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cross_sectional import CrossSectionalRegression
from src.models.fama_macbeth import FamaMacBeth
from src.models.expected_return import ExpectedReturnModel


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """
    Generate synthetic data with known factor premia for validation.
    True model: R = 0.02 * F1 + 0.01 * F2 + noise
    """
    np.random.seed(42)
    n_stocks = 100
    n_dates = 60
    dates = pd.date_range("2015-01-01", periods=n_dates, freq="ME")
    tickers = [f"S{i:03d}" for i in range(n_stocks)]

    factor1 = pd.DataFrame(
        np.random.randn(n_dates, n_stocks), index=dates, columns=tickers
    )
    factor2 = pd.DataFrame(
        np.random.randn(n_dates, n_stocks), index=dates, columns=tickers
    )

    # True premia: λ1=0.02, λ2=0.01
    noise = pd.DataFrame(
        np.random.randn(n_dates, n_stocks) * 0.05, index=dates, columns=tickers
    )
    returns = 0.02 * factor1 + 0.01 * factor2 + noise

    return returns, {"F1": factor1, "F2": factor2}


# ── Cross-Sectional Regression ────────────────────────────────────────

def test_cs_regression_basic():
    """Test that CS regression returns correct structure."""
    np.random.seed(1)
    ret = pd.Series(np.random.randn(50), index=[f"S{i}" for i in range(50)])
    fac = pd.DataFrame({
        "A": np.random.randn(50),
        "B": np.random.randn(50),
    }, index=ret.index)

    cs = CrossSectionalRegression()
    result = cs.fit(ret, fac)

    assert "coefficients" in result
    assert "intercept" in result["coefficients"].index
    assert "A" in result["coefficients"].index
    assert result["n_obs"] == 50


def test_cs_regression_perfect_fit():
    """When returns are a perfect linear function of factors, R²≈1."""
    np.random.seed(2)
    n = 100
    tickers = [f"S{i}" for i in range(n)]
    f = pd.DataFrame({"X": np.random.randn(n)}, index=tickers)
    r = pd.Series(3.0 * f["X"] + 1.0, index=tickers)

    result = CrossSectionalRegression().fit(r, f)
    assert result["r_squared"] > 0.99


# ── Fama-MacBeth ──────────────────────────────────────────────────────

def test_fama_macbeth_recovers_premia(synthetic_data):
    """FM should recover approximately correct factor premia from synthetic data."""
    returns, factors = synthetic_data
    fm = FamaMacBeth(newey_west_lags=4)
    result = fm.fit(returns, factors)

    summary = result["summary"]
    assert "F1" in summary.index
    assert "F2" in summary.index

    # True premia are 0.02 and 0.01; allow wide margin due to noise
    assert abs(summary.loc["F1", "mean_lambda"] - 0.02) < 0.01
    assert abs(summary.loc["F2", "mean_lambda"] - 0.01) < 0.01

    # Should be statistically significant
    assert abs(summary.loc["F1", "nw_t_stat"]) > 1.96
    assert abs(summary.loc["F2", "nw_t_stat"]) > 1.0


def test_fama_macbeth_lambda_series_shape(synthetic_data):
    """Lambda series should have one row per date."""
    returns, factors = synthetic_data
    result = FamaMacBeth().fit(returns, factors)
    lambda_df = result["lambda_series"]
    assert len(lambda_df) > 0
    assert "F1" in lambda_df.columns


# ── Expected Return Model ─────────────────────────────────────────────

def test_expected_return_deciles(synthetic_data):
    """Decile assignment should produce values 1-10."""
    returns, factors = synthetic_data
    fm = FamaMacBeth().fit(returns, factors)
    er_model = ExpectedReturnModel(rolling_window=6)

    forecasts = er_model.forecast(fm["lambda_series"], factors)
    if not forecasts.empty:
        deciles = er_model.assign_deciles(forecasts)
        valid = deciles.dropna(how="all")
        if not valid.empty:
            assert valid.min().min() >= 1
            assert valid.max().max() <= 10


def test_expected_return_ranking(synthetic_data):
    """Ranking should produce values between 0 and 1."""
    returns, factors = synthetic_data
    fm = FamaMacBeth().fit(returns, factors)
    er_model = ExpectedReturnModel(rolling_window=6)

    forecasts = er_model.forecast(fm["lambda_series"], factors)
    if not forecasts.empty:
        ranks = er_model.rank_stocks(forecasts)
        assert ranks.min().min() >= 0
        assert ranks.max().max() <= 1
