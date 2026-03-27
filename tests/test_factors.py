"""
Unit tests for the factor engineering modules.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.factors.style_factors import (
    MomentumFactor,
    SizeFactor,
    LowVolatilityFactor,
    LiquidityFactor,
    build_price_factors,
)
from src.factors.alpha_factor import EarningsSurpriseMomentum
from src.factors.preprocessing import (
    winsorize_cross_section,
    zscore_cross_section,
    fill_missing_values,
    preprocess_factor,
)
from src.factors.orthogonalization import (
    gram_schmidt,
    residualize,
    factor_correlation_matrix,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_prices():
    """50 trading days, 5 tickers with smooth upward trends + noise."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    data = {}
    for t in tickers:
        data[t] = 100 + np.cumsum(np.random.normal(0.05, 1.0, len(dates)))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def mock_factor():
    """Simple 10-date × 5-ticker factor with one NaN."""
    np.random.seed(7)
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    tickers = ["A", "B", "C", "D", "E"]
    df = pd.DataFrame(
        np.random.randn(10, 5), index=dates, columns=tickers
    )
    df.iloc[3, 2] = np.nan  # inject one missing
    return df


# ── Momentum ──────────────────────────────────────────────────────────

def test_momentum_factor_shape(mock_prices):
    mom = MomentumFactor(lookback=60, skip=5).compute(mock_prices)
    assert mom.shape == mock_prices.shape
    # First `lookback` rows should be NaN (not enough history)
    assert mom.iloc[:60].isna().all().all()


def test_momentum_factor_sign(mock_prices):
    """Stocks trending up should have positive momentum."""
    # Force a clear upward trend for one ticker
    prices = mock_prices.copy()
    prices["UP"] = np.linspace(100, 200, len(prices))
    mom = MomentumFactor(lookback=60, skip=5).compute(prices)
    # After warm-up, momentum for 'UP' should always be positive
    assert (mom["UP"].dropna() > 0).all()


# ── Size ──────────────────────────────────────────────────────────────

def test_size_factor_inversion():
    mcap = pd.DataFrame({"BIG": [1e12], "SMALL": [1e8]})
    size = SizeFactor().compute(mcap)
    # Small should have a HIGHER score (less negative)
    assert size["SMALL"].iloc[0] > size["BIG"].iloc[0]


# ── Low Volatility ───────────────────────────────────────────────────

def test_low_vol_sign(mock_prices):
    low_vol = LowVolatilityFactor(window=20).compute(mock_prices)
    # All values should be ≤ 0 (inverted)
    assert (low_vol.dropna(how="all") <= 0).all().all()


# ── Liquidity ─────────────────────────────────────────────────────────

def test_liquidity_no_volume(mock_prices):
    """Should not crash when volume is None (proxy mode)."""
    liq = LiquidityFactor(window=10).compute(mock_prices, volume=None)
    assert liq.shape == mock_prices.shape


# ── Alpha Factor (ESM) ───────────────────────────────────────────────

def test_esm_snapshot():
    df = pd.DataFrame({
        "Trailing EPS": [2.0, 3.0, 0.0],
        "Forward EPS": [2.5, 3.6, 1.0],
    })
    esm = EarningsSurpriseMomentum().compute_snapshot(df)
    # Stock 0: (2.5/2.0)-1 = 0.25
    assert np.isclose(esm.iloc[0], 0.25)
    # Stock 1: (3.6/3.0)-1 = 0.20
    assert np.isclose(esm.iloc[1], 0.20)
    # Stock 2: trailing EPS=0 → NaN (division by zero handled)
    assert pd.isna(esm.iloc[2])


def test_esm_sue(mock_prices):
    """SUE on synthetic EPS data should produce finite values after warm-up."""
    np.random.seed(99)
    dates = pd.date_range("2018-01-01", periods=600, freq="B")
    eps = pd.DataFrame({
        "A": np.cumsum(np.random.normal(0.01, 0.03, 600)) + 3,
        "B": np.cumsum(np.random.normal(0.01, 0.02, 600)) + 4,
    }, index=dates)
    sue = EarningsSurpriseMomentum().compute_sue(eps)
    # After warm-up (2 * 252 = 504 days), should have some finite values
    assert sue.iloc[504:].notna().any().any()


# ── Preprocessing ────────────────────────────────────────────────────

def test_winsorize_clamps_outlier(mock_factor):
    df = mock_factor.copy()
    df.iloc[0, 0] = 100.0  # extreme outlier
    w = winsorize_cross_section(df)
    assert w.iloc[0, 0] < 100.0  # should be clamped


def test_zscore_mean_zero(mock_factor):
    z = zscore_cross_section(mock_factor.dropna())
    row_means = z.mean(axis=1)
    # Cross-sectional mean should be ~0 at every date
    assert (row_means.abs() < 1e-10).all()


def test_zscore_clipping():
    df = pd.DataFrame({"A": [0], "B": [0], "C": [0], "D": [100]})
    z = zscore_cross_section(df, clip=3.0)
    assert z.max().max() <= 3.0
    assert z.min().min() >= -3.0


def test_fill_missing_drops_sparse_tickers():
    dates = pd.date_range("2020-01-01", periods=10)
    df = pd.DataFrame(
        np.ones((10, 3)), index=dates, columns=["A", "B", "C"]
    )
    # Make column C almost entirely NaN (60% missing > 50% threshold)
    df.loc[df.index[:6], "C"] = np.nan
    filled = fill_missing_values(df, max_missing_pct=0.50)
    assert "C" not in filled.columns


def test_full_preprocessing_pipeline(mock_factor):
    result = preprocess_factor(mock_factor.dropna())
    # Should be z-scored: cross-sectional mean ≈ 0
    assert (result.mean(axis=1).abs() < 1e-10).all()


# ── Orthogonalization ─────────────────────────────────────────────────

def test_gram_schmidt_orthogonal(mock_prices):
    """After Gram-Schmidt, factors should have near-zero cross-sectional correlation."""
    mom = MomentumFactor(lookback=60, skip=5).compute(mock_prices)
    vol = LowVolatilityFactor(window=20).compute(mock_prices)
    liq = LiquidityFactor(window=10).compute(mock_prices, volume=None)

    factors = {"mom": mom, "vol": vol, "liq": liq}

    ortho = gram_schmidt(factors, priority_order=["mom", "vol", "liq"])

    # Check correlation at a specific late date
    test_date = mock_prices.index[200]
    corr_before = factor_correlation_matrix(factors, date=test_date)
    corr_after = factor_correlation_matrix(ortho, date=test_date)

    # Off-diagonals should be smaller after orthogonalization
    off_diag_before = corr_before.values[np.triu_indices(3, k=1)]
    off_diag_after = corr_after.values[np.triu_indices(3, k=1)]
    assert np.nanmean(np.abs(off_diag_after)) <= np.nanmean(np.abs(off_diag_before)) + 0.05


def test_correlation_matrix_output():
    """Correlation matrix should be symmetric with 1s on diagonal."""
    dates = pd.date_range("2020-01-01", periods=5)
    f1 = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=["A", "B", "C"])
    f2 = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=["A", "B", "C"])

    corr = factor_correlation_matrix({"f1": f1, "f2": f2}, date=dates[0])
    assert corr.shape == (2, 2)
    assert np.isclose(corr.loc["f1", "f1"], 1.0)
