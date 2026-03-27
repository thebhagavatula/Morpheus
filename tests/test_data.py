import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.universe import UniverseBuilder
from src.data.prices import PriceEngine
from src.data.fundamentals import FundamentalsEngine


def test_universe_builder_fallback():
    """Test that UniverseBuilder returns data even if web scraper fails (fallback list)."""
    builder = UniverseBuilder()
    tickers = builder.get_universe()
    assert len(tickers) > 0
    assert isinstance(tickers, list)
    assert all(isinstance(t, str) for t in tickers)


def test_price_engine_forward_returns():
    """Test that forward return calculation correctly shifts future prices back."""
    engine = PriceEngine()

    # Create mock prices
    dates = pd.date_range("2020-01-01", periods=10)
    prices = pd.DataFrame(
        {"AAPL": [100, 101, 102, 105, 110, 115, 120, 118, 116, 120]},
        index=dates,
    )

    fwd_ret = engine.calculate_forward_returns(prices, periods=2)

    # Index 0 (price 100), 2-period forward price is 102 → return = 0.02
    assert np.isclose(fwd_ret.iloc[0]["AAPL"], 0.02)

    # Last 2 periods must be NaN (no future data)
    assert pd.isna(fwd_ret.iloc[-1]["AAPL"])
    assert pd.isna(fwd_ret.iloc[-2]["AAPL"])


def test_fundamental_lagging():
    """Test that fundamentals are correctly lagged by FUNDAMENTAL_LAG_DAYS."""
    engine = FundamentalsEngine()

    df = pd.DataFrame({"Report Date": ["2020-03-31", "2020-06-30"]})

    lagged_df = engine.lag_fundamentals_point_in_time(df, report_date_col="Report Date")

    from src.config import FUNDAMENTAL_LAG_DAYS

    expected_date = pd.to_datetime("2020-03-31") + pd.Timedelta(days=FUNDAMENTAL_LAG_DAYS)
    assert lagged_df.iloc[0]["Available Date"] == expected_date


def test_price_engine_extract_adj_close_flat():
    """Test _extract_adj_close with a flat (single-ticker) DataFrame."""
    engine = PriceEngine()

    dates = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame(
        {"Close": [100, 101, 102, 103, 104], "Volume": [1e6] * 5},
        index=dates,
    )
    result = engine._extract_adj_close(data, ["AAPL"])
    assert "AAPL" in result.columns
    assert len(result) == 5
