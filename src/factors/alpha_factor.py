"""
Custom Alpha Factor: Earnings Surprise Momentum (ESM)

This is the mandatory *original signal* required by the research spec.

Economic Rationale
------------------
Post-earnings-announcement drift (PEAD) is one of the most persistent
anomalies in equity markets.  Stocks that beat earnings expectations tend
to continue drifting upward, and vice versa.

Since consensus EPS estimates are not available in free datasets, we
approximate the signal using **standardised unexpected earnings (SUE)**:

    SUE_t = (EPS_t - EPS_{t-4}) / σ(EPS_t - EPS_{t-4})

where:
  - EPS_t          is trailing twelve-month EPS at time t
  - EPS_{t-4}      is the same metric four quarters (~252 trading days) ago
  - σ              is the rolling standard deviation of the seasonal difference

A high SUE means earnings grew more than the recent historical norm —
a positive surprise proxy.

When only snapshot fundamentals are available (yfinance fallback), we
derive a simpler variant:

    ESM = (Forward EPS / Trailing EPS) - 1

which captures the market's embedded expectation of earnings growth.
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


class EarningsSurpriseMomentum:
    """
    Computes the Earnings Surprise Momentum (ESM) alpha factor.

    Supports two paths:
      1. **Time-series EPS** (SimFin-style): uses SUE (standardised unexpected earnings).
      2. **Snapshot EPS** (yfinance fallback): uses Forward/Trailing EPS ratio.
    """

    # Number of trading days corresponding to 4 quarters (~1 year)
    SEASONAL_LAG_DAYS = 252
    # Rolling window for std of seasonal difference
    STD_WINDOW = 8  # quarters (when data is quarterly-frequency)

    # ------------------------------------------------------------------
    # Path 1: Time-series SUE
    # ------------------------------------------------------------------
    def compute_sue(self, eps_series: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        eps_series : DataFrame
            index = date, columns = tickers, values = trailing EPS.

        Returns
        -------
        DataFrame of SUE scores, same shape.
        """
        logger.info("Computing Standardised Unexpected Earnings (SUE)...")

        seasonal_diff = eps_series - eps_series.shift(self.SEASONAL_LAG_DAYS)

        # Rolling std of the seasonal difference
        rolling_std = seasonal_diff.rolling(
            window=self.SEASONAL_LAG_DAYS * 2,
            min_periods=self.SEASONAL_LAG_DAYS,
        ).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        sue = seasonal_diff / rolling_std

        return sue

    # ------------------------------------------------------------------
    # Path 2: Snapshot forward/trailing ratio
    # ------------------------------------------------------------------
    def compute_snapshot(self, fundamentals: pd.DataFrame) -> pd.Series:
        """
        Parameters
        ----------
        fundamentals : DataFrame
            Must contain 'Forward EPS' and 'Trailing EPS' columns.

        Returns
        -------
        Series of ESM scores.
        """
        logger.info("Computing Earnings Surprise Momentum (snapshot mode)...")

        required = ["Forward EPS", "Trailing EPS"]
        for col in required:
            if col not in fundamentals.columns:
                logger.warning(f"Column '{col}' missing — returning NaN.")
                return pd.Series(np.nan, index=fundamentals.index)

        trailing = fundamentals["Trailing EPS"].replace(0, np.nan)
        esm = (fundamentals["Forward EPS"] / trailing) - 1.0

        return esm

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------
    def compute(
        self,
        eps_series: pd.DataFrame = None,
        fundamentals: pd.DataFrame = None,
    ) -> pd.DataFrame | pd.Series:
        """
        Automatically chooses the best available computation path.
        """
        if eps_series is not None and not eps_series.empty:
            return self.compute_sue(eps_series)

        if fundamentals is not None and not fundamentals.empty:
            return self.compute_snapshot(fundamentals)

        raise ValueError(
            "Either eps_series (time-series) or fundamentals (snapshot) must be provided."
        )


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    dates = pd.date_range("2018-01-01", periods=504, freq="B")
    np.random.seed(42)
    eps = pd.DataFrame(
        {
            "AAPL": np.cumsum(np.random.normal(0.01, 0.05, len(dates))) + 5,
            "MSFT": np.cumsum(np.random.normal(0.01, 0.04, len(dates))) + 4,
        },
        index=dates,
    )
    esm = EarningsSurpriseMomentum()
    sue = esm.compute(eps_series=eps)
    print("SUE tail:")
    print(sue.tail())
