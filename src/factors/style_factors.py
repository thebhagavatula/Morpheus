"""
Style Factor Construction

Constructs the six canonical style factors from price + fundamental data:
  1. Value      — composite of B/P, E/P, EBITDA/EV
  2. Momentum   — 12-month return skipping most recent month (12-1)
  3. Size       — log market cap (inverted: small = high score)
  4. Low Vol    — trailing realised volatility (inverted: low vol = high score)
  5. Quality    — composite of ROE, inverse debt/equity, earnings stability
  6. Liquidity  — average daily dollar volume (inverted: illiquid = high score)

All factors are returned as cross-sectional DataFrames:
    index = date,  columns = ticker,  values = raw factor exposure.
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

try:
    from src.config import (
        MOMENTUM_LOOKBACK_DAYS,
        MOMENTUM_SKIP_DAYS,
        VOLATILITY_WINDOW_DAYS,
        LIQUIDITY_WINDOW_DAYS,
        QUALITY_WEIGHTS,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config import (
        MOMENTUM_LOOKBACK_DAYS,
        MOMENTUM_SKIP_DAYS,
        VOLATILITY_WINDOW_DAYS,
        LIQUIDITY_WINDOW_DAYS,
        QUALITY_WEIGHTS,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================================================================
# Price-derived factors (require only a price DataFrame)
# ======================================================================

class MomentumFactor:
    """
    12-1 Momentum: cumulative return over the past 12 months excluding
    the most recent 1 month.  The skip avoids the well-documented
    short-term reversal effect.
    """

    def __init__(
        self,
        lookback: int = MOMENTUM_LOOKBACK_DAYS,
        skip: int = MOMENTUM_SKIP_DAYS,
    ):
        self.lookback = lookback
        self.skip = skip

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing 12-1 Momentum factor...")

        # Price at t-skip (the recent end we exclude)
        price_recent = prices.shift(self.skip)
        # Price at t-lookback (the far end)
        price_far = prices.shift(self.lookback)

        # Return from t-lookback to t-skip
        momentum = (price_recent / price_far) - 1.0

        return momentum


class SizeFactor:
    """
    Size = -log(market_cap).
    Negative sign so that *small* stocks have a *high* factor score,
    consistent with the small-cap premium hypothesis.
    """

    def compute(self, market_cap: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing Size factor (inverted log market cap)...")
        return -np.log(market_cap.replace(0, np.nan))


class LowVolatilityFactor:
    """
    Low Volatility = -σ  (trailing realised volatility, inverted).
    A stock with *lower* volatility gets a *higher* factor score.
    """

    def __init__(self, window: int = VOLATILITY_WINDOW_DAYS):
        self.window = window

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Computing Low Volatility factor (window={self.window})...")

        daily_returns = prices.pct_change()
        rolling_vol = daily_returns.rolling(window=self.window, min_periods=self.window // 2).std()

        # Annualise (optional, doesn't affect cross-sectional rank, but aids interpretation)
        annualised_vol = rolling_vol * np.sqrt(252)

        # Invert: low vol ⇒ high score
        return -annualised_vol


class LiquidityFactor:
    """
    Liquidity = -avg_dollar_volume.
    *Illiquid* stocks score higher, consistent with the illiquidity premium.

    If volume data is unavailable, falls back to a turnover proxy using
    price * (price_change amplitude).
    """

    def __init__(self, window: int = LIQUIDITY_WINDOW_DAYS):
        self.window = window

    def compute(
        self, prices: pd.DataFrame, volume: pd.DataFrame = None
    ) -> pd.DataFrame:
        logger.info(f"Computing Liquidity factor (window={self.window})...")

        if volume is not None and not volume.empty:
            dollar_volume = prices * volume
        else:
            # Proxy: use absolute daily return as a rough activity proxy
            logger.warning(
                "Volume data not provided — using |daily return| proxy for liquidity."
            )
            dollar_volume = prices.pct_change().abs() * prices

        avg_dv = dollar_volume.rolling(window=self.window, min_periods=self.window // 2).mean()

        # Invert: illiquid ⇒ higher score
        return -avg_dv


# ======================================================================
# Fundamental-derived factors (require merged fundamental data)
# ======================================================================

class ValueFactor:
    """
    Composite Value = average z-score of B/P, E/P, and EBITDA/EV.
    All three are *higher-is-cheaper* ratios, so no inversion needed.

    Input: a DataFrame where each row is (date, ticker) and columns include
    the raw fundamental fields.  The caller (pipeline) is responsible for
    aligning fundamentals to the right dates.
    """

    REQUIRED_FIELDS = {
        "book_value_per_share": "Book Value Per Share",
        "trailing_eps": "Trailing EPS",
        "ebitda": "EBITDA",
        "enterprise_value": "Enterprise Value",
        "price": "Current Price",
    }

    def compute(self, fundamentals: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        fundamentals : DataFrame
            Must contain columns: Ticker, date index (or Available Date),
            and at least some of the required fundamental fields.

        Returns
        -------
        DataFrame  (index=date, columns=ticker, values=composite value score)
        """
        logger.info("Computing composite Value factor (B/P, E/P, EBITDA/EV)...")

        df = fundamentals.copy()

        # --- B/P (Book-to-Price) ---
        bp = self._safe_ratio(df, "Book Value Per Share", "Current Price")

        # --- E/P (Earnings-to-Price) ---
        ep = self._safe_ratio(df, "Trailing EPS", "Current Price")

        # --- EBITDA/EV ---
        ebitda_ev = self._safe_ratio(df, "EBITDA", "Enterprise Value")

        # Composite: simple average of available sub-factors
        components = pd.concat(
            [bp.rename("bp"), ep.rename("ep"), ebitda_ev.rename("ebitda_ev")],
            axis=1,
        )
        composite = components.mean(axis=1, skipna=True)

        return composite

    @staticmethod
    def _safe_ratio(df: pd.DataFrame, numerator_col: str, denominator_col: str) -> pd.Series:
        """Computes numerator/denominator, returning NaN where denominator is zero or missing."""
        if numerator_col not in df.columns or denominator_col not in df.columns:
            logger.warning(f"Missing columns for {numerator_col}/{denominator_col}, returning NaN.")
            return pd.Series(np.nan, index=df.index)
        denom = df[denominator_col].replace(0, np.nan)
        return df[numerator_col] / denom


class QualityFactor:
    """
    Composite Quality = weighted combination of:
      - ROE (higher = better)
      - Inverse Debt/Equity (lower leverage = better)
      - Earnings Stability (lower std of trailing EPS = better)
    """

    def __init__(self, weights: dict = None):
        self.weights = weights or QUALITY_WEIGHTS

    def compute(self, fundamentals: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing composite Quality factor...")

        df = fundamentals.copy()
        components = {}

        # ROE (higher is better, no inversion)
        if "Return On Equity" in df.columns:
            components["roe"] = df["Return On Equity"]
        else:
            logger.warning("Return On Equity column missing; skipping ROE sub-factor.")

        # Inverted Debt/Equity (lower leverage = higher score)
        if "Debt To Equity" in df.columns:
            dte = df["Debt To Equity"].replace(0, np.nan)
            components["debt_equity_inv"] = -dte
        else:
            logger.warning("Debt To Equity column missing; skipping leverage sub-factor.")

        # Earnings Stability — lower std of trailing EPS is better
        # This requires a time-series dimension that a single snapshot doesn't have.
        # When available, we'll use rolling std of EPS.  For now, skip gracefully.
        if "Earnings Stability" in df.columns:
            components["earnings_stability"] = -df["Earnings Stability"]

        if not components:
            logger.warning("No quality sub-factors available; returning NaN.")
            return pd.Series(np.nan, index=df.index)

        comp_df = pd.DataFrame(components)

        # Weighted average (only over available components)
        total_weight = 0.0
        weighted_sum = pd.Series(0.0, index=df.index)
        for key, series in components.items():
            w = self.weights.get(key, 0.0)
            mask = series.notna()
            weighted_sum = weighted_sum.add(series.fillna(0) * w)
            total_weight += w

        if total_weight == 0:
            return pd.Series(np.nan, index=df.index)

        return weighted_sum / total_weight


# ======================================================================
# Convenience: build all price-based factors in one call
# ======================================================================

def build_price_factors(
    prices: pd.DataFrame,
    market_cap: pd.DataFrame = None,
    volume: pd.DataFrame = None,
) -> dict[str, pd.DataFrame]:
    """
    Builds all factors that only need price (and optionally volume / market cap).

    Returns a dict  { factor_name: DataFrame(index=date, columns=ticker) }
    """
    factors = {}

    factors["momentum"] = MomentumFactor().compute(prices)
    factors["low_volatility"] = LowVolatilityFactor().compute(prices)

    if market_cap is not None and not market_cap.empty:
        factors["size"] = SizeFactor().compute(market_cap)
    else:
        logger.warning("Market cap not provided — Size factor skipped.")

    factors["liquidity"] = LiquidityFactor().compute(prices, volume)

    return factors
