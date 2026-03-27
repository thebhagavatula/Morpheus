"""
Portfolio Construction

Builds long/short portfolios from expected return rankings with:
  - Market neutrality (dollar neutral)
  - Sector neutrality (equal long/short within each sector)
  - Beta neutrality (hedge market beta exposure)
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

try:
    from src.config import SECTOR_COLUMN
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config import SECTOR_COLUMN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioConstructor:
    """
    Builds neutralised long/short portfolios from expected return forecasts.
    """

    def long_short_decile(
        self,
        decile_assignments: pd.DataFrame,
        top_decile: int = 10,
        bottom_decile: int = 1,
    ) -> dict[str, pd.DataFrame]:
        """
        Returns equal-weighted long (top decile) and short (bottom decile)
        portfolio weight DataFrames.
        """
        logger.info(f"Building long/short portfolio: long D{top_decile}, short D{bottom_decile}")

        long_weights = pd.DataFrame(0.0, index=decile_assignments.index, columns=decile_assignments.columns)
        short_weights = pd.DataFrame(0.0, index=decile_assignments.index, columns=decile_assignments.columns)

        for date in decile_assignments.index:
            row = decile_assignments.loc[date]
            longs = row[row == top_decile].index
            shorts = row[row == bottom_decile].index

            # Only build positions when both legs are available (dollar neutral)
            if len(longs) > 0 and len(shorts) > 0:
                long_weights.loc[date, longs] = 1.0 / len(longs)
                short_weights.loc[date, shorts] = -1.0 / len(shorts)

        combined = long_weights + short_weights
        return {"long": long_weights, "short": short_weights, "net": combined}

    def market_neutral(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures dollar neutrality: sum of weights = 0 at each date.
        Adjusts by subtracting the mean weight from all positions.
        """
        logger.info("Applying market neutrality constraint...")
        return weights.sub(weights.mean(axis=1), axis=0)

    def sector_neutral(
        self,
        expected_returns: pd.DataFrame,
        sector_map: pd.Series,
        n_long: int = 2,
        n_short: int = 2,
    ) -> pd.DataFrame:
        """
        Within each sector, go long top-n and short bottom-n stocks.
        Equal-weighted within each leg, dollar-neutral across sectors.
        """
        logger.info("Building sector-neutral portfolio...")

        weights = pd.DataFrame(0.0, index=expected_returns.index, columns=expected_returns.columns)
        sectors = sector_map.dropna().unique()

        for date in expected_returns.index:
            er = expected_returns.loc[date].dropna()
            for sector in sectors:
                sector_tickers = [t for t in er.index if sector_map.get(t) == sector]
                if len(sector_tickers) < n_long + n_short:
                    continue

                sector_er = er[sector_tickers].sort_values(ascending=False)
                top = sector_er.head(n_long).index
                bottom = sector_er.tail(n_short).index

                weights.loc[date, top] = 1.0 / (len(sectors) * n_long)
                weights.loc[date, bottom] = -1.0 / (len(sectors) * n_short)

        return weights

    def beta_neutral(
        self,
        weights: pd.DataFrame,
        betas: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Adjusts portfolio weights to achieve zero portfolio beta.

        β_portfolio = Σ w_i · β_i → target = 0
        Adjustment: w_i' = w_i - (β_port / N) for simplicity,
        or more precisely via the minimum-variance hedge.
        """
        logger.info("Applying beta-neutrality constraint...")

        adjusted = weights.copy()
        for date in weights.index:
            w = weights.loc[date]
            b = betas.loc[date] if date in betas.index else pd.Series(1.0, index=w.index)

            active = w[w != 0]
            if active.empty:
                continue

            port_beta = (w * b).sum()
            n_active = (w != 0).sum()

            if n_active > 0 and not pd.isna(port_beta):
                # Distribute the beta adjustment across active positions
                adjustment = port_beta * b / (b ** 2).sum()
                adjusted.loc[date] = w - adjustment

        return adjusted


class PortfolioReturns:
    """
    Computes portfolio returns from weights and actual returns.
    """

    @staticmethod
    def compute(
        weights: pd.DataFrame,
        actual_returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Portfolio return = Σ w_i · r_i at each date.
        Weights from date t are applied to returns from t to t+1.
        """
        common = weights.index.intersection(actual_returns.index)
        port_ret = (weights.loc[common] * actual_returns.loc[common]).sum(axis=1)
        return port_ret
