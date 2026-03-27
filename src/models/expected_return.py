"""
Expected Return Model

Combines estimated factor premia with current factor exposures to produce
stock-level expected return forecasts:

    E[R_i] = Σ_k  λ̂_k · F_{ik}

where λ̂_k is the (shrunk or rolling-average) factor premium estimate
and F_{ik} is stock i's current exposure to factor k.

Provides:
  - Stock-level expected return ranking
  - Decile portfolio assignment
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


class ExpectedReturnModel:
    """
    Produces stock-level expected return forecasts from Fama-MacBeth output.
    """

    def __init__(self, rolling_window: int = 12):
        """
        Parameters
        ----------
        rolling_window : number of past periods (months) to average
            factor premia over. Smooths noisy single-period estimates.
        """
        self.rolling_window = rolling_window

    def estimate_premia(self, lambda_series: pd.DataFrame) -> pd.DataFrame:
        """
        Computes rolling-average factor premia for each date.
        """
        logger.info(f"Estimating premia with {self.rolling_window}-period rolling mean...")

        # Exclude intercept from premia if present
        cols = [c for c in lambda_series.columns if c != "intercept"]
        premia = lambda_series[cols].rolling(
            window=self.rolling_window, min_periods=max(3, self.rolling_window // 2)
        ).mean()

        return premia

    def forecast(
        self,
        lambda_series: pd.DataFrame,
        factors: dict[str, pd.DataFrame],
        dates: list = None,
    ) -> pd.DataFrame:
        """
        Generates stock-level expected return forecast for each date.

        Parameters
        ----------
        lambda_series : DataFrame from FamaMacBeth (index=date, cols=factors)
        factors : dict {name: DataFrame(index=date, columns=ticker)}

        Returns
        -------
        DataFrame (index=date, columns=ticker) of E[R]
        """
        premia = self.estimate_premia(lambda_series)
        factor_names = [c for c in premia.columns if c in factors]

        if dates is None:
            dates = sorted(premia.dropna(how="all").index)

        logger.info(f"Forecasting expected returns for {len(dates)} dates, {len(factor_names)} factors...")

        forecasts = {}
        for date in dates:
            if date not in premia.index or premia.loc[date].isna().all():
                continue

            lam = premia.loc[date]
            er = pd.Series(0.0, index=factors[factor_names[0]].columns, dtype=float)

            for fname in factor_names:
                if fname not in lam.index or pd.isna(lam[fname]):
                    continue
                if date not in factors[fname].index:
                    continue
                er += lam[fname] * factors[fname].loc[date]

            forecasts[date] = er

        return pd.DataFrame(forecasts).T

    def rank_stocks(self, expected_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional percentile rank (0=worst, 1=best) at each date.
        """
        return expected_returns.rank(axis=1, pct=True)

    def assign_deciles(self, expected_returns: pd.DataFrame, n_quantiles: int = 10) -> pd.DataFrame:
        """
        Assigns each stock to a decile (1=lowest E[R], 10=highest).
        """
        logger.info(f"Assigning stocks to {n_quantiles} quantile portfolios...")

        def _decile_row(row):
            valid = row.dropna()
            if len(valid) < n_quantiles:
                return pd.Series(np.nan, index=row.index)
            labels = pd.qcut(valid, q=n_quantiles, labels=False, duplicates="drop") + 1
            return labels.reindex(row.index)

        return expected_returns.apply(_decile_row, axis=1)

    def decile_returns(
        self,
        decile_assignments: pd.DataFrame,
        actual_returns: pd.DataFrame,
        n_quantiles: int = 10,
    ) -> pd.DataFrame:
        """
        Computes equal-weighted average return for each decile at each date.

        Returns
        -------
        DataFrame (index=date, columns=1..n_quantiles)
        """
        logger.info("Computing decile portfolio returns...")

        common_dates = decile_assignments.index.intersection(actual_returns.index)
        results = {}

        for date in common_dates:
            d = decile_assignments.loc[date]
            r = actual_returns.loc[date]
            row = {}
            for q in range(1, n_quantiles + 1):
                mask = d == q
                if mask.sum() > 0:
                    row[q] = r[mask].mean()
                else:
                    row[q] = np.nan
            results[date] = row

        return pd.DataFrame(results).T.sort_index()
