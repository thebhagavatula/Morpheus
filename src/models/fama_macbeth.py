"""
Fama-MacBeth Two-Pass Regression

The gold-standard method for estimating factor risk premia:

  Pass 1 (Time-series of cross-sections):
      At each date t, run  R_t = α_t + Σ λ_{kt} · F_{kt} + ε_t
      This yields a time-series of λ̂_{kt} for each factor k.

  Pass 2 (Inference across time):
      λ̄_k  = (1/T) Σ λ̂_{kt}
      t(λ̄) = λ̄_k / SE(λ̄_k)

      SE uses Newey-West correction for serial correlation.

References:
  Fama & MacBeth (1973), "Risk, Return, and Equilibrium: Empirical Tests"
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

try:
    from src.models.cross_sectional import CrossSectionalRegression
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.models.cross_sectional import CrossSectionalRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FamaMacBeth:
    """
    Two-pass Fama-MacBeth estimator for cross-sectional factor premia.
    """

    def __init__(self, newey_west_lags: int = 6):
        self.newey_west_lags = newey_west_lags
        self.cs_reg = CrossSectionalRegression()
        self.lambda_series = None  # populated after fit()

    def fit(
        self,
        returns: pd.DataFrame,
        factors: dict[str, pd.DataFrame],
        dates: list = None,
    ) -> dict:
        """
        Parameters
        ----------
        returns : DataFrame (index=date, columns=ticker) of forward returns
        factors : dict {name: DataFrame(index=date, columns=ticker)}
        dates   : optional subset of dates to use

        Returns
        -------
        dict with:
          - lambda_series: DataFrame (index=date, columns=factor) of estimated premia
          - summary: DataFrame with mean_lambda, t_stat, nw_t_stat, std_error, nw_std_error
        """
        if dates is None:
            dates = sorted(returns.index)

        factor_names = list(factors.keys())
        all_lambdas = []

        logger.info(f"Running Fama-MacBeth over {len(dates)} periods for {len(factor_names)} factors...")

        for date in dates:
            if date not in returns.index:
                continue

            # Build factor exposure matrix for this date
            factor_row = {}
            skip = False
            for name in factor_names:
                if date not in factors[name].index:
                    skip = True
                    break
                factor_row[name] = factors[name].loc[date]

            if skip:
                continue

            ret_t = returns.loc[date]
            fac_t = pd.DataFrame(factor_row)

            result = self.cs_reg.fit(ret_t, fac_t, add_intercept=True)

            if result["n_obs"] > 0:
                lambdas = result["coefficients"]
                lambdas.name = date
                all_lambdas.append(lambdas)

        if not all_lambdas:
            logger.warning("No valid cross-sections found.")
            return {"lambda_series": pd.DataFrame(), "summary": pd.DataFrame()}

        lambda_df = pd.DataFrame(all_lambdas)
        self.lambda_series = lambda_df

        # --- Pass 2: time-series inference ---
        summary = self._compute_summary(lambda_df)

        return {"lambda_series": lambda_df, "summary": summary}

    def _compute_summary(self, lambda_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes mean λ, OLS t-stat, and Newey-West adjusted t-stat.
        """
        results = []

        for col in lambda_df.columns:
            series = lambda_df[col].dropna()
            T = len(series)
            if T < 3:
                results.append({
                    "factor": col, "mean_lambda": np.nan,
                    "std_error": np.nan, "t_stat": np.nan,
                    "nw_std_error": np.nan, "nw_t_stat": np.nan, "n_periods": T,
                })
                continue

            mean_lam = series.mean()
            # OLS standard error: σ/√T
            ols_se = series.std(ddof=1) / np.sqrt(T)
            ols_t = mean_lam / ols_se if ols_se > 0 else np.nan

            # Newey-West standard error
            nw_se = self._newey_west_se(series.values, self.newey_west_lags)
            nw_t = mean_lam / nw_se if nw_se > 0 else np.nan

            results.append({
                "factor": col,
                "mean_lambda": mean_lam,
                "std_error": ols_se,
                "t_stat": ols_t,
                "nw_std_error": nw_se,
                "nw_t_stat": nw_t,
                "n_periods": T,
            })

        return pd.DataFrame(results).set_index("factor")

    @staticmethod
    def _newey_west_se(x: np.ndarray, max_lags: int) -> float:
        """
        Newey-West (1987) HAC standard error for the mean of x.
        """
        T = len(x)
        x_dm = x - x.mean()  # de-meaned

        # Variance term (lag 0)
        gamma_0 = np.sum(x_dm ** 2) / T

        # Auto-covariance terms with Bartlett kernel
        nw_sum = gamma_0
        for lag in range(1, min(max_lags, T - 1) + 1):
            weight = 1.0 - lag / (max_lags + 1)
            gamma_l = np.sum(x_dm[lag:] * x_dm[:-lag]) / T
            nw_sum += 2 * weight * gamma_l

        # SE of the mean
        var_mean = nw_sum / T
        return np.sqrt(max(var_mean, 0))


def run_fama_macbeth(
    returns: pd.DataFrame,
    factors: dict[str, pd.DataFrame],
    newey_west_lags: int = 6,
) -> dict:
    """Convenience function."""
    fm = FamaMacBeth(newey_west_lags=newey_west_lags)
    return fm.fit(returns, factors)
