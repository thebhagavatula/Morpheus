"""
Cross-Sectional Regression

Single-period cross-sectional OLS:
    R_t = α + β₁F₁ + β₂F₂ + … + ε

Provides the building block for the Fama-MacBeth two-pass procedure.
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


class CrossSectionalRegression:
    """
    Runs a single cross-sectional OLS at one point in time.
    """

    def fit(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
        add_intercept: bool = True,
    ) -> dict:
        """
        Parameters
        ----------
        returns : Series  (index=tickers, values=forward returns)
        factors : DataFrame  (index=tickers, columns=factor names)
        add_intercept : bool

        Returns
        -------
        dict with keys: coefficients, residuals, r_squared, t_stats, std_errors
        """
        # Align and drop NaN
        combined = pd.concat([returns.rename("ret"), factors], axis=1).dropna()

        if len(combined) < factors.shape[1] + 2:
            return self._empty_result(factors.columns, add_intercept)

        y = combined["ret"].values
        X = combined[factors.columns].values.astype(float)

        if add_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
            col_names = ["intercept"] + list(factors.columns)
        else:
            col_names = list(factors.columns)

        # OLS: β = (X'X)⁻¹ X'y
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            return self._empty_result(factors.columns, add_intercept)

        beta = XtX_inv @ (X.T @ y)
        fitted = X @ beta
        residuals = y - fitted

        n, k = X.shape
        dof = n - k

        # Residual variance
        s2 = (residuals @ residuals) / max(dof, 1)

        # Standard errors
        var_beta = s2 * np.diag(XtX_inv)
        se = np.sqrt(np.maximum(var_beta, 0))

        # t-statistics
        t_stats = np.where(se > 0, beta / se, np.nan)

        # R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return {
            "coefficients": pd.Series(beta, index=col_names),
            "std_errors": pd.Series(se, index=col_names),
            "t_stats": pd.Series(t_stats, index=col_names),
            "residuals": pd.Series(residuals, index=combined.index),
            "r_squared": r2,
            "n_obs": n,
        }

    @staticmethod
    def _empty_result(factor_names, add_intercept):
        col_names = (["intercept"] + list(factor_names)) if add_intercept else list(factor_names)
        return {
            "coefficients": pd.Series(np.nan, index=col_names),
            "std_errors": pd.Series(np.nan, index=col_names),
            "t_stats": pd.Series(np.nan, index=col_names),
            "residuals": pd.Series(dtype=float),
            "r_squared": np.nan,
            "n_obs": 0,
        }
