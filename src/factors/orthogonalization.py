"""
Factor Orthogonalization

Multicollinearity between style factors inflates variance of Fama-MacBeth
premia estimates and makes it impossible to attribute returns to individual
factors.  This module provides two standard approaches:

  1. **Sequential Gram-Schmidt** orthogonalization
  2. **Cross-sectional residualization** (regress each factor on all others)

Both operate cross-sectionally (independently at each date).
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


# ======================================================================
# 1. Gram-Schmidt Sequential Orthogonalization
# ======================================================================

def gram_schmidt(
    factors: dict[str, pd.DataFrame],
    priority_order: list[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Applies sequential Gram-Schmidt to a dict of factor DataFrames.

    The first factor in `priority_order` is left untouched.  Each
    subsequent factor has its projection onto all preceding factors
    removed.

    Parameters
    ----------
    factors : dict  {name: DataFrame(index=date, columns=ticker)}
    priority_order : list of factor names in desired precedence.
        Defaults to dict insertion order if None.

    Returns
    -------
    dict of orthogonalized factor DataFrames.

    Notes
    -----
    The ordering matters economically:
      - Factors listed first retain their full variance.
      - Later factors capture only the *residual* exposure unexplained
        by preceding factors.
    Choose ordering based on economic conviction (e.g. Value first if
    your thesis centres on value premia).
    """
    if priority_order is None:
        priority_order = list(factors.keys())

    logger.info(
        f"Gram-Schmidt orthogonalization in order: {priority_order}"
    )

    # Convert to a 3-D structure: for each date, build NxK matrix
    # where N=stocks, K=factors, then orthogonalise columns.
    dates = factors[priority_order[0]].index
    tickers = factors[priority_order[0]].columns

    ortho = {name: factors[name].copy() for name in priority_order}

    for i, name_i in enumerate(priority_order[1:], start=1):
        for j in range(i):
            name_j = priority_order[j]
            # For each date, project factor_i onto factor_j and subtract
            ortho[name_i] = _subtract_projection_df(
                ortho[name_i], ortho[name_j]
            )

    return ortho


def _subtract_projection_df(
    target: pd.DataFrame, basis: pd.DataFrame
) -> pd.DataFrame:
    """
    Row-wise (cross-sectional) projection removal:
      target_perp = target - (target · basis / basis · basis) * basis
    """
    result = target.copy()
    for date_idx in target.index:
        t = target.loc[date_idx].values.astype(float)
        b = basis.loc[date_idx].values.astype(float)

        # Mask NaN for dot product
        valid = np.isfinite(t) & np.isfinite(b)
        if valid.sum() < 3:
            continue

        dot_tb = np.nansum(t[valid] * b[valid])
        dot_bb = np.nansum(b[valid] * b[valid])
        if dot_bb == 0:
            continue

        coeff = dot_tb / dot_bb
        result.loc[date_idx, valid] = t[valid] - coeff * b[valid]

    return result


# ======================================================================
# 2. Cross-Sectional Residualization
# ======================================================================

def residualize(
    factors: dict[str, pd.DataFrame],
    target_factor: str,
) -> pd.DataFrame:
    """
    Residualizes `target_factor` against all other factors in the dict,
    independently at each date.

    Runs a cross-sectional OLS at each date:
        target = α + Σ β_k · other_k + ε

    Returns the residual ε as the orthogonalised factor.
    """
    logger.info(f"Residualizing '{target_factor}' against all other factors...")

    other_names = [n for n in factors if n != target_factor]
    target_df = factors[target_factor]

    residual = target_df.copy()

    for date_idx in target_df.index:
        y = target_df.loc[date_idx].values.astype(float)

        # Build regressor matrix X [N x K]
        X_parts = []
        for name in other_names:
            X_parts.append(factors[name].loc[date_idx].values.astype(float))
        X = np.column_stack(X_parts)

        # Valid mask (no NaN in y or any X column)
        valid = np.isfinite(y)
        for col in range(X.shape[1]):
            valid &= np.isfinite(X[:, col])

        if valid.sum() < X.shape[1] + 2:
            continue  # not enough observations

        y_v = y[valid]
        X_v = X[valid]

        # Add intercept
        X_v = np.column_stack([np.ones(X_v.shape[0]), X_v])

        # OLS: β = (X'X)^{-1} X'y
        try:
            beta = np.linalg.lstsq(X_v, y_v, rcond=None)[0]
            fitted = X_v @ beta
            eps = y_v - fitted
            residual.iloc[residual.index.get_loc(date_idx), valid] = eps
        except np.linalg.LinAlgError:
            continue  # singular matrix, skip

    return residual


def residualize_all(
    factors: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Residualize every factor against all others.
    """
    logger.info(f"Residualizing all {len(factors)} factors against each other...")
    return {
        name: residualize(factors, name) for name in factors
    }


# ======================================================================
# Diagnostics
# ======================================================================

def factor_correlation_matrix(
    factors: dict[str, pd.DataFrame],
    date: str = None,
) -> pd.DataFrame:
    """
    Computes the cross-sectional correlation matrix of all factors
    at a specific date, or the average correlation across all dates.

    Useful for before/after orthogonalization comparison.
    """
    names = list(factors.keys())

    if date is not None:
        # Correlation at a single date
        combined = pd.DataFrame(
            {n: factors[n].loc[date] for n in names}
        )
        return combined.corr()

    # Average correlation across all dates
    all_corrs = []
    sample_dates = factors[names[0]].index
    for d in sample_dates:
        try:
            combined = pd.DataFrame({n: factors[n].loc[d] for n in names})
            c = combined.corr()
            if not c.isna().all().all():
                all_corrs.append(c)
        except Exception:
            continue

    if not all_corrs:
        return pd.DataFrame(np.nan, index=names, columns=names)

    avg_corr = sum(all_corrs) / len(all_corrs)
    return avg_corr
