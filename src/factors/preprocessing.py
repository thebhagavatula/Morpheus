"""
Factor Preprocessing Pipeline

Applies the standard quant preprocessing chain to raw factor exposures:
  1. Winsorization  — clamp outliers at configurable percentile thresholds
  2. Missing data   — drop tickers with >50% NaN, else fill with sector median
  3. Z-score        — cross-sectional standardisation (mean 0, std 1) per date
  4. Clipping       — hard cap at ±Z_SCORE_CLIP standard deviations

All operations are *cross-sectional* (applied independently at each point
in time) to avoid information leakage across the time axis.
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

try:
    from src.config import WINSORIZE_LIMITS, Z_SCORE_CLIP, SECTOR_COLUMN
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config import WINSORIZE_LIMITS, Z_SCORE_CLIP, SECTOR_COLUMN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def winsorize_cross_section(
    factor: pd.DataFrame,
    lower: float = WINSORIZE_LIMITS[0],
    upper: float = WINSORIZE_LIMITS[1],
) -> pd.DataFrame:
    """
    Row-wise (cross-sectional) winsorization.

    For each date, values below the `lower` percentile are set to that
    percentile value, and values above the `upper` percentile are capped
    at that value.
    """
    logger.info(f"Winsorizing at ({lower:.2%}, {upper:.2%})...")

    def _winsorize_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if valid.empty:
            return row
        lo = valid.quantile(lower)
        hi = valid.quantile(upper)
        return row.clip(lower=lo, upper=hi)

    return factor.apply(_winsorize_row, axis=1)


def fill_missing_values(
    factor: pd.DataFrame,
    sector_map: pd.Series = None,
    max_missing_pct: float = 0.50,
) -> pd.DataFrame:
    """
    Two-step missing data treatment:
      1. Drop tickers where more than `max_missing_pct` of dates are NaN.
      2. Fill remaining NaN with sector-median (if sector_map provided)
         or cross-sectional median.

    Parameters
    ----------
    factor : DataFrame (index=date, columns=ticker)
    sector_map : Series  ticker -> sector label  (optional)
    max_missing_pct : float  threshold for dropping a ticker entirely
    """
    logger.info("Treating missing factor values...")

    # Step 1: prune tickers with excessive missing data
    missing_pct = factor.isna().mean()
    drop_tickers = missing_pct[missing_pct > max_missing_pct].index.tolist()
    if drop_tickers:
        logger.info(
            f"Dropping {len(drop_tickers)} tickers with >{max_missing_pct:.0%} missing data."
        )
        factor = factor.drop(columns=drop_tickers)

    # Step 2: fill remaining NaN
    if sector_map is not None and not sector_map.empty:
        factor = _fill_with_sector_median(factor, sector_map)
    else:
        # Simple cross-sectional median fill
        factor = factor.apply(lambda row: row.fillna(row.median()), axis=1)

    return factor


def _fill_with_sector_median(
    factor: pd.DataFrame, sector_map: pd.Series
) -> pd.DataFrame:
    """
    For each date, fills NaN values with the median of the same sector.
    Falls back to the global cross-sectional median if the sector median
    is also NaN.
    """
    filled = factor.copy()
    for date_idx in filled.index:
        row = filled.loc[date_idx]
        global_median = row.median()
        for ticker in row.index[row.isna()]:
            sector = sector_map.get(ticker)
            if sector is not None:
                sector_peers = [
                    t for t in row.index
                    if sector_map.get(t) == sector and pd.notna(row[t])
                ]
                if sector_peers:
                    filled.at[date_idx, ticker] = row[sector_peers].median()
                    continue
            # Fallback
            filled.at[date_idx, ticker] = global_median
    return filled


def zscore_cross_section(
    factor: pd.DataFrame,
    clip: float = Z_SCORE_CLIP,
) -> pd.DataFrame:
    """
    Cross-sectional z-score normalisation per date, then clip to ±clip.

    z_i = (x_i - μ) / σ     (computed across stocks at each date)
    """
    logger.info(f"Z-scoring cross-sectionally, clip=±{clip}...")

    mean = factor.mean(axis=1)
    std = factor.std(axis=1).replace(0, np.nan)

    z = factor.sub(mean, axis=0).div(std, axis=0)

    # Hard clip
    z = z.clip(lower=-clip, upper=clip)

    return z


def preprocess_factor(
    raw_factor: pd.DataFrame,
    sector_map: pd.Series = None,
    winsorize: bool = True,
    fill_missing: bool = True,
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for a single factor.

    Parameters
    ----------
    raw_factor : DataFrame (index=date, columns=ticker, values=raw exposure)
    sector_map : optional Series ticker->sector for sector-aware fill
    winsorize, fill_missing, standardize : bool toggles for each step

    Returns
    -------
    DataFrame of preprocessed factor exposures.
    """
    result = raw_factor.copy()

    if winsorize:
        result = winsorize_cross_section(result)
    if fill_missing:
        result = fill_missing_values(result, sector_map=sector_map)
    if standardize:
        result = zscore_cross_section(result)

    return result


def preprocess_all_factors(
    factors: dict[str, pd.DataFrame],
    sector_map: pd.Series = None,
) -> dict[str, pd.DataFrame]:
    """
    Applies the full preprocessing pipeline to every factor in the dict.
    """
    logger.info(f"Preprocessing {len(factors)} factors...")
    return {
        name: preprocess_factor(raw, sector_map=sector_map)
        for name, raw in factors.items()
    }
