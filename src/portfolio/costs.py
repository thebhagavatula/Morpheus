"""
Transaction Cost and Turnover Analysis

Models trading friction to produce cost-adjusted portfolio returns:
  - Configurable cost in basis points (one-way)
  - Turnover calculation (portfolio weight changes)
  - Capacity analysis (estimated max AUM before returns degrade)
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

try:
    from src.config import TRANSACTION_COST_BPS
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config import TRANSACTION_COST_BPS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CostModel:
    """
    Linear transaction cost model.
    """

    def __init__(self, cost_bps: float = TRANSACTION_COST_BPS):
        self.cost_rate = cost_bps / 10_000  # Convert bps to decimal

    def compute_turnover(self, weights: pd.DataFrame) -> pd.Series:
        """
        Turnover = (1/2) Σ |w_t - w_{t-1}| at each rebalance.
        One-way turnover (the proportion of the portfolio traded).
        """
        weight_changes = weights.diff().abs()
        turnover = weight_changes.sum(axis=1, min_count=1) / 2.0
        return turnover

    def compute_costs(self, weights: pd.DataFrame) -> pd.Series:
        """
        Trading cost at each rebalance = turnover × cost_rate × 2
        (×2 because we pay on both the buy and sell side).
        """
        turnover = self.compute_turnover(weights)
        costs = turnover.fillna(0.0) * self.cost_rate * 2
        return costs

    def adjust_returns(
        self,
        gross_returns: pd.Series,
        weights: pd.DataFrame,
    ) -> pd.Series:
        """
        Net return = gross return - transaction costs.
        """
        costs = self.compute_costs(weights)
        common = gross_returns.index.intersection(costs.index)
        net = gross_returns.loc[common] - costs.loc[common]
        return net

    def turnover_summary(self, weights: pd.DataFrame) -> dict:
        """
        Summary statistics for turnover.
        """
        turnover = self.compute_turnover(weights).dropna()
        if turnover.empty:
            return {"mean": np.nan, "median": np.nan, "max": np.nan, "total_cost_drag_annualized": np.nan}

        mean_to = turnover.mean()
        return {
            "mean_turnover": mean_to,
            "median_turnover": turnover.median(),
            "max_turnover": turnover.max(),
            "annualized_cost_drag_bps": mean_to * self.cost_rate * 2 * 12 * 10_000,  # monthly → annual
        }

    def capacity_analysis(
        self,
        weights: pd.DataFrame,
        avg_daily_volume: pd.DataFrame,
        max_participation_rate: float = 0.10,
    ) -> float:
        """
        Estimates maximum portfolio AUM (in USD) such that trading
        does not exceed `max_participation_rate` of average daily volume.

        Returns estimated capacity in dollars.
        """
        turnover = self.compute_turnover(weights)
        mean_to = turnover.mean()
        if mean_to == 0 or avg_daily_volume.empty:
            return np.inf

        # Average daily volume across all held stocks
        mean_adv = avg_daily_volume.mean().mean()

        # Max AUM ≈ (max_participation × mean_ADV × n_stocks) / mean_turnover
        n_stocks = (weights != 0).sum(axis=1).mean()
        capacity = (max_participation_rate * mean_adv * n_stocks) / mean_to

        logger.info(f"Estimated capacity: ${capacity:,.0f}")
        return capacity
