"""
Backtesting Engine

Walk-forward backtesting framework with:
  - Configurable in-sample / out-of-sample split
  - Rolling window evaluation
  - Integration with CostModel for net returns
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

try:
    from src.portfolio.construction import PortfolioConstructor, PortfolioReturns
    from src.portfolio.costs import CostModel
    from src.models.expected_return import ExpectedReturnModel
    from src.models.fama_macbeth import FamaMacBeth
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.portfolio.construction import PortfolioConstructor, PortfolioReturns
    from src.portfolio.costs import CostModel
    from src.models.expected_return import ExpectedReturnModel
    from src.models.fama_macbeth import FamaMacBeth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """
    Walk-forward backtesting engine.

    At each rebalance date t:
      1. Use data up to t (in-sample window) to estimate factor premia.
      2. Forecast expected returns from current exposures.
      3. Construct portfolio.
      4. Measure realised return over the next period.
    """

    def __init__(
        self,
        in_sample_months: int = 36,
        rebalance_every: int = 1,  # months
        cost_bps: float = None,
    ):
        self.in_sample_months = in_sample_months
        self.rebalance_every = rebalance_every
        self.cost_model = CostModel(cost_bps) if cost_bps is not None else CostModel()
        self.er_model = ExpectedReturnModel(rolling_window=12)
        self.constructor = PortfolioConstructor()

    def run(
        self,
        returns: pd.DataFrame,
        factors: dict[str, pd.DataFrame],
        oos_start: str = None,
    ) -> dict:
        """
        Runs the full walk-forward backtest.

        Parameters
        ----------
        returns : DataFrame (index=date, columns=ticker) monthly returns
        factors : dict {name: DataFrame}
        oos_start : optional start date for out-of-sample period

        Returns
        -------
        dict with:
          - gross_returns: Series of portfolio gross returns
          - net_returns: Series of cost-adjusted returns
          - weights: DataFrame of portfolio weights over time
          - turnover: Series
          - is_oos: Series(bool) indicating out-of-sample periods
        """
        all_dates = sorted(returns.index)

        if oos_start is not None:
            oos_start_dt = pd.Timestamp(oos_start)
        else:
            # Default: first 60% is in-sample
            split_idx = int(len(all_dates) * 0.6)
            oos_start_dt = all_dates[split_idx]

        logger.info(f"Walk-forward backtest: OOS starts at {oos_start_dt}")

        # Rebalance dates (every rebalance_every months)
        rebal_dates = all_dates[::self.rebalance_every]

        all_weights = {}
        is_oos = {}

        for i, date in enumerate(rebal_dates):
            # Define in-sample window
            is_end = date
            is_start_idx = max(0, all_dates.index(date) - self.in_sample_months)
            is_start = all_dates[is_start_idx]
            is_window = [d for d in all_dates if is_start <= d <= is_end]

            if len(is_window) < 12:
                continue

            # Run Fama-MacBeth on in-sample
            fm = FamaMacBeth(newey_west_lags=4)
            result = fm.fit(returns, factors, dates=is_window)
            lambda_series = result["lambda_series"]

            if lambda_series.empty:
                continue

            # Forecast E[R] using current exposures
            er = self.er_model.forecast(lambda_series, factors, dates=[date])

            if er.empty:
                continue

            # Build decile portfolios
            deciles = self.er_model.assign_deciles(er)
            portfolio = self.constructor.long_short_decile(deciles)

            all_weights[date] = portfolio["net"].loc[date] if date in portfolio["net"].index else None
            is_oos[date] = date >= oos_start_dt

        # Assemble weight DataFrame
        weights_df = pd.DataFrame(all_weights).T.fillna(0)

        # Compute returns
        gross = PortfolioReturns.compute(weights_df, returns)

        # Cost-adjusted returns
        net = self.cost_model.adjust_returns(gross, weights_df)

        # Turnover
        turnover = self.cost_model.compute_turnover(weights_df)

        return {
            "gross_returns": gross,
            "net_returns": net,
            "weights": weights_df,
            "turnover": turnover,
            "is_oos": pd.Series(is_oos),
            "oos_start": oos_start_dt,
        }

    @staticmethod
    def split_results(results: dict) -> tuple[pd.Series, pd.Series]:
        """
        Splits returns into in-sample and out-of-sample.
        """
        oos_mask = results["is_oos"]
        net = results["net_returns"]

        common = net.index.intersection(oos_mask.index)
        is_ret = net.loc[common][~oos_mask.loc[common]]
        oos_ret = net.loc[common][oos_mask.loc[common]]

        return is_ret, oos_ret
