"""
Main Pipeline Runner

End-to-end execution of the cross-sectional expected return model:
  1. Fetch universe & price data
  2. Construct factors
  3. Preprocess & orthogonalize
  4. Run Fama-MacBeth regression
  5. Generate expected return forecasts
  6. Build portfolios
  7. Backtest with costs
  8. Compute performance metrics
  9. Generate research-grade visualizations

Usage:
    python -m src.main
    python -m src.main --skip-download   (use cached data)
"""

import argparse
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CACHE_DIR, PROCESSED_DATA_DIR, UNIVERSE_START_DATE, UNIVERSE_END_DATE,
    TRANSACTION_COST_BPS, BASE_DIR,
)
from src.data.universe import UniverseBuilder
from src.data.prices import PriceEngine
from src.factors.style_factors import (
    MomentumFactor, SizeFactor, LowVolatilityFactor, LiquidityFactor,
    build_price_factors,
)
from src.factors.alpha_factor import EarningsSurpriseMomentum
from src.factors.preprocessing import preprocess_all_factors
from src.factors.orthogonalization import gram_schmidt, factor_correlation_matrix
from src.models.fama_macbeth import FamaMacBeth
from src.models.expected_return import ExpectedReturnModel
from src.portfolio.construction import PortfolioConstructor, PortfolioReturns
from src.portfolio.costs import CostModel
from src.analytics.metrics import (
    information_coefficient, ic_summary, sharpe_ratio, sortino_ratio,
    max_drawdown, alpha_beta, rolling_t_stat, full_performance_report,
)
from src.analytics.visualizations import (
    plot_rolling_factor_premia, plot_correlation_heatmap, plot_decile_performance,
    plot_drawdown, plot_rolling_t_stat, plot_regime_comparison, plot_cumulative_returns,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = BASE_DIR / "reports"


def main(skip_download: bool = False):
    """Full pipeline execution."""

    # ── Step 1: Universe & Price Data ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Data Acquisition")
    logger.info("=" * 60)

    universe_builder = UniverseBuilder()
    tickers = universe_builder.get_universe(bypass_cache=not skip_download)
    logger.info(f"Universe: {len(tickers)} tickers")

    price_engine = PriceEngine()
    prices = price_engine.load_prices(
        bypass_cache=not skip_download,
        tickers=tickers,
    )
    logger.info(f"Prices: {prices.shape[0]} dates × {prices.shape[1]} tickers")

    # Monthly returns (for Fama-MacBeth)
    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna(how="all")

    # Forward returns (for IC calculations)
    fwd_returns = price_engine.calculate_forward_returns(monthly_prices, periods=1)

    # ── Step 2: Factor Construction ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Factor Construction")
    logger.info("=" * 60)

    # Price-based factors on daily data, then resample to monthly
    daily_factors = build_price_factors(prices)

    # Resample to month-end
    factors = {}
    for name, df in daily_factors.items():
        factors[name] = df.resample("ME").last()

    # Custom alpha: Earnings Surprise Momentum (using price data as proxy)
    # Since we don't have EPS time-series without SimFin, we approximate
    # using price momentum divergence as an earnings surprise proxy
    esm = EarningsSurpriseMomentum()
    # Use daily price data to create a synthetic EPS proxy
    rolling_earnings_proxy = prices.rolling(63).mean()  # 3-month rolling avg
    sue = esm.compute_sue(rolling_earnings_proxy)
    factors["earnings_surprise"] = sue.resample("ME").last()

    logger.info(f"Factors constructed: {list(factors.keys())}")

    # ── Step 3: Preprocessing ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Factor Preprocessing")
    logger.info("=" * 60)

    processed_factors = preprocess_all_factors(factors)
    logger.info("Preprocessing complete (winsorize → fill → z-score)")

    # ── Step 4: Factor Diagnostics & Orthogonalization ────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Factor Diagnostics")
    logger.info("=" * 60)

    # Correlation before orthogonalization
    corr_before = factor_correlation_matrix(processed_factors)
    logger.info(f"Pre-orthogonalization correlation matrix:\n{corr_before.round(3)}")
    plot_correlation_heatmap(corr_before, "Factor Correlation (Before Orthogonalization)", "corr_before.png")

    # Orthogonalize
    ortho_order = ["momentum", "low_volatility", "liquidity", "earnings_surprise"]
    # Only orthogonalize factors that exist
    ortho_order = [f for f in ortho_order if f in processed_factors]
    ortho_factors = gram_schmidt(processed_factors, priority_order=ortho_order) if ortho_order else processed_factors

    corr_after = factor_correlation_matrix(ortho_factors)
    logger.info(f"Post-orthogonalization correlation matrix:\n{corr_after.round(3)}")
    plot_correlation_heatmap(corr_after, "Factor Correlation (After Orthogonalization)", "corr_after.png")

    # ── Step 5: Fama-MacBeth Regression ───────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Fama-MacBeth Regression")
    logger.info("=" * 60)

    fm = FamaMacBeth(newey_west_lags=6)
    fm_result = fm.fit(monthly_returns, ortho_factors)
    lambda_series = fm_result["lambda_series"]
    summary = fm_result["summary"]

    logger.info(f"\n{'=' * 50}")
    logger.info("FACTOR PREMIA SUMMARY")
    logger.info(f"{'=' * 50}")
    logger.info(f"\n{summary.to_string()}")

    # Save summary
    summary.to_csv(REPORTS_DIR / "factor_premia_summary.csv")
    plot_rolling_factor_premia(lambda_series)

    # ── Step 6: Expected Return Forecasts ─────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Expected Return Model")
    logger.info("=" * 60)

    er_model = ExpectedReturnModel(rolling_window=12)
    forecasts = er_model.forecast(lambda_series, ortho_factors)
    deciles = er_model.assign_deciles(forecasts)
    decile_ret = er_model.decile_returns(deciles, monthly_returns)

    logger.info(f"Forecasts generated for {len(forecasts)} periods")
    plot_decile_performance(decile_ret)

    # ── Step 7: Portfolio Construction ────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: Portfolio Construction")
    logger.info("=" * 60)

    constructor = PortfolioConstructor()
    portfolio = constructor.long_short_decile(deciles)
    weights = portfolio["net"]

    # Apply market neutrality
    weights = constructor.market_neutral(weights)

    gross_returns = PortfolioReturns.compute(weights, monthly_returns)

    # ── Step 8: Cost-Adjusted Returns ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 8: Transaction Costs & Backtest")
    logger.info("=" * 60)

    cost_model = CostModel(cost_bps=TRANSACTION_COST_BPS)
    net_returns = cost_model.adjust_returns(gross_returns, weights)
    turnover_stats = cost_model.turnover_summary(weights)

    logger.info(f"Turnover stats: {turnover_stats}")

    # ── Step 9: Performance Evaluation ────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 9: Performance Evaluation")
    logger.info("=" * 60)

    # IC analysis
    ic = information_coefficient(forecasts, monthly_returns)
    ic_stats = ic_summary(ic)
    logger.info(f"IC Summary: {ic_stats}")

    # Portfolio performance
    gross_report = full_performance_report(gross_returns)
    net_report = full_performance_report(net_returns)

    logger.info(f"\nGross Performance:\n{pd.Series(gross_report).to_string()}")
    logger.info(f"\nNet Performance:\n{pd.Series(net_report).to_string()}")

    # Rolling t-stat
    t_stat = rolling_t_stat(net_returns, window=24)

    # ── Step 10: Visualizations ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 10: Generating Research Visualizations")
    logger.info("=" * 60)

    plot_drawdown(net_returns, "Portfolio Drawdown (Net of Costs)")
    plot_rolling_t_stat(t_stat)
    plot_regime_comparison(net_returns)
    plot_cumulative_returns({
        "Gross L/S": gross_returns,
        "Net L/S": net_returns,
    })

    # ── Save Results ──────────────────────────────────────────────────
    results = {
        "gross_report": gross_report,
        "net_report": net_report,
        "ic_summary": ic_stats,
        "turnover": turnover_stats,
        "factor_premia": summary.to_dict(),
    }

    pd.DataFrame([gross_report]).to_csv(REPORTS_DIR / "gross_performance.csv", index=False)
    pd.DataFrame([net_report]).to_csv(REPORTS_DIR / "net_performance.csv", index=False)
    ic.to_csv(REPORTS_DIR / "ic_series.csv")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Reports saved to: {REPORTS_DIR}")
    logger.info(f"Figures saved to: {REPORTS_DIR / 'figures'}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Sectional Expected Return Model Pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Use cached data instead of re-downloading")
    args = parser.parse_args()
    main(skip_download=args.skip_download)
