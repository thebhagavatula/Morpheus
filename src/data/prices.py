"""
Price Data and Return Calculation Engine

Extracts historical prices, handling splits and dividends via adjusted close.
Computes point-in-time forward returns for the expected return model.

Compatibility:
  - yfinance >= 0.2.x (handles both auto_adjust=True and legacy Adj Close)
  - Handles single-ticker and multi-ticker downloads gracefully
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Robust import: works whether run as module or standalone script
try:
    from src.config import CACHE_DIR, UNIVERSE_START_DATE, UNIVERSE_END_DATE, PROXIES, REQUEST_HEADERS
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config import CACHE_DIR, UNIVERSE_START_DATE, UNIVERSE_END_DATE, PROXIES, REQUEST_HEADERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceEngine:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.prices_file = self.cache_dir / "prices_cache.parquet"

    def _extract_adj_close(self, data: pd.DataFrame, tickers: list) -> pd.DataFrame:
        """
        Robustly extracts adjusted close prices from yfinance output.
        Handles:
          - MultiIndex columns (multi-ticker download)
          - Flat columns (single-ticker download)
          - Both 'Adj Close' (legacy) and 'Close' (auto_adjust=True) column names
        """
        prices_dict = {}

        if isinstance(data.columns, pd.MultiIndex):
            # Multi-ticker: columns are (Ticker, Field) or (Field, Ticker)
            top_level = data.columns.get_level_values(0).unique().tolist()

            # Determine axis orientation by checking if tickers appear in level 0
            tickers_in_top = [t for t in tickers if t in top_level]

            if tickers_in_top:
                # Grouped by ticker: (Ticker, Field)
                for ticker in tickers:
                    if ticker not in top_level:
                        continue
                    ticker_data = data[ticker]
                    col = self._pick_close_column(ticker_data.columns)
                    if col is not None:
                        prices_dict[ticker] = ticker_data[col]
            else:
                # Grouped by field: (Field, Ticker)
                col = self._pick_close_column(top_level)
                if col is not None:
                    for ticker in tickers:
                        if ticker in data[col].columns:
                            prices_dict[ticker] = data[col][ticker]
        else:
            # Single ticker — flat columns
            col = self._pick_close_column(data.columns)
            if col is not None and len(tickers) == 1:
                prices_dict[tickers[0]] = data[col]

        return pd.DataFrame(prices_dict)

    @staticmethod
    def _pick_close_column(columns) -> str | None:
        """Pick the best available close-price column name."""
        for candidate in ['Adj Close', 'Close']:
            if candidate in columns:
                return candidate
        return None

    def download_prices(
        self,
        tickers: list,
        start_date: str = UNIVERSE_START_DATE,
        end_date: str = UNIVERSE_END_DATE,
    ) -> pd.DataFrame:
        """
        Batch downloads adjusted close prices for the provided tickers.
        Uses proxy settings from config if available.
        """
        logger.info(
            f"Downloading price data for {len(tickers)} tickers "
            f"from {start_date} to {end_date}..."
        )

        # Inject proxy into environment so yfinance's internal requests use it
        import os
        if PROXIES:
            logger.info(f"Using proxy for yfinance: {list(PROXIES.keys())}")
            if "https" in PROXIES:
                os.environ["HTTPS_PROXY"] = PROXIES["https"]
            if "http" in PROXIES:
                os.environ["HTTP_PROXY"] = PROXIES["http"]

        # Build download kwargs — handle yfinance version differences
        download_kwargs = dict(
            tickers=tickers,
            start=start_date,
            end=end_date,
            group_by="ticker",
            threads=True,
            progress=False,
        )

        # Prefer auto_adjust=False to get 'Adj Close' explicitly,
        # but fall back if the kwarg is no longer supported.
        try:
            data = yf.download(**download_kwargs, auto_adjust=False)
        except TypeError:
            data = yf.download(**download_kwargs)
        except Exception as e:
            if "403" in str(e):
                logger.error(
                    f"HTTP 403 Forbidden from Yahoo Finance: {e}. "
                    "Try setting HTTP_PROXY/HTTPS_PROXY in your .env file."
                )
            raise

        df_prices = self._extract_adj_close(data, tickers)

        if df_prices.empty:
            logger.warning("No price data was retrieved. Check tickers or date range.")
            return df_prices

        # Drop columns that are entirely NaN (delisted / never traded)
        df_prices = df_prices.dropna(axis=1, how="all")

        # Save to parquet cache
        df_prices.to_parquet(self.prices_file)
        logger.info(
            f"Price data shape {df_prices.shape} cached to {self.prices_file}"
        )

        return df_prices

    def load_prices(
        self, bypass_cache: bool = False, tickers: list = None
    ) -> pd.DataFrame:
        """
        Loads cached prices or downloads if necessary.
        """
        if self.prices_file.exists() and not bypass_cache:
            logger.info("Loading prices from cache...")
            return pd.read_parquet(self.prices_file)

        if tickers is None or len(tickers) == 0:
            raise ValueError(
                "Tickers must be provided if cache is empty or bypassed."
            )

        return self.download_prices(tickers)

    def calculate_forward_returns(
        self, prices: pd.DataFrame, periods: int = 21
    ) -> pd.DataFrame:
        """
        Calculates expected forward returns.
        periods=21 ≈ 1 trading month.
        Return at time t = (P_{t+periods} / P_t) - 1
        """
        logger.info(f"Computing {periods}-day forward returns...")

        future_prices = prices.shift(-periods)
        fwd_returns = (future_prices / prices) - 1.0

        return fwd_returns


if __name__ == "__main__":
    from src.data.universe import UniverseBuilder

    builder = UniverseBuilder()
    tickers = builder.get_universe()[:20]  # Test with 20

    engine = PriceEngine()
    prices = engine.load_prices(bypass_cache=True, tickers=tickers)
    fwd_ret = engine.calculate_forward_returns(prices)

    print("Latest Prices:")
    print(prices.tail())
