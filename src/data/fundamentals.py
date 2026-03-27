"""
Fundamental Data Ingestion

Handles acquiring corporate fundamental metrics (B/P, E/P, ROE, etc.)
from SimFin or Yahoo Finance as a fallback.
Critically, this lags fundamentals by 'FUNDAMENTAL_LAG_DAYS' from period end
date to approximate the point-in-time point they became public.

Compatibility:
  - simfin >= 0.10 (bulk download API)
  - Falls back to yfinance .info dict when simfin is unavailable
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Robust import
try:
    from src.config import CACHE_DIR, FUNDAMENTAL_LAG_DAYS, PROXIES, REQUEST_HEADERS, get_robust_session
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config import CACHE_DIR, FUNDAMENTAL_LAG_DAYS, PROXIES, REQUEST_HEADERS, get_robust_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FundamentalsEngine:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.fundamentals_file = self.cache_dir / "fundamentals_cache.parquet"

        # Attempt SimFin import — lazy, non-fatal
        self.sf = None
        self.simfin_available = False
        try:
            import simfin as sf

            simfin_data_dir = self.cache_dir / "simfin"
            simfin_data_dir.mkdir(parents=True, exist_ok=True)
            sf.set_data_dir(str(simfin_data_dir))
            sf.set_api_key("free")
            self.sf = sf
            self.simfin_available = True
        except ImportError:
            logger.warning(
                "simfin library not installed. "
                "Install it via `pip install simfin` or use the yfinance fallback."
            )
        except Exception as e:
            logger.warning(f"simfin initialisation failed: {e}")

    # ------------------------------------------------------------------
    # SimFin path
    # ------------------------------------------------------------------
    def fetch_simfin_fundamentals(self) -> pd.DataFrame:
        """
        Fetches US income statement and balance sheet data from SimFin bulk
        downloads. Merges them on Ticker + Report Date.
        """
        if not self.simfin_available:
            raise ImportError("simfin library is required for this method.")

        logger.info("Downloading SimFin datasets for US equities...")

        # Determine available API based on simfin version
        load_income = getattr(self.sf, "load_income", None)
        load_balance = getattr(self.sf, "load_balance", None)

        if load_income is None or load_balance is None:
            raise AttributeError(
                "Your simfin version does not expose load_income / load_balance. "
                "Please upgrade: pip install --upgrade simfin"
            )

        df_income = load_income(dataset="us", variant="ttm-asreported")
        df_balance = load_balance(dataset="us", variant="quarterly-asreported")

        # Both DataFrames use a MultiIndex (Ticker, Report Date) by default.
        # Reset so we can merge on explicit columns.
        if isinstance(df_income.index, pd.MultiIndex):
            df_income = df_income.reset_index()
        if isinstance(df_balance.index, pd.MultiIndex):
            df_balance = df_balance.reset_index()

        # Find common merge keys
        merge_keys = [
            c for c in ["Ticker", "Report Date"] if c in df_income.columns and c in df_balance.columns
        ]
        if not merge_keys:
            raise KeyError(
                "Cannot find common merge keys (Ticker, Report Date) in SimFin data."
            )

        df = pd.merge(
            df_income,
            df_balance,
            on=merge_keys,
            how="inner",
            suffixes=("", "_bs"),
        )

        return df

    # ------------------------------------------------------------------
    # yfinance fallback path
    # ------------------------------------------------------------------
    def fetch_yfinance_fundamentals(self, tickers: list) -> pd.DataFrame:
        """
        Lightweight fallback: pulls the latest snapshot of key fundamental
        ratios from yfinance's .info dict.  This only gives *current* values
        so it is less useful for a backtest, but lets the pipeline run
        without simfin.

        Uses proxy settings from config if available.
        """
        import os
        import yfinance as yf

        # Inject proxy into environment so yfinance routes through it
        if PROXIES:
            logger.info(f"Using proxy for yfinance fundamentals: {list(PROXIES.keys())}")
            if "https" in PROXIES:
                os.environ["HTTPS_PROXY"] = PROXIES["https"]
            if "http" in PROXIES:
                os.environ["HTTP_PROXY"] = PROXIES["http"]

        logger.info(
            f"Fetching yfinance fundamentals for {len(tickers)} tickers (snapshot only)..."
        )

        # Create a proxy-aware session for yf.Ticker
        session = get_robust_session()

        records = []
        for ticker_symbol in tickers:
            try:
                ticker_obj = yf.Ticker(ticker_symbol, session=session)
                info = ticker_obj.info
                records.append(
                    {
                        "Ticker": ticker_symbol,
                        "Report Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                        "Market Cap": info.get("marketCap"),
                        "Book Value Per Share": info.get("bookValue"),
                        "Trailing EPS": info.get("trailingEps"),
                        "Forward EPS": info.get("forwardEps"),
                        "EBITDA": info.get("ebitda"),
                        "Enterprise Value": info.get("enterpriseValue"),
                        "Return On Equity": info.get("returnOnEquity"),
                        "Debt To Equity": info.get("debtToEquity"),
                        "Current Price": info.get("currentPrice"),
                    }
                )
            except Exception as exc:
                if "403" in str(exc):
                    logger.warning(
                        f"{ticker_symbol}: HTTP 403 — blocked by Yahoo Finance. "
                        "Consider setting HTTP_PROXY/HTTPS_PROXY in .env"
                    )
                else:
                    logger.debug(f"Skipping {ticker_symbol}: {exc}")

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Point-in-time lagging
    # ------------------------------------------------------------------
    def lag_fundamentals_point_in_time(
        self, df: pd.DataFrame, report_date_col: str = "Report Date"
    ) -> pd.DataFrame:
        """
        Lags fundamentals to prevent look-ahead bias.
        A Q1 report ending March 31 isn't available until ~May.
        By adding FUNDAMENTAL_LAG_DAYS we place the value at the date it
        became tradable information.
        """
        logger.info(
            f"Lagging fundamentals by {FUNDAMENTAL_LAG_DAYS} days "
            "to simulate point-in-time..."
        )

        df = df.copy()
        df[report_date_col] = pd.to_datetime(df[report_date_col], errors="coerce")
        df["Available Date"] = df[report_date_col] + pd.Timedelta(
            days=FUNDAMENTAL_LAG_DAYS
        )

        return df

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def get_fundamentals(self, tickers: list = None) -> pd.DataFrame:
        """
        Returns fundamental data, using cache if available.
        Falls back from SimFin -> yfinance if needed.
        """
        if self.fundamentals_file.exists():
            logger.info("Loading fundamentals from cache...")
            return pd.read_parquet(self.fundamentals_file)

        # Try SimFin first
        if self.simfin_available:
            try:
                df = self.fetch_simfin_fundamentals()
                df = self.lag_fundamentals_point_in_time(df)
                df.to_parquet(self.fundamentals_file)
                return df
            except Exception as e:
                logger.warning(f"SimFin fetch failed ({e}), falling back to yfinance.")

        # yfinance fallback
        if tickers is None or len(tickers) == 0:
            raise ValueError(
                "Tickers must be provided for yfinance fallback "
                "when SimFin is unavailable and no cache exists."
            )

        df = self.fetch_yfinance_fundamentals(tickers)
        df = self.lag_fundamentals_point_in_time(df)
        df.to_parquet(self.fundamentals_file)
        return df


if __name__ == "__main__":
    engine = FundamentalsEngine()
    print(
        "FundamentalsEngine initialised. "
        f"SimFin available: {engine.simfin_available}. "
        "Call get_fundamentals(tickers=...) to fetch data."
    )
