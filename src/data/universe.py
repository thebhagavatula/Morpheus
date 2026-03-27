"""
Universe Selection Module

Responsible for defining the investable universe of stocks.
Supports multiple indexes:
  - S&P 500 (US)
  - NIFTY 50 (India)
  - NIFTY NEXT 50 (India)

Due to the limitations of free public data, we approximate survivorship-bias
control by collecting current constituents from Wikipedia.
"""

import pandas as pd
import logging
import sys
from pathlib import Path

# Robust import: works whether run as module or standalone script
try:
    from src.config import PROCESSED_DATA_DIR, get_robust_session, REQUEST_HEADERS
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config import PROCESSED_DATA_DIR, get_robust_session, REQUEST_HEADERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Supported Index Registry ─────────────────────────────────────────
SUPPORTED_INDEXES = {
    "S&P 500": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "suffix": "",            # US tickers — no suffix for yfinance
        "fallback": ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                     "JPM", "JNJ", "V", "PG", "XOM"],
        "description": "US Large-Cap (500 stocks)",
    },
    "NIFTY 50": {
        "url": "https://en.wikipedia.org/wiki/NIFTY_50",
        "suffix": ".NS",         # NSE suffix for yfinance
        "fallback": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
                     "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS",
                     "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS"],
        "description": "India Large-Cap (50 stocks)",
    },
    "NIFTY NEXT 50": {
        "url": "https://en.wikipedia.org/wiki/NIFTY_Next_50",
        "suffix": ".NS",
        "fallback": ["ADANIGREEN.NS", "BANKBARODA.NS", "CHOLAFIN.NS",
                     "DLF.NS", "GAIL.NS", "GODREJCP.NS",
                     "INDIGO.NS", "NAUKRI.NS", "PEL.NS", "SIEMENS.NS"],
        "description": "India Mid-Cap (next 50 stocks)",
    },
}


class UniverseBuilder:
    def __init__(self, cache_dir: Path = PROCESSED_DATA_DIR):
        self.cache_dir = cache_dir

    def _cache_file(self, index_name: str) -> Path:
        safe = index_name.lower().replace(" ", "_").replace("&", "and")
        return self.cache_dir / f"{safe}_universe.csv"

    # ── Fetch from Wikipedia ──────────────────────────────────────────
    def fetch_index_constituents(self, index_name: str = "S&P 500") -> list:
        """
        Scrapes Wikipedia for the current constituents of the given index.
        Uses a proxy-aware session with retry to handle HTTP 403 blocks.
        """
        index_info = SUPPORTED_INDEXES.get(index_name)
        if index_info is None:
            raise ValueError(
                f"Unknown index '{index_name}'. "
                f"Supported: {list(SUPPORTED_INDEXES.keys())}"
            )

        logger.info(f"Fetching {index_name} constituents from Wikipedia...")
        url = index_info["url"]
        suffix = index_info["suffix"]

        try:
            session = get_robust_session()
            response = session.get(url, timeout=30)
            response.raise_for_status()

            tables = pd.read_html(response.text)
            df = tables[0]

            symbol_col = self._find_symbol_column(df)

            # Clean tickers
            tickers = df[symbol_col].astype(str).str.strip()
            # Fix BRK.B -> BRK-B for yfinance (US only)
            if not suffix:
                tickers = tickers.str.replace('.', '-', regex=False)
            else:
                # Ensure NSE suffix is appended
                tickers = tickers.apply(
                    lambda t: t if t.endswith(suffix) else t + suffix
                )

            tickers = tickers.tolist()

            # Cache
            cache_file = self._cache_file(index_name)
            df.to_csv(cache_file, index=False)
            logger.info(f"Successfully retrieved {len(tickers)} tickers for {index_name}.")
            return tickers

        except Exception as e:
            logger.error(f"Failed to fetch {index_name}: {e}")
            if "403" in str(e):
                logger.warning(
                    "HTTP 403 Forbidden — try setting HTTP_PROXY/HTTPS_PROXY in .env"
                )
            return index_info["fallback"]

    # ── Legacy S&P 500 method (backward compatible) ───────────────────
    def fetch_sp500_constituents(self) -> list:
        return self.fetch_index_constituents("S&P 500")

    # ── Get universe (with cache) ─────────────────────────────────────
    def get_universe(self, index_name: str = "S&P 500",
                     bypass_cache: bool = False) -> list:
        """
        Gets the universe from cache if it exists, otherwise fetches it.
        """
        cache_file = self._cache_file(index_name)

        if cache_file.exists() and not bypass_cache:
            df = pd.read_csv(cache_file)
            symbol_col = self._find_symbol_column(df)
            suffix = SUPPORTED_INDEXES.get(index_name, {}).get("suffix", "")
            tickers = df[symbol_col].astype(str).str.strip()
            if not suffix:
                tickers = tickers.str.replace('.', '-', regex=False)
            else:
                tickers = tickers.apply(
                    lambda t: t if t.endswith(suffix) else t + suffix
                )
            tickers = tickers.tolist()
            logger.info(f"Loaded {len(tickers)} tickers for {index_name} from cache.")
            return tickers

        return self.fetch_index_constituents(index_name)

    @staticmethod
    def _find_symbol_column(df: pd.DataFrame) -> str:
        """
        Finds the ticker/symbol column regardless of minor Wikipedia markup changes.
        """
        for col in df.columns:
            if col.lower() in ('symbol', 'ticker', 'ticker symbol',
                               'company name', 'company'):
                return col
        # Indian indexes sometimes use different column names
        for col in df.columns:
            if 'symbol' in col.lower() or 'ticker' in col.lower():
                return col
        raise KeyError(
            f"Could not find a symbol column in {df.columns.tolist()}. "
            "Wikipedia table structure may have changed."
        )


if __name__ == "__main__":
    builder = UniverseBuilder()

    for idx_name in SUPPORTED_INDEXES:
        tickers = builder.get_universe(index_name=idx_name)
        print(f"\n{idx_name}: {len(tickers)} tickers")
        print(f"  Sample: {tickers[:5]}")
