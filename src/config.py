"""
Project Configuration Parameters

Defines the central configurable elements for the research pipeline.
"""
import os
from pathlib import Path

# --- DIRECTORY STRUCTURE ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Ensure data directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- UNIVERSE CONFIGURATION ---
UNIVERSE_START_DATE = "2010-01-01"
UNIVERSE_END_DATE = "2023-12-31"

# We target a broad US universe. In a commercial setting this would be e.g. Russell 3000.
# We will construct a synthetic broad universe list.
MIN_MARKET_CAP_USD = 500_000_000  # $500M micro/small cap cutoff
MIN_PRICE = 2.0  # Avoid penny stocks
MIN_LIQUIDITY_30D = 1_000_000  # $1M daily average turnover

# --- FACTOR CONFIGURATION ---
WINSORIZE_LIMITS = (0.01, 0.99)  # 1st and 99th percentiles
Z_SCORE_CLIP = 3.0  # Max z-score (standard deviations)

FUNDAMENTAL_LAG_DAYS = 90  # Typical reporting delay assumption for point-in-time

# Momentum: 12-month return skipping the most recent month (12-1 momentum)
MOMENTUM_LOOKBACK_DAYS = 252  # ~12 months of trading days
MOMENTUM_SKIP_DAYS = 21       # ~1 month to skip (short-term reversal)

# Volatility
VOLATILITY_WINDOW_DAYS = 60   # Trailing window for realized volatility

# Liquidity
LIQUIDITY_WINDOW_DAYS = 30    # Trailing average daily dollar volume window

# Quality composite weights (ROE, Debt/Equity inverted, Earnings Stability)
QUALITY_WEIGHTS = {
    "roe": 0.40,
    "debt_equity_inv": 0.30,
    "earnings_stability": 0.30,
}

# Sector column name in fundamentals data (used for sector-neutral operations)
SECTOR_COLUMN = "Sector"

# --- PORTFOLIO & BACKTEST CONFIGURATION ---
REBALANCE_FREQ = 'M'  # 'M' for monthly, 'W' for weekly
TRANSACTION_COST_BPS = 10  # 10 bps per trade one-way

# --- VISUALIZATION SETTINGS ---
# "Old-money" investment banking aesthetic colors for future frontend and plots
IB_PALETTE = {
    "navy_blue": "#002147",
    "gold": "#CFB53B",
    "dark_green": "#004B23",
    "beige": "#F5F5DC",
    "dark_grey": "#333333",
    "burgundy": "#800020"
}

# --- PROXY & NETWORK CONFIGURATION ---
# Reads proxy settings from environment variables or a .env file.
# Set these in your shell or in the project-root .env file:
#   HTTP_PROXY=http://user:pass@proxy-host:port
#   HTTPS_PROXY=http://user:pass@proxy-host:port
#
# If no proxy is configured, direct connections are used.

def _load_dotenv():
    """Load .env file from project root if it exists (lightweight, no dependency)."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("\"'")
            if key and value and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

# Build proxy dict from environment
_http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
_https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

PROXIES = {}
if _http_proxy:
    PROXIES["http"] = _http_proxy
if _https_proxy:
    PROXIES["https"] = _https_proxy

# User-Agent header to avoid 403 blocks from websites that reject default Python UA
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 0.5  # seconds: 0.5, 1.0, 2.0
RETRY_STATUS_FORCELIST = [403, 429, 500, 502, 503, 504]


def get_robust_session():
    """
    Returns a requests.Session pre-configured with:
      - Proxy settings (if any)
      - Browser-like User-Agent header
      - Automatic retry with exponential backoff on 403/429/5xx

    Usage in data modules:
        from src.config import get_robust_session
        session = get_robust_session()
        response = session.get("https://example.com")
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()

    # Proxy
    if PROXIES:
        session.proxies.update(PROXIES)

    # Headers
    session.headers.update(REQUEST_HEADERS)

    # Retry strategy
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_FORCELIST,
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session
