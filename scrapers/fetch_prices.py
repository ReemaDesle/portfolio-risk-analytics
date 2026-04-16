"""
Portfolio Risk Analytics — Stock Data Scraper
==============================================
Scrapes historical OHLCV data for a curated set of tickers across
four portfolio archetypes using yfinance.

Portfolio archetypes
--------------------
  1. Geopolitical-Sensitive  → Gold, Oil, Defence, Emerging markets
  2. Tech-Heavy              → Big Tech + semiconductors
  3. Balanced                → Mix of equities, bonds, gold
  4. Conservative            → Bonds, utilities, consumer staples

Outputs (saved to ./data/raw/)
-------------------------------
  prices_daily.csv           — Adjusted close prices (wide format)
  returns_daily.csv          — Log daily returns (wide format)
  ohlcv_<TICKER>.csv         — Full OHLCV per ticker (long format)
  metadata.json              — Ticker info snapshot (sector, name …)

Usage
-----
  python scrape_portfolio_data.py
  python scrape_portfolio_data.py --start 2020-01-01 --end 2024-12-31

Dependencies
------------
  pip install yfinance pandas
"""

import os
import json
import argparse
import logging
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_START_DATE = "2026-03-20"   # inclusive
DEFAULT_END_DATE   = "2026-04-15"   # inclusive (yfinance end is exclusive, handled below)

# ──────────────────────────────────────────────
# ❷  TICKER UNIVERSE
# ──────────────────────────────────────────────
PORTFOLIOS = {
    # --- Geopolitical-Sensitive ---
    "geopolitical": {
        "GLD":  "SPDR Gold Shares ETF",
        "USO":  "United States Oil Fund ETF",
        "LMT":  "Lockheed Martin (Defence)",
        "RTX":  "Raytheon Technologies (Defence)",
        "EEM":  "iShares MSCI Emerging Markets ETF",
        "GC=F": "Gold Futures",          # spot gold cross-check
        "CL=F": "Crude Oil Futures",
    },

    # --- Tech-Heavy ---
    "tech": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "NVDA": "NVIDIA",
        "GOOGL":"Alphabet (Google)",
        "META": "Meta Platforms",
        "AMZN": "Amazon",
        "TSLA": "Tesla",
        "SOXX": "iShares Semiconductor ETF",
        "QQQ":  "Invesco NASDAQ-100 ETF",
    },

    # --- Balanced ---
    "balanced": {
        "SPY":  "SPDR S&P 500 ETF",
        "AGG":  "iShares Core US Aggregate Bond ETF",
        "GLD":  "SPDR Gold Shares ETF",
        "VTI":  "Vanguard Total Stock Market ETF",
        "EFA":  "iShares MSCI EAFE ETF",
        "BND":  "Vanguard Total Bond Market ETF",
    },

    # --- Conservative ---
    "conservative": {
        "TLT":  "iShares 20+ Year Treasury Bond ETF",
        "IEF":  "iShares 7-10 Year Treasury Bond ETF",
        "VPU":  "Vanguard Utilities ETF",
        "KO":   "Coca-Cola (Consumer Staples)",
        "JNJ":  "Johnson & Johnson (Healthcare)",
        "PG":   "Procter & Gamble (Consumer Staples)",
        "XLP":  "Consumer Staples Select Sector SPDR",
    },
}

# Flat deduplicated list of all tickers
ALL_TICKERS = sorted({
    ticker
    for tickers in PORTFOLIOS.values()
    for ticker in tickers
})

# ──────────────────────────────────────────────
# ❸  OUTPUT DIRECTORY
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "tickers")

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def resolve_dates(start: str, end: str):
    """
    Validate and normalise date strings.
    yfinance's `end` is exclusive, so we add one calendar day.
    """
    fmt = "%Y-%m-%d"
    s = datetime.strptime(start, fmt).date()
    e = datetime.strptime(end, fmt).date()
    if s >= e:
        raise ValueError(f"START ({s}) must be before END ({e}).")
    # yfinance end is exclusive — add 1 day so our end date IS included
    e_yf = pd.Timestamp(e) + pd.Timedelta(days=1)
    log.info("Study window: %s → %s  (yfinance end param: %s)", s, e, e_yf.date())
    return str(s), str(e_yf.date())


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted-close prices for all tickers in one batch call.
    Returns a DataFrame indexed by date, one column per ticker.
    """
    log.info("Downloading price data for %d tickers …", len(tickers))
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # gives adjusted OHLCV directly
        progress=True,
        threads=True,
    )

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        close.columns = tickers

    close.index = pd.to_datetime(close.index).normalize()
    close.index.name = "date"
    log.info("Price matrix shape: %s", close.shape)
    return close


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download full OHLCV for a single ticker."""
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, auto_adjust=True)
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df.index.name = "date"
    df.columns = [c.lower() for c in df.columns]
    df.insert(0, "ticker", ticker)
    return df


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns: ln(P_t / P_{t-1})."""
    returns = np.log(prices / prices.shift(1))
    returns.index.name = "date"
    return returns.dropna(how="all")


def fetch_metadata(tickers: list[str]) -> dict:
    """Fetch sector / industry / name info for each ticker."""
    meta = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            meta[t] = {
                "short_name":     info.get("shortName", ""),
                "long_name":      info.get("longName", ""),
                "sector":         info.get("sector", ""),
                "industry":       info.get("industry", ""),
                "exchange":       info.get("exchange", ""),
                "currency":       info.get("currency", ""),
                "quote_type":     info.get("quoteType", ""),
                "description":    info.get("longBusinessSummary", "")[:300],
            }
            log.info("  Metadata: %-8s  %s", t, meta[t]["short_name"])
        except Exception as exc:
            log.warning("  Could not fetch metadata for %s: %s", t, exc)
            meta[t] = {}
    return meta


def add_portfolio_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Return a helper DataFrame mapping each ticker to its portfolio(s).
    A ticker can appear in multiple portfolios (e.g. GLD).
    """
    rows = []
    for portfolio, tickers in PORTFOLIOS.items():
        for ticker in tickers:
            rows.append({"ticker": ticker, "portfolio": portfolio})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════

def run(start_date: str, end_date: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Resolve dates ──────────────────────
    start_yf, end_yf = resolve_dates(start_date, end_date)

    # ── 2. Download prices (batch) ────────────
    prices = download_prices(ALL_TICKERS, start_yf, end_yf)

    # ── 3. Log returns ────────────────────────
    returns = compute_log_returns(prices)

    # ── 4. Save wide-format CSVs ──────────────
    prices_path  = os.path.join(OUTPUT_DIR, "prices_daily.csv")
    returns_path = os.path.join(OUTPUT_DIR, "returns_daily.csv")

    prices.to_csv(prices_path)
    returns.to_csv(returns_path)
    log.info("Saved: %s", prices_path)
    log.info("Saved: %s", returns_path)

    # ── 5. Save per-ticker OHLCV ──────────────
    log.info("Downloading per-ticker OHLCV …")
    failed_tickers = []
    for ticker in ALL_TICKERS:
        try:
            ohlcv = download_ohlcv(ticker, start_yf, end_yf)
            out   = os.path.join(OUTPUT_DIR, f"ohlcv_{ticker.replace('=','_')}.csv")
            ohlcv.to_csv(out)
            log.info("  Saved OHLCV: %s  (%d rows)", ticker, len(ohlcv))
        except Exception as exc:
            log.warning("  OHLCV failed for %s: %s", ticker, exc)
            failed_tickers.append(ticker)

    # ── 6. Portfolio membership map ───────────
    portfolio_map = add_portfolio_labels(prices)
    portfolio_map.to_csv(os.path.join(OUTPUT_DIR, "portfolio_map.csv"), index=False)
    log.info("Saved: portfolio_map.csv")

    # ── 7. Metadata ───────────────────────────
    log.info("Fetching ticker metadata …")
    meta = fetch_metadata(ALL_TICKERS)
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved: %s", meta_path)

    # ── 8. Summary report ─────────────────────
    log.info("")
    log.info("═" * 55)
    log.info("  SCRAPE COMPLETE")
    log.info("  Study window : %s → %s", start_date, end_date)
    log.info("  Tickers      : %d", len(ALL_TICKERS))
    log.info("  Trading days : %d", len(prices))
    log.info("  Missing data : %d cells",  int(prices.isna().sum().sum()))
    if failed_tickers:
        log.warning("  OHLCV failed : %s", failed_tickers)
    log.info("  Output dir   : %s", os.path.abspath(OUTPUT_DIR))
    log.info("═" * 55)

    return prices, returns, meta


# ══════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape yfinance data for Portfolio Risk Analytics"
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        help=f"Study start date YYYY-MM-DD  (default: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END_DATE,
        help=f"Study end date YYYY-MM-DD    (default: {DEFAULT_END_DATE})",
    )
    args = parser.parse_args()

    run(start_date=args.start, end_date=args.end)