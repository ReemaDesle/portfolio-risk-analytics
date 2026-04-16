"""
Portfolio Risk Analytics — News Data Scraper
============================================
Collects technology news from TechCrunch via RSS.

  Source              Domain          Method
  ──────────────────────────────────────────────────
  TechCrunch          Technology      RSS scraping

Each source saves its own raw CSV to data/raw/news/.
A combiner function merges all into raw_news_combined.csv.

Usage
-----
  python scrape_news.py                          # all sources, default dates
  python scrape_news.py --start 2022-01-01 --end 2023-12-31

Dependencies
───────────────────────────────────────────────
  pip install requests pandas feedparser python-dotenv
"""

import os
import sys
import time
import logging
import argparse
import warnings
from datetime import datetime, timedelta
from io import StringIO, BytesIO
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ──────────────────────────────────────────────
# ❶  DATE RANGE  ← change these two variables
# ──────────────────────────────────────────────
DEFAULT_START_DATE = "2017-01-01"
DEFAULT_END_DATE   = "2024-12-31"

# ──────────────────────────────────────────────
# ❷  OUTPUT DIRECTORY
# ──────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent.parent
RAW_DIR   = ROOT_DIR / "data" / "raw" / "news"

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
# Utility helpers
# ══════════════════════════════════════════════

def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

def save(df: pd.DataFrame, filename: str) -> Path:
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    log.info("  ✔  Saved %s  (%d rows)", path.name, len(df))
    return path


# ══════════════════════════════════════════════
# SOURCE — TechCrunch RSS
# ══════════════════════════════════════════════

def scrape_techcrunch() -> pd.DataFrame:
    """
    TechCrunch publishes a full-text RSS feed — no login needed.
    Covers recent articles; historical depth ~a few weeks.
    """
    log.info("── TechCrunch RSS ───────────────────────────")
    try:
        import feedparser
    except ImportError:
        log.warning("  feedparser not installed. Run: pip install feedparser")
        return pd.DataFrame()

    FEEDS = [
        "https://techcrunch.com/feed/",
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "https://techcrunch.com/category/startups/feed/",
    ]

    rows = []
    for feed_url in FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                date_str = (
                    datetime(*pub[:3]).strftime("%Y-%m-%d") if pub
                    else datetime.today().strftime("%Y-%m-%d")
                )
                title   = entry.get("title", "")
                summary = entry.get("summary", "")
                rows.append({
                    "date":       date_str,
                    "source":     "techcrunch",
                    "domain":     "technology",
                    "headline":   f"{title}. {summary}"[:500],
                    "tone_score": None,
                    "url":        entry.get("link", ""),
                })
            log.info("  Feed: %s  → %d articles", feed_url.split("/")[-2], len(feed.entries))
        except Exception as exc:
            log.warning("  TechCrunch error (%s): %s", feed_url, exc)

    df = pd.DataFrame(rows)
    log.info("  TechCrunch: %d rows collected", len(df))
    if not df.empty:
        save(df, "techcrunch_raw.csv")
    return df


# ══════════════════════════════════════════════
# Combiner
# ══════════════════════════════════════════════

STANDARD_COLS = ["date", "source", "domain", "headline", "tone_score", "url"]

def combine_all(frames: list) -> pd.DataFrame:
    """Merge all source DataFrames into a single raw_news_combined.csv."""
    log.info("── Combining all sources ────────────────────")
    frames = [f for f in frames if not f.empty]
    if not frames:
        log.warning("  No data collected from any source!")
        return pd.DataFrame(columns=STANDARD_COLS)

    combined = pd.concat(frames, ignore_index=True)

    # Ensure standard columns
    for col in STANDARD_COLS:
        if col not in combined.columns:
            combined[col] = None

    combined = combined[STANDARD_COLS]
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined = combined.sort_values("date").reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["headline", "date"])

    out_path = RAW_DIR.parent / "raw_news_combined.csv"
    combined.to_csv(out_path, index=False)
    log.info("  Combined rows: %d  → %s", len(combined), out_path)
    return combined


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def run(start_date: str, end_date: str, sources: list = None):
    ensure_dirs()
    all_sources = ["techcrunch"]
    sources = [s.lower() for s in sources] if sources else all_sources

    log.info("═" * 55)
    log.info("  NEWS SCRAPER  |  %s → %s", start_date, end_date)
    log.info("  Sources: %s", ", ".join(sources))
    log.info("═" * 55)

    frames = []

    if "techcrunch" in sources:
        frames.append(scrape_techcrunch())

    combined = combine_all(frames)

    log.info("")
    log.info("═" * 55)
    log.info("  SCRAPE COMPLETE")
    log.info("  Total articles  : %d", len(combined))
    log.info("  Date range      : %s → %s",
             combined['date'].min() if not combined.empty else "N/A",
             combined['date'].max() if not combined.empty else "N/A")
    if not combined.empty:
        log.info("  Domain split    :\n%s", combined['domain'].value_counts().to_string())
    log.info("  Output dir      : %s", RAW_DIR.parent.resolve())
    log.info("═" * 55)

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape news data for Portfolio Risk Analytics"
    )
    parser.add_argument("--start",   default=DEFAULT_START_DATE)
    parser.add_argument("--end",     default=DEFAULT_END_DATE)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        choices=["techcrunch"],
        help="Which sources to scrape (default: all)",
    )
    args = parser.parse_args()
    run(start_date=args.start, end_date=args.end, sources=args.sources)