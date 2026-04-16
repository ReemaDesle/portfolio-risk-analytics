"""
Portfolio Risk Analytics — Technology News Scraper
==================================================
Fetches technology news from TechCrunch and Hacker News.
Applies cleaning and filtering similar to the global news pipeline.

Output columns
--------------
  date        YYYY-MM-DD
  source      "techcrunch" | "hackernews"
  domain      "technology"
  headline    article title
  tone_score  None (downstream step enriches this)
  url         link to original article

Usage
-----
  python scrape_tech.py --start 2026-03-20 --end 2026-04-15

Output saved to:  data/raw/news/tech_news.csv
"""

import os
import re
import sys
import time
import logging
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DEFAULT_START_DATE = "2021-05-15"
DEFAULT_END_DATE   = "2021-06-15"

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT_DIR / "data" / "raw" / "news"

TECH_KEYWORDS = [
    "AI", "artificial intelligence", "machine learning", "semiconductor",
    "chip shortage", "tech earnings", "NVIDIA", "Apple", "Microsoft",
    "cybersecurity", "data breach", "IPO", "tech layoffs", "OpenAI",
    "cryptocurrency", "bitcoin", "cloud computing", "regulation",
]

# Regex patterns for junk rows (similar to preprocess_finance.py)
JUNK_PATTERNS = [
    r"^(live updates?|live blog)",
    r"(results? (live|announced)|live result)",
    r"^(word of the day|quote of the day)",
    r"(wishes|greetings|messages|images quotes)",
    r"^(sensex today|gold rate today|nifty today)",
    r"(election \d{4} result)",
    r"(board results \d{4})",
    r"^oscars? (winner|results?)",
]
JUNK_RE = re.compile("|".join(JUNK_PATTERNS), re.IGNORECASE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# Utilities & Preprocessing
# ══════════════════════════════════════════════

def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def save(df: pd.DataFrame, filename: str) -> Path:
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    log.info("  ✔  Saved %s  (%d rows)", path.name, len(df))
    return path


def _safe_get(url: str, params=None, headers=None,
              retries: int = 3, wait: float = 2.0) -> requests.Response | None:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers,
                             timeout=30, verify=False)
            r.raise_for_status()
            return r
        except Exception as exc:
            log.warning("  Request failed (%d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(wait * attempt)
    return None


def is_junk(headline: str) -> bool:
    if not isinstance(headline, str): return True
    return bool(JUNK_RE.search(headline))


def clean_headline(text: str) -> str:
    if not isinstance(text, str): return ""
    # Decode HTML entities
    text = text.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    # Remove branding separators
    text = re.sub(r"\s*[|\u2014\u2013]\s*(TechCrunch|Hacker News|Reuters|NYT|BBC|CNN|AP|Guardian).*$", "", text, flags=re.I)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════
# SOURCE 1 — TechCrunch RSS
# ══════════════════════════════════════════════

def scrape_techcrunch() -> pd.DataFrame:
    """Fetch recent TechCrunch stories via RSS."""
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
        except Exception as exc:
            log.warning("  TechCrunch error (%s): %s", feed_url, exc)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["url"])
    log.info("  TechCrunch: %d articles collected (raw)", len(df))
    return df


# ══════════════════════════════════════════════
# SOURCE 2 — Hacker News (Algolia API)
# ══════════════════════════════════════════════

def scrape_hackernews(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch tech stories from Hacker News using keyword list."""
    log.info("── Hacker News (tech keywords)  %s → %s", start_date, end_date)
    BASE = "https://hn.algolia.com/api/v1/search_by_date"

    s_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    e_ts = int(datetime.strptime(end_date,   "%Y-%m-%d").timestamp())

    rows = []
    seen_urls = set()

    # Keywords for tech-specific HN search
    hn_keywords = [
        "artificial intelligence", "semiconductor", "layoffs",
        "OpenAI", "bitcoin", "cybersecurity", "IPO", "startup", "NVIDIA"
    ]

    for kw in hn_keywords:
        page = 0
        while page < 5:    # max 5 pages (1000 hits) per keyword
            params = {
                "query":          kw,
                "tags":           "story",
                "numericFilters": f"created_at_i>{s_ts},created_at_i<{e_ts}",
                "hitsPerPage":    200,
                "page":           page,
            }
            r = _safe_get(BASE, params=params)
            if r is None:
                break

            data    = r.json()
            hits    = data.get("hits", [])
            nb_pages = data.get("nbPages", 1)

            for h in hits:
                url = h.get("url", "") or ""
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                ts = h.get("created_at_i", 0)
                date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else start_date
                title    = (h.get("title") or h.get("story_title") or "").strip()
                
                if not title:
                    continue

                rows.append({
                    "date":       date_str,
                    "source":     "hackernews",
                    "domain":     "technology",
                    "headline":   title[:400],
                    "tone_score": None,
                    "url":        url,
                })

            page += 1
            if page >= min(nb_pages, 5):
                break
            time.sleep(0.3)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["headline", "date"])
    log.info("  Hacker News: %d articles collected (raw)", len(df))
    return df


# ══════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════

OUTPUT_COLS = ["date", "source", "domain", "headline", "tone_score", "url"]

def run(start_date: str = DEFAULT_START_DATE,
        end_date:   str = DEFAULT_END_DATE):

    ensure_dirs()

    log.info("═" * 60)
    log.info("  TECH SCRAPER |  %s → %s", start_date, end_date)
    log.info("═" * 60)

    # 1. Scrape
    frames = []
    
    tc_df = scrape_techcrunch()
    if not tc_df.empty:
        frames.append(tc_df)
        
    hn_df = scrape_hackernews(start_date, end_date)
    if not hn_df.empty:
        frames.append(hn_df)

    if not frames:
        log.error("No articles collected. Exiting.")
        return pd.DataFrame(columns=OUTPUT_COLS)

    df = pd.concat(frames, ignore_index=True)

    # 2. Preprocess
    before_p = len(df)
    
    # Filter junk
    df = df[~df["headline"].apply(is_junk)].copy()
    
    # Clean headlines
    df["headline"] = df["headline"].apply(clean_headline)
    
    # Length filter
    df = df[df["headline"].str.len() >= 40].copy()
    
    # Deduplicate on cleaned content
    df = df.drop_duplicates(subset=["headline", "date"]).reset_index(drop=True)
    
    log.info("── Preprocessing Complete:")
    log.info("   Dropped %d junk/short/duplicate rows", before_p - len(df))
    log.info("   Remaining: %d articles", len(df))

    # 3. Finalize columns
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = None
    df = df[OUTPUT_COLS].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # 4. Save
    save(df, "tech_news.csv")

    log.info("")
    log.info("═" * 60)
    log.info("  COMPLETE")
    log.info("  Total articles : %d", len(df))
    log.info("  Output         : %s", RAW_DIR / "tech_news.csv")
    log.info("═" * 60)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Technology news scraper: TechCrunch + Hacker News with preprocessing"
    )
    parser.add_argument("--start",  default=DEFAULT_START_DATE,
                        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START_DATE})")
    parser.add_argument("--end",    default=DEFAULT_END_DATE,
                        help=f"End date YYYY-MM-DD (default: {DEFAULT_END_DATE})")
    args = parser.parse_args()
    run(start_date=args.start, end_date=args.end)
