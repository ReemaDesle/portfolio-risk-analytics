"""

Collects historical and recent news from five sources:

  Source              Domain          Method
  ──────────────────────────────────────────────────
  GDELT               Geopolitical    Bulk CSV download (2001–present)
  NewsAPI             Both            REST API (keyword search, ~1 month back on free tier)
  Reuters RSS         Geopolitical    RSS feed parsing
  TechCrunch          Technology      RSS scraping
  Hacker News         Technology      Algolia search API (no key needed)

Each source saves its own raw CSV to data/raw/news/.
A combiner function merges all into raw_news_combined.csv.

Usage
-----
  python scrape_news.py                          # all sources, default dates
  python scrape_news.py --start 2022-01-01 --end 2023-12-31
  python scrape_news.py --sources gdelt hackernews   # specific sources only

Required env vars (create a .env file or export before running)
───────────────────────────────────────────────
  NEWSAPI_KEY         → https://newsapi.org/register  (free)

Dependencies
───────────────────────────────────────────────
  pip install requests pandas feedparser newsapi-python python-dotenv
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
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE   = "2024-12-31"

# ──────────────────────────────────────────────
# ❷  KEYWORDS
# ──────────────────────────────────────────────
GEO_KEYWORDS = [
    "war", "conflict", "sanctions", "geopolitical", "invasion",
    "military", "NATO", "Russia", "Ukraine", "Middle East",
    "oil price", "gold price", "inflation", "federal reserve",
    "interest rate", "recession", "GDP", "trade war", "tariff",
]
TECH_KEYWORDS = [
    "AI", "artificial intelligence", "machine learning", "semiconductor",
    "chip shortage", "tech earnings", "NVIDIA", "Apple", "Microsoft",
    "cybersecurity", "data breach", "IPO", "tech layoffs", "OpenAI",
    "cryptocurrency", "bitcoin", "cloud computing", "regulation",
]

# ──────────────────────────────────────────────
# ❸  OUTPUT DIRECTORY
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

def safe_request(url: str, params=None, headers=None,
                 retries: int = 3, wait: float = 2.0):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            return r
        except Exception as exc:
            log.warning("  Request failed (attempt %d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(wait * attempt)
    return None


# ══════════════════════════════════════════════
# SOURCE 1 — GDELT
# ══════════════════════════════════════════════

def scrape_gdelt(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download GDELT GKG (Global Knowledge Graph) daily CSV files.
    """
    log.info("── GDELT ────────────────────────────────────")

    s = datetime.strptime(start_date, "%Y-%m-%d").date()
    e = datetime.strptime(end_date,   "%Y-%m-%d").date()

    GDELT_MASTER = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

    log.info("  Fetching GDELT master file list …")
    r = safe_request(GDELT_MASTER)
    if r is None:
        log.warning("  GDELT master list unavailable. Skipping.")
        return pd.DataFrame()

    lines  = r.text.strip().splitlines()
    gkg_urls = [
        line.split()[-1]
        for line in lines
        if ".gkg.csv.zip" in line or ".gkg.csv.gz" in line
    ]
    log.info("  Total GKG files in master list: %d", len(gkg_urls))

    def url_date(url: str):
        fname = url.split("/")[-1]
        try:
            return datetime.strptime(fname[:8], "%Y%m%d").date()
        except Exception:
            return None

    in_range = [u for u in gkg_urls if url_date(u) and s <= url_date(u) <= e]
    log.info("  GKG files in date range: %d  (downloading …)", len(in_range))

    rows = []
    for i, url in enumerate(in_range[:500], 1):
        r2 = safe_request(url)
        if r2 is None:
            continue
        try:
            import zipfile, io as _io, gzip
            raw_bytes = r2.content
            if url.endswith(".zip"):
                with zipfile.ZipFile(_io.BytesIO(raw_bytes)) as z:
                    name = z.namelist()[0]
                    text = z.read(name).decode("latin-1", errors="replace")
            else:
                text = gzip.decompress(raw_bytes).decode("latin-1", errors="replace")

            for line in text.splitlines()[:5000]:
                parts = line.split("\t")
                if len(parts) < 10:
                    continue
                date_str = parts[0][:8]
                themes   = parts[7]  if len(parts) > 7  else ""
                tone_str = parts[9]  if len(parts) > 9  else ""
                try:
                    tone_val = float(tone_str.split(",")[0]) if tone_str else None
                except Exception:
                    tone_val = None

                themes_lc = themes.lower()
                if any(k.lower().replace(" ", "_") in themes_lc
                       or k.lower() in themes_lc
                       for k in GEO_KEYWORDS + TECH_KEYWORDS):
                    rows.append({
                        "date":       date_str,
                        "source":     "gdelt",
                        "domain":     _classify_domain(themes_lc),
                        "headline":   themes[:200],
                        "tone_score": tone_val,
                        "url":        "",
                    })
        except Exception as exc:
            log.debug("  GDELT parse error for %s: %s", url, exc)

        if i % 50 == 0:
            log.info("    Processed %d / %d GDELT files …", i, len(in_range))

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("  No GDELT rows matched keyword filter.")
        return df

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.date.astype(str)
    log.info("  GDELT: %d rows collected", len(df))
    save(df, "gdelt_raw.csv")
    return df


# ══════════════════════════════════════════════
# SOURCE 2 — NewsAPI
# ══════════════════════════════════════════════

def scrape_newsapi(start_date: str, end_date: str) -> pd.DataFrame:
    """
    NewsAPI free tier: ~1 month back, max 100 results per query.
    """
    log.info("── NewsAPI ──────────────────────────────────")
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        log.warning("  NEWSAPI_KEY not set. Skipping NewsAPI.")
        return pd.DataFrame()

    BASE = "https://newsapi.org/v2/everything"
    rows = []

    free_cutoff = (datetime.today() - timedelta(days=29)).strftime("%Y-%m-%d")
    effective_start = max(start_date, free_cutoff)
    if effective_start > end_date:
        log.warning("  NewsAPI free tier: date range is outside the 30-day window. Skipping.")
        return pd.DataFrame()

    all_keywords = GEO_KEYWORDS + TECH_KEYWORDS
    batches = [all_keywords[i:i+3] for i in range(0, len(all_keywords), 3)]

    for batch in batches:
        query = " OR ".join(f'"{k}"' for k in batch)
        params = {
            "q":        query,
            "from":     effective_start,
            "to":       end_date,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": 100,
            "apiKey":   api_key,
        }
        r = safe_request(BASE, params=params)
        if r is None:
            continue
        data = r.json()
        articles = data.get("articles", [])
        for a in articles:
            pub = a.get("publishedAt", "")[:10]
            rows.append({
                "date":       pub,
                "source":     "newsapi",
                "domain":     _classify_domain(" ".join(batch).lower()),
                "headline":   (a.get("title") or "") + " " + (a.get("description") or ""),
                "tone_score": None,
                "url":        a.get("url", ""),
            })
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("  NewsAPI returned no articles.")
        return df
    log.info("  NewsAPI: %d rows collected", len(df))
    save(df, "newsapi_raw.csv")
    return df


# ══════════════════════════════════════════════
# SOURCE 3 — Reuters RSS
# ══════════════════════════════════════════════

def scrape_reuters_rss() -> pd.DataFrame:
    """
    Parse multiple Reuters RSS feeds.
    """
    log.info("── Reuters RSS ──────────────────────────────")
    try:
        import feedparser
    except ImportError:
        log.warning("  feedparser not installed. Run: pip install feedparser")
        return pd.DataFrame()

    FEEDS = {
        "https://feeds.reuters.com/reuters/businessNews":       "geopolitical",
        "https://feeds.reuters.com/reuters/technologyNews":     "technology",
        "https://feeds.reuters.com/reuters/worldNews":          "geopolitical",
        "https://feeds.reuters.com/reuters/UKBusinessNews":     "geopolitical",
        "https://feeds.reuters.com/reuters/financialNewsHeads": "geopolitical",
    }

    rows = []
    for feed_url, domain in FEEDS.items():
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
                    "source":     "reuters_rss",
                    "domain":     domain,
                    "headline":   f"{title}. {summary}"[:500],
                    "tone_score": None,
                    "url":        entry.get("link", ""),
                })
            log.info("  Feed: %s  → %d articles", feed_url.split("/")[-1], len(feed.entries))
        except Exception as exc:
            log.warning("  Reuters RSS error (%s): %s", feed_url, exc)

    df = pd.DataFrame(rows)
    log.info("  Reuters RSS: %d rows collected", len(df))
    save(df, "reuters_rss_raw.csv")
    return df


# ══════════════════════════════════════════════
# SOURCE 4 — TechCrunch RSS
# ══════════════════════════════════════════════

def scrape_techcrunch() -> pd.DataFrame:
    """
    TechCrunch publishes a full-text RSS feed.
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
    save(df, "techcrunch_raw.csv")
    return df


# ══════════════════════════════════════════════
# SOURCE 5 — Hacker News (Algolia API)
# ══════════════════════════════════════════════

def scrape_hackernews(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Uses HN's public Algolia search API — no key required.
    """
    log.info("── Hacker News ──────────────────────────────")
    BASE = "https://hn.algolia.com/api/v1/search_by_date"

    s_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    e_ts = int(datetime.strptime(end_date,   "%Y-%m-%d").timestamp())

    rows = []
    hn_keywords = [
        "AI chip semiconductor",
        "tech layoffs earnings",
        "OpenAI regulation",
        "bitcoin crypto market",
        "cybersecurity breach",
        "startup IPO venture",
        "war conflict sanctions",
        "federal reserve inflation",
    ]

    for kw in hn_keywords:
        page = 0
        while True:
            params = {
                "query":         kw,
                "tags":          "story",
                "numericFilters": f"created_at_i>{s_ts},created_at_i<{e_ts}",
                "hitsPerPage":   200,
                "page":          page,
            }
            r = safe_request(BASE, params=params)
            if r is None:
                break
            data    = r.json()
            hits    = data.get("hits", [])
            nb_pages = data.get("nbPages", 1)

            for h in hits:
                ts = h.get("created_at_i", 0)
                date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""
                title    = h.get("title", "") or h.get("story_title", "")
                rows.append({
                    "date":       date_str,
                    "source":     "hackernews",
                    "domain":     _classify_domain(title.lower() + " " + kw.lower()),
                    "headline":   title[:400],
                    "tone_score": None,
                    "url":        h.get("url", ""),
                })

            page += 1
            if page >= min(nb_pages, 5):
                break
            time.sleep(0.3)

    df = pd.DataFrame(rows).drop_duplicates(subset=["url"])
    log.info("  Hacker News: %d rows collected", len(df))
    save(df, "hackernews_raw.csv")
    return df


# ══════════════════════════════════════════════
# Domain classifier
# ══════════════════════════════════════════════

GEO_TERMS  = {k.lower() for k in GEO_KEYWORDS}
TECH_TERMS = {k.lower() for k in TECH_KEYWORDS}

def _classify_domain(text: str) -> str:
    text = text.lower()
    geo_hit  = sum(1 for t in GEO_TERMS  if t in text)
    tech_hit = sum(1 for t in TECH_TERMS if t in text)
    if tech_hit > geo_hit:
        return "technology"
    if geo_hit > 0:
        return "geopolitical"
    return "general"


# ══════════════════════════════════════════════
# Combiner
# ══════════════════════════════════════════════

STANDARD_COLS = ["date", "source", "domain", "headline", "tone_score", "url"]

def combine_all(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge all source DataFrames into a single raw_news_combined.csv."""
    log.info("── Combining all sources ────────────────────")
    frames = [f for f in frames if not f.empty]
    if not frames:
        log.warning("  No data collected from any source!")
        return pd.DataFrame(columns=STANDARD_COLS)

    combined = pd.concat(frames, ignore_index=True)

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

def run(start_date: str, end_date: str, sources: list[str] = None):
    ensure_dirs()
    # Reddit removed from all_sources
    all_sources = ["gdelt", "newsapi", "reuters", "techcrunch", "hackernews"]
    sources = [s.lower() for s in sources] if sources else all_sources

    log.info("═" * 55)
    log.info("  NEWS SCRAPER  |  %s → %s", start_date, end_date)
    log.info("  Sources: %s", ", ".join(sources))
    log.info("═" * 55)

    frames = []

    if "gdelt"      in sources: frames.append(scrape_gdelt(start_date, end_date))
    if "newsapi"    in sources: frames.append(scrape_newsapi(start_date, end_date))
    if "reuters"    in sources: frames.append(scrape_reuters_rss())
    if "techcrunch" in sources: frames.append(scrape_techcrunch())
    if "hackernews" in sources: frames.append(scrape_hackernews(start_date, end_date))

    combined = combine_all(frames)

    log.info("")
    log.info("═" * 55)
    log.info("  SCRAPE COMPLETE")
    log.info("  Total articles  : %d", len(combined))
    log.info("  Date range      : %s → %s", combined['date'].min() if not combined.empty else "N/A",
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
        # Reddit removed from choices
        choices=["gdelt", "newsapi", "reuters", "techcrunch", "hackernews"],
        help="Which sources to scrape (default: all)",
    )
    args = parser.parse_args()
    run(start_date=args.start, end_date=args.end, sources=args.sources)