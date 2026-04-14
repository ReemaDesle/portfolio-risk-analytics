"""
Portfolio Risk Analytics — News Data Scraper
============================================
Collects historical and recent news from four sources:

  Source              Domain          Method
  ──────────────────────────────────────────────────
  GDELT DOC API       Geopolitical    Real article titles via DOC 2.0 API (2015–present)
                                      + GKG tone score fallback for pre-2015 years
  Reuters RSS         Geopolitical    RSS feed parsing
  TechCrunch          Technology      RSS scraping
  Hacker News         Technology      Algolia search API (no key needed)


  This version uses the GDELT DOC 2.0 API which returns real article
  titles scraped from the global news web — proper English sentences
  that FinBERT scores correctly.

  Strategy:
    2015–present → GDELT DOC API  → real titles  → FinBERT scoring
    2001–2014    → GDELT GKG tone → normalised    → used as-is (no FinBERT)
  Both flows produce the same output columns so they merge cleanly.

Each source saves its own raw CSV to data/raw/news/.
A combiner function merges all into raw_news_combined.csv.

Usage
-----
  python scrape_news.py                          # all sources, default dates
  python scrape_news.py --start 2022-01-01 --end 2023-12-31
  python scrape_news.py --sources gdelt hackernews   # specific sources only

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
# SOURCE 1 — GDELT  (DOC 2.0 API + GKG tone fallback)
# ══════════════════════════════════════════════

# ── GDELT DOC API constants ────────────────────
_GDELT_DOC_URL      = "https://api.gdeltproject.org/api/v2/doc/doc"
_GDELT_DOC_CUTOFF   = "2015-01-01"   # DOC API reliable from this date onward
_GDELT_DOC_MAX_REC  = 250            # hard cap per single API call (API limit)
_GDELT_DOC_DELAY    = 3.0            # was 1.0 — triple the courtesy delay
_GDELT_DOC_RETRIES  = 5              # was 3 — more retries before giving up

# Keyword groups for the DOC API.
# Each group is issued as one OR query. Smaller groups = more targeted results.
_GDELT_GEO_GROUPS = [
    ["war", "invasion", "military conflict"],
    ["sanctions", "geopolitical", "NATO"],
    ["Russia Ukraine", "Middle East conflict"],
    ["oil price shock", "crude oil market"],
    ["gold price", "gold surge", "gold safe haven"],
    ["inflation", "federal reserve rate"],
    ["interest rate hike", "monetary policy"],
    ["recession", "GDP growth"],
    ["trade war", "tariff", "trade sanctions"],
]
_GDELT_TECH_GROUPS = [
    ["artificial intelligence", "AI regulation"],
    ["semiconductor", "chip shortage", "NVIDIA"],
    ["tech earnings", "Apple Microsoft earnings"],
    ["cybersecurity breach", "data breach hack"],
    ["tech layoffs", "technology job cuts"],
    ["OpenAI", "ChatGPT", "large language model"],
    ["cryptocurrency bitcoin", "crypto market"],
    ["cloud computing", "AWS Azure"],
]


def _gdelt_doc_month_windows(start_date: str, end_date: str):
    """
    Generate (window_start, window_end) pairs in monthly chunks.
    Smaller windows → more results per keyword because the DOC API
    ranks by recency within the window.
    """
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date,   "%Y-%m-%d")
    current = s
    while current <= e:
        # End of this month
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        window_end = min(next_month - timedelta(days=1), e)
        yield current.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")
        current = next_month


def _gdelt_doc_fetch(query: str, start: str, end: str) -> list[dict]:
    """
    Single GDELT DOC API call for one keyword group + one time window.
    Returns list of article dicts with keys: date, headline, url, source.
    """
    # API datetime format: YYYYMMDDHHMMSS
    s_str = start.replace("-", "") + "000000"
    e_str = end.replace("-",   "") + "235959"

    params = {
        "query":           query,
        "mode":            "artlist",       # article list mode — returns titles
        "maxrecords":      _GDELT_DOC_MAX_REC,
        "startdatetime":   s_str,
        "enddatetime":     e_str,
        "sort":            "DateDesc",
        "format":          "json",
    }

    import time  # Just in case

    for attempt in range(1, _GDELT_DOC_RETRIES + 1):
        try:
            r = requests.get(_GDELT_DOC_URL, params=params, timeout=30)
            if r.status_code == 429:
                wait = 60 * attempt        # 60s, 120s, 180s ...
                log.warning("    Rate limited. Waiting %ds before retry.", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data     = r.json()
            articles = data.get("articles", [])
            rows = []
            for a in articles:
                raw_date = a.get("seendate", "")    # format: YYYYMMDDTHHmmssZ
                if len(raw_date) >= 8:
                    try:
                        date_str = datetime.strptime(raw_date[:8], "%Y%m%d").strftime("%Y-%m-%d")
                    except Exception:
                        date_str = ""
                else:
                    date_str = ""

                title = (a.get("title") or "").strip()
                if not title or len(title) < 10:
                    continue   # skip empty / junk titles

                rows.append({
                    "date":       date_str,
                    "headline":   title,
                    "url":        a.get("url",    ""),
                    "source_domain": a.get("domain", ""),
                    "language":   a.get("language", ""),
                })
            return rows

        except Exception as exc:
            log.warning("    GDELT DOC attempt %d/%d failed: %s", attempt, _GDELT_DOC_RETRIES, exc)
            if attempt < _GDELT_DOC_RETRIES:
                time.sleep(_GDELT_DOC_DELAY * attempt * 3)

    return []


def _scrape_gdelt_doc(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Scrape GDELT DOC API for real article titles.
    Iterates: keyword_group × monthly_window
    Total calls ≈ (9 geo + 8 tech groups) × (months in range)
    For 2018–2024 (84 months) that is ~1,428 calls at ~1s delay ≈ 25 min.
    Reduce by narrowing date range or using --sources gdelt with a short window.
    """
    log.info("  [GDELT DOC API]  %s → %s", start_date, end_date)

    all_rows   = []
    all_groups = (
        [("geopolitical", q) for group in _GDELT_GEO_GROUPS  for q in [" OR ".join(f'"{k}"' for k in group)]]
      + [("technology",   q) for group in _GDELT_TECH_GROUPS for q in [" OR ".join(f'"{k}"' for k in group)]]
    )
    windows    = list(_gdelt_doc_month_windows(start_date, end_date))
    total_calls = len(all_groups) * len(windows)
    call_n      = 0

    log.info("  Keyword groups: %d  |  Monthly windows: %d  |  Total API calls: %d",
             len(all_groups), len(windows), total_calls)
    log.info("  Estimated time at 1s/call: ~%d minutes", total_calls // 60 + 1)

    for domain, query in all_groups:
        for win_start, win_end in windows:
            call_n += 1
            rows = _gdelt_doc_fetch(query, win_start, win_end)
            for row in rows:
                row["source"] = "gdelt_doc"
                row["domain"] = domain
                row["tone_score"] = None   # FinBERT will score from headline
                all_rows.append(row)

            if call_n % 50 == 0:
                log.info("    API calls: %d / %d  |  Articles so far: %d",
                         call_n, total_calls, len(all_rows))
            time.sleep(_GDELT_DOC_DELAY)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"source_domain": "news_domain"})
    df = df[df["date"] != ""]

    # Keep English articles only (FinBERT is English-only)
    if "language" in df.columns:
        df = df[df["language"].str.lower().isin(["english", "eng", ""]) | df["language"].isna()]

    # Deduplicate on (headline, date) — same article may appear in multiple keyword queries
    df = df.drop_duplicates(subset=["headline", "date"])

    # Reorder to standard columns
    df = df[["date", "source", "domain", "headline", "tone_score", "url"]].copy()
    log.info("  GDELT DOC: %d unique articles collected", len(df))
    return df


# ── GKG tone fallback for pre-2015 data ────────

def _normalise_gdelt_tone(raw_tone: float) -> float:
    """
    Map GDELT raw tone (roughly -30 to +30) to [-1, +1] using tanh.
    tanh(x/10) gives a smooth, bounded mapping:
      raw  -20 → -0.96,  -10 → -0.76,  0 → 0,  +10 → +0.76,  +20 → +0.96
    This makes it compatible with FinBERT compound scores downstream.
    """
    import math
    try:
        return round(math.tanh(float(raw_tone) / 10.0), 6)
    except Exception:
        return 0.0


def _scrape_gdelt_gkg_tone(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pre-2015 fallback: download GDELT GKG bulk CSVs and extract the
    built-in tone score (no FinBERT — tone is used directly as sentiment).
    The tone score is normalised to [-1, +1] so it is compatible with
    the FinBERT compound column used by the rest of the pipeline.

    GKG tone field format: tone, pos_score, neg_score, polarity, ...
    We extract field[0] (overall tone) and fields [1,2] (pos/neg density).
    """
    log.info("  [GDELT GKG tone fallback]  %s → %s  (pre-2015)", start_date, end_date)

    import zipfile, io as _io, gzip as _gzip

    GDELT_MASTER = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
    r = safe_request(GDELT_MASTER)
    if r is None:
        log.warning("  GDELT master list unavailable. Skipping GKG fallback.")
        return pd.DataFrame()

    s = datetime.strptime(start_date, "%Y-%m-%d").date()
    e = datetime.strptime(end_date,   "%Y-%m-%d").date()

    def url_date(url):
        fname = url.split("/")[-1]
        try:
            return datetime.strptime(fname[:8], "%Y%m%d").date()
        except Exception:
            return None

    gkg_urls = [
        ln.split()[-1]
        for ln in r.text.strip().splitlines()
        if ".gkg.csv.zip" in ln or ".gkg.csv.gz" in ln
    ]
    in_range = [u for u in gkg_urls if url_date(u) and s <= url_date(u) <= e]
    log.info("  GKG files in range: %d", len(in_range))

    rows = []
    for i, url in enumerate(in_range[:300], 1):    # cap: 300 files ≈ ~3 days of 15-min blocks
        r2 = safe_request(url)
        if r2 is None:
            continue
        try:
            raw_bytes = r2.content
            if url.endswith(".zip"):
                with zipfile.ZipFile(_io.BytesIO(raw_bytes)) as z:
                    text = z.read(z.namelist()[0]).decode("latin-1", errors="replace")
            else:
                text = _gzip.decompress(raw_bytes).decode("latin-1", errors="replace")

            for line in text.splitlines()[:3000]:
                parts = line.split("\t")
                if len(parts) < 10:
                    continue
                date_str = parts[0][:8]
                themes   = parts[7]  if len(parts) > 7  else ""
                tone_str = parts[9]  if len(parts) > 9  else ""

                # Only keep rows relevant to our keywords
                themes_lc = themes.lower()
                if not any(k.lower().replace(" ", "_") in themes_lc or k.lower() in themes_lc
                           for k in GEO_KEYWORDS + TECH_KEYWORDS):
                    continue

                try:
                    tone_raw  = float(tone_str.split(",")[0]) if tone_str else None
                    tone_norm = _normalise_gdelt_tone(tone_raw) if tone_raw is not None else None
                except Exception:
                    tone_norm = None

                # Build a minimal readable pseudo-headline from the theme codes
                # e.g. "ECON_TAXATION;WB_1243_GOLD" → "gold taxation"
                readable = (
                    themes.replace(";", " ")
                          .replace("_", " ")
                          .lower()[:200]
                )

                rows.append({
                    "date":       date_str,
                    "source":     "gdelt_gkg",
                    "domain":     _classify_domain(themes_lc),
                    "headline":   readable,
                    "tone_score": tone_norm,   # pre-computed, skip FinBERT
                    "url":        "",
                })
        except Exception as exc:
            log.debug("  GKG parse error: %s", exc)

        if i % 50 == 0:
            log.info("    GKG files processed: %d / %d", i, len(in_range))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(subset=["headline", "date"])
    log.info("  GDELT GKG tone: %d rows", len(df))
    return df


def scrape_gdelt(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Main GDELT dispatcher.

    Routing logic
    -------------
      start ≥ 2015  →  DOC API only          (real titles, FinBERT-ready)
      end   < 2015  →  GKG tone only         (theme codes, normalised tone)
      span crosses  →  DOC API for ≥2015 half
                       GKG tone for <2015 half
                       both merged into one DataFrame

    The output always has the standard columns:
      date, source, domain, headline, tone_score, url
    tone_score is None for DOC rows (FinBERT will fill it) and a
    normalised float for GKG rows (used directly as sentiment signal).
    """
    log.info("── GDELT ────────────────────────────────────")

    cutoff = _GDELT_DOC_CUTOFF   # "2015-01-01"
    frames = []

    # ── Portion handled by DOC API ────────────
    doc_start = max(start_date, cutoff)
    if doc_start <= end_date:
        log.info("  DOC API window: %s → %s", doc_start, end_date)
        doc_df = _scrape_gdelt_doc(doc_start, end_date)
        if not doc_df.empty:
            frames.append(doc_df)
    else:
        log.info("  DOC API skipped (entire range is pre-%s)", cutoff)

    # ── Portion handled by GKG tone fallback ──
    if start_date < cutoff:
        gkg_end = min(end_date, "2014-12-31")
        log.info("  GKG tone window: %s → %s", start_date, gkg_end)
        gkg_df = _scrape_gdelt_gkg_tone(start_date, gkg_end)
        if not gkg_df.empty:
            frames.append(gkg_df)
    else:
        log.info("  GKG tone fallback skipped (range starts at/after %s)", cutoff)

    if not frames:
        log.warning("  GDELT: no data collected.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["headline", "date"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info("  GDELT total: %d rows  (doc=%d, gkg=%d)",
             len(df),
             len(df[df["source"] == "gdelt_doc"]),
             len(df[df["source"] == "gdelt_gkg"]))

    save(df, "gdelt_raw.csv")
    return df


# ══════════════════════════════════════════════
# SOURCE 2 — Reuters RSS
# ══════════════════════════════════════════════

def scrape_reuters_rss() -> pd.DataFrame:
    """
    Parse multiple Reuters RSS feeds.
    RSS gives ~20–40 recent articles per feed; no auth required.
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
    if not df.empty:
        save(df, "reuters_rss_raw.csv")
    return df


# ══════════════════════════════════════════════
# SOURCE 3 — TechCrunch RSS
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
# SOURCE 4 — Hacker News (Algolia API)
# ══════════════════════════════════════════════

def scrape_hackernews(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Uses HN's public Algolia search API — no key required.
    Good depth: covers years of HN history for tech keywords.
    """
    log.info("── Hacker News ──────────────────────────────")
    BASE = "https://hn.algolia.com/api/v1/search_by_date"

    s_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    e_ts = int(datetime.strptime(end_date,   "%Y-%m-%d").timestamp())

    rows = []
    # Batch HN queries by keyword group to stay within rate limits
    hn_keywords = [
        "artificial intelligence",
        "semiconductor",
        "layoffs",
        "OpenAI",
        "bitcoin",
        "cybersecurity",
        "IPO",
        "sanctions",
        "inflation",
        "recession",
        "ai",
        "startup",
        "war",
        "tech",
        "startup",
        "federal",
        "reserve",
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
            if page >= min(nb_pages, 5):   # max 5 pages (1000 articles) per keyword
                break
            time.sleep(0.3)

    df = pd.DataFrame(rows).drop_duplicates(subset=["url"])
    log.info("  Hacker News: %d rows collected", len(df))
    if not df.empty:
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

def run(start_date: str, end_date: str, sources: list[str] = None):
    ensure_dirs()
    all_sources = ["gdelt", "reuters", "techcrunch", "hackernews"]
    sources = [s.lower() for s in sources] if sources else all_sources

    log.info("═" * 55)
    log.info("  NEWS SCRAPER  |  %s → %s", start_date, end_date)
    log.info("  Sources: %s", ", ".join(sources))
    log.info("═" * 55)

    frames = []

    if "gdelt"       in sources: frames.append(scrape_gdelt(start_date, end_date))
    if "reuters"     in sources: frames.append(scrape_reuters_rss())
    if "techcrunch"  in sources: frames.append(scrape_techcrunch())
    if "hackernews"  in sources: frames.append(scrape_hackernews(start_date, end_date))

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
        choices=["gdelt", "reuters", "techcrunch", "hackernews"],
        help="Which sources to scrape (default: all)",
    )
    args = parser.parse_args()
    run(start_date=args.start, end_date=args.end, sources=args.sources)