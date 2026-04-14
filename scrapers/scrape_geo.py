"""
Portfolio Risk Analytics — Geopolitical News Scraper
=====================================================
Replaces GDELT (heavy rate-limits) with three reliable, freely accessible
sources that work cleanly for historical date ranges:

  Source                    Coverage          Auth required?
  ─────────────────────────────────────────────────────────
  Wikipedia Current Events  Daily world events  None (MediaWiki API)
  ReliefWeb API (UN OCHA)   Crisis / conflict   None (appname only)
  Hacker News Algolia       Tech / geopolitics  None (public API)

After collection each headline is classified by country + world region
using the Groq LLM API (llama-3.3-70b-versatile).

Output columns
--------------
  date        YYYY-MM-DD
  source      "wikipedia" | "reliefweb" | "hackernews"
  domain      always "geopolitical"
  headline    article / event title
  tone_score  None  (FinBERT / downstream step enriches this)
  url         link to original article or Wikipedia page

Usage
-----
  python scrape_geo.py                          # defaults below
  python scrape_geo.py --start 2021-05-15 --end 2021-06-15
  python scrape_geo.py --start 2021-05-15 --end 2021-06-15 --batch 20

Output saved to:  data/raw/news/gdelt_geo.csv

Dependencies (already in requirements.txt):
  pip install requests pandas python-dotenv beautifulsoup4
"""

import os
import re
import sys
import json
import time
import math
import logging
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DEFAULT_START_DATE = "2021-05-15"
DEFAULT_END_DATE   = "2021-06-15"

GROQ_API_KEY  = os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY")
GROQ_MODEL    = "llama-3.3-70b-versatile"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_BATCH    = 25      # headlines per Groq call
GROQ_RPM      = 30      # free-tier rpm limit

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT_DIR / "data" / "raw" / "news"

# Wikipedia geopolitical section headers to keep
WIKI_GEO_SECTIONS = {
    "armed conflicts and attacks",
    "wars and conflicts",
    "politics and elections",
    "international relations",
    "disasters and accidents",
    "economy and business",
    "law and crime",
    "terrorism",
    "military",
    "diplomacy",
}

# ReliefWeb: themes relevant to geopolitics
RW_THEMES = [
    "Conflict", "Politics", "Humanitarian Financing",
    "Security", "Peacekeeping", "War and Conflict",
    "Terrorism", "Displacement", "Refugees",
]

# Hacker News keywords that are geo-relevant
HN_GEO_KEYWORDS = [
    "sanctions", "war", "invasion", "geopolitical",
    "NATO", "Russia", "China", "Iran", "Palestine",
    "Israel", "Taliban", "coup", "election fraud",
    "nuclear", "tariff", "trade war", "terrorism",
    "inflation policy", "federal reserve", "OPEC",
    "oil embargo", "United Nations", "G7", "G20",
    "refugee", "diplomacy", "treaty", "military",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════

def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def save(df: pd.DataFrame, filename: str) -> Path:
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    log.info("  ✔  Saved %s  (%d rows)", path.name, len(df))
    return path


def _date_range(start: str, end: str):
    """Yield each date between start and end inclusive as YYYY-MM-DD strings."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    d = s
    while d <= e:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


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


# ══════════════════════════════════════════════
# SOURCE 1 — Wikipedia Current Events Portal
# ══════════════════════════════════════════════
# Wikipedia stores each day's world events at:
#   https://en.wikipedia.org/wiki/Portal:Current_events/YYYY_Month_D
# The MediaWiki API returns rendered HTML which we parse with BeautifulSoup.
# Sections like "Armed conflicts", "Politics", etc. give clean geopolitical groupings.

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_UA  = "PortfolioRiskAnalytics/1.0 (academic research; contact: researcher@example.com)"


def _wiki_fetch_day(date_str: str) -> list[dict]:
    """
    Fetch Wikipedia Current Events for a single day.
    Returns list of {date, headline, url, section} dicts.
    """
    dt     = datetime.strptime(date_str, "%Y-%m-%d")
    # Wikipedia naming:  "May 15" → "Portal:Current_events/2021_May_15"
    month  = dt.strftime("%B")      # "May"
    day    = str(dt.day)            # "15"  (no leading zero)
    year   = str(dt.year)
    page   = f"Portal:Current_events/{year}_{month}_{day}"

    params = {
        "action":    "parse",
        "page":      page,
        "prop":      "text",
        "format":    "json",
        "disablelimitreport": "1",
    }
    headers = {"User-Agent": _WIKI_UA}

    r = _safe_get(_WIKI_API, params=params, headers=headers)
    if r is None:
        return []

    try:
        html = r.json().get("parse", {}).get("text", {}).get("*", "")
    except Exception:
        return []

    if not html:
        return []

    soup    = BeautifulSoup(html, "html.parser")
    rows    = []
    current_section = "general"

    # Each day page is structured as <h2>/<h3> section headers + <ul><li> bullet items
    for tag in soup.find_all(["h2", "h3", "li"]):
        if tag.name in ("h2", "h3"):
            raw = tag.get_text(separator=" ", strip=True).lower()
            # Strip "[edit]" or similar suffixes
            raw = re.sub(r"\[.*?\]", "", raw).strip()
            current_section = raw
        elif tag.name == "li":
            sect_clean = current_section.strip().lower()
            # Only keep geopolitical sections
            if not any(s in sect_clean for s in WIKI_GEO_SECTIONS):
                continue

            text = tag.get_text(separator=" ", strip=True)
            if len(text) < 20:
                continue

            # Grab the first external/internal link in the bullet as the URL
            link = tag.find("a", href=True)
            if link:
                href = link["href"]
                if href.startswith("/wiki/"):
                    url = "https://en.wikipedia.org" + href
                elif href.startswith("http"):
                    url = href
                else:
                    url = ""
            else:
                url = f"https://en.wikipedia.org/wiki/{page.replace(' ', '_')}"

            rows.append({
                "date":     date_str,
                "headline": text[:500],
                "url":      url,
                "section":  current_section,
            })

    return rows


def scrape_wikipedia(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Scrape Wikipedia Current Events for every day in [start_date, end_date].
    Rate: ~1.5s per day → 32 days ≈ ~48s total.
    """
    log.info("── Wikipedia Current Events  %s → %s", start_date, end_date)

    all_rows = []
    dates    = list(_date_range(start_date, end_date))
    log.info("  Days to fetch: %d", len(dates))

    for i, date_str in enumerate(dates, 1):
        rows = _wiki_fetch_day(date_str)
        for row in rows:
            row["source"] = "wikipedia"
            row["domain"] = "geopolitical"
            row["tone_score"] = None
        all_rows.extend(rows)

        if i % 5 == 0 or i == len(dates):
            log.info("  Day %d/%d  |  Events so far: %d", i, len(dates), len(all_rows))
        time.sleep(1.0)   # be a polite Wikipedia citizen

    if not all_rows:
        log.warning("  Wikipedia returned no events.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["headline", "date"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info("  Wikipedia: %d unique events collected", len(df))
    return df


# ══════════════════════════════════════════════
# SOURCE 2 — ReliefWeb API  (UN OCHA)
# ══════════════════════════════════════════════
# ReliefWeb indexes crisis and humanitarian news from hundreds of sources.
# API: https://apidoc.rwlabs.org/
# No auth key needed; just include appname.
# Limit: 1000 req/day (we will use ~20–30 calls max).

_RW_ENDPOINT = "https://api.reliefweb.int/v1/reports"
_RW_APPNAME  = "portfolio-risk-analytics"
_RW_FIELDS   = ["title", "date.created", "url", "source.name", "theme.name", "primary_country.name"]


def _rw_fetch_page(from_date: str, to_date: str, offset: int = 0, limit: int = 1000) -> dict:
    """Single ReliefWeb API page request."""
    payload = {
        "appname": _RW_APPNAME,
        "filter": {
            "operator": "AND",
            "conditions": [
                {
                    "field": "date.created",
                    "value": {
                        "from": f"{from_date}T00:00:00+00:00",
                        "to":   f"{to_date}T23:59:59+00:00",
                    }
                },
                {
                    "field": "language.code",
                    "value": "en",
                }
            ],
        },
        "fields": {"include": _RW_FIELDS},
        "sort":   ["date.created:desc"],
        "limit":  limit,
        "offset": offset,
    }
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(
            _RW_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=30,
            verify=False,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("  ReliefWeb request failed (offset=%d): %s", offset, exc)
        return {}


def scrape_reliefweb(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch all English-language reports in the date range from ReliefWeb.
    Paginates automatically; max 1000/call, 1000 req/day limit (we use ~10 calls max).
    """
    log.info("── ReliefWeb API  %s → %s", start_date, end_date)

    all_rows = []
    offset   = 0
    limit    = 1000
    page_n   = 0

    while True:
        page_n += 1
        data    = _rw_fetch_page(start_date, end_date, offset=offset, limit=limit)
        items   = data.get("data", [])
        total   = data.get("totalCount", 0)

        if page_n == 1:
            log.info("  Total reports available: %d", total)

        if not items:
            break

        for item in items:
            fields = item.get("fields", {})
            title  = fields.get("title", "").strip()
            if not title:
                continue

            raw_date = fields.get("date", {}).get("created", "")
            try:
                date_str = datetime.strptime(raw_date[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
            except Exception:
                date_str = start_date

            url = fields.get("url", "")

            all_rows.append({
                "date":      date_str,
                "source":    "reliefweb",
                "domain":    "geopolitical",
                "headline":  title[:500],
                "tone_score": None,
                "url":       url,
            })

        log.info("  Page %d  |  fetched=%d / %d  |  rows so far=%d",
                 page_n, offset + len(items), total, len(all_rows))

        offset += limit
        if offset >= total:
            break
        time.sleep(1.5)

    if not all_rows:
        log.warning("  ReliefWeb returned no reports.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["headline", "date"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info("  ReliefWeb: %d unique reports collected", len(df))
    return df


# ══════════════════════════════════════════════
# SOURCE 3 — Hacker News (Algolia API)
# ══════════════════════════════════════════════
# HN's Algolia endpoint covers the full history of HN posts,
# queryable by keyword + timestamp range. Completely free, no key.

_HN_BASE = "https://hn.algolia.com/api/v1/search_by_date"


def scrape_hackernews_geo(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch geo-relevant Hacker News stories using geopolitical keyword list.
    """
    log.info("── Hacker News (geo keywords)  %s → %s", start_date, end_date)

    s_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    e_ts = int(datetime.strptime(end_date,   "%Y-%m-%d").timestamp())

    all_rows = []
    seen_urls = set()

    for kw in HN_GEO_KEYWORDS:
        page = 0
        while page < 3:    # max 3 pages (600 hits) per keyword
            params = {
                "query":          kw,
                "tags":           "story",
                "numericFilters": f"created_at_i>{s_ts},created_at_i<{e_ts}",
                "hitsPerPage":    200,
                "page":           page,
            }
            r = _safe_get(_HN_BASE, params=params)
            if r is None:
                break

            data    = r.json()
            hits    = data.get("hits", [])
            nb_pages = data.get("nbPages", 1)

            for h in hits:
                url   = h.get("url", "") or ""
                title = (h.get("title") or h.get("story_title") or "").strip()
                if not title or url in seen_urls:
                    continue
                seen_urls.add(url)

                ts       = h.get("created_at_i", 0)
                date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else start_date

                all_rows.append({
                    "date":      date_str,
                    "source":    "hackernews",
                    "domain":    "geopolitical",
                    "headline":  title[:400],
                    "tone_score": None,
                    "url":       url,
                })

            page += 1
            if page >= min(nb_pages, 3):
                break
            time.sleep(0.3)

    if not all_rows:
        log.warning("  Hacker News returned no geo-relevant posts.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["headline", "date"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info("  Hacker News: %d unique articles collected", len(df))
    return df


# ══════════════════════════════════════════════
# Groq Categorisation (country + region)
# ══════════════════════════════════════════════

WORLD_REGIONS = [
    "Europe", "Middle East", "North Africa", "Sub-Saharan Africa",
    "North America", "Latin America", "South Asia", "East Asia",
    "Southeast Asia", "Central Asia", "Oceania", "Global / Multiple",
]

_SYSTEM_PROMPT = """\
You are a geopolitical news classifier. Given a numbered list of news headlines, \
classify each one with:
  1. country  – the single most relevant country (use full English name, e.g. "Russia", \
"United States", "Iran"). If no specific country, write "Global".
  2. region   – choose exactly one from: """ + ", ".join(f'"{r}"' for r in WORLD_REGIONS) + """.

Return ONLY a JSON array with one object per headline, in the same order, like:
[
  {"country": "Russia", "region": "Europe"},
  {"country": "Global", "region": "Global / Multiple"},
  ...
]
Do not add any commentary, markdown fences, or extra text — just the raw JSON array."""


def _groq_classify_batch(headlines: list[str]) -> list[dict]:
    """Send a batch of headlines to Groq → list of {country, region} dicts."""
    fallback = [{"country": "Unknown", "region": "Global / Multiple"}] * len(headlines)

    if not GROQ_API_KEY:
        log.warning("  GROK_API_KEY not set — skipping Groq classification.")
        return fallback

    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    payload  = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": f"Classify these {len(headlines)} headlines:\n\n{numbered}"},
        ],
        "temperature": 0.0,
        "max_tokens":  len(headlines) * 25 + 150,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }

    for attempt in range(1, 4):
        try:
            r = requests.post(GROQ_ENDPOINT, json=payload, headers=headers,
                              timeout=60, verify=False)
            if r.status_code == 429:
                wait = 10 * (2 ** attempt)
                log.warning("    Groq rate-limited — waiting %ds", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()

            # Strip markdown fences if model wraps output
            if content.startswith("```"):
                content = "\n".join(content.split("\n")[1:])
                content = content.rstrip("`").strip()

            result = json.loads(content)
            if isinstance(result, list):
                # Pad / trim to exact batch size
                while len(result) < len(headlines):
                    result.append({"country": "Unknown", "region": "Global / Multiple"})
                return result[:len(headlines)]

        except Exception as exc:
            log.warning("    Groq attempt %d/3 failed: %s", attempt, exc)
            time.sleep(5)

    return fallback


def categorise_with_groq(df: pd.DataFrame, batch_size: int = GROQ_BATCH) -> pd.DataFrame:
    """Add 'country' and 'region' columns to df via Groq LLM, processing in batches."""
    if df.empty:
        df["country"] = pd.Series(dtype=str)
        df["region"]  = pd.Series(dtype=str)
        return df

    headlines  = df["headline"].tolist()
    n          = len(headlines)
    n_batches  = math.ceil(n / batch_size)
    interval   = 60.0 / GROQ_RPM   # min seconds between calls

    log.info("── Groq Classification  (%d articles, %d batches) ──", n, n_batches)

    countries, regions = [], []

    for i in range(n_batches):
        t0     = time.time()
        batch  = headlines[i * batch_size : (i + 1) * batch_size]
        result = _groq_classify_batch(batch)

        countries.extend(r.get("country", "Unknown") for r in result)
        regions.extend(r.get("region",  "Global / Multiple") for r in result)

        elapsed = time.time() - t0
        sleep_t = max(0.0, interval - elapsed)
        log.info("  Batch %d/%d  (%d headlines)  → sleep %.1fs",
                 i + 1, n_batches, len(batch), sleep_t)

        if sleep_t > 0 and (i + 1) < n_batches:
            time.sleep(sleep_t)

    df = df.copy()
    df["country"] = countries
    df["region"]  = regions
    return df


# ══════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════

OUTPUT_COLS = ["date", "source", "domain", "headline", "tone_score", "url", "country", "region"]


def run(start_date: str = DEFAULT_START_DATE,
        end_date:   str = DEFAULT_END_DATE,
        batch_size: int = GROQ_BATCH):

    ensure_dirs()

    log.info("═" * 60)
    log.info("  GEO SCRAPER  |  %s → %s", start_date, end_date)
    log.info("  Sources      :  Wikipedia · ReliefWeb · Hacker News")
    log.info("  Groq model   :  %s", GROQ_MODEL)
    log.info("  Groq key     :  %s", "✔ loaded" if GROQ_API_KEY else "✘ MISSING — check .env")
    log.info("═" * 60)

    # ── 1. Scrape all 3 sources ────────────────
    frames = []

    wiki_df = scrape_wikipedia(start_date, end_date)
    if not wiki_df.empty:
        frames.append(wiki_df)

    rw_df = scrape_reliefweb(start_date, end_date)
    if not rw_df.empty:
        frames.append(rw_df)

    hn_df = scrape_hackernews_geo(start_date, end_date)
    if not hn_df.empty:
        frames.append(hn_df)

    if not frames:
        log.error("No articles collected from any source. Exiting.")
        return pd.DataFrame(columns=OUTPUT_COLS)

    # ── 2. Combine & deduplicate ───────────────
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(subset=["headline", "date"]).reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)
    log.info("── Combined: %d unique articles from %d sources", len(df), len(frames))
    log.info("  Source breakdown:\n%s", df["source"].value_counts().to_string())

    # ── 3. Groq country/region classification ──
    df = categorise_with_groq(df, batch_size=batch_size)

    # ── 4. Ensure standard columns ─────────────
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = None
    df = df[OUTPUT_COLS].copy()

    # ── 5. Save ────────────────────────────────
    save(df, "gdelt_geo.csv")

    log.info("")
    log.info("═" * 60)
    log.info("  COMPLETE")
    log.info("  Total articles : %d", len(df))
    log.info("  Date range     : %s → %s",
             df["date"].min() if not df.empty else "N/A",
             df["date"].max() if not df.empty else "N/A")
    if not df.empty:
        log.info("  Top countries  :\n%s", df["country"].value_counts().head(10).to_string())
        log.info("  Region split   :\n%s", df["region"].value_counts().to_string())
    log.info("  Output         : %s", RAW_DIR / "gdelt_geo.csv")
    log.info("═" * 60)

    return df


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geopolitical news scraper: Wikipedia + ReliefWeb + HN → Groq classification"
    )
    parser.add_argument("--start",  default=DEFAULT_START_DATE,
                        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START_DATE})")
    parser.add_argument("--end",    default=DEFAULT_END_DATE,
                        help=f"End date YYYY-MM-DD (default: {DEFAULT_END_DATE})")
    parser.add_argument("--batch",  type=int, default=GROQ_BATCH,
                        help=f"Headlines per Groq call (default: {GROQ_BATCH})")
    args = parser.parse_args()
    run(start_date=args.start, end_date=args.end, batch_size=args.batch)
