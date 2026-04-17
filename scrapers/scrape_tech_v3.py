"""
Portfolio Risk Analytics — Technology News Scraper (v2)
======================================================
High-volume historical scraper using the Hacker News (Algolia) API.
Scales by segmenting the search into monthly buckets to bypass 1,000-hit limits.

Target Date Range: 2021-01-01 onwards.
Output: data/raw/news/tech_news_v2.csv
"""

import os
import re
import sys
import time
import json
import logging
import random
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from dateutil.relativedelta import relativedelta

import requests
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DEFAULT_START_DATE = "2021-01-01"
DEFAULT_END_DATE   = datetime.today().strftime("%Y-%m-%d")

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT_DIR / "data" / "raw" / "news"
CHECKPOINT_DIR = ROOT_DIR / "scrapers" / "checkpoints"
OUTPUT_FILE = RAW_DIR / "tech_news_v3.csv"
CHECKPOINT_FILE = CHECKPOINT_DIR / "tech_v3_checkpoint.json"

TECH_KEYWORDS = [
    "Artificial Intelligence", "NVIDIA", "OpenAI", "Semiconductor",
    "TSMC", "IPO", "Tech Layoffs", "Cloud Computing", "Machine Learning",
    "Cybersecurity", "Startup", "Venture Capital", "Big Tech", "Regulation",
    "Quantum Computing", "Automation", "SaaS", "Fintech"
]

# Strict relevance patterns (must match as whole words)
RELEVANCE_WORDS = [
    "AI", "Artificial Intelligence", "NVIDIA", "OpenAI", "Semiconductor",
    "TSMC", "IPO", "Layoff", "Cloud", "Computing", "Machine Learning",
    "Cybersecurity", "Startup", "Venture Capital", "Big Tech", "Regulation",
    "Quantum", "Automation", "SaaS", "Fintech"
]
RELEVANCE_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in RELEVANCE_WORDS) + r")\b", re.IGNORECASE)

# Regex patterns for junk rows (similar to existing v1 scrapers)
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
# Utilities
# ══════════════════════════════════════════════

def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def _safe_get(url: str, params=None, retries: int = 3, wait: float = 2.0) -> requests.Response | None:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30, verify=False)
            r.raise_for_status()
            return r
        except Exception as exc:
            log.warning("  Request failed (%d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(wait * attempt)
    return None

def clean_headline(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    text = re.sub(r"\s*[|\u2014\u2013]\s*(TechCrunch|Hacker News|Reuters|NYT|BBC|CNN|AP|Guardian).*$", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_junk(headline: str) -> bool:
    if not isinstance(headline, str): return True
    return bool(JUNK_RE.search(headline))

def is_relevant(headline: str) -> bool:
    """Strict word-boundary check to avoid 'AI' matching 'Aircraft'."""
    if not isinstance(headline, str): return False
    return bool(RELEVANCE_RE.search(headline))

# ══════════════════════════════════════════════
# Pagination & Checkpointing
# ══════════════════════════════════════════════

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text())
        except:
            pass
    return {"last_month": None, "seen_urls": []}

def save_checkpoint(last_month, seen_urls):
    # We only store a sample of URLs to avoid huge checkpoint files, 
    # or just use it to track the month.
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_month": last_month, "seen_urls": list(seen_urls)[-1000:]}, f)

def get_monthly_ranges(start_date: str, end_date: str):
    """Generate (start_ts, end_ts, month_str) tuples."""
    curr = datetime.strptime(start_date, "%Y-%m-%d")
    end  = datetime.strptime(end_date, "%Y-%m-%d")
    
    while curr <= end:
        m_start = curr
        m_end   = curr + relativedelta(months=1) - timedelta(seconds=1)
        if m_end > end:
            m_end = end
        
        yield (
            int(m_start.timestamp()),
            int(m_end.timestamp()),
            m_start.strftime("%Y-%m")
        )
        curr += relativedelta(months=1)

# ══════════════════════════════════════════════
# Core Scraper
# ══════════════════════════════════════════════

def scrape_hn_month(start_ts, end_ts, month_str, keywords, seen_urls):
    BASE_URL = "https://hn.algolia.com/api/v1/search_by_date"
    rows = []
    
    log.info(f"  Fetching: {month_str}...")
    
    for kw in keywords:
        page = 0
        while page < 50: # Algolia allows up to 1000 hits (50 pages * 20 hits)
            params = {
                "query": kw,
                "tags": "story",
                "numericFilters": f"created_at_i>={start_ts},created_at_i<={end_ts}",
                "hitsPerPage": 20, # Default is 20
                "page": page
            }
            r = _safe_get(BASE_URL, params=params)
            if not r: break
            
            data = r.json()
            hits = data.get("hits", [])
            nb_pages = data.get("nbPages", 1)
            
            for h in hits:
                url = h.get("url", "")
                if not url or url in seen_urls: continue
                
                title = h.get("title", "").strip()
                if not title: continue
                
                cleaned = clean_headline(title)
                if len(cleaned) < 40 or is_junk(cleaned):
                    continue
                
                # New strict relevance check
                if not is_relevant(cleaned):
                    continue
                
                dt = datetime.utcfromtimestamp(h.get("created_at_i")).strftime("%Y-%m-%d")
                rows.append({
                    "date": dt,
                    "source": "hackernews",
                    "domain": "technology",
                    "headline": cleaned,
                    "tone_score": None,
                    "url": url
                })
                seen_urls.add(url)
            
            page += 1
            if page >= nb_pages: break
            time.sleep(random.uniform(0.6, 1.4))
            
    return rows

def run_v2(start_date, end_date, test=False):
    ensure_dirs()
    checkpoint = load_checkpoint()
    
    # If checkpoint exists and is within range, resume
    resume_month = checkpoint.get("last_month")
    actual_start = start_date
    if resume_month and not test:
        log.info(f"Resuming from after {resume_month}")
        actual_start = (datetime.strptime(resume_month, "%Y-%m") + relativedelta(months=1)).strftime("%Y-%m-%d")

    seen_urls = set(checkpoint.get("seen_urls", []))
    
    log.info("═" * 60)
    log.info(f" TECH SCRAPER V2 | {start_date} → {end_date}")
    if test: log.info(" *** TEST MODE ENABLED (Top 5 items per month) ***")
    log.info("═" * 60)

    # Initialize CSV if not exists
    if not OUTPUT_FILE.exists():
        pd.DataFrame(columns=["date", "source", "domain", "headline", "tone_score", "url"]).to_csv(OUTPUT_FILE, index=False)

    total_added = 0
    for s_ts, e_ts, month_str in get_monthly_ranges(actual_start, end_date):
        # In test mode, we only use a few keywords and 1 page
        kws = TECH_KEYWORDS[:3] if test else TECH_KEYWORDS
        
        month_rows = scrape_hn_month(s_ts, e_ts, month_str, kws, seen_urls)
        
        if month_rows:
            df_month = pd.DataFrame(month_rows)
            df_month.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            total_added += len(month_rows)
            log.info(f"  ✔ Added {len(month_rows)} records for {month_str}")
        
        save_checkpoint(month_str, seen_urls)
        
        if test:
            log.info("Test mode: Stopping after one month.")
            break

    log.info("═" * 60)
    log.info(f" COMPLETE | Total New Records: {total_added}")
    log.info(f" Output: {OUTPUT_FILE}")
    log.info("═" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end",   default=DEFAULT_END_DATE)
    parser.add_argument("--test",  action="store_true", help="Run in test mode (one month only)")
    args = parser.parse_args()
    
    run_v2(args.start, args.end, test=args.test)
