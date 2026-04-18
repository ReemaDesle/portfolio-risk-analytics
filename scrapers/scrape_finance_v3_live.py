"""
scrape_finance_v3_live.py (Hybrid Live Filler)
==============================================
Fills the 2024-2026 financial news gap using direct-site archives.
Faster and bypasses Wayback Machine IP blocks.
"""

import os
import re
import csv
import time
import random
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT_DIR / "data" / "raw" / "news"
CP_DIR   = ROOT_DIR / "scrapers" / "checkpoints"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CP_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DIR / "financial_news_v3.csv"
PROGRESS_FILE = CP_DIR / "finance_live_progress.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
}

# ── Preprocessing ─────────────────────────────────────────────────────────────
JUNK_RE = re.compile(r"(live updates?|live blog|word of the day|quote of the day|sensex today|nifty updates)", re.IGNORECASE)

def is_junk(headline: str) -> bool:
    if not headline or len(headline) < 30: return True
    return bool(JUNK_RE.search(headline))

def clean_headline(text: str) -> str:
    if not text: return ""
    text = re.sub(r"\s*[|\u2014\u2013]\s*(ET Markets?|Moneycontrol|Livemint|BSE|NSE|Reuters).*$", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()

# ── Direct-Site Parsers ────────────────────────────────────────────────────────
def fetch_et_archive(dt):
    """Fetch from Economic Times Daily Archive."""
    # Pattern: https://economictimes.indiatimes.com/archive/year-2024,month-3,day-18.cms
    url = f"https://economictimes.indiatimes.com/archive/year-{dt.year},month-{dt.month},day-{dt.day}.cms"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200: return []
        soup = BeautifulSoup(r.text, "html.parser")
        found = []
        # ET Archive 2024 uses article-list or direct class table
        for a in soup.select("ul.article-list a, .content a, table.pda-archive a"):
            text = clean_headline(a.get_text(strip=True))
            if not is_junk(text) and ("/markets/" in a['href'] or "stocks" in a['href']):
                found.append({"headline": text, "url": a['href']})
        return found
    except Exception as e:
        log.error(f"ET Error {dt.date()}: {e}")
        return []

def fetch_livemint_archive(dt):
    """Fetch from Livemint Daily Archive."""
    # Pattern: https://www.livemint.com/news/archive/2024-03-18
    url = f"https://www.livemint.com/news/archive/{dt.strftime('%Y-%m-%d')}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200: return []
        soup = BeautifulSoup(r.text, "html.parser")
        found = []
        # Livemint 2024 uses listing_row or headline selectors
        for it in soup.select("div.listing_row, h2.headline"):
            a = it.find('a', href=True) if it.name != 'a' else it
            if a:
                text = clean_headline(a.get_text(strip=True))
                if not is_junk(text):
                    found.append({"headline": text, "url": a['href']})
        return found
    except Exception as e:
        log.error(f"Livemint Error {dt.date()}: {e}")
        return []

def fetch_moneycontrol_page(page):
    """Fetch from Moneycontrol Live Pagination."""
    url = f"https://www.moneycontrol.com/news/business/markets/page-{page}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200: return []
        soup = BeautifulSoup(r.text, "html.parser")
        found = []
        for it in soup.select("li.clearfix"):
            h2 = it.find(['h2', 'h3'])
            a = h2.find('a', href=True) if h2 else it.find('a', href=True)
            if a:
                text = clean_headline(a.get_text(strip=True))
                # For Moneycontrol live, we also need to pull the date from the span
                date_span = it.find("span")
                date_str = date_span.get_text(strip=True) if date_span else ""
                if not is_junk(text):
                    found.append({"headline": text, "url": a['href'], "raw_date": date_str})
        return found
    except Exception as e:
        log.error(f"Moneycontrol Page {page} Error: {e}")
        return []

# ── Main ──────────────────────────────────────────────────────────────────────
def run_filler(start_date_str, end_date_str):
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # 1. Fill ET and Livemint (Date-based is easy)
    current_dt = start_dt
    day_count = (end_dt - start_dt).days + 1
    
    log.info(f"Filling ET and Livemint from {start_date_str} to {end_date_str}...")
    for _ in tqdm(range(day_count), desc="Filling Archives"):
        dt_str = current_dt.strftime("%Y-%m-%d")
        
        # Livemint
        heads = fetch_livemint_archive(current_dt)
        if heads:
            with open(OUTPUT_FILE, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                for h in heads:
                    writer.writerow([dt_str, "Livemint", "financial", h['headline'], h['url']])
        
        # ET
        heads = fetch_et_archive(current_dt)
        if heads:
            with open(OUTPUT_FILE, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                for h in heads:
                    writer.writerow([dt_str, "EconomicTimes", "financial", h['headline'], h['url']])
        
        current_dt += timedelta(days=1)
        time.sleep(random.uniform(0.5, 1.5))

    # 2. Moneycontrol (Page-based, go back until target date)
    log.info("Filling Moneycontrol from Live Pagination...")
    page = 1
    reached_target = False
    while not reached_target and page < 500: # Safety cap
        log.info(f"  Processing Moneycontrol Page {page}...")
        heads = fetch_moneycontrol_page(page)
        if not heads: break
        
        new_rows = []
        for h in heads:
            # Try to parse date from Moneycontrol span (e.g. "April 18, 2024")
            # This is heuristic. If we can't parse it, we skip.
            try:
                # Basic string contains check for the year
                if str(start_dt.year-1) in h['raw_date']:
                    reached_target = True
                    break
                
                # If it's a valid recent date, add it
                # (Actual date parsing logic would be better but this is a fast filler)
                new_rows.append(["2024-2026", "Moneycontrol", "financial", h['headline'], h['url']])
            except:
                continue
        
        if new_rows:
            with open(OUTPUT_FILE, 'a', encoding='utf-8', newline='') as f:
                csv.writer(f).writerows(new_rows)
        
        page += 1
        time.sleep(random.uniform(1.0, 2.0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-03-18")
    parser.add_argument("--end",   default="2026-04-16")
    args = parser.parse_args()
    
    run_filler(args.start, args.end)
