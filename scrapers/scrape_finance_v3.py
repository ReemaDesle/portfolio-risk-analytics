"""
scrape_finance_v3.py (Archive Scraper)
======================================
Historical Financial News Scraper using the Wayback Machine (CDX API).
Targets: Moneycontrol, Economic Times, Livemint.
Frequency: 3 snapshots per day aligned with IST Market Hours.
"""

import os
import re
import json
import time
import random
import hashlib
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
CHECKPOINT_FILE = CP_DIR / "finance_v3_checkpoint.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
}

SOURCES = {
    "Moneycontrol": "https://www.moneycontrol.com/news/business/markets",
    "EconomicTimes": "https://economictimes.indiatimes.com/markets/stocks/news",
    "Livemint": "https://www.livemint.com/market/stock-market-news"
}

# ── Preprocessing Logic ───────────────────────────────────────────────────────
JUNK_PATTERNS = [
    r"^(live updates?|live blog)",
    r"(results? (live|announced)|live result)",
    r"^(word of the day|quote of the day)",
    r"(wishes|greetings|messages|images quotes)",
    r"^(sensex today|gold rate today|nifty today)",
    r"(election \d{4} result)",
    r"(board results \d{4})",
    r"^oscars? (winner|results?)",
    r"nifty updates: ",
    r"stock market today: "
]
JUNK_RE = re.compile("|".join(JUNK_PATTERNS), re.IGNORECASE)

def is_junk(headline: str) -> bool:
    if not headline or len(headline) < 30: return True
    return bool(JUNK_RE.search(headline))

def clean_headline(text: str) -> str:
    if not text: return ""
    text = text.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    text = re.sub(r"\s*[|\u2014\u2013]\s*(ET Markets?|Moneycontrol|Livemint|BSE|NSE|Reuters|NYT|BBC|CNN|AP|Guardian).*$", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Wayback Discovery (CDX) ───────────────────────────────────────────────────
def discover_snapshots(url, start_date, end_date):
    """Fetch daily snapshots from CDX API."""
    cdx_url = "https://web.archive.org/cdx/search/cdx"
    params = {
        "url": url,
        "from": start_date.replace("-", ""),
        "to": end_date.replace("-", ""),
        "output": "json",
        "collapse": "timestamp:10" # This gives roughly hourly snapshots
    }
    
    log.info(f"Discovering snapshots for {url}...")
    try:
        r = requests.get(cdx_url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data or len(data) < 2: return []
        
        # Data format: [["urlkey", "timestamp", "original", ...], [...]]
        snapshots = []
        for row in data[1:]:
            timestamp = row[1]
            original = row[2]
            snapshots.append({"timestamp": timestamp, "url": original})
        
        # Subsample to 3 per day (aiming for 04:00, 07:00, 10:00 UTC ~ 9:30, 12:30, 15:30 IST)
        target_hours = ["04", "07", "10"]
        grouped = {}
        for s in snapshots:
            day = s["timestamp"][:8]
            hour = s["timestamp"][8:10]
            if day not in grouped: grouped[day] = {}
            
            # Find closest to each target hour
            for target in target_hours:
                diff = abs(int(hour) - int(target))
                if target not in grouped[day] or diff < grouped[day][target]["diff"]:
                    grouped[day][target] = {"snap": s, "diff": diff}
        
        final_snaps = []
        for day in sorted(grouped.keys()):
            for target in target_hours:
                if target in grouped[day]:
                    final_snaps.append(grouped[day][target]["snap"])
        
        log.info(f"  Found {len(final_snaps)} subsampled snapshots.")
        return final_snaps
    except Exception as e:
        log.error(f"CDX discovery failed: {e}")
        return []

# ── Extraction ────────────────────────────────────────────────────────────────
def extract_headlines(html, source_name):
    soup = BeautifulSoup(html, "html.parser")
    headlines = []
    
    # Generic news seekers (H1-H4) or common list selectors
    # High-signal headlines are often in <a> tags inside <h3> or <h4>
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'a']):
        text = tag.get_text(strip=True)
        cleaned = clean_headline(text)
        
        if not is_junk(cleaned):
            # Try to find a link if it's an H-tag
            href = ""
            if tag.name.startswith('h'):
                a = tag.find('a', href=True)
                if a: href = a['href']
            elif tag.name == 'a' and tag.get('href'):
                href = tag['href']
                
            if href and (len(cleaned) > 35):
                headlines.append({
                    "headline": cleaned,
                    "url": href
                })
    
    # Simple deduplication within the snapshot
    seen = set()
    unique = []
    for h in headlines:
        if h['headline'] not in seen:
            unique.append(h)
            seen.add(h['headline'])
    return unique

# ── Main Run ──────────────────────────────────────────────────────────────────
def run_scrape(start_date, end_date):
    if not OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("date,source,domain,headline,url\n")
            
    seen_headlines = set() # Global dedup
    
    for source_name, base_url in SOURCES.items():
        snaps = discover_snapshots(base_url, start_date, end_date)
        
        for s in tqdm(snaps, desc=f"Processing {source_name}"):
            ts = s["timestamp"]
            orig = s["url"]
            # id_ modifier gets raw content
            play_url = f"https://web.archive.org/web/{ts}id_/{orig}"
            
            try:
                # Polite retry wrapper
                for attempt in range(3):
                    try:
                        r = requests.get(play_url, headers=HEADERS, timeout=20)
                        if r.status_code == 200: break
                        if r.status_code == 429:
                            wait = 60 * (attempt + 1)
                            log.warning(f"  Rate limited. Waiting {wait}s...")
                            time.sleep(wait)
                        else:
                            time.sleep(5)
                    except:
                        time.sleep(5)
                
                if r.status_code != 200: continue
                
                extracted = extract_headlines(r.text, source_name)
                
                # Filter of new headlines only
                new_rows = []
                # Format timestamp for date column
                date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
                
                for h in extracted:
                    if h['headline'] not in seen_headlines:
                        new_rows.append([
                            date_str, source_name, "financial", h['headline'], h['url']
                        ])
                        seen_headlines.add(h['headline'])
                
                # Append in chunks
                if new_rows:
                    with open(OUTPUT_FILE, 'a', encoding='utf-8', newline='') as f:
                        import csv
                        writer = csv.writer(f)
                        writer.writerows(new_rows)
                
                time.sleep(random.uniform(1.0, 2.5)) # Jitter
                
            except Exception as e:
                log.error(f"Error fetching snapshot {ts}: {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2021-01-15") # Test default
    args = parser.parse_args()
    
    run_scrape(args.start, args.end)
