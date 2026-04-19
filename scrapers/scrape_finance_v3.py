"""
scrape_finance_v3.py (Manifest-Based Archive Scraper)
====================================================
Robust scraper using Wayback CDX Manifests and Site-Specific Parsers.
Targets: Moneycontrol, Economic Times, Livemint (2021-2026).
"""

import os
import re
import json
import csv
import time
import random
import hashlib
import logging
import argparse
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

OUTPUT_FILE    = RAW_DIR / "financial_news_v3.csv"
MANIFEST_FILE  = CP_DIR / "finance_manifest_v3.json"
PROGRESS_FILE  = CP_DIR / "finance_progress_v3.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
}

SOURCES = {
    "Moneycontrol": "https://www.moneycontrol.com/news/business/markets",
    "EconomicTimes": "https://economictimes.indiatimes.com/markets/stocks/news",
    "Livemint": "https://www.livemint.com/market/stock-market-news"
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

# ── Manifest Discovery ────────────────────────────────────────────────────────
def discover_manifest(start_date, end_date):
    """Build a Master Manifest of snapshots to process."""
    if MANIFEST_FILE.exists():
        log.info("Loading existing manifest...")
        with open(MANIFEST_FILE, 'r') as f: return json.load(f)

    log.info(f"Generating new manifest ({start_date} to {end_date})...")
    manifest = []
    
    for name, url in SOURCES.items():
        cdx_url = "https://web.archive.org/cdx/search/cdx"
        params = {
            "url": url, "from": start_date.replace("-", ""), "to": end_date.replace("-", ""),
            "output": "json", "collapse": "timestamp:10" # Roughly hourly
        }
        
        try:
            r = requests.get(cdx_url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if not data or len(data) < 2: continue
            
            # Subsample to 3 IST snapshots: 04:00 (9:30 IST), 07:00 (12:30 IST), 10:00 (3:30 IST)
            target_hours = ["04", "07", "10"]
            grouped = {}
            for row in data[1:]:
                ts = row[1]
                orig = row[2]
                day = ts[:8]
                hour = ts[8:10]
                if day not in grouped: grouped[day] = {}
                for target in target_hours:
                    diff = abs(int(hour) - int(target))
                    if target not in grouped[day] or diff < grouped[day][target]["diff"]:
                        grouped[day][target] = {"ts": ts, "url": orig, "diff": diff}
            
            for day in sorted(grouped.keys()):
                for target in target_hours:
                    if target in grouped[day]:
                        manifest.append({
                            "source": name,
                            "timestamp": grouped[day][target]["ts"],
                            "url": grouped[day][target]["url"]
                        })
        except Exception as e:
            log.error(f"Discovery for {name} failed: {e}")

    with open(MANIFEST_FILE, 'w') as f: json.dump(manifest, f, indent=2)
    log.info(f"Manifest saved with {len(manifest)} snapshots.")
    return manifest

# ── Site-Specific Parsing ─────────────────────────────────────────────────────
def parse_snapshot(html, source):
    soup = BeautifulSoup(html, "html.parser")
    found = []
    
    if source == "Moneycontrol":
        # Moneycontrol stores headlines in MT15 blocks or clearfix news lists
        for it in soup.select("div.MT15, li.clearfix, h2, h3"):
            a = it.find('a', href=True) if it.name != 'a' else it
            if a:
                text = clean_headline(a.get_text(strip=True))
                if not is_junk(text):
                    found.append({"headline": text, "url": a['href']})
                    
    elif source == "EconomicTimes":
        # ET uses eachStory or articlelist containers
        for it in soup.select("div.eachStory, div.main-content li"):
            a = it.find('a', href=True)
            if a:
                text = clean_headline(a.get_text(strip=True))
                if not is_junk(text):
                    found.append({"headline": text, "url": a['href']})
                    
    elif source == "Livemint":
        # Livemint has listing_row containers
        for it in soup.select("div.listing_row"):
            a = it.find('a', href=True)
            if a:
                text = clean_headline(a.get_text(strip=True))
                if not is_junk(text):
                    found.append({"headline": text, "url": a['href']})
                    
    return found

# ── Main Loop ─────────────────────────────────────────────────────────────────
def run(start, end):
    manifest = discover_manifest(start, end)
    
    if not OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("date,source,domain,headline,url\n")
            
    progress = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f: progress = json.load(f)
    
    # ── Cross-Run Deduplication ───────────────────────────────────────────────
    seen_heads = set()
    if OUTPUT_FILE.exists():
        log.info("Indexing existing headlines for deduplication...")
        try:
            # Use 'errors=ignore' for robust reading of potentially malformed bits
            with open(OUTPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    h = row.get('headline', '')
                    if h:
                        # Normalize: strip spaces and quotes for a robust hash
                        clean_h = h.strip().strip('"').strip("'").lower()
                        seen_heads.add(hashlib.md5(clean_h.encode()).hexdigest()[:12])
            log.info(f"  Indexed {len(seen_heads):,} unique headlines.")
        except Exception as e:
            log.warning(f"  Deduplication indexing failed (starting fresh): {e}")

    # ── Main Harvest Loop ─────────────────────────────────────────────────────
    for s in tqdm(manifest, desc="Processing Archive"):
        task_id = f"{s['source']}_{s['timestamp']}"
        if task_id in progress: continue
        
        # Sanitize URL (Remove trailing dots often found in archive CDX results)
        clean_url = s['url'].replace(".com./", ".com/").replace(".co./", ".co/")
        play_url = f"https://web.archive.org/web/{s['timestamp']}id_/{clean_url}"
        date_str = f"{s['timestamp'][:4]}-{s['timestamp'][4:6]}-{s['timestamp'][6:8]}"
        
        try:
            r = None
            # Polite retry wrapper
            for attempt in range(4):
                try:
                    r = requests.get(play_url, headers=HEADERS, timeout=25)
                    if r.status_code == 200: break
                    if r.status_code == 429:
                        wait = 300 * (attempt + 1)
                        log.warning(f"  Rate limited (429). Waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        time.sleep(15)
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as ce:
                    # Target machine actively refused (10061) or Timeout
                    wait = 600 # 10 Minute Deep Sleep
                    log.error(f"  Connection Refused/Timeout on {task_id}. Triggering 10min DEEP SLEEP to clear block...")
                    time.sleep(wait)
                except Exception as e:
                    time.sleep(15)
            
            if r and r.status_code == 200:
                results = parse_snapshot(r.text, s['source'])
                new_rows = []
                for h in results:
                    # Apply identical normalization for a perfect match
                    clean_h = h['headline'].strip().strip('"').strip("'").lower()
                    h_hash = hashlib.md5(clean_h.encode()).hexdigest()[:12]
                    
                    if h_hash not in seen_heads:
                        new_rows.append([date_str, s['source'], "financial", h['headline'], h['url']])
                        seen_heads.add(h_hash)
                
                if new_rows:
                    with open(OUTPUT_FILE, 'a', encoding='utf-8', newline='') as f:
                        import csv
                        csv.writer(f).writerows(new_rows)
                
                progress[task_id] = True
                with open(PROGRESS_FILE, 'w') as f: json.dump(progress, f)
                
            time.sleep(random.uniform(1.5, 4.0)) # Politer jitter
            
        except Exception as e:
            log.error(f"Error on {task_id}: {e}")
            time.sleep(30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end",   default="2026-04-16")
    args = parser.parse_args()
    run(args.start, args.end)
