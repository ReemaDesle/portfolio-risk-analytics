"""
scrape_finance.py

Paginated direct scraper for financial news headlines (v4). 
Targets recent history (Jan 2024) by paginating Markets news lists.
Bypasses Archive.org.

Output: data/raw/news/financial_news_final.csv
"""

import argparse
import csv
import hashlib
import json
import logging
import re
import time
from datetime import datetime, date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_RAW_DIR    = ROOT_DIR / "data" / "raw" / "news"
CHECKPOINT_DIR  = Path(__file__).parent / "checkpoints"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE     = DATA_RAW_DIR / "financial_news_final.csv"
CHECKPOINT_FILE = CHECKPOINT_DIR / "financial_news_checkpoint.json"

# ── Config ────────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
}

SOURCES = {
    "Moneycontrol": {
        "url_pattern": "https://www.moneycontrol.com/news/business/markets/page-{page}/",
        "container_selector": "li.clearfix, div.f3",
        "date_stop": "2024-01-01"
    },
    "Economic Times": {
        "url_pattern": "https://economictimes.indiatimes.com/markets/stocks/news/articlelist/1715249553.cms?curpg={page}",
        "container_selector": "div.eachStory, ul.content li",
        "date_stop": "2024-01-01"
    },
    "Livemint": {
        "url_pattern": "https://www.livemint.com/market/stock-market-news?page={page}",
        "container_selector": "div.listing_row, div.headline_list",
        "date_stop": "2024-01-01"
    }
}

# ── Utilities ─────────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text())
            return {
                "processed": set(data.get("processed", [])),
                "ids":       set(data.get("ids", []))
            }
        except: pass
    return {"processed": set(), "ids": set()}

def save_checkpoint(state: dict):
    CHECKPOINT_FILE.write_text(json.dumps({
        "processed": list(state["processed"]),
        "ids":       list(state["ids"])
    }, indent=2))

def init_csv():
    if not OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "source", "domain", "headline", "url"])
            writer.writeheader()

def append_rows(rows: list[dict]):
    if not rows: return
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "source", "domain", "headline", "url"])
        writer.writerows(rows)

def parse_date_heuristic(tag):
    """Attempt to find a date within or near a news tag."""
    # Look for data attributes
    date_str = tag.get("data-date") or tag.get("data-created") or tag.get("data-modified")
    if date_str:
        return date_str[:10]
    
    # Text search in small/span tags nearby
    time_tag = tag.find("span", class_=re.compile(r"date|time|posted", re.I)) or tag.find("time")
    if time_tag:
        txt = time_tag.get_text()
        # Look for patterns like Jan 12, 2024
        match = re.search(r"(\w+ \d{1,2}, 20\d{2})", txt)
        if match:
            try:
                dt = datetime.strptime(match.group(1), "%b %d, %Y")
                return dt.strftime("%Y-%m-%d")
            except: pass
    return None

# ── Scraper ───────────────────────────────────────────────────────────────────

def run_pagination_scrape(source_name, config, session, max_pages, state):
    log.info(f"=== Starting Source: {source_name} ===")
    
    stop_date = config["date_stop"]
    
    for page in range(1, max_pages + 1):
        url = config["url_pattern"].format(page=page)
        log.info(f"  Fetching Page {page}: {url}")
        
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 404:
                log.info(f"  End of pages reached (404) at page {page}")
                break
            r.raise_for_status()
            
            soup = BeautifulSoup(r.text, "html.parser")
            # Multiple selectors for different sites
            containers = soup.select(config["container_selector"])
            if not containers:
                # Fallback to pure links if selector fails
                containers = soup.find_all("a", href=True)
            
            page_articles = []
            oldest_date_on_page = None

            for it in containers:
                a = it if it.name == "a" else it.find("a", href=True)
                if not a: continue
                
                href = a["href"]
                if not href.startswith("http"):
                    domain = url.split("/")[2]
                    href = f"https://{domain}{href}"
                
                headline = a.get_text(strip=True) or a.get("title", "").strip()
                if len(headline) < 25 or "/news/" not in href and "/markets/" not in href:
                    continue
                
                # Check IDs for deduplication
                rid = hashlib.md5(href.encode()).hexdigest()[:12]
                if rid in state["ids"]:
                    continue

                # Try to get date
                found_date = parse_date_heuristic(it)
                if found_date:
                    oldest_date_on_page = found_date
                    if found_date < stop_date:
                        # We reached older content, stop this source soon
                        log.info(f"    Reached date {found_date} (< {stop_date}). Stopping source.")
                        return page_articles

                page_articles.append({
                    "date": found_date or "2024-01-XX", # Placeholder if unknown
                    "source": source_name,
                    "domain": "financial",
                    "headline": headline,
                    "url": href
                })
                state["ids"].add(rid)

            if page_articles:
                append_rows(page_articles)
                log.info(f"    ✔ Saved {len(page_articles)} articles from Page {page}")
                save_checkpoint(state)
            else:
                log.info(f"    No new articles found on page {page}")

            time.sleep(2.0) # Polite delay
            
        except Exception as e:
            log.warning(f"  Error on page {page}: {e}")
            break
            
    return []

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01", help="YYYY-MM-DD - stop scraping when reaching articles older than this")
    parser.add_argument("--end", default="2024-01-31", help="YYYY-MM-DD - ignore articles newer than this")
    parser.add_argument("--max-pages", type=int, default=100)
    args = parser.parse_args()

    init_csv()
    state = load_checkpoint()
    session = requests.Session()
    session.headers.update(HEADERS)
    
    for name, config in SOURCES.items():
        # Override the stop date from CLI
        config["date_stop"] = args.start
        
        log.info(f"=== Starting Source: {name} (Window: {args.start} to {args.end}) ===")
        
        for page in range(1, args.max_pages + 1):
            url = config["url_pattern"].format(page=page)
            log.info(f"  Fetching Page {page}: {url}")
            
            try:
                r = session.get(url, timeout=15)
                if r.status_code == 404: break
                r.raise_for_status()
                
                soup = BeautifulSoup(r.text, "html.parser")
                containers = soup.select(config["container_selector"]) or soup.find_all("a", href=True)
                
                page_articles = []
                found_date = None
                for it in containers:
                    a = it if it.name == "a" else it.find("a", href=True)
                    if not a: continue
                    
                    href = a["href"]
                    if not href.startswith("http"):
                        domain = url.split("/")[2]
                        href = f"https://{domain}{href}"
                    
                    headline = a.get_text(strip=True) or a.get("title", "").strip()
                    if len(headline) < 25 or "/news/" not in href and "/markets/" not in href:
                        continue
                    
                    rid = hashlib.md5(href.encode()).hexdigest()[:12]
                    if rid in state["ids"]: continue

                    found_date = parse_date_heuristic(it)
                    if found_date:
                        if found_date < args.start:
                            log.info(f"    Reached date {found_date} (< {args.start}). Stopping {name}.")
                            break # Move to next source
                        if found_date > args.end:
                            continue # Skip newer articles

                    page_articles.append({
                        "date": found_date or ds_placeholder(args.start),
                        "source": name,
                        "domain": "financial",
                        "headline": headline,
                        "url": href
                    })
                    state["ids"].add(rid)

                if page_articles:
                    append_rows(page_articles)
                    log.info(f"    ✔ Saved {len(page_articles)} articles from Page {page}")
                    save_checkpoint(state)
                
                time.sleep(2.5)
                
                # If we broke out of the inner loop due to date
                if found_date and found_date < args.start:
                    break

            except Exception as e:
                log.warning(f"  Error on page {page}: {e}")
                break

    log.info(f"\nScrape complete. Output: {OUTPUT_FILE}")

def ds_placeholder(start_date):
    # Returns 2024-01-XX style placeholder based on year/month of start_date
    return start_date[:8] + "XX"

if __name__ == "__main__":
    main()
