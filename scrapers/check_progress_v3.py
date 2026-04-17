"""
Portfolio Risk Analytics — Progress Tracker (v3 - Memory Optimized)
==================================================================
Lightweight utility to check scraper volume without loading pandas.
"""

import os
import json
import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT_DIR / "data" / "raw" / "news"
CP_DIR   = ROOT_DIR / "scrapers" / "checkpoints"

FILES = {
    "Technology (v3)":    {
        "csv": RAW_DIR / "tech_news_v3.csv",
        "checkpoint": CP_DIR / "tech_v3_checkpoint.json",
        "keywords": ["AI", "NVIDIA", "OpenAI", "Semiconductor", "Startup", "Cloud", "GPT", "LLM", "Algorithm"]
    },
    "Geopolitical (v3)":  {
        "csv": RAW_DIR / "geo_news_v3.csv",
        "checkpoint": CP_DIR / "geo_v3_checkpoint.json",
        "keywords": ["war", "sanctions", "invasion", "Russia", "China", "NATO", "Military", "Conflict"]
    }
}

def has_exact_kw(headline, keywords, pattern):
    return bool(pattern.search(headline))

def check_progress_light(label, cfg):
    print(f"--- {label} ---")
    path = cfg["csv"]
    cp_path = cfg["checkpoint"]
    
    # 1. Check Checkpoint
    if cp_path.exists():
        try:
            with open(cp_path, 'r') as f:
                cp = json.load(f)
                print(f"  Last Month Done: {cp.get('last_month', 'N/A')}")
        except:
            print("  Checkpoint: Found but unreadable")
    else:
        print("  Checkpoint: Not found (processing month...)")

    # 2. Check CSV (Line by Line)
    if not path.exists():
        print(f"  [!] CSV File not found: {path.name}\n")
        return

    try:
        total_rows = 0
        valid_hits = 0
        min_date = None
        max_date = None
        
        # Pre-compile pattern for speed
        pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in cfg["keywords"]) + r")\b", re.IGNORECASE)
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline() # Skip header
            for line in f:
                parts = line.split(',')
                if len(parts) < 4: continue
                
                total_rows += 1
                date_str = parts[0].strip()
                headline = parts[3].strip()
                
                # Check keywords
                if pattern.search(headline):
                    valid_hits += 1
                
                # Update date range
                if not min_date or date_str < min_date: min_date = date_str
                if not max_date or date_str > max_date: max_date = date_str

        print(f"  Total Rows     : {total_rows:,}")
        if total_rows > 0:
            print(f"  Date Range     : {min_date} to {max_date}")
            print(f"  Signal Validity: {valid_hits/total_rows:.1%} (Exact word matches)")
        
    except Exception as e:
        print(f"  [!] Error: {e}")
    print()

def main():
    print("=" * 55)
    print(" V3 HIGH-SIGNAL SCRAPER PROGRESS (Memory Optimized)")
    print("=" * 55)
    print()
    
    for label, cfg in FILES.items():
        check_progress_light(label, cfg)
    
    print("=" * 55)

if __name__ == "__main__":
    main()
