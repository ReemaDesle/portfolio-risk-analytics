"""
Portfolio Risk Analytics — Financial Progress Tracker (v3)
==========================================================
Lightweight utility focused exclusively on the 2021-2026 Financial News Harvest.
"""

import os
import csv
from pathlib import Path
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT_DIR / "data" / "raw" / "news"
FIN_FILE = RAW_DIR / "financial_news_v3.csv"

def check_finance_progress():
    print("=" * 60)
    print(" V3 FINANCIAL HARVEST PROGRESS (2021-2026)")
    print("=" * 60)
    
    if not FIN_FILE.exists():
        print(f"  [!] Archive file not found: {FIN_FILE.name}")
        return

    try:
        total_rows = 0
        min_date = None
        max_date = None
        sources = Counter()
        
        # Read line by line to be memory efficient
        with open(FIN_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                dt = row.get('date', '').strip()
                src = row.get('source', 'Unknown').strip()
                
                sources[src] += 1
                
                if dt:
                    if not min_date or dt < min_date: min_date = dt
                    if not max_date or dt > max_date: max_date = dt

        print(f"  Total Rows Collected : {total_rows:,}")
        if total_rows > 0:
            print(f"  Timeline Coverage    : {min_date} to {max_date}")
            print("\n  --- Source Distribution ---")
            for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                print(f"  {src:<15}: {count:,} ({count/total_rows:.1%})")
        
    except Exception as e:
        print(f"  [!] Error parsing file: {e}")
    
    print("-" * 60)
    print(" Status: Active (Hybrid Filler Task Running)")
    print("=" * 60)

if __name__ == "__main__":
    check_finance_progress()
