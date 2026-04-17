import requests
from datetime import datetime

def debug_geo():
    start_ts = int(datetime(2021, 1, 1).timestamp())
    end_ts = int(datetime(2021, 1, 31).timestamp())
    kw = "sanctions"
    
    url = "https://hn.algolia.com/api/v1/search_by_date"
    params = {
        "query": kw,
        "tags": "story",
        "numericFilters": f"created_at_i>={start_ts},created_at_i<={end_ts}",
        "hitsPerPage": 10
    }
    
    r = requests.get(url, params=params)
    data = r.json()
    hits = data.get("hits", [])
    
    print(f"Total hits found for '{kw}' in Jan 2021: {len(hits)}")
    for h in hits:
        print(f"  - {h.get('title')}")

if __name__ == "__main__":
    debug_geo()
