"""
setup_project.py
----------------
Run this once inside your cloned GitHub repo to create the full
portfolio-risk-analytics project structure with all starter files.

Usage:
    git clone https://github.com/your-org/portfolio-risk-analytics.git
    cd portfolio-risk-analytics
    python setup_project.py
"""

import os

ROOT = "."

FOLDERS = [
    "data/raw",
    "data/processed",
    "scrapers",
    "pipeline",
    "models",
    "notebooks",
    "dashboard",
    "reports",
]

FILES = {

# ── ROOT ─────────────────────────────────────────────────────────────────────

"requirements.txt": """\
yfinance==0.2.40
pandas==2.2.2
requests==2.32.3
beautifulsoup4==4.12.3
feedparser==6.0.11
vaderSentiment==3.3.2
scikit-learn==1.5.0
statsmodels==0.14.2
matplotlib==3.9.0
seaborn==0.13.2
joblib==1.4.2
python-dotenv==1.0.1
""",

".gitignore": """\
# Data files — never commit raw or processed data
data/
models/*.pkl
models/*.joblib

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Environment
.env
.venv/
venv/
env/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Reports (generate locally)
reports/*.pdf
""",

"README.md": """\
# Portfolio Risk Analytics

A data science project analysing how geopolitical and technology news shocks
affect stock portfolio volatility and risk.

## Project structure

```
portfolio-risk-analytics/
├── data/
│   ├── raw/              # raw CSVs (gitignored)
│   └── processed/        # cleaned and merged CSVs (gitignored)
├── scrapers/
│   ├── fetch_prices.py   # yfinance: stocks, gold, VIX
│   ├── scrape_news.py    # Reuters, BBC, Hacker News
│   └── incremental_updater.py
├── pipeline/
│   ├── preprocess.py     # merge + clean → master_data.csv
│   ├── sentiment_score.py# VADER scoring → sentiment_scored.csv
│   └── score_compute.py  # Risk Score, Safety Score, SRI
├── models/
│   └── train_models.py   # train ARIMA + RF, save .pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_experiments.ipynb
├── dashboard/
│   └── app.py            # Streamlit dashboard
└── reports/              # PDF deliverables
```

## Quick start

```bash
pip install -r requirements.txt

# Step 1 — scrape data (do once)
python scrapers/fetch_prices.py
python scrapers/scrape_news.py

# Step 2 — build master dataset
python pipeline/sentiment_score.py
python pipeline/preprocess.py

# Step 3 — compute scores
python pipeline/score_compute.py

# Step 4 — train models
python models/train_models.py

# Step 5 — launch dashboard
streamlit run dashboard/app.py
```

## Team

| Person   | Owns                                      |
|----------|-------------------------------------------|
| Teammate | fetch_prices.py, incremental_updater.py   |
| You      | scrape_news.py, sentiment_score.py        |
| Both     | pipeline/, notebooks/, dashboard/         |
""",

".env.example": """\
# Copy this to .env and fill in your values
# .env is gitignored — never commit real keys

NEWSAPI_KEY=your_newsapi_key_here
""",

# ── SCRAPERS ──────────────────────────────────────────────────────────────────

"scrapers/__init__.py": "",

"scrapers/fetch_prices.py": """\
\"\"\"
fetch_prices.py
Fetches historical OHLCV data from yfinance and computes
daily_return, rolling_vol_20d, and drawdown.
Output: data/raw/raw_prices.csv
\"\"\"

import yfinance as yf
import pandas as pd
import os

DATA_PATH   = "data/raw/raw_prices.csv"
START_DATE  = "2019-01-01"
END_DATE    = "2024-12-31"

TICKERS = {
    "AAPL":  "tech",
    "NVDA":  "tech",
    "MSFT":  "tech",
    "GOOGL": "tech",
    "GC=F":  "gold",
    "SPY":   "market",
    "^VIX":  "volatility",
}

ROLLING_WINDOW = 20


def compute_drawdown(series):
    rolling_max = series.cummax()
    return (series - rolling_max) / rolling_max


def fetch_all():
    frames = []
    for ticker, domain in TICKERS.items():
        print(f"  Fetching {ticker}...")
        raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                          auto_adjust=False, progress=False)
        if raw.empty:
            print(f"  WARNING: no data for {ticker}")
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = ["_".join(c).strip("_").lower() for c in raw.columns]
        else:
            raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

        if "adj_close" not in raw.columns:
            raw["adj_close"] = raw.get("close")

        raw["daily_return"]    = raw["adj_close"].pct_change()
        raw["rolling_vol_20d"] = raw["daily_return"].rolling(ROLLING_WINDOW, min_periods=5).std()
        raw["drawdown"]        = compute_drawdown(raw["adj_close"])

        raw = raw.reset_index()
        raw.rename(columns={"Date": "date"}, inplace=True)
        raw["date"]   = pd.to_datetime(raw["date"]).dt.date
        raw["ticker"] = ticker
        raw["domain"] = domain

        keep = ["date","ticker","domain","open","high","low","close",
                "adj_close","volume","daily_return","rolling_vol_20d","drawdown"]
        keep = [c for c in keep if c in raw.columns]
        frames.append(raw[keep])

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["adj_close"]).sort_values(["date","ticker"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows → {DATA_PATH}")


if __name__ == "__main__":
    fetch_all()
""",

"scrapers/scrape_news.py": """\
\"\"\"
scrape_news.py
Scrapes geopolitical news (Reuters, BBC) and tech news (Hacker News)
and saves to data/raw/raw_news.csv.
Output columns: date | headline | source | domain | url
\"\"\"

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

DATA_PATH = "data/raw/raw_news.csv"
HEADERS   = {"User-Agent": "Mozilla/5.0 (research project)"}


def scrape_hacker_news(pages=5):
    \"\"\"Tech news via Algolia API — no key needed, full history.\"\"\"
    rows = []
    tags = ["AI", "semiconductor", "chip", "tech regulation", "nvidia", "apple"]
    for tag in tags:
        url = (f"https://hn.algolia.com/api/v1/search_by_date"
               f"?tags=story&query={tag}&hitsPerPage=100")
        try:
            r = requests.get(url, timeout=10)
            for hit in r.json().get("hits", []):
                rows.append({
                    "date":     hit.get("created_at","")[:10],
                    "headline": hit.get("title",""),
                    "source":   "hackernews",
                    "domain":   "technology",
                    "url":      hit.get("url",""),
                })
        except Exception as e:
            print(f"  HN error for '{tag}': {e}")
        time.sleep(0.5)
    return rows


def scrape_rss(url, source, domain):
    \"\"\"Generic RSS parser — works for BBC, Reuters, TechCrunch etc.\"\"\"
    try:
        import feedparser
        feed = feedparser.parse(url)
        rows = []
        for entry in feed.entries:
            date = entry.get("published","")[:10] if entry.get("published") else ""
            rows.append({
                "date":     date,
                "headline": entry.get("title",""),
                "source":   source,
                "domain":   domain,
                "url":      entry.get("link",""),
            })
        return rows
    except Exception as e:
        print(f"  RSS error ({source}): {e}")
        return []


def main():
    all_rows = []

    print("Scraping Hacker News (tech)...")
    all_rows += scrape_hacker_news()

    print("Scraping BBC World RSS (geo)...")
    all_rows += scrape_rss(
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        source="bbc", domain="geopolitical"
    )

    print("Scraping Reuters World RSS (geo)...")
    all_rows += scrape_rss(
        "https://feeds.reuters.com/reuters/worldNews",
        source="reuters", domain="geopolitical"
    )

    print("Scraping TechCrunch RSS (tech)...")
    all_rows += scrape_rss(
        "https://techcrunch.com/feed/",
        source="techcrunch", domain="technology"
    )

    df = pd.DataFrame(all_rows)
    df = df[df["headline"].str.strip() != ""]
    df = df.dropna(subset=["date","headline"])
    df = df.drop_duplicates(subset=["headline"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows → {DATA_PATH}")


if __name__ == "__main__":
    main()
""",

"scrapers/incremental_updater.py": """\
\"\"\"
incremental_updater.py
Run daily (via cron) to append only new rows to both raw CSVs.
Cron: 30 18 * * 1-5  cd /path/to/project && python scrapers/incremental_updater.py
\"\"\"

import pandas as pd
import yfinance as yf
import requests
import feedparser
import time
import os
from datetime import date, timedelta

PRICES_PATH = "data/raw/raw_prices.csv"
NEWS_PATH   = "data/raw/raw_news.csv"

TICKERS = {
    "AAPL":"tech","NVDA":"tech","MSFT":"tech","GOOGL":"tech",
    "GC=F":"gold","SPY":"market","^VIX":"volatility",
}

def last_date(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df["date"].max().date()

def update_prices():
    if not os.path.exists(PRICES_PATH):
        print("No prices file — run fetch_prices.py first.")
        return
    from_date = last_date(PRICES_PATH) + timedelta(days=1)
    to_date   = date.today()
    if from_date >= to_date:
        print(f"Prices up to date ({from_date}).")
        return
    print(f"Updating prices: {from_date} → {to_date}")
    frames = []
    for ticker, domain in TICKERS.items():
        raw = yf.download(ticker, start=str(from_date), end=str(to_date),
                          auto_adjust=False, progress=False)
        if raw.empty: continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = ["_".join(c).strip("_").lower() for c in raw.columns]
        else:
            raw.columns = [c.lower().replace(" ","_") for c in raw.columns]
        if "adj_close" not in raw.columns: raw["adj_close"] = raw.get("close")
        raw["daily_return"]    = raw["adj_close"].pct_change()
        raw["rolling_vol_20d"] = raw["daily_return"].rolling(20, min_periods=5).std()
        raw["drawdown"]        = (raw["adj_close"] - raw["adj_close"].cummax()) / raw["adj_close"].cummax()
        raw = raw.reset_index().rename(columns={"Date":"date"})
        raw["date"] = pd.to_datetime(raw["date"]).dt.date
        raw["ticker"] = ticker; raw["domain"] = domain
        keep = ["date","ticker","domain","open","high","low","close",
                "adj_close","volume","daily_return","rolling_vol_20d","drawdown"]
        frames.append(raw[[c for c in keep if c in raw.columns]])
    if not frames: print("No new price rows."); return
    existing = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    combined = pd.concat([existing, pd.concat(frames)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date","ticker"]).sort_values(["date","ticker"])
    combined.to_csv(PRICES_PATH, index=False)
    print(f"Prices updated: {len(combined):,} total rows.")

def update_news():
    if not os.path.exists(NEWS_PATH):
        print("No news file — run scrape_news.py first.")
        return
    new_rows = []
    for url, source, domain in [
        ("http://feeds.bbci.co.uk/news/world/rss.xml",    "bbc",        "geopolitical"),
        ("https://feeds.reuters.com/reuters/worldNews",   "reuters",    "geopolitical"),
        ("https://techcrunch.com/feed/",                  "techcrunch", "technology"),
    ]:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                d = entry.get("published","")[:10]
                new_rows.append({"date":d,"headline":entry.get("title",""),
                                 "source":source,"domain":domain,"url":entry.get("link","")})
        except: pass
        time.sleep(0.3)
    existing  = pd.read_csv(NEWS_PATH)
    new_df    = pd.DataFrame(new_rows)
    new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce").dt.date
    combined  = pd.concat([existing, new_df], ignore_index=True)
    combined  = combined.drop_duplicates(subset=["headline"]).sort_values("date")
    combined.to_csv(NEWS_PATH, index=False)
    print(f"News updated: {len(combined):,} total rows.")

if __name__ == "__main__":
    update_prices()
    update_news()
""",

# ── PIPELINE ──────────────────────────────────────────────────────────────────

"pipeline/__init__.py": "",

"pipeline/sentiment_score.py": """\
\"\"\"
sentiment_score.py
Runs VADER on raw_news.csv headlines.
Output: data/processed/sentiment_scored.csv
Columns: date | geo_sentiment | tech_sentiment | geo_count | tech_count
\"\"\"

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

NEWS_PATH = "data/raw/raw_news.csv"
OUT_PATH  = "data/processed/sentiment_scored.csv"


def main():
    df  = pd.read_csv(NEWS_PATH, parse_dates=["date"])
    sia = SentimentIntensityAnalyzer()

    print("Scoring headlines with VADER...")
    df["compound"] = df["headline"].apply(
        lambda h: sia.polarity_scores(str(h))["compound"]
    )

    geo  = df[df["domain"]=="geopolitical"].groupby("date").agg(
        geo_sentiment=("compound","mean"), geo_count=("compound","count")).reset_index()
    tech = df[df["domain"]=="technology"].groupby("date").agg(
        tech_sentiment=("compound","mean"), tech_count=("compound","count")).reset_index()

    scored = pd.merge(geo, tech, on="date", how="outer").sort_values("date")
    scored["date"] = pd.to_datetime(scored["date"]).dt.date

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    scored.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(scored):,} rows → {OUT_PATH}")


if __name__ == "__main__":
    main()
""",

"pipeline/preprocess.py": """\
\"\"\"
preprocess.py
Merges raw_prices.csv and sentiment_scored.csv on trading calendar.
Output: data/processed/master_data.csv
\"\"\"

import pandas as pd
import os

PRICES_PATH    = "data/raw/raw_prices.csv"
SENTIMENT_PATH = "data/processed/sentiment_scored.csv"
OUT_PATH       = "data/processed/master_data.csv"


def main():
    prices    = pd.read_csv(PRICES_PATH,    parse_dates=["date"])
    sentiment = pd.read_csv(SENTIMENT_PATH, parse_dates=["date"])

    trading_dates = prices["date"].unique()
    sentiment = (sentiment
                 .set_index("date")
                 .reindex(trading_dates, method="ffill")
                 .reset_index()
                 .rename(columns={"index":"date"}))

    master = pd.merge(prices, sentiment, on="date", how="left")
    master = master.sort_values(["date","ticker"]).reset_index(drop=True)

    for col in ["geo_sentiment","tech_sentiment"]:
        master[col] = master[col].fillna(0.0)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    master.to_csv(OUT_PATH, index=False)
    print(f"master_data.csv: {len(master):,} rows, {len(master.columns)} columns")
    print(f"Columns: {list(master.columns)}")


if __name__ == "__main__":
    main()
""",

"pipeline/score_compute.py": """\
\"\"\"
score_compute.py
Computes Risk Score, Safety Score, and Shock Recovery Index (SRI).
Weights are placeholders — replace with EDA-derived values from notebook 02.
Output: data/processed/scores.csv
\"\"\"

import pandas as pd
import numpy as np
import os

MASTER_PATH = "data/processed/master_data.csv"
OUT_PATH    = "data/processed/scores.csv"

# Replace these weights after running 02_feature_analysis.ipynb
W_VOL        = 0.40
W_DRAWDOWN   = 0.35
W_GEO_SHOCK  = 0.25


def risk_score(row):
    vol      = abs(row.get("rolling_vol_20d", 0) or 0)
    dd       = abs(row.get("drawdown", 0) or 0)
    geo      = abs(row.get("geo_sentiment", 0) or 0)
    raw      = W_VOL * vol + W_DRAWDOWN * dd + W_GEO_SHOCK * geo
    return round(min(raw * 100, 100), 2)


def safety_score(row):
    # Higher when volatility is low, sentiment is positive, drawdown is small
    vol_inv  = 1 - min(abs(row.get("rolling_vol_20d", 0) or 0) * 20, 1)
    sent_pos = (row.get("geo_sentiment", 0) or 0 + 1) / 2
    dd_inv   = 1 - min(abs(row.get("drawdown", 0) or 0), 1)
    raw      = (vol_inv + sent_pos + dd_inv) / 3
    return round(raw * 100, 2)


def main():
    df = pd.read_csv(MASTER_PATH, parse_dates=["date"])
    df["risk_score"]   = df.apply(risk_score, axis=1)
    df["safety_score"] = df.apply(safety_score, axis=1)

    out = df[["date","ticker","domain","risk_score","safety_score",
              "daily_return","rolling_vol_20d","drawdown",
              "geo_sentiment","tech_sentiment"]].copy()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out):,} rows → {OUT_PATH}")


if __name__ == "__main__":
    main()
""",

# ── MODELS ────────────────────────────────────────────────────────────────────

"models/train_models.py": """\
\"\"\"
train_models.py
Trains ARIMA (volatility forecasting) and Random Forest (regime classification).
Saves models to models/arima.pkl and models/regime_classifier.pkl
Change MODE to 'retrain' weekly.
\"\"\"

import pandas as pd
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MASTER_PATH = "data/processed/master_data.csv"
ARIMA_PATH  = "models/arima.pkl"
RF_PATH     = "models/regime_classifier.pkl"


def train_arima(df, ticker="SPY"):
    series = (df[df["ticker"]==ticker]
              .sort_values("date")["rolling_vol_20d"]
              .dropna())
    print(f"Training ARIMA on {ticker} volatility ({len(series)} points)...")
    model  = ARIMA(series, order=(1,1,1)).fit()
    joblib.dump(model, ARIMA_PATH)
    print(f"ARIMA saved → {ARIMA_PATH}")
    print(f"AIC: {model.aic:.2f}")


def train_regime_classifier(df):
    # Label regime: high geo shock = 1, high tech sentiment = 2, else = 0
    df = df.dropna(subset=["geo_sentiment","tech_sentiment","rolling_vol_20d"])
    df["regime"] = 0
    df.loc[df["geo_sentiment"] < -0.2, "regime"] = 1   # geopolitical stress
    df.loc[df["tech_sentiment"] > 0.3,  "regime"] = 2  # tech optimism

    features = ["rolling_vol_20d","drawdown","geo_sentiment","tech_sentiment"]
    X = df[features].fillna(0)
    y = df["regime"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print("\\nRegime classifier report:")
    print(classification_report(y_test, clf.predict(X_test),
          target_names=["Neutral","Geo stress","Tech optimism"]))

    joblib.dump(clf, RF_PATH)
    print(f"Classifier saved → {RF_PATH}")


def main():
    df = pd.read_csv(MASTER_PATH, parse_dates=["date"])
    os.makedirs("models", exist_ok=True)
    train_arima(df)
    train_regime_classifier(df)


if __name__ == "__main__":
    main()
""",

# ── DASHBOARD ─────────────────────────────────────────────────────────────────

"dashboard/app.py": """\
\"\"\"
app.py — Streamlit dashboard
Run: streamlit run dashboard/app.py
\"\"\"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Portfolio Risk Analytics", layout="wide")
st.title("Portfolio Risk Analytics Dashboard")

MASTER = "data/processed/master_data.csv"
SCORES = "data/processed/scores.csv"

if not os.path.exists(MASTER):
    st.warning("Run the pipeline first: preprocess.py → score_compute.py")
    st.stop()

master = pd.read_csv(MASTER, parse_dates=["date"])
scores = pd.read_csv(SCORES, parse_dates=["date"])

tickers = scores["ticker"].unique().tolist()
selected = st.sidebar.multiselect("Select tickers", tickers, default=tickers[:2])

df = scores[scores["ticker"].isin(selected)]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Rolling volatility over time")
    fig, ax = plt.subplots(figsize=(8,3))
    for t in selected:
        sub = df[df["ticker"]==t].sort_values("date")
        ax.plot(sub["date"], sub["rolling_vol_20d"], label=t)
    ax.legend(); ax.set_ylabel("20d rolling vol")
    st.pyplot(fig)

with col2:
    st.subheader("Risk Score over time")
    fig, ax = plt.subplots(figsize=(8,3))
    for t in selected:
        sub = df[df["ticker"]==t].sort_values("date")
        ax.plot(sub["date"], sub["risk_score"], label=t)
    ax.legend(); ax.set_ylabel("Risk Score (0-100)")
    st.pyplot(fig)

st.subheader("Geopolitical sentiment vs gold volatility")
gold = df[df["ticker"]=="GC=F"].sort_values("date")
if not gold.empty:
    fig, ax1 = plt.subplots(figsize=(12,3))
    ax2 = ax1.twinx()
    ax1.fill_between(gold["date"], gold["geo_sentiment"], alpha=0.3, color="coral", label="Geo sentiment")
    ax2.plot(gold["date"], gold["rolling_vol_20d"], color="gold", label="Gold vol")
    ax1.set_ylabel("Sentiment"); ax2.set_ylabel("Volatility")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    st.pyplot(fig)
""",

# ── NOTEBOOKS (stubs) ─────────────────────────────────────────────────────────

"notebooks/.gitkeep": "",
"reports/.gitkeep":   "",

}


def create_structure():
    print("Creating project structure...\n")

    for folder in FOLDERS:
        path = os.path.join(ROOT, folder)
        os.makedirs(path, exist_ok=True)
        print(f"  Created  {path}/")

    print()
    for filepath, content in FILES.items():
        full = os.path.join(ROOT, filepath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Written  {filepath}")

    print("\nDone. Next steps:")
    print("  1. pip install -r requirements.txt")
    print("  2. python scrapers/fetch_prices.py")
    print("  3. python scrapers/scrape_news.py")
    print("  4. python pipeline/sentiment_score.py")
    print("  5. python pipeline/preprocess.py")
    print("  6. python pipeline/score_compute.py")
    print("  7. python models/train_models.py")
    print("  8. streamlit run dashboard/app.py")


if __name__ == "__main__":
    create_structure()