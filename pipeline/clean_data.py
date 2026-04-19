"""
Portfolio Risk Analytics — Data Cleaning & Feature Engineering
==============================================================
Transform stage (ETL): loads all raw/scored data, applies cleaning,
and outputs clean feature tables ready for risk score computation.

Cleaning steps applied
-----------------------
  1.  Load & combine scored news  (financial, geo, tech)
  2.  Fix domain labels            (financial_news → domain='financial')
  3.  Deduplicate headlines
  4.  Roll weekend news → next Monday  (markets closed Sat/Sun)
  5.  Aggregate daily sentiment per domain (weighted by article count)
  6.  Clip sentiment outliers at ±3 std  (per domain)
  7.  Load stock prices (yfinance), forward-fill gaps
  8.  Compute daily log returns
  9.  Compute 5-day rolling volatility
  10. Add 1-day lagged sentiment features  (for ML)
  11. Forward-fill single-day holiday gaps in sentiment

Inputs
------
  data/raw/news/financial_news_scored.csv
  data/raw/news/geo_news_scored.csv
  data/raw/news/tech_news_scored.csv
  data/raw/tickers/prices_daily.csv

Outputs  →  data/processed/
-------
  master_news.csv   — all scored articles combined & cleaned
  master_data.csv   — trading-day rows: sentiment + prices + returns + vol + lags

Usage
-----
  python pipeline/clean_data.py
  python pipeline/clean_data.py --start 2026-03-20 --end 2026-04-15
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
RAW_NEWS_DIR  = ROOT_DIR / "data" / "raw" / "news"
RAW_DIR       = ROOT_DIR / "data" / "raw" / "tickers"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

FINANCIAL_SCORED = RAW_NEWS_DIR / "financial_news_v3_scored.csv"
GEO_SCORED       = RAW_NEWS_DIR / "geo_news_scored_v3.csv"
TECH_SCORED      = RAW_NEWS_DIR / "tech_news_scored_v3.csv"
PRICES_CSV       = RAW_DIR / "prices_daily.csv"

# Standard columns expected from every scored CSV
STANDARD_COLS = ["date", "source", "domain", "headline", "url",
                 "tone_label", "tone_score", "prob_pos", "prob_neg", "prob_neu"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 1 & 2 — Load + fix domain labels
# ══════════════════════════════════════════════

def load_scored_news() -> pd.DataFrame:
    """
    Load all three scored CSVs and combine into one master DataFrame.
    Forces correct domain labels and standardises column names.
    """
    log.info("── [1/6] Loading scored news files ─────────────────")

    sources = [
        (FINANCIAL_SCORED, "financial"),   
        (GEO_SCORED,       "geopolitical"),
        (TECH_SCORED,      "technology"),
    ]

    frames = []
    for path, forced_domain in sources:
        if not path.exists():
            log.warning("  MISSING: %s — skipping.", path.name)
            continue

        # Load with mangle_dupe_cols to handle duplicate 'tone_score'
        df = pd.read_csv(path)
        log.info("  Loaded %-35s  %d rows", path.name, len(df))

        # ── Handle column differences ───────────────────────────────────
        # Financial v3 uses 'sentiment_score' / 'sentiment_label'
        if "sentiment_score" in df.columns and "tone_score" not in df.columns:
            df = df.rename(columns={"sentiment_score": "tone_score", "sentiment_label": "tone_label"})

        # Geo/Tech v3 may have duplicate 'tone_score' (Pandas might rename to 'tone_score.1')
        # If both exist, use the one with data
        if "tone_score.1" in df.columns:
            df["tone_score"] = df["tone_score"].fillna(df["tone_score.1"])

        if forced_domain:
            df["domain"] = forced_domain

        # finance script adds clean_headline — use it as the canonical headline
        if "clean_headline" in df.columns:
            df["headline"] = df["clean_headline"].fillna(df["headline"])

        # ── Normalise column names across v3 schemas ──────────────────────
        # financial_v3 : 'sentiment_score' / 'sentiment_label'  (no prob_*)
        # geo/tech_v3  : 'tone_score' (NaN placeholder) + 'tone_score.1'
        #                 real values in tone_score.1; prob_pos/neg/neu present

        # 1. Rename sentiment_score → tone_score (financial v3)
        if "sentiment_score" in df.columns and "tone_score" not in df.columns:
            df.rename(columns={"sentiment_score": "tone_score"}, inplace=True)
        if "sentiment_label" in df.columns and "tone_label" not in df.columns:
            df.rename(columns={"sentiment_label": "tone_label"}, inplace=True)

        # 2. geo/tech_v3: real scores live in 'tone_score.1'; drop the NaN column
        if "tone_score.1" in df.columns:
            df["tone_score"] = df["tone_score.1"].fillna(df.get("tone_score"))
            df.drop(columns=["tone_score.1"], inplace=True)

        # 3. Derive prob_* for files that don't have them (financial v3)
        if "prob_pos" not in df.columns:
            if "tone_label" in df.columns and "tone_score" in df.columns:
                score = df["tone_score"].fillna(0.0)
                lbl   = df["tone_label"].str.lower().fillna("neutral")
                df["prob_pos"] = np.where(lbl == "positive", score,
                                 np.where(lbl == "negative", 1 - score, 0.33))
                df["prob_neg"] = np.where(lbl == "negative", score,
                                 np.where(lbl == "positive", 1 - score, 0.33))
                df["prob_neu"] = (1.0 - df["prob_pos"] - df["prob_neg"]).clip(0.0, 1.0)
            else:
                df["prob_pos"] = 0.33
                df["prob_neg"] = 0.33
                df["prob_neu"] = 0.34
        # ─────────────────────────────────────────────────────────────────

        for col in STANDARD_COLS:
            if col not in df.columns:
                df[col] = 0 if col.startswith("prob_") else None

        frames.append(df[STANDARD_COLS])

    if not frames:
        raise FileNotFoundError("No scored news files found!")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    
    # Ensure tone_score is numeric
    combined["tone_score"] = pd.to_numeric(combined["tone_score"], errors="coerce")
    
    combined = combined.dropna(subset=["date", "tone_score"])

    # ── 3. Deduplicate
    before = len(combined)
    combined = combined.drop_duplicates(subset=["headline", "date"]).reset_index(drop=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    log.info("  Dropped %d duplicates → %d unique articles",
             before - len(combined), len(combined))
    log.info("  Domain counts:\n%s", combined["domain"].value_counts().to_string())
    return combined


# ══════════════════════════════════════════════
# 4 — Roll weekend articles to next Monday
# ══════════════════════════════════════════════

def roll_weekend_to_monday(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Markets are closed Saturday & Sunday.
    Shift all weekend articles to the next trading day (Monday)
    so their sentiment signal arrives on the first day the market
    can react to it.

        Saturday (+2 days) → Monday
        Sunday   (+1 day)  → Monday
        Weekday            → unchanged
    """
    log.info("── [2/6] Rolling weekend news to Monday ─────────────")
    df = news_df.copy()

    sat_mask = df["date"].dt.dayofweek == 5
    sun_mask = df["date"].dt.dayofweek == 6

    df.loc[sat_mask, "date"] += pd.Timedelta(days=2)
    df.loc[sun_mask, "date"] += pd.Timedelta(days=1)

    log.info("  Rolled %d Saturday + %d Sunday articles → Monday",
             sat_mask.sum(), sun_mask.sum())
    return df


# ══════════════════════════════════════════════
# 5 & 6 — Aggregate + clip outliers
# ══════════════════════════════════════════════

def compute_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse all articles per (date, domain) to a single row.
    Applies outlier clipping at ±3 std per domain.

    Output columns per domain:
      avg_prob_pos, avg_prob_neg, avg_prob_neu,
      article_count, sentiment_score  (= prob_pos − prob_neg)
    """
    log.info("── [3/6] Aggregating daily sentiment ────────────────")

    agg = news_df.groupby(["date", "domain"]).agg(
        avg_prob_pos  = ("prob_pos",  "mean"),
        avg_prob_neg  = ("prob_neg",  "mean"),
        avg_prob_neu  = ("prob_neu",  "mean"),
        article_count = ("headline",  "count"),
    ).reset_index()

    agg["sentiment_score"] = (agg["avg_prob_pos"] - agg["avg_prob_neg"]).round(6)

    # ── Clip ±3 std outliers per domain
    for domain in agg["domain"].unique():
        mask = agg["domain"] == domain
        col  = agg.loc[mask, "sentiment_score"]
        mu, sigma = col.mean(), col.std()
        if sigma > 0:
            lo, hi = mu - 3 * sigma, mu + 3 * sigma
            n_clipped = ((col < lo) | (col > hi)).sum()
            agg.loc[mask, "sentiment_score"] = col.clip(lo, hi)
            if n_clipped:
                log.info("  Clipped %d outlier(s) in domain '%s'  (±3σ window [%.4f, %.4f])",
                         n_clipped, domain, lo, hi)

    agg = agg.sort_values(["date", "domain"]).reset_index(drop=True)
    log.info("  Sentiment rows: %d  |  Domains: %s",
             len(agg), list(agg["domain"].unique()))
    return agg


# ══════════════════════════════════════════════
# 7, 8, 9 — Prices, returns, volatility
# ══════════════════════════════════════════════

def load_prices_with_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load prices_daily.csv then compute:
      - Log returns      (ret_<TICKER>)
      - 5-day rolling σ  (vol5_<TICKER>)

    Returns: (prices, returns, volatility)  — all DatetimeIndexed.
    """
    log.info("── [4/6] Loading prices & computing returns/vol ─────")

    if not PRICES_CSV.exists():
        raise FileNotFoundError(
            f"Missing: {PRICES_CSV}\nRun: python scrapers/fetch_prices.py"
        )

    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index).normalize()
    prices.index.name = "date"
    prices = prices.sort_index().ffill()

    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    returns.columns = [f"ret_{c}" for c in returns.columns]

    vol = returns.rolling(5, min_periods=2).std()
    vol.columns = [f"vol5_{c.replace('ret_', '')}" for c in vol.columns]

    log.info("  %d trading days  |  %d tickers", len(prices), len(prices.columns))
    log.info("  Log returns + 5-day vol computed.")
    return prices, returns, vol


# ══════════════════════════════════════════════
# 10 & 11 — Build master_data with lags + ffill
# ══════════════════════════════════════════════

def build_master_data(daily_sentiment: pd.DataFrame,
                      prices: pd.DataFrame,
                      returns: pd.DataFrame,
                      vol: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot daily sentiment wide, add 1-day lag features,
    join with prices + returns + vol on trading-day index.
    Forward-fill single-day holiday gaps in sentiment (limit=1).
    """
    log.info("── [5/6] Building master_data (trading-day aligned) ─")

    # Pivot sentiment wide
    pivot = daily_sentiment.pivot_table(
        index="date",
        columns="domain",
        values=["sentiment_score", "avg_prob_neg", "avg_prob_pos",
                "avg_prob_neu", "article_count"],
        aggfunc="first",
    )
    pivot.columns = ["_".join(col).strip() for col in pivot.columns]
    pivot.index   = pd.to_datetime(pivot.index)
    pivot.index.name = "date"

    # ── 10. Add 1-day lagged sentiment
    lag_cols = [c for c in pivot.columns if c.startswith("sentiment_score_")]
    for col in lag_cols:
        pivot[f"lag1_{col}"] = pivot[col].shift(1)
    log.info("  Added %d lag-1 sentiment feature(s).", len(lag_cols))

    # ── Join on trading-day index from prices
    prices.index  = pd.to_datetime(prices.index)
    returns.index = pd.to_datetime(returns.index)
    vol.index     = pd.to_datetime(vol.index)

    master = prices.join(returns, how="left") \
                   .join(vol,     how="left") \
                   .join(pivot,   how="left")

    # ── 11. Forward-fill single-day holiday gaps in sentiment
    sentiment_cols = [c for c in master.columns if any(
        c.startswith(p) for p in
        ["sentiment_score_", "avg_prob_", "article_count_", "lag1_"]
    )]
    filled = master[sentiment_cols].ffill(limit=1)
    n_filled = filled.notna().sum().sum() - master[sentiment_cols].notna().sum().sum()
    master[sentiment_cols] = filled
    if n_filled > 0:
        log.info("  Forward-filled %d holiday-gap cell(s) in sentiment columns.", n_filled)

    master = master.sort_index()
    log.info("  master_data: %d rows × %d cols", *master.shape)
    return master


# ══════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════

def save_outputs(master_news: pd.DataFrame, master: pd.DataFrame):
    log.info("── [6/6] Saving cleaned outputs ─────────────────────")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    news_out = master_news.copy()
    news_out["date"] = news_out["date"].dt.strftime("%Y-%m-%d")
    news_out.to_csv(PROCESSED_DIR / "master_news_v3.csv", index=False)
    log.info("  ✔  master_news_v3.csv  (%d rows × %d cols)",
             len(news_out), len(news_out.columns))

    master_out = master.reset_index()
    master_out.to_csv(PROCESSED_DIR / "master_data_v3.csv", index=False)
    log.info("  ✔  master_data_v3.csv  (%d rows × %d cols)",
             len(master_out), len(master_out.columns))


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def run(start_date: str = None, end_date: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("═" * 60)
    log.info("  CLEAN DATA  |  Portfolio Risk Analytics")
    log.info("═" * 60)

    master_news = load_scored_news()
    master_news = roll_weekend_to_monday(master_news)

    if start_date:
        master_news = master_news[master_news["date"] >= pd.to_datetime(start_date)]
    if end_date:
        master_news = master_news[master_news["date"] <= pd.to_datetime(end_date)]

    log.info("  Articles in window: %d  (%s → %s)",
             len(master_news),
             master_news["date"].min().strftime("%Y-%m-%d"),
             master_news["date"].max().strftime("%Y-%m-%d"))

    daily_sentiment             = compute_daily_sentiment(master_news)
    prices, returns, vol        = load_prices_with_features()
    master                      = build_master_data(daily_sentiment, prices, returns, vol)

    save_outputs(master_news, master)

    log.info("")
    log.info("═" * 60)
    log.info("  COMPLETE  — master_data ready for score_compute.py")
    log.info("  Articles   : %d", len(master_news))
    log.info("  Trading days : %d", len(master))
    log.info("  Columns    : %d  (prices + returns + vol + sentiment + lags)", len(master.columns))
    log.info("  Outputs    → %s", PROCESSED_DIR.resolve())
    log.info("═" * 60)

    return master_news, master


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean & engineer features from scored news + yfinance prices."
    )
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=None, help="Filter end date YYYY-MM-DD")
    args = parser.parse_args()
    run(start_date=args.start, end_date=args.end)
