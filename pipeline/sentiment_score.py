"""

Processing steps
----------------
  1. Load raw_news_combined.csv
  2. Clean headline text (strip HTML, URLs)
  3. Batch-score with FinBERT (GPU if available, else CPU)
  4. Aggregate per domain per trading day
  5. Forward-fill weekends onto Monday
  6. 7-day rolling mean, std, shock flags
  7. Save data/processed/sentiment_scored.csv

Dependencies
------------
  pip install transformers torch pandas numpy
"""

import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# -----------------------------------------------
# 1  DATE RANGE  <- change these two variables
# -----------------------------------------------
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE   = "2024-12-31"

# -----------------------------------------------
# 2  FinBERT settings
# -----------------------------------------------
FINBERT_MODEL = "ProsusAI/finbert"   # downloads ~440 MB on first run
BATCH_SIZE    = 32                   # lower to 8 if CPU RAM is tight
MAX_TOKEN_LEN = 512

# -----------------------------------------------
# 3  Paths
# -----------------------------------------------
ROOT_DIR      = Path(__file__).resolve().parent.parent   # pipeline/ → project root
RAW_NEWS_PATH = ROOT_DIR / "data" / "raw" / "raw_news_combined.csv"
PRICES_PATH   = ROOT_DIR / "data" / "raw" / "prices_daily.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_PATH   = PROCESSED_DIR / "sentiment_scored.csv"

# ===============================================
# Text cleaner
# ===============================================

_URL_RE   = re.compile(r"https?://\S+")
_HTML_RE  = re.compile(r"<[^>]+>")
_JUNK_RE  = re.compile(r"[^a-zA-Z0-9\s.,!?\'\"%-]")
_MULTI_RE = re.compile(r"\s{2,}")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _JUNK_RE.sub(" ", text)
    text = _MULTI_RE.sub(" ", text)
    return text.strip()[:400]   # 400 chars ~ 80 tokens, well within FinBERT 512 limit


# ===============================================
# FinBERT loader
# ===============================================

def load_finbert():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        raise ImportError("Run:  pip install transformers torch")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("  Loading FinBERT on %s  (first run downloads ~440 MB)", device.upper())

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model     = model.to(device)
    model.eval()

    # ProsusAI/finbert label order (from model card): 0=positive, 1=negative, 2=neutral
    label_order = ["positive", "negative", "neutral"]
    return tokenizer, model, device, label_order


# ===============================================
# Batch scoring
# ===============================================

def score_with_finbert(texts, tokenizer, model, device, label_order):
    """
    Returns DataFrame with columns:
      finbert_compound  in [-1, +1]  = P(pos) - P(neg)
      finbert_positive  in [0,  1]
      finbert_negative  in [0,  1]
      finbert_neutral   in [0,  1]
      finbert_label     str
    """
    import torch
    import torch.nn.functional as F

    results = []
    n = len(texts)

    for i in range(0, n, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc   = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_TOKEN_LEN, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            probs = F.softmax(model(**enc).logits, dim=-1).cpu().numpy()

        for row in probs:
            p_pos = float(row[label_order.index("positive")])
            p_neg = float(row[label_order.index("negative")])
            p_neu = float(row[label_order.index("neutral")])
            results.append({
                "finbert_compound": round(p_pos - p_neg, 6),
                "finbert_positive": round(p_pos, 6),
                "finbert_negative": round(p_neg, 6),
                "finbert_neutral":  round(p_neu, 6),
                "finbert_label":    label_order[int(row.argmax())],
            })

        done = min(i + BATCH_SIZE, n)
        if done % 500 < BATCH_SIZE or done == n:
            log.info("    Scored %d / %d articles", done, n)

    return pd.DataFrame(results)


# ===============================================
# Trading-day alignment
# ===============================================

def get_trading_days(start, end, prices_path=None):
    if prices_path and prices_path.exists():
        idx = pd.read_csv(prices_path, index_col=0, parse_dates=True).index.normalize()
        idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
        log.info("  Trading days from prices CSV: %d", len(idx))
        return pd.DatetimeIndex(idx)
    return pd.bdate_range(start=start, end=end)


def align_to_trading_days(daily_df, trading_days):
    """Forward-fill weekends/holidays onto next trading day."""
    daily_df.index = pd.to_datetime(daily_df.index)
    full_idx = pd.date_range(
        start=min(daily_df.index.min(), trading_days.min()),
        end=max(daily_df.index.max(),   trading_days.max()),
        freq="D",
    )
    return daily_df.reindex(full_idx).ffill().reindex(trading_days)


# ===============================================
# Daily aggregation per domain
# ===============================================

def aggregate_daily(df, domain):
    sub = df[df["domain"] == domain].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.set_index("date")
    if sub.empty:
        return pd.DataFrame()

    prefix = domain[:4]   # "geo_" or "tech"
    grp = sub.groupby(sub.index).agg(
        sentiment_mean = ("finbert_compound", "mean"),
        sentiment_std  = ("finbert_compound", "std"),
        article_count  = ("finbert_compound", "count"),
        avg_positive   = ("finbert_positive", "mean"),
        avg_negative   = ("finbert_negative", "mean"),
        avg_neutral    = ("finbert_neutral",  "mean"),
        dominant_label = ("finbert_label",
                          lambda x: x.mode()[0] if not x.empty else "neutral"),
    )
    grp.index.name = "date"
    return grp.add_prefix(f"{prefix}_")


# ===============================================
# Rolling features + shock flags
# ===============================================

def add_rolling_features(df, window=7):
    """
    For each domain:
      - {prefix}_roll7          7-day rolling mean of sentiment
      - {prefix}_roll7_std      7-day rolling std
      - {prefix}_shock_flag     1 when |sentiment - roll_mean| > 2 * roll_std
      - {prefix}_sentiment_delta  day-over-day change
    """
    for col in [c for c in df.columns if c.endswith("_sentiment_mean")]:
        prefix = col.replace("_sentiment_mean", "")
        df[f"{prefix}_roll{window}"]       = df[col].rolling(window, min_periods=1).mean()
        df[f"{prefix}_roll{window}_std"]   = df[col].rolling(window, min_periods=1).std()
        roll_mean = df[col].rolling(window, min_periods=3).mean()
        roll_std  = df[col].rolling(window, min_periods=3).std()
        df[f"{prefix}_shock_flag"]         = ((df[col] - roll_mean).abs() > 2 * roll_std).astype(int)
        df[f"{prefix}_sentiment_delta"]    = df[col].diff()
        df[f"{prefix}_direction"]          = np.sign(df[col]).astype(int)

    geo_col  = next((c for c in df.columns if "geo"  in c and "sentiment_mean" in c), None)
    tech_col = next((c for c in df.columns if "tech" in c and "sentiment_mean" in c), None)
    if geo_col and tech_col:
        df["combined_sentiment"] = (df[geo_col] + df[tech_col]) / 2
    elif geo_col:
        df["combined_sentiment"] = df[geo_col]
    elif tech_col:
        df["combined_sentiment"] = df[tech_col]
    return df


# ===============================================
# Main pipeline
# ===============================================

def run(
    start_date  = DEFAULT_START_DATE,
    end_date    = DEFAULT_END_DATE,
    news_path   = RAW_NEWS_PATH,
    prices_path = PRICES_PATH,
    limit       = None,
):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not news_path.exists():
        raise FileNotFoundError(f"Run scrape_news.py first. Missing: {news_path}")

    log.info("Loading raw news …")
    news = pd.read_csv(news_path, low_memory=False)
    news["date"] = pd.to_datetime(news["date"], errors="coerce")
    news = news.dropna(subset=["date"])
    news = news[(news["date"] >= pd.Timestamp(start_date)) &
                (news["date"] <= pd.Timestamp(end_date))]
                
    if limit:
        news = news.head(limit)
        
    log.info("  Articles in date range: %d", len(news))

    if news.empty:
        return pd.DataFrame()

    # Clean
    news["headline_clean"] = news["headline"].apply(clean_text)
    news = news[news["headline_clean"].str.strip() != ""].reset_index(drop=True)

    # Score
    tokenizer, model, device, label_order = load_finbert()
    scores = score_with_finbert(news["headline_clean"].tolist(), tokenizer, model, device, label_order)
    news   = pd.concat([news, scores], axis=1)

    # Save full scored file for EDA
    news.to_csv(PROCESSED_DIR / "news_finbert_raw.csv", index=False)
    log.info("  Label dist:\n%s", news["finbert_label"].value_counts().to_string())

    # Normalise domain labels
    news["domain"] = news["domain"].str.lower().str.strip()
    news.loc[news["domain"].str.contains("geo|polit|war|conflict", na=False), "domain"] = "geopolitical"
    news.loc[news["domain"].str.contains("tech|software|ai|semi",  na=False), "domain"] = "technology"

    # Daily aggregation
    geo_daily  = aggregate_daily(news, "geopolitical")
    tech_daily = aggregate_daily(news, "technology")

    # Align to trading days
    trading_days = get_trading_days(start_date, end_date, prices_path)
    geo_aln  = align_to_trading_days(geo_daily,  trading_days) if not geo_daily.empty  else pd.DataFrame(index=trading_days)
    tech_aln = align_to_trading_days(tech_daily, trading_days) if not tech_daily.empty else pd.DataFrame(index=trading_days)

    sentiment = geo_aln.join(tech_aln, how="outer")
    sentiment.index.name = "date"
    sentiment = add_rolling_features(sentiment, window=7)
    sentiment = sentiment.ffill().fillna(0)
    sentiment.index = sentiment.index.strftime("%Y-%m-%d")
    sentiment.to_csv(OUTPUT_PATH)

    log.info("Saved %d rows, %d cols → %s", len(sentiment), len(sentiment.columns), OUTPUT_PATH)
    return sentiment


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start",      default=DEFAULT_START_DATE)
    p.add_argument("--end",        default=DEFAULT_END_DATE)
    p.add_argument("--news",       default=str(RAW_NEWS_PATH))
    p.add_argument("--prices",     default=str(PRICES_PATH))
    p.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    p.add_argument("--limit",      default=None, type=int, help="Limit number of articles scored for quick testing")
    args = p.parse_args()
    
    # Update BATCH_SIZE if provided (must update global before run)
    BATCH_SIZE = args.batch_size
    
    run(start_date=args.start, end_date=args.end,
        news_path=Path(args.news), prices_path=Path(args.prices), limit=args.limit)