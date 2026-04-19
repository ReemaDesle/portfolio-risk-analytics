import os
import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.stattools import adfuller
from datetime import datetime

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def prepare_time_series(scored_csv_path, output_path=None):
    """
    Transforms scored news data into a continuous daily time series.
    """
    if not os.path.exists(scored_csv_path):
        logger.error(f"Scored data not found at {scored_csv_path}")
        return None

    logger.info(f"Loading scored data from {scored_csv_path}...")
    df = pd.read_csv(scored_csv_path)

    # 1. Date Conversion
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Aggregation Logic
    # We aggregate by 'date' and calculate:
    # - mean_sentiment: Average probability (1.0 = positive, -1.0 = negative, 0.0 = neutral)
    # - headline_count: Signal density (high volume = high attention)
    
    # Convert FinBERT labels to scores if they exist
    if 'sentiment_label' in df.columns:
        # Map: Positive=1, Neutral=0, Negative=-1
        label_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
        df['sentiment_score_mapped'] = df['sentiment_label'].map(label_map)
    else:
        logger.warning("No 'sentiment_label' found! Falling back to raw probabilities if available.")
        # Fallback if categories aren't present
        if 'positive' in df.columns and 'negative' in df.columns:
            df['sentiment_score_mapped'] = df['positive'] - df['negative']
        else:
            df['sentiment_score_mapped'] = 0.0

    logger.info("Resampling to daily frequency...")
    daily_stats = df.groupby('date').agg({
        'sentiment_score_mapped': ['mean', 'std', 'count']
    })
    
    # Flatten columns
    daily_stats.columns = ['sentiment_avg', 'sentiment_volatility', 'news_volume']
    
    # 3. Handle Gaps (Weekends & Holidays)
    # We want a continuous range from 2021 to 2026
    all_dates = pd.date_range(start=daily_stats.index.min(), end=daily_stats.index.max(), freq='D')
    daily_full = daily_stats.reindex(all_dates)
    
    # Fill missing news gaps (days with zero news)
    # For sentiment, we use forward fill (mood persists) or fill with 0 (neutral)
    daily_full['news_volume'] = daily_full['news_volume'].fillna(0)
    daily_full['sentiment_avg'] = daily_full['sentiment_avg'].ffill().fillna(0)
    daily_full['sentiment_volatility'] = daily_full['sentiment_volatility'].fillna(0)

    # 4. Stationarity Test (ADF)
    logger.info("Performing Augmented Dickey-Fuller (ADF) Test...")
    res = adfuller(daily_full['sentiment_avg'])
    adf_stat, p_value = res[0], res[1]
    
    logger.info(f"ADF Statistic: {adf_stat:.4f}")
    logger.info(f"P-Value: {p_value:.4f}")
    
    if p_value < 0.05:
        logger.info("Result: Data is STATIONARY (p < 0.05). Ready for ARIMA.")
    else:
        logger.warning("Result: Data is NON-STATIONARY. ARIMA will require differencing (d=1).")

    # 5. Save output
    if output_path:
        daily_full.to_csv(output_path, index_label='date')
        logger.info(f"Daily signal saved to {output_path}")
        
    return daily_full

if __name__ == "__main__":
    import sys
    
    # Default paths within the project structure
    SCORED_FILE = "data/raw/news/financial_news_v3_scored_MOCK.csv"
    OUTPUT_FILE = "data/processed/sentiment_daily_index.csv"
    
    if len(sys.argv) > 1:
        SCORED_FILE = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_FILE = sys.argv[2]
        
    # Ensure processed dir exists
    processed_dir = os.path.dirname(OUTPUT_FILE)
    if processed_dir:
        os.makedirs(processed_dir, exist_ok=True)
    
    prepare_time_series(SCORED_FILE, OUTPUT_FILE)
