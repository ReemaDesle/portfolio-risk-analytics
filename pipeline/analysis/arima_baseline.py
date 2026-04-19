import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_arimax_forecast(sentiment_path=None, prices_path=None, target_ticker='SPY', forecast_steps=10, lag=1, use_zscore=True):
    """
    Fits an ARIMAX model using Sentiment as an exogenous variable.
    Returns a dictionary of results for API consumption.
    """
    if sentiment_path is None:
        sentiment_path = os.path.join(os.path.dirname(__file__), "../../data/processed/sentiment_daily_index.csv")
    if prices_path is None:
        prices_path = os.path.join(os.path.dirname(__file__), "../../data/raw/tickers/prices_daily.csv")

    try:
        logger.info(f"Loading data: Ticker={target_ticker}, Lag={lag}, Z-Score={use_zscore}")
        
        # 1. Load Datasets
        sentiment_df = pd.read_csv(sentiment_path, index_col='date', parse_dates=True)
        prices_df = pd.read_csv(prices_path, index_col='date', parse_dates=True)
        
        if target_ticker not in prices_df.columns:
            return {"error": f"Ticker {target_ticker} not found in prices data"}

        # Apply Lag to sentiment if specified
        if lag > 0:
            sentiment_df['sentiment_avg'] = sentiment_df['sentiment_avg'].shift(lag)
            sentiment_df = sentiment_df.dropna()

        # Apply Z-Score normalization if specified
        if use_zscore:
            s_mean = sentiment_df['sentiment_avg'].mean()
            s_std = sentiment_df['sentiment_avg'].std()
            sentiment_df['sentiment_avg'] = (sentiment_df['sentiment_avg'] - s_mean) / s_std

        # 2. Alignment & Preprocessing
        target_series = prices_df[target_ticker].pct_change().dropna()
        merged = pd.merge(target_series, sentiment_df['sentiment_avg'], left_index=True, right_index=True, how='inner')
        merged.columns = ['returns', 'sentiment']
        
        # Enforce business day frequency to silence statsmodels warnings
        merged.index = pd.DatetimeIndex(merged.index).to_period('B').to_timestamp()
        
        if len(merged) < 20:
            return {"error": "Insufficient data for ARIMA modeling"}

        # 3. Fit ARIMAX Model on available data
        model = SARIMAX(merged['returns'], 
                        exog=merged['sentiment'], 
                        order=(1, 0, 1), 
                        enforce_stationarity=False, 
                        enforce_invertibility=False)
        results = model.fit(disp=False)
        
        # 4. Forecast the next 'forecast_steps'
        # For simplicity in "live" mode, we treat the last known sentiment as the exog for forecast
        last_sentiment = merged['sentiment'].iloc[-1]
        exog_forecast = np.full((forecast_steps, 1), last_sentiment)
        
        forecast = results.get_forecast(steps=forecast_steps, exog=exog_forecast)
        summary = forecast.summary_frame()
        
        # 5. Metrics
        sentiment_coeff = results.params['sentiment']
        p_value = results.pvalues['sentiment']
        
        # 6. Prepare Return Dict
        forecast_list = []
        for i, (idx, row) in enumerate(summary.iterrows()):
            forecast_list.append({
                "step": i + 1,
                "value": round(float(row['mean']), 6),
                "lower": round(float(row['mean_ci_lower']), 6),
                "upper": round(float(row['mean_ci_upper']), 6)
            })

        return {
            "ticker": target_ticker,
            "model": "ARIMAX(1,0,1)",
            "sentiment_coeff": round(float(sentiment_coeff), 6),
            "p_value": round(float(p_value), 6),
            "significant": bool(p_value < 0.10),
            "significance_level": "90%" if p_value < 0.10 else None,
            "lag": lag,
            "z_scored": use_zscore,
            "forecast": forecast_list,
            "interpretation": f"Sentiment impact is {('significant' if p_value < 0.10 else 'not significant')} with a coefficient of {sentiment_coeff:.4f}."
        }

    except Exception as e:
        logger.error(f"ARIMAX Error for {target_ticker}: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    res = run_arimax_forecast(target_ticker='NVDA')
    print(res)
