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

def run_arimax_forecast(sentiment_path, prices_path, target_ticker='SPY', forecast_steps=30, lag=0, use_zscore=False):
    """
    Fits an ARIMAX model using Sentiment as an exogenous variable.
    Includes optional 'lag' and 'zscore' parameters for feature engineering.
    """
    logger.info(f"Loading data: Ticker={target_ticker}, Lag={lag}, Z-Score={use_zscore}")
    
    # 1. Load Datasets
    sentiment_df = pd.read_csv(sentiment_path, index_col='date', parse_dates=True)
    prices_df = pd.read_csv(prices_path, index_col='date', parse_dates=True)
    
    # Apply Lag to sentiment if specified
    if lag > 0:
        sentiment_df['sentiment_avg'] = sentiment_df['sentiment_avg'].shift(lag)
        sentiment_df = sentiment_df.dropna()

    # Apply Z-Score normalization if specified
    if use_zscore:
        s_mean = sentiment_df['sentiment_avg'].mean()
        s_std = sentiment_df['sentiment_avg'].std()
        sentiment_df['sentiment_avg'] = (sentiment_df['sentiment_avg'] - s_mean) / s_std
        logger.info(f"Sentiment Z-Scored: Mean={s_mean:.4f}, Std={s_std:.4f}")

    # 2. Alignment & Preprocessing
    target_series = prices_df[target_ticker].pct_change().dropna()
    merged = pd.merge(target_series, sentiment_df['sentiment_avg'], left_index=True, right_index=True, how='inner')
    merged.columns = ['returns', 'sentiment']
    
    logger.info(f"Aligned dataset size: {len(merged)} trading days.")

    # 3. Train/Test Split (80/20)
    split_idx = int(len(merged) * 0.8)
    train_data = merged.iloc[:split_idx]
    test_data = merged.iloc[split_idx:]

    # 4. Fit ARIMAX Model
    model = SARIMAX(train_data['returns'], 
                    exog=train_data['sentiment'], 
                    order=(1, 0, 1), 
                    enforce_stationarity=False, 
                    enforce_invertibility=False)
    results = model.fit(disp=False)
    
    # 5. Forecast
    forecast = results.get_forecast(steps=len(test_data), exog=test_data['sentiment'])
    mean_forecast = forecast.summary_frame()['mean']
    conf_int = forecast.summary_frame()[['mean_ci_lower', 'mean_ci_upper']]
    
    # 6. Evaluation
    mse = mean_squared_error(test_data['returns'], mean_forecast)
    logger.info(f"Model Mean Squared Error: {mse:.6f}")
    
    # 7. Sentiment Impact Analysis
    sentiment_coef = results.params['sentiment']
    p_value = results.pvalues['sentiment']
    logger.info(f"Sentiment Coefficient: {sentiment_coef:.4f} (p-value: {p_value:.4f})")
    
    if p_value < 0.05:
        logger.info(f"RESULT: Sentiment (Lag={lag}, Z={use_zscore}) is STATISTICALLY SIGNIFICANT on {target_ticker}!")
    elif p_value < 0.10:
        logger.info(f"RESULT: Sentiment (Lag={lag}, Z={use_zscore}) approaches boundary significance (90%) for {target_ticker}.")
    else:
        logger.info(f"RESULT: Sentiment (Lag={lag}, Z={use_zscore}) not significant at 90% for {target_ticker}.")

    # 8. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, test_data['returns'], label='Actual Returns', alpha=0.5)
    plt.plot(test_data.index, mean_forecast, label=f'ARIMAX Forecast', color='red')
    plt.fill_between(test_data.index, conf_int['mean_ci_lower'], conf_int['mean_ci_upper'], color='pink', alpha=0.3)
    plt.title(f"ARIMAX Forecast: {target_ticker} (Lag={lag}, Z={use_zscore})")
    plt.legend()
    
    output_dir = "reports/plots"
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_lag{lag}_z" if use_zscore else f"_lag{lag}"
    plt.savefig(f"{output_dir}/arimax_{target_ticker}{suffix}.png")
    plt.savefig(f"{output_dir}/arimax_{target_ticker}.png") # Overwrite primary
    
    return results

if __name__ == "__main__":
    SENTIMENT_INDEX = "data/processed/sentiment_daily_index.csv"
    PRICES_DATA = "data/raw/tickers/prices_daily.csv"
    
    # Optimized Discovery: Run NVDA and SPY with Lag=1 + Z-Score
    print(f"\n{'='*40}")
    print(f" OPTIMIZED FORECASTING: NVDA (LAG-1, Z-SCORE)")
    print(f"{'='*40}")
    try:
        run_arimax_forecast(SENTIMENT_INDEX, PRICES_DATA, target_ticker='NVDA', lag=1, use_zscore=True)
    except Exception as e:
        logger.error(f"Failed to process NVDA: {e}")

    print(f"\n{'='*40}")
    print(f" OPTIMIZED FORECASTING: SPY (LAG-1, Z-SCORE)")
    print(f"{'='*40}")
    try:
        run_arimax_forecast(SENTIMENT_INDEX, PRICES_DATA, target_ticker='SPY', lag=1, use_zscore=True)
    except Exception as e:
        logger.error(f"Failed to process SPY: {e}")
