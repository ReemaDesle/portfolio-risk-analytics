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
