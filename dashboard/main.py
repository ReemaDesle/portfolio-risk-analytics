import sys, pathlib, io, json, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from pipeline.ml.infer import run_inference, classify_portfolio, compute_weights, PORTFOLIOS

app = FastAPI(title="Portfolio Risk Analytics API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static paths ────────────────────────────────────────────────────────────
ROOT                 = pathlib.Path(__file__).resolve().parent.parent
MASTER_DATA          = ROOT / "data" / "processed" / "master_data.csv"
ML_FEATURES          = ROOT / "data" / "processed" / "ml_features.csv"
SENTIMENT_INDEX      = ROOT / "data" / "processed" / "sentiment_daily_index.csv"
REPORTS_DIR          = ROOT / "reports"
ML_RESULTS_TXT       = REPORTS_DIR / "ml_results_summary.txt"
ARIMAX_RESULTS_TXT   = REPORTS_DIR / "arimax_results.txt"
STAT_TESTS_TXT       = ROOT / "statistical_tests_results.txt"
HYPOTHESIS_TXT       = ROOT / "hypothesis_testing_results.txt"
MODELS_DIR           = ROOT / "models" / "ml"

LEAD_TICKERS = {
    "tech":         "NVDA",
    "geopolitical": "GLD",
    "balanced":     "SPY",
    "conservative": "TLT",
}


# ── Helpers ─────────────────────────────────────────────────────────────────
def _sanitize(obj):
    """Recursively replace NaN / ±Inf floats with None."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _load_master(target_date: str = None) -> pd.DataFrame:
    if not MASTER_DATA.exists():
        return None
    df = pd.read_csv(MASTER_DATA, parse_dates=["date"]).sort_values("date")
    if target_date:
        df = df[df["date"] <= pd.Timestamp(target_date)]
    return df.fillna(0)


def _read_text(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════
# ROOT
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
def read_root():
    return {
        "message": "Portfolio Risk Analytics API — v4",
        "models":  "M1–M6 trained on 1,329 trading days (2021–2026)",
        "status":  "live",
        "docs":    "/docs",
    }


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1 — CORE USER MODE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/tickers")
def get_tickers():
    """
    Returns tickers grouped by archetype, plus the valid date range.
    Powers the grouped UI mentioned in requirements.
    """
    if not MASTER_DATA.exists():
        raise HTTPException(status_code=500, detail="master_data.csv not found")

    df = pd.read_csv(MASTER_DATA, parse_dates=["date"])
    
    # Filter tickers that actually exist in the data
    grouped = {}
    for arch, tickers in PORTFOLIOS.items():
        grouped[arch] = [t for t in tickers if t in df.columns]

    dates = df["date"].dropna().dt.strftime("%Y-%m-%d").tolist()
    return {
        "grouped":     grouped,
        "lead_tickers": LEAD_TICKERS,
        "date_range":  {"min": min(dates), "max": max(dates)},
        "valid_dates": sorted(set(dates)),
    }


# ── Request model for /api/analyze ──────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    tickers:    List[str]
    quantities: List[int]
    date:       Optional[str] = None   # ISO date string, e.g. "2026-03-15"


@app.post("/api/analyze")
def analyze_portfolio(req: AnalyzeRequest):
    """
    User Mode core endpoint.
    Accepts custom tickers + quantities + optional date.

    1. Classifies the portfolio to the best-fit archetype (or "balanced")
    2. Filters data up to the given date
    3. Runs full ML inference (M1–M6)
    4. Returns action, justification (short + detailed), XAI model contributions,
       recovery justification, ticker line chart data, stock table
    """
    if len(req.tickers) != len(req.quantities):
        raise HTTPException(status_code=400, detail="tickers and quantities must be the same length")
    if not req.tickers:
        raise HTTPException(status_code=400, detail="At least one ticker required")

    # Classify to archetype (default: balanced for unknown tickers)
    archetype = classify_portfolio(req.tickers, req.quantities)
    weights   = compute_weights(req.tickers, req.quantities)

    try:
        result = run_inference(
            portfolio=archetype,
            target_date=req.date,
            user_tickers=req.tickers,
            user_quantities=req.quantities,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    buysell = result.get("buysell", {})
    m2      = result.get("m2", {})

    # Build structured response
    response = {
        "mapped_archetype": archetype,
        "weights":          weights,
        "target_date":      result.get("target_date"),
        "data_rows_used":   result.get("data_rows_used"),

        # Recommended action
        "action":           buysell.get("action"),
        "confidence":       buysell.get("confidence"),

        # Justification (toggle: short / detailed)
        "justification": {
            "short":    buysell.get("short_reason"),
            "detailed": buysell.get("reasoning"),
        },

        # Recovery justification (user mode req #4)
        "recovery": {
            "band_label":     m2.get("band_label"),
            "justification":  m2.get("note"),
            "p25_days":       m2.get("p25_days"),
            "p50_days":       m2.get("p50_days"),
            "p75_days":       m2.get("p75_days"),
        },

        # XAI — model contributions (user mode req #5, visible on detailed toggle)
        "model_contributions": result.get("model_contributions"),

        # Charts & table
        "ticker_chart_data": result.get("market_chart"),
        "stock_table":       result.get("stock_table"),

        # Raw model outputs (for technical mode / advanced use)
        "m1": result.get("m1"),
        "m2": result.get("m2"),
        "m3": result.get("m3"),
        "m4_m5": result.get("m4_m5"),
        "m6": result.get("m6"),
    }

    return _sanitize(response)


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2 — TECHNICAL MODE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/metrics")
def get_metrics():
    """
    Technical Mode: Returns structured ML performance metrics + statistical test results.
    Sources: ml_results_summary.txt, arimax_results.txt, statistical_tests_results.txt,
             hypothesis_testing_results.txt
    """
    # ── M1–M6 metrics (structured from known training output) ──────────────
    ml_models = {
        "M1": {
            "name":       "Shock Classifier",
            "algorithm":  "RandomForest (shock_events < 100)",
            "f1_default": 0.0714,
            "f1_calibrated": 0.3913,
            "auc_roc":    0.8209,
            "auc_pr":     0.3185,
            "optimal_threshold": 0.296,
            "n_shock_events": 39,
            "note": "Calibrated F1 vs default F1 shows threshold tuning is critical for rare events.",
        },
        "M2": {
            "name":       "Recovery Predictor",
            "algorithm":  "QuantileRegressor (P25/P50/P75) + GradientBoosting",
            "mae_days":   2.23,
            "r2":         -0.167,
            "band_width": 5.4,
            "n_shock_events": 244,
            "note": "Negative R² expected for quantile regression; MAE of 2.23 days is operationally useful.",
        },
        "M3": {
            "name":       "Risk Scorer (next-day vol)",
            "algorithm":  "Ridge + QuantileRegressor per portfolio",
            "per_portfolio": {
                "geopolitical": {"r2": 0.2045, "mae": 0.00198, "band_width": 0.003766},
                "tech":         {"r2": 0.3239, "mae": 0.003001, "band_width": 0.005829},
                "balanced":     {"r2": 0.138,  "mae": 0.001171, "band_width": 0.002239},
                "conservative": {"r2": 0.3039, "mae": 0.001484, "band_width": 0.002419},
            },
            "note": "Sentiment-only features prevent circularity. MLP severely overfit (R²<<0).",
        },
        "M4": {
            "name":       "Cross-Domain Lag (Granger Causality)",
            "algorithm":  "VAR (BIC lag selection) + Granger tests",
            "significant_pairs": [
                {"cause": "geopolitical", "effect": "tech",     "lag": 3, "p_value": 0.0062},
                {"cause": "financial",    "effect": "tech",     "lag": 1, "p_value": 0.0366},
                {"cause": "financial",    "effect": "balanced", "lag": 1, "p_value": 0.0045},
            ],
            "adf_all_stationary": True,
            "note": "All series were stationary (ADF p=0.0). Granger confirms geo news leads tech vol by 3 days.",
        },
        "M5": {
            "name":       "Sentiment-Vol Regression (Domain Sensitivity)",
            "algorithm":  "RidgeCV per portfolio + SHAP values",
            "per_portfolio": {
                "geopolitical": {"r2": 0.728,  "alpha": 10.0, "dominant": "financial"},
                "tech":         {"r2": 0.5981, "alpha": 10.0, "dominant": "geopolitical"},
                "balanced":     {"r2": 0.5478, "alpha": 10.0, "dominant": "financial"},
                "conservative": {"r2": 0.64,   "alpha": 10.0, "dominant": "geopolitical"},
            },
            "note": "High R² because lagged_vol (t-1) is a strong autocorrelation control. SHAP values saved.",
        },
        "M6": {
            "name":       "Portfolio Clustering",
            "algorithm":  "KMeans + Hierarchical (AgglomerativeClustering)",
            "best_k":     2,
            "silhouette": 0.256,
            "assignments": {
                "geopolitical": "Geopolitical-Sensitive",
                "tech":         "Geopolitical-Sensitive",
                "balanced":     "Geopolitical-Sensitive",
                "conservative": "Geopolitical-Sensitive",
            },
            "note": "All 4 archetypes clustered similarly — data from 2021–2026 is geo-shock dominated.",
        },
    }

    # ── Statistical tests ──────────────────────────────────────────────────
    statistical_tests = {
        "wilcoxon_rank_sum": {
            "description": "Tech vs Defensive returns (H5)",
            "statistic":   1.3261,
            "p_value":     0.1848,
            "significant": False,
            "note":        "p=0.18 > 0.05. Trend visible in EDA but not statistically significant at n=18.",
        },
        "anova_one_way": {
            "description": "Cross-portfolio return comparison (H6)",
            "f_statistic": 0.7517,
            "p_value":     0.5254,
            "significant": False,
            "note":        "p=0.53 — archetypes behave similarly in raw returns during this window.",
        },
        "chi_square": {
            "description": "High Risk News vs Negative Market Direction (H4)",
            "statistic":   1.1077,
            "p_value":     0.2926,
            "significant": False,
            "note":        "p=0.29 — no strong coupling found at n=18. Granger (M4) provides stronger evidence.",
        },
    }

    # ── Hypothesis results ─────────────────────────────────────────────────
    hypotheses = {
        "H1_safe_haven":         {"result": "Partially Confirmed", "correlation": 0.1467,
                                  "detail": "Positive but weak correlation between geo sentiment and safe-haven returns."},
        "H2_tech_sensitivity":   {"result": "Confirmed", "vol_shock": 0.0249, "vol_normal": 0.0214,
                                  "detail": "16% vol spike in tech portfolios during negative sentiment shocks."},
        "H3_portfolio_comparison":{"result": "Confirmed", "tech_sensitivity": 0.40,
                                  "detail": "Tech has highest geo sensitivity (0.40) vs all other archetypes."},
        "H4_lag_effect":         {"result": "Confirmed", "lag0_corr": -0.2383, "lag1_corr": -0.1789,
                                  "detail": "Lag-0 correlation stronger than Lag-1 — market prices news same day."},
        "H5_predictive_dominance":{"result": "Confirmed (this window)", "geo_corr": 0.3427, "fin_corr": 0.0522,
                                  "detail": "Geopolitical news has 7× more impact on SPY than fin/tech news."},
        "H6_recovery_time":      {"result": "Partially Confirmed", "avg_recovery_days": "2–7",
                                  "detail": "Fast recovery (M2: median 4 days). 'Day-0' efficiency observed."},
    }

    # ── ARIMA metrics ──────────────────────────────────────────────────────
    arima_metrics = {
        "model":   "ARIMAX(1,0,1) with FinBERT sentiment as exogenous signal",
        "dataset": "financial_news_v3_scored.csv (20,537 headlines)",
        "tickers": {
            "NVDA": {"coefficient": 0.0067, "p_value": 0.0973, "significant_90pct": True,
                     "note": "1-day lagged sentiment significant at 90%. Tech headlines lead price by ~24h."},
            "SPY":  {"coefficient": 0.0031, "p_value": 0.2012, "significant_90pct": False,
                     "note": "Broad market sensitive to extreme sentiment (outliers), not daily average."},
            "GLD":  {"coefficient": -0.0005, "p_value": 0.6413, "significant_90pct": False,
                     "note": "Gold insensitive to financial news — driven by macro/geopolitical factors."},
            "TLT":  {"coefficient": 0.0008, "p_value": 0.4691, "significant_90pct": False,
                     "note": "Bonds largely insulated from daily equity-centric sentiment shocks."},
        },
    }

    return _sanitize({
        "ml_models":         ml_models,
        "statistical_tests": statistical_tests,
        "hypotheses":        hypotheses,
        "arima":             arima_metrics,
    })


@app.get("/api/model-mapping")
def get_model_mapping():
    """
    Technical Mode: Returns how each ML model (M1–M6) maps to
    EDA findings, hypotheses, and statistical tests.
    Powers the 'models → hypothesis → EDA' section in Technical Mode.
    """
    return {
        "mappings": [
            {
                "model":       "M1",
                "name":        "Shock Classifier",
                "algorithm":   "RandomForest / XGBoost (adaptive threshold, walk-forward CV)",
                "hypotheses":  ["H2: Tech 16% vol spike during shocks", "H4: Lag-0 market efficiency"],
                "eda_finding": "EDA confirmed tech portfolios experience outsized vol during negative news days",
                "stat_test":   "Wilcoxon (p=0.18) — trend visible but sample too small for significance",
                "feature_importance": "vol10_tech, vol20_tech, vol5_tech are top-3 predictors",
                "xai_note":    "Lower shock prob → safer entry. High prob (>75%) triggers REDUCE signal.",
            },
            {
                "model":       "M2",
                "name":        "Recovery Predictor",
                "algorithm":   "QuantileRegressor (P25/P50/P75) + GradientBoosting",
                "hypotheses":  ["H6: 'Day-0' recovery efficiency confirmed"],
                "eda_finding": "Recovery time near 0 days — market fully prices shocks within the session",
                "stat_test":   "No direct statistical test; empirically measured from 244 pooled shock events",
                "feature_importance": "shock_magnitude, sector_label, pre_shock_baseline",
                "xai_note":    "Recovery band (P25–P75) provides confidence interval for 'how long to wait'.",
            },
            {
                "model":       "M3",
                "name":        "Risk Scorer",
                "algorithm":   "Ridge regression (sentiment-only features → next-day vol)",
                "hypotheses":  ["H4: News impacts vol same day (Lag-0 correlation = -0.24)"],
                "eda_finding": "Sentiment score drop correlates with next-day vol spike across all archetypes",
                "stat_test":   "R²: tech=0.32, conservative=0.30 — moderate predictive power",
                "feature_importance": "lagged_vol, sentiment_3d_financial, market_regime_flag",
                "xai_note":    "Risk label (NORMAL / ELEVATED / HIGH RISK) + P25–P75 band for uncertainty.",
            },
            {
                "model":       "M4",
                "name":        "Cross-Domain Lag (Granger Causality)",
                "algorithm":   "VAR (BIC lag selection) + Granger tests on stationary series",
                "hypotheses":  ["H5: Geopolitical news is the #1 predictor (SPY corr=0.34)"],
                "eda_finding": "Geo sentiment leads tech portfolio vol by 3–5 days (Granger p=0.006)",
                "stat_test":   "ADF tests: all series stationary (p=0.0). Granger confirms causal direction.",
                "feature_importance": "sentiment_score_geopolitical, vol5_tech (endogenous)",
                "xai_note":    "Indicates WHICH domain of news to watch and HOW MANY DAYS in advance.",
            },
            {
                "model":       "M5",
                "name":        "Domain Sensitivity Regression",
                "algorithm":   "RidgeCV + SHAP LinearExplainer per portfolio",
                "hypotheses":  ["H3: Tech has highest geo sensitivity (0.40) vs all archetypes",
                                "H1: Safe-haven assets react to geopolitical uncertainty"],
                "eda_finding": "Tech portfolio most responsive to news — highest beta to sentiment swings",
                "stat_test":   "Chi-Square (p=0.29) — directional but not yet significant at n=18",
                "feature_importance": "lagged_vol (autocorrelation control), sentiment_3d_financial, article_zscore",
                "xai_note":    "Coefficients show sensitivity per domain. SHAP values saved for per-day advice.",
            },
            {
                "model":       "M6",
                "name":        "Portfolio Clustering",
                "algorithm":   "KMeans (k=2, silhouette=0.26) + Hierarchical (Ward linkage)",
                "hypotheses":  ["H1: Safe-haven clustering reflects real portfolio behavior"],
                "eda_finding": "All 4 archetypes cluster as 'Geopolitical-Sensitive' in 2021–2026 data",
                "stat_test":   "ANOVA (p=0.53) — archetypes not statistically different in raw returns",
                "feature_importance": "M5 domain coefficients, shock_frequency, intra_portfolio_correlation",
                "xai_note":    "Shows which portfolio 'type' user falls into and expansion suggestions.",
            },
        ]
    }


from pipeline.analysis.arima_baseline import run_arimax_forecast

@app.get("/api/arima/{ticker}")
def get_arima_forecast(ticker: str, steps: int = 10, lag: int = 1, use_zscore: bool = True):
    """
    Technical Mode: Returns live ARIMA forecast results for a given ticker or archetype.
    If ticker is an archetype name (e.g. 'tech'), it uses the lead ticker (e.g. 'NVDA').
    """
    t_key = ticker.lower()
    if t_key in LEAD_TICKERS:
        target = LEAD_TICKERS[t_key]
        log_msg = f"Mapping archetype '{ticker}' to lead ticker '{target}'"
    else:
        target = ticker.upper()
        log_msg = f"Running ARIMA for ticker '{target}'"
    
    # Run live forecast
    res = run_arimax_forecast(target_ticker=target, forecast_steps=steps, lag=lag, use_zscore=use_zscore)
    
    if "error" in res:
        precomputed_notes = {
            "NVDA": "NVDA has the highest sensitivity to news sentiment. Tech headlines lead price by ~24 hours.",
            "SPY":  "Broad market sensitive to extreme sentiment (outliers), not daily average.",
            "GLD":  "Gold insensitive to financial news — driven by macro/geopolitical factors.",
            "TLT":  "Bonds largely insulated from daily equity-centric sentiment shocks.",
        }
        return {
            "ticker": target,
            "available": False,
            "error": res["error"],
            "note": precomputed_notes.get(target, f"No precomputed insight for {target}.")
        }

    res["available"] = True
    res["ticker"]    = target
    # Check if a plot exists
    suffix = f"_lag{lag}_z" if use_zscore else f"_lag{lag}"
    res["plot_available"] = (REPORTS_DIR / "plots" / f"arimax_{target}{suffix}.png").exists()
    
    return _sanitize(res)


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3 — POWER BI DATA FEED ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/data-feed/master")
def feed_master(date: str = None):
    """
    Power BI data feed: returns master_data.csv as downloadable CSV.
    Connect in Power BI Desktop via: Get Data → Web → this URL.
    Optional ?date=YYYY-MM-DD to filter up to a specific date.
    """
    if not MASTER_DATA.exists():
        raise HTTPException(status_code=500, detail="master_data.csv not found")

    df = pd.read_csv(MASTER_DATA)
    if date:
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.Timestamp(date)]

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=master_data.csv"},
    )


@app.get("/api/data-feed/ml-features")
def feed_ml_features():
    """
    Power BI data feed: returns ml_features.csv as downloadable CSV.
    Contains all engineered features: vol_zscore, sentiment_velocity,
    intra_corr, market_regime_flag, shock labels, etc.
    """
    if not ML_FEATURES.exists():
        raise HTTPException(status_code=500, detail="ml_features.csv not found")

    df = pd.read_csv(ML_FEATURES)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ml_features.csv"},
    )


@app.get("/api/data-feed/sentiment-index")
def feed_sentiment_index():
    """
    Power BI data feed: returns sentiment_daily_index.csv as downloadable CSV.
    Contains daily sentiment scores per domain (geo / financial / technology).
    """
    if not SENTIMENT_INDEX.exists():
        raise HTTPException(status_code=500, detail="sentiment_daily_index.csv not found")

    df = pd.read_csv(SENTIMENT_INDEX)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sentiment_daily_index.csv"},
    )


@app.get("/api/data-feed/model-metrics")
def feed_model_metrics():
    """
    Power BI data feed: returns a flat CSV of all model performance metrics.
    Useful for building a 'Model Comparison' bar chart in Power BI.
    """
    rows = [
        {"model": "M1", "metric": "F1_calibrated", "value": 0.3913, "portfolio": "tech"},
        {"model": "M1", "metric": "AUC_ROC",        "value": 0.8209, "portfolio": "tech"},
        {"model": "M1", "metric": "AUC_PR",          "value": 0.3185, "portfolio": "tech"},
        {"model": "M2", "metric": "MAE_days",         "value": 2.23,   "portfolio": "all"},
        {"model": "M2", "metric": "band_width_days",  "value": 5.4,    "portfolio": "all"},
        {"model": "M3", "metric": "R2",               "value": 0.3239, "portfolio": "tech"},
        {"model": "M3", "metric": "R2",               "value": 0.2045, "portfolio": "geopolitical"},
        {"model": "M3", "metric": "R2",               "value": 0.1380, "portfolio": "balanced"},
        {"model": "M3", "metric": "R2",               "value": 0.3039, "portfolio": "conservative"},
        {"model": "M5", "metric": "R2",               "value": 0.5981, "portfolio": "tech"},
        {"model": "M5", "metric": "R2",               "value": 0.7280, "portfolio": "geopolitical"},
        {"model": "M5", "metric": "R2",               "value": 0.5478, "portfolio": "balanced"},
        {"model": "M5", "metric": "R2",               "value": 0.6400, "portfolio": "conservative"},
        {"model": "M6", "metric": "silhouette",       "value": 0.256,  "portfolio": "all"},
    ]
    df = pd.DataFrame(rows)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=model_metrics.csv"},
    )


@app.get("/api/data-feed/hypothesis-results")
def feed_hypothesis_results():
    """
    Power BI data feed: returns hypothesis validation results as CSV.
    Use this in Power BI to build a 'Hypothesis Status' funnel/table.
    """
    rows = [
        {"hypothesis": "H1 Safe-Haven Reactivity", "result": "Partially Confirmed",
         "metric": "correlation", "value": 0.1467, "p_value": None, "significant": False},
        {"hypothesis": "H2 Tech Sensitivity",       "result": "Confirmed",
         "metric": "vol_spike_pct", "value": 16.0, "p_value": None, "significant": True},
        {"hypothesis": "H3 Portfolio Sensitivity",  "result": "Confirmed",
         "metric": "tech_geo_sensitivity", "value": 0.40, "p_value": None, "significant": True},
        {"hypothesis": "H4 Lag Effect",             "result": "Confirmed",
         "metric": "lag0_correlation", "value": -0.2383, "p_value": None, "significant": True},
        {"hypothesis": "H5 Predictive Dominance",   "result": "Confirmed (this window)",
         "metric": "geo_corr_SPY", "value": 0.3427, "p_value": None, "significant": True},
        {"hypothesis": "H6 Recovery Time",          "result": "Partially Confirmed",
         "metric": "avg_recovery_days", "value": 4.0, "p_value": None, "significant": False},
        {"hypothesis": "Stat: Wilcoxon (Tech vs Defensive)", "result": "Not Significant",
         "metric": "p_value", "value": 0.1848, "p_value": 0.1848, "significant": False},
        {"hypothesis": "Stat: ANOVA (Portfolio Returns)",    "result": "Not Significant",
         "metric": "p_value", "value": 0.5254, "p_value": 0.5254, "significant": False},
        {"hypothesis": "Stat: Chi-Square (News↔Market)",    "result": "Not Significant",
         "metric": "p_value", "value": 0.2926, "p_value": 0.2926, "significant": False},
    ]
    df = pd.DataFrame(rows)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=hypothesis_results.csv"},
    )


# ══════════════════════════════════════════════════════════════════════════
# LEGACY ENDPOINTS (preserved for backward compatibility)
# ══════════════════════════════════════════════════════════════════════════

@app.get("/portfolios")
def get_portfolios():
    return {"portfolios": list(PORTFOLIOS.keys()), "tickers": PORTFOLIOS}


@app.get("/analytics/{portfolio}")
def get_analytics(portfolio: str):
    if portfolio not in PORTFOLIOS:
        raise HTTPException(status_code=404, detail=f"Unknown portfolio: {portfolio}")
    master_df = _load_master()
    if master_df is None:
        raise HTTPException(status_code=500, detail="master_data.csv not found")
    tickers    = PORTFOLIOS[portfolio]
    price_cols = [t for t in tickers if t in master_df.columns]
    sent_cols  = [c for c in master_df.columns if c.startswith("sentiment_score_")]
    vol_col    = f"vol5_{portfolio}"
    shock_col  = f"shock_{portfolio}"
    keep = ["date"] + price_cols + sent_cols
    if vol_col in master_df.columns:   keep.append(vol_col)
    if shock_col in master_df.columns: keep.append(shock_col)
    market_data = master_df[keep].tail(60).to_dict(orient="records")
    return {"portfolio": portfolio, "tickers": tickers, "market_data": market_data}


@app.get("/inference/{portfolio}")
def get_inference(portfolio: str):
    if portfolio not in PORTFOLIOS:
        raise HTTPException(status_code=404, detail=f"Unknown portfolio '{portfolio}'")
    try:
        result = run_inference(portfolio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return _sanitize(result)


@app.get("/suggestions/{portfolio}")
def get_suggestions(portfolio: str):
    if portfolio not in PORTFOLIOS:
        raise HTTPException(status_code=404, detail=f"Unknown portfolio: {portfolio}")
    try:
        result = run_inference(portfolio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    m1 = result.get("m1", {})
    m2 = result.get("m2", {})
    m3 = result.get("m3", {})
    m4_m5 = result.get("m4_m5", {})
    m6 = result.get("m6", {})
    bs = result.get("buysell", {})
    shock_prob = m1.get("shock_probability", 0.0)
    sri_value  = round(shock_prob * 20, 2)

    return _sanitize({
        "status":    m3.get("risk_label", "NORMAL"),
        "action":    bs.get("action", "HOLD"),
        "reasoning": bs.get("reasoning", "No signal."),
        "sri_value": sri_value,
        "category":  m6.get("category_label", "Unknown"),
        "shock": {
            "probability":       m1.get("shock_probability"),
            "signal":            m1.get("signal"),
            "note":              m1.get("note"),
            "optimal_threshold": m1.get("optimal_threshold"),
        },
        "recovery": {
            "band_label": m2.get("band_label"),
            "p25_days":   m2.get("p25_days"),
            "p50_days":   m2.get("p50_days"),
            "p75_days":   m2.get("p75_days"),
        },
        "risk_score": {
            "predicted_vol": m3.get("predicted_vol"),
            "risk_label":    m3.get("risk_label"),
            "p25_vol":       m3.get("p25_vol"),
            "p75_vol":       m3.get("p75_vol"),
        },
        "domain_sensitivity": {
            "dominant_domain":  m4_m5.get("dominant_domain"),
            "coefficients":     m4_m5.get("coefficients"),
            "ranked_domains":   m4_m5.get("ranked_domains"),
            "recent_sentiment": m4_m5.get("recent_sentiment"),
            "news_to_watch":    m4_m5.get("news_to_watch"),
            "granger_note":     m4_m5.get("granger_confirmed"),
        },
        "portfolio_category": {
            "label":               m6.get("category_label"),
            "shock_frequency_pct": m6.get("shock_frequency"),
            "intra_correlation":   m6.get("intra_correlation"),
            "safe_haven_pct":      m6.get("safe_haven_weight"),
            "expansion":           m6.get("expansion_suggestions", []),
        },
        "buysell":      bs,
        "stock_table":  result.get("stock_table", []),
        "market_chart": result.get("market_chart", []),
        "model_contributions": result.get("model_contributions", {}),
    })


@app.get("/model-status")
def get_model_status():
    files = {
        "m1_shock_classifier":   (MODELS_DIR / "m1_shock_classifier.pkl").exists(),
        "m2_recovery_predictor": (MODELS_DIR / "m2_recovery_predictor.pkl").exists(),
        "m3_ridge_tech":         (MODELS_DIR / "m3_ridge_tech.pkl").exists(),
        "m3_ridge_geopolitical": (MODELS_DIR / "m3_ridge_geopolitical.pkl").exists(),
        "m3_ridge_balanced":     (MODELS_DIR / "m3_ridge_balanced.pkl").exists(),
        "m3_ridge_conservative": (MODELS_DIR / "m3_ridge_conservative.pkl").exists(),
        "m4_ccf_results":        (MODELS_DIR / "m4_ccf_results.csv").exists(),
        "m5_domain_coefficients":(MODELS_DIR / "m5_domain_coefficients.csv").exists(),
        "m6_clusters":           (MODELS_DIR / "m6_clusters.csv").exists(),
        "master_data":           MASTER_DATA.exists(),
        "ml_features":           ML_FEATURES.exists(),
    }
    all_ready = all(files.values())
    return {"all_ready": all_ready, "files": files}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
