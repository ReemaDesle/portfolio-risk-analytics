import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pipeline.ml.infer import run_inference, PORTFOLIOS

app = FastAPI(title="Portfolio Risk Analytics API")


def _sanitize(obj):
    """Recursively replace NaN / ±Inf floats with None so json.dumps won't crash."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static data paths ──────────────────────────────────────────────────────
ROOT            = pathlib.Path(__file__).resolve().parent.parent
MASTER_DATA     = ROOT / "data" / "processed" / "master_data.csv"
ML_RESULTS      = ROOT / "reports" / "ml_results_summary.txt"


# ── Helpers ────────────────────────────────────────────────────────────────
def _load_master():
    if not MASTER_DATA.exists():
        return None
    return pd.read_csv(MASTER_DATA).fillna(0)


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
def read_root():
    return {
        "message": "Portfolio Risk Analytics API — v3",
        "models":  "M1–M6 trained on 1,329 trading days (2021–2026)",
        "status":  "live",
    }


@app.get("/portfolios")
def get_portfolios():
    """Return list of available portfolio archetypes."""
    return {
        "portfolios": list(PORTFOLIOS.keys()),
        "tickers":    PORTFOLIOS,
    }


@app.get("/analytics/{portfolio}")
def get_analytics(portfolio: str):
    """
    Return raw market data (last 60d prices + sentiment) for the selected portfolio.
    Used by the Analytics tab chart.
    """
    if portfolio not in PORTFOLIOS:
        raise HTTPException(status_code=404, detail=f"Unknown portfolio: {portfolio}")

    master_df = _load_master()
    if master_df is None:
        raise HTTPException(status_code=500, detail="master_data.csv not found")

    tickers     = PORTFOLIOS[portfolio]
    price_cols  = [t for t in tickers if t in master_df.columns]
    sent_cols   = [c for c in master_df.columns if c.startswith("sentiment_score_")]
    vol_col     = f"vol5_{portfolio}"
    shock_col   = f"shock_{portfolio}"

    keep = ["date"] + price_cols + sent_cols
    if vol_col in master_df.columns:   keep.append(vol_col)
    if shock_col in master_df.columns: keep.append(shock_col)

    market_data = master_df[keep].tail(60).to_dict(orient="records")

    return {
        "portfolio":   portfolio,
        "tickers":     tickers,
        "market_data": market_data,
    }


@app.get("/inference/{portfolio}")
def get_inference(portfolio: str):
    """
    Run full ML inference pipeline for the selected portfolio.
    Returns all 5 output modules: shock, recovery, risk score,
    domain sensitivity, portfolio category, and buy/sell signal.
    """
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
    """
    Legacy endpoint — now backed by real ML inference (M1–M6).
    Returns a flattened summary for backward compatibility with the frontend.
    """
    if portfolio not in PORTFOLIOS:
        raise HTTPException(status_code=404, detail=f"Unknown portfolio: {portfolio}")

    try:
        result = run_inference(portfolio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    m1      = result.get("m1", {})
    m2      = result.get("m2", {})
    m3      = result.get("m3", {})
    m4_m5   = result.get("m4_m5", {})
    m6      = result.get("m6", {})
    bs      = result.get("buysell", {})

    # Map to the shape the current frontend expects
    shock_prob = m1.get("shock_probability", 0.0)
    risk_label = m3.get("risk_label", "NORMAL")

    # SRI proxy: scaled shock prob × 20 so it sits in the existing 0–20 SRI range
    sri_value  = round(shock_prob * 20, 2)

    return _sanitize({
        # Legacy fields (frontend already uses these)
        "status":     risk_label,
        "action":     bs.get("action", "HOLD"),
        "reasoning":  bs.get("reasoning", "No signal."),
        "sri_value":  sri_value,
        "category":   m6.get("category_label", "Unknown"),

        # New fields (frontend will use after App.jsx update)
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
        "buysell": bs,
        "stock_table":  result.get("stock_table", []),
        "market_chart": result.get("market_chart", []),
    })


@app.get("/model-status")
def get_model_status():
    """Return which trained model files are present."""
    files = {
        "m1_shock_classifier":   (ROOT / "models/ml/m1_shock_classifier.pkl").exists(),
        "m2_recovery_predictor": (ROOT / "models/ml/m2_recovery_predictor.pkl").exists(),
        "m3_risk_scorer_tech":   (ROOT / "models/ml/m3_risk_scorer_tech.pkl").exists(),
        "m4_ccf_results":        (ROOT / "models/ml/m4_ccf_results.csv").exists(),
        "m5_ridge_tech":         (ROOT / "models/ml/m5_ridge_tech.pkl").exists(),
        "m6_clusters":           (ROOT / "models/ml/m6_clusters.csv").exists(),
        "master_data":           MASTER_DATA.exists(),
        "ml_features":           (ROOT / "data/processed/ml_features.csv").exists(),
    }
    all_ready = all(files.values())
    return {"all_ready": all_ready, "files": files}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
