"""
Portfolio Risk Analytics — ML Modelling Pipeline
=================================================
Implements all 6 models from ml_modeling_plan.txt

  M1 — Shock Classifier         (XGBoost binary classifier)
  M2 — Recovery Predictor       (Gradient Boosting Regressor)
  M3 — Portfolio Risk Scorer    (Ridge + MLP Regressor)
  M4 — Cross-Domain Lag Predictor (Granger Causality + VAR)
  M5 — Sentiment-Volatility Regression (Lasso)
  M6 — Portfolio Clustering     (KMeans + HDBSCAN)

Feature Engineering:
  - Rolling windows (5, 10, 20 day)
  - Cross-domain sentiment lags (1–5 days)
  - Portfolio-level volatility aggregations
  - Z-score normalisation

Inputs:
  data/processed/master_data.csv

Outputs:
  data/processed/ml_features.csv       — engineered feature table
  models/ml/m1_shock_classifier.*      — XGBoost shock model + metrics
  models/ml/m2_recovery_predictor.*    — GBR recovery model + metrics
  models/ml/m3_risk_scorer_ridge.*     — Ridge risk model + metrics
  models/ml/m3_risk_scorer_mlp.*       — MLP risk model  + metrics
  models/ml/m4_granger_results.csv     — Granger causality p-values
  models/ml/m5_lasso_coefficients.csv  — Lasso domain importance
  models/ml/m6_clusters.csv           — Portfolio cluster assignments
  reports/ml_results_summary.txt      — Human-readable results summary

Usage:
  python pipeline/ml/train_models.py
  python pipeline/ml/train_models.py --skip-granger   # faster, skips M4
"""

import argparse
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR    = ROOT_DIR / "models" / "ml"
REPORTS_DIR   = ROOT_DIR / "reports"

MASTER_DATA   = PROCESSED_DIR / "master_data.csv"
FEATURES_OUT  = PROCESSED_DIR / "ml_features.csv"
RESULTS_TXT   = REPORTS_DIR / "ml_results_summary.txt"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Portfolio definitions (matching EDA / score_compute)
# ──────────────────────────────────────────────
PORTFOLIOS = {
    "geopolitical": ["GLD", "USO", "LMT", "RTX", "EEM", "GC=F", "CL=F"],
    "tech":         ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "SOXX", "QQQ"],
    "balanced":     ["SPY", "AGG", "GLD", "VTI", "EFA", "BND"],
    "conservative": ["TLT", "IEF", "VPU", "KO", "JNJ", "PG", "XLP"],
}

DOMAINS = ["financial", "geopolitical", "technology"]

# ══════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the full ML feature table from master_data.csv.

    New features added:
      - Portfolio returns / volatility (equal-weight averaged per archetype)
      - Rolling windows: 10-day and 20-day volatility per ticker
      - Cross-domain lagged sentiments: lag-2 to lag-5 for each domain
      - Article count spike flags (z-score > 2)
      - Z-score normalisation for all sentiment features
    """
    log.info("── [FEATURE ENGINEERING] ──────────────────────────────")
    df = df.copy()

    # ── 1. Portfolio-level return aggregations ───────────────────────────────
    log.info("  Building portfolio-level return & vol aggregations...")
    for pname, tickers in PORTFOLIOS.items():
        ret_cols = [f"ret_{t}" for t in tickers if f"ret_{t}" in df.columns]
        vol_cols = [f"vol5_{t}" for t in tickers if f"vol5_{t}" in df.columns]
        if ret_cols:
            pf_ret = df[ret_cols].mean(axis=1)
            df[f"ret_{pname}"]   = pf_ret
            df[f"vol10_{pname}"] = pf_ret.rolling(10, min_periods=3).std()
            df[f"vol20_{pname}"] = pf_ret.rolling(20, min_periods=5).std()
        if vol_cols:
            df[f"vol5_{pname}"] = df[vol_cols].mean(axis=1)

    # ── 2. Cross-domain lags (lag-2 to lag-5) ───────────────────────────────
    log.info("  Adding cross-domain lag features (lag-2 to lag-5)...")
    for domain in DOMAINS:
        col = f"sentiment_score_{domain}"
        if col in df.columns:
            for lag in range(2, 6):
                df[f"lag{lag}_sentiment_score_{domain}"] = df[col].shift(lag)

    # ── 3. Article count spike flag (z-score > 2 = "news burst") ────────────
    log.info("  Computing article count spike z-scores...")
    for domain in DOMAINS:
        col = f"article_count_{domain}"
        if col in df.columns:
            rolling_mean = df[col].rolling(20, min_periods=3).mean()
            rolling_std  = df[col].rolling(20, min_periods=3).std().replace(0, np.nan)
            df[f"article_spike_{domain}"] = (
                ((df[col] - rolling_mean) / rolling_std) > 2
            ).astype(int)

    # ── 4. Shock indicator (rolling z-score on portfolio vol) ────────────────
    log.info("  Deriving shock flags per portfolio...")
    SHOCK_Z = 2.0
    for pname in PORTFOLIOS:
        col = f"vol5_{pname}"
        if col in df.columns:
            roll_mean = df[col].rolling(20, min_periods=5).mean()
            roll_std  = df[col].rolling(20, min_periods=5).std().replace(0, np.nan)
            df[f"shock_{pname}"] = (
                ((df[col] - roll_mean) / roll_std) > SHOCK_Z
            ).astype(int)

    # ── 5. Z-score normalise sentiment features ──────────────────────────────
    log.info("  Z-score normalising sentiment features...")
    sent_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["sentiment_score_", "avg_prob_neg_", "avg_prob_pos_",
                                   "avg_prob_neu_", "lag"]
    )]
    for col in sent_cols:
        mu, sigma = df[col].mean(), df[col].std()
        if sigma > 0:
            df[f"z_{col}"] = (df[col] - mu) / sigma

    log.info("  Feature engineering complete. Shape: %d rows × %d cols",
             *df.shape)
    return df


# ══════════════════════════════════════════════
# DATA PREP HELPERS
# ══════════════════════════════════════════════

def time_series_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological split — NO shuffling (prevents data leakage)."""
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def get_feature_columns(df: pd.DataFrame, patterns: list[str]) -> list[str]:
    """Return column names that start with any of the given prefixes."""
    return [c for c in df.columns if any(c.startswith(p) for p in patterns)]


# ══════════════════════════════════════════════
# M1 — SHOCK CLASSIFIER (XGBoost)
# ══════════════════════════════════════════════

def train_m1_shock_classifier(df: pd.DataFrame, results: dict) -> None:
    """
    Binary classifier: predict if tomorrow will be a shock day.
    Target: shock_tech (or shock_geopolitical) based on vol z-score > 2.
    """
    log.info("── [M1] Shock Classifier (XGBoost) ────────────────────")

    try:
        from xgboost import XGBClassifier
        from sklearn.metrics import classification_report, roc_auc_score
    except ImportError:
        log.error("  xgboost not installed. Run: pip install xgboost")
        results["M1"] = "SKIPPED — xgboost not installed"
        return

    TARGET = "shock_tech"
    if TARGET not in df.columns:
        log.warning("  Target column '%s' missing. Skipping M1.", TARGET)
        results["M1"] = f"SKIPPED — {TARGET} column missing"
        return

    # Features
    feature_prefixes = [
        "z_sentiment_score_", "z_avg_prob_neg_",
        "z_lag1_", "z_lag2_", "z_lag3_",
        "article_spike_",
        "vol5_tech", "vol10_tech", "vol20_tech",
        "ret_tech",
    ]
    feat_cols = [c for c in df.columns if any(c.startswith(p) for p in feature_prefixes)]

    # Use next-day shock as target (shift target back by 1 for prediction)
    df_m1 = df[feat_cols + [TARGET]].copy()
    df_m1[TARGET] = df_m1[TARGET].shift(-1)  # predict TOMORROW's shock
    df_m1 = df_m1.dropna()

    if len(df_m1) < 10:
        log.warning("  Insufficient rows after dropna (%d). Skipping M1.", len(df_m1))
        results["M1"] = f"SKIPPED — only {len(df_m1)} valid rows"
        return

    X = df_m1[feat_cols]
    y = df_m1[TARGET].astype(int)
    X_train, X_test = time_series_split(X)
    y_train, y_test = time_series_split(y.to_frame())
    y_train, y_test = y_train.squeeze(), y_test.squeeze()

    # Handle class imbalance
    pos_weight = max(1, (y_train == 0).sum() / max((y_train == 1).sum(), 1))

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, zero_division=0)
    auc    = roc_auc_score(y_test, y_prob) if len(y_test.unique()) > 1 else float("nan")

    log.info("  M1 AUC: %.4f\n%s", auc, report)

    # Save model + feature importances
    joblib.dump(model, MODELS_DIR / "m1_shock_classifier.pkl")
    fi = pd.DataFrame({"feature": feat_cols, "importance": model.feature_importances_})
    fi.sort_values("importance", ascending=False).to_csv(
        MODELS_DIR / "m1_feature_importance.csv", index=False
    )

    results["M1"] = {
        "AUC":    round(auc, 4),
        "report": report,
        "top_features": fi.head(5).to_dict("records"),
    }
    log.info("  ✔ M1 saved → models/ml/m1_shock_classifier.pkl")


# ══════════════════════════════════════════════
# M2 — RECOVERY PREDICTOR (Gradient Boosting)
# ══════════════════════════════════════════════

def train_m2_recovery_predictor(df: pd.DataFrame, results: dict) -> None:
    """
    Regression: how many days until vol returns to normal after a shock?
    Only runs on rows that are shock days.
    """
    log.info("── [M2] Recovery Predictor (GBR) ──────────────────────")

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    TARGET_SHOCK = "shock_tech"
    VOL_COL      = "vol5_tech"

    if TARGET_SHOCK not in df.columns or VOL_COL not in df.columns:
        log.warning("  Required columns missing. Skipping M2.")
        results["M2"] = "SKIPPED — shock_tech or vol5_tech missing"
        return

    # Build recovery_days label: from each shock row, count days until vol
    # drops below rolling mean again.
    log.info("  Computing recovery day labels...")
    df_m2 = df[[VOL_COL, TARGET_SHOCK]].copy().dropna().reset_index(drop=True)
    roll_mean = df_m2[VOL_COL].rolling(20, min_periods=5).mean()

    recovery_days = []
    for i in range(len(df_m2)):
        if df_m2[TARGET_SHOCK].iloc[i] == 1:
            days = 0
            for j in range(i + 1, len(df_m2)):
                if df_m2[VOL_COL].iloc[j] < roll_mean.iloc[j]:
                    break
                days += 1
            recovery_days.append(days)
        else:
            recovery_days.append(np.nan)

    df_m2["recovery_days"] = recovery_days
    df_m2 = df_m2[df_m2[TARGET_SHOCK] == 1].dropna(subset=["recovery_days"])

    if len(df_m2) < 5:
        log.warning("  Only %d shock events found — not enough to train M2.", len(df_m2))
        results["M2"] = f"SKIPPED — only {len(df_m2)} shock events (need ≥ 5)"
        return

    # Features for shock rows
    feature_prefixes = [
        "z_sentiment_score_", "z_avg_prob_neg_",
        "z_lag1_", "article_spike_",
        "vol5_", "vol10_", "vol20_",
    ]
    feat_cols = [c for c in df.columns if any(c.startswith(p) for p in feature_prefixes)]
    df_feat   = df.iloc[df_m2.index if hasattr(df_m2.index, '__iter__') else df_m2.index.tolist()]
    df_feat   = df.loc[df_m2.index, feat_cols].copy() if set(df_m2.index).issubset(set(df.index)) else df[feat_cols].iloc[:len(df_m2)]

    X = df_feat.fillna(df_feat.median())
    y = df_m2["recovery_days"].values

    if len(X) < 5:
        results["M2"] = "SKIPPED — insufficient aligned shock rows"
        return

    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    model = GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred) if len(y_test) > 1 else float("nan")
        log.info("  M2 MAE: %.2f days | R²: %.4f", mae, r2)
        results["M2"] = {"MAE_days": round(mae, 2), "R2": round(r2, 4),
                         "n_shock_events": len(df_m2)}
    else:
        log.info("  M2 trained (no test set — too few shock events).")
        results["M2"] = {"note": "Trained on all shock events (no test split)",
                         "n_shock_events": len(df_m2)}

    joblib.dump(model, MODELS_DIR / "m2_recovery_predictor.pkl")
    log.info("  ✔ M2 saved → models/ml/m2_recovery_predictor.pkl")


# ══════════════════════════════════════════════
# M3 — PORTFOLIO RISK SCORER (Ridge + MLP)
# ══════════════════════════════════════════════

def train_m3_risk_scorer(df: pd.DataFrame, results: dict) -> None:
    """
    Regression: predict portfolio-level volatility from sentiment features.
    Trains Ridge (interpretable) and MLP (non-linear) and compares.
    """
    log.info("── [M3] Portfolio Risk Scorer (Ridge + MLP) ───────────")

    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler

    results["M3"] = {}

    for pname in PORTFOLIOS:
        target_col = f"vol5_{pname}"
        if target_col not in df.columns:
            continue

        feature_prefixes = [
            "z_sentiment_score_", "z_avg_prob_neg_", "z_avg_prob_pos_",
            "z_lag1_sentiment_score_", "z_lag2_sentiment_score_",
            "article_spike_", "article_count_",
        ]
        feat_cols = [c for c in df.columns if any(c.startswith(p) for p in feature_prefixes)]

        df_m3 = df[feat_cols + [target_col]].dropna()
        if len(df_m3) < 10:
            continue

        X = df_m3[feat_cols].values
        y = df_m3[target_col].values

        split   = max(1, int(len(X) * 0.8))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_preds = ridge.predict(X_test) if len(X_test) else ridge.predict(X_train)
        ridge_y     = y_test if len(X_test) else y_train
        ridge_mae   = mean_absolute_error(ridge_y, ridge_preds)
        ridge_r2    = r2_score(ridge_y, ridge_preds) if len(ridge_y) > 1 else float("nan")

        # MLP
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", max_iter=500,
            learning_rate_init=0.001, random_state=42, early_stopping=True
        )
        mlp.fit(X_train, y_train)
        mlp_preds   = mlp.predict(X_test) if len(X_test) else mlp.predict(X_train)
        mlp_mae     = mean_absolute_error(ridge_y, mlp_preds)
        mlp_r2      = r2_score(ridge_y, mlp_preds) if len(ridge_y) > 1 else float("nan")

        nonlinear = (mlp_r2 - ridge_r2) > 0.05

        log.info("  %s | Ridge R²=%.3f MAE=%.4f | MLP R²=%.3f MAE=%.4f | Non-linear: %s",
                 pname, ridge_r2, ridge_mae, mlp_r2, mlp_mae, nonlinear)

        joblib.dump({"model": ridge, "scaler": scaler, "features": feat_cols},
                    MODELS_DIR / f"m3_ridge_{pname}.pkl")
        joblib.dump({"model": mlp,   "scaler": scaler, "features": feat_cols},
                    MODELS_DIR / f"m3_mlp_{pname}.pkl")

        results["M3"][pname] = {
            "Ridge": {"R2": round(ridge_r2, 4), "MAE": round(ridge_mae, 6)},
            "MLP":   {"R2": round(mlp_r2, 4),   "MAE": round(mlp_mae, 6)},
            "relationship_is_nonlinear": bool(nonlinear),
        }

    log.info("  ✔ M3 saved for all portfolios.")


# ══════════════════════════════════════════════
# M4 — CROSS-DOMAIN LAG PREDICTOR (Granger + VAR)
# ══════════════════════════════════════════════

def train_m4_cross_domain_lag(df: pd.DataFrame, results: dict,
                               skip: bool = False) -> None:
    """
    Granger Causality: does domain A sentiment Granger-cause domain B volatility?
    + VAR model for joint lag estimation.
    """
    log.info("── [M4] Cross-Domain Lag Predictor (Granger + VAR) ────")

    if skip:
        log.info("  Skipped (--skip-granger flag set).")
        results["M4"] = "SKIPPED by user flag"
        return

    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        from statsmodels.tsa.api import VAR
    except ImportError:
        log.error("  statsmodels not installed. Run: pip install statsmodels")
        results["M4"] = "SKIPPED — statsmodels not installed"
        return

    MAX_LAG = 5
    granger_rows = []

    domain_pairs = [
        ("geopolitical", "tech"),
        ("financial",    "tech"),
        ("geopolitical", "balanced"),
        ("technology",   "geopolitical"),
    ]

    for cause_domain, effect_pf in domain_pairs:
        cause_col  = f"sentiment_score_{cause_domain}"
        effect_col = f"vol5_{effect_pf}"
        if cause_col not in df.columns or effect_col not in df.columns:
            continue

        pair_df = df[[cause_col, effect_col]].dropna()
        if len(pair_df) < MAX_LAG * 3 + 5:
            log.warning("  Skipping %s→%s (too few rows: %d)", cause_domain, effect_pf, len(pair_df))
            continue

        try:
            test = grangercausalitytests(pair_df[[effect_col, cause_col]],
                                         maxlag=MAX_LAG, verbose=False)
            for lag in range(1, MAX_LAG + 1):
                p_val = test[lag][0]["ssr_ftest"][1]
                granger_rows.append({
                    "cause":  cause_domain,
                    "effect": effect_pf,
                    "lag":    lag,
                    "p_value": round(p_val, 4),
                    "significant": p_val < 0.05,
                })
        except Exception as e:
            log.warning("  Granger test failed for %s→%s: %s", cause_domain, effect_pf, e)

    if granger_rows:
        granger_df = pd.DataFrame(granger_rows)
        granger_df.to_csv(MODELS_DIR / "m4_granger_results.csv", index=False)
        sig = granger_df[granger_df["significant"]]
        log.info("  Significant Granger causalities found:\n%s",
                 sig.to_string(index=False) if len(sig) else "  None at p<0.05")
        results["M4"] = {
            "significant_pairs": sig[["cause", "effect", "lag", "p_value"]].to_dict("records"),
            "total_tested": len(granger_rows),
        }
    else:
        results["M4"] = "No pairs had sufficient data"

    # VAR model on available sentiment+vol columns
    var_cols = [c for c in df.columns if ("sentiment_score_" in c or "vol5_" in c)
                and not c.startswith("lag")]
    var_df   = df[var_cols].dropna()

    if len(var_df) >= 20:
        try:
            var_model = VAR(var_df)
            fitted    = var_model.fit(maxlags=5, ic="aic")
            joblib.dump(fitted, MODELS_DIR / "m4_var_model.pkl")
            log.info("  VAR model fitted (selected lag=%d by AIC).", fitted.k_ar)
            results["M4"]["VAR_lag_order"] = fitted.k_ar
        except Exception as e:
            log.warning("  VAR fitting failed: %s", e)

    log.info("  ✔ M4 Granger results saved → models/ml/m4_granger_results.csv")


# ══════════════════════════════════════════════
# M5 — SENTIMENT-VOLATILITY REGRESSION (Lasso)
# ══════════════════════════════════════════════

def train_m5_sentiment_vol_regression(df: pd.DataFrame, results: dict) -> None:
    """
    Lasso regression: which domain's sentiment best predicts each portfolio's vol?
    Non-zero Lasso coefficients reveal the most predictive domains.
    """
    log.info("── [M5] Sentiment-Volatility Regression (Lasso) ───────")

    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    lasso_rows  = []
    results["M5"] = {}

    # Lasso feature columns: raw sentiment scores + probs per domain
    feat_cols = [c for c in df.columns if any(
        c.startswith(p) for p in [
            "sentiment_score_", "avg_prob_neg_", "avg_prob_pos_",
            "lag1_sentiment_score_", "article_count_",
        ]
    )]

    for pname in PORTFOLIOS:
        target_col = f"vol5_{pname}"
        if target_col not in df.columns:
            continue

        sub = df[feat_cols + [target_col]].dropna()
        if len(sub) < 10:
            continue

        X = sub[feat_cols].values
        y = sub[target_col].values

        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lasso = LassoCV(cv=min(5, len(sub)), random_state=42, max_iter=5000)
        lasso.fit(X_scaled, y)

        y_pred = lasso.predict(X_scaled)
        r2     = r2_score(y, y_pred) if len(y) > 1 else float("nan")

        coef_df = pd.DataFrame({
            "portfolio": pname,
            "feature":   feat_cols,
            "coefficient": lasso.coef_,
        })
        lasso_rows.append(coef_df)

        nonzero = coef_df[coef_df["coefficient"] != 0].sort_values(
            "coefficient", key=abs, ascending=False
        )
        log.info("  %s → R²=%.4f | Non-zero features: %d/%d",
                 pname, r2, len(nonzero), len(feat_cols))
        results["M5"][pname] = {
            "R2": round(r2, 4),
            "alpha": round(lasso.alpha_, 6),
            "top_predictors": nonzero.head(5)[["feature", "coefficient"]].to_dict("records"),
        }

    if lasso_rows:
        all_coefs = pd.concat(lasso_rows, ignore_index=True)
        all_coefs.to_csv(MODELS_DIR / "m5_lasso_coefficients.csv", index=False)
        log.info("  ✔ M5 Lasso coefficients saved → models/ml/m5_lasso_coefficients.csv")


# ══════════════════════════════════════════════
# M6 — PORTFOLIO CLUSTERING (KMeans + HDBSCAN)
# ══════════════════════════════════════════════

def train_m6_portfolio_clustering(df: pd.DataFrame, results: dict) -> None:
    """
    Cluster portfolios by their sentiment-sensitivity profile.
    Features: average beta to each domain, shock frequency, avg recovery time.
    """
    log.info("── [M6] Portfolio Clustering (KMeans + HDBSCAN) ───────")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    portfolio_profiles = []

    for pname, tickers in PORTFOLIOS.items():
        row = {"portfolio": pname}

        # Mean vol for this portfolio
        vol_col = f"vol5_{pname}"
        if vol_col in df.columns:
            row["mean_vol"] = df[vol_col].mean()

        # Shock frequency
        shock_col = f"shock_{pname}"
        if shock_col in df.columns:
            row["shock_freq"] = df[shock_col].mean()

        # Beta to each domain: correlation of portfolio return with domain sentiment
        ret_col = f"ret_{pname}"
        for domain in DOMAINS:
            sent_col = f"sentiment_score_{domain}"
            if ret_col in df.columns and sent_col in df.columns:
                pair = df[[ret_col, sent_col]].dropna()
                if len(pair) > 3:
                    r, _ = stats.pearsonr(pair[ret_col], pair[sent_col])
                    row[f"beta_{domain}"] = round(r, 4)

        portfolio_profiles.append(row)

    profile_df = pd.DataFrame(portfolio_profiles).set_index("portfolio")
    profile_df = profile_df.dropna(how="all").fillna(0)

    if len(profile_df) < 2:
        log.warning("  Not enough portfolios for clustering. Skipping M6.")
        results["M6"] = "SKIPPED — fewer than 2 portfolios with data"
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profile_df.values)

    # KMeans
    n_clusters = min(3, len(profile_df))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    profile_df["kmeans_cluster"] = km_labels

    CLUSTER_LABELS = {0: "Safe-Haven Heavy", 1: "Tech-Reactive", 2: "Balanced/Defensive"}
    profile_df["kmeans_label"] = [CLUSTER_LABELS.get(l, f"Cluster-{l}") for l in km_labels]

    # HDBSCAN (optional)
    try:
        import hdbscan
        hdb = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
        hdb_labels = hdb.fit_predict(X_scaled)
        profile_df["hdbscan_cluster"] = hdb_labels
    except ImportError:
        log.info("  hdbscan not installed — skipping HDBSCAN (KMeans only).")
        log.info("  Install with: pip install hdbscan")

    profile_df.to_csv(MODELS_DIR / "m6_clusters.csv")
    log.info("  Cluster assignments:\n%s", profile_df[["kmeans_label"]].to_string())
    log.info("  ✔ M6 saved → models/ml/m6_clusters.csv")

    results["M6"] = {
        "assignments": profile_df["kmeans_label"].to_dict(),
        "profiles": profile_df.drop(columns=["kmeans_label"], errors="ignore").to_dict(),
    }


# ══════════════════════════════════════════════
# SAVE RESULTS SUMMARY
# ══════════════════════════════════════════════

def save_results_summary(results: dict) -> None:
    """Write a plain-text human-readable summary of all model results."""
    lines = [
        "=" * 65,
        "  ML MODELLING RESULTS SUMMARY — Portfolio Risk Analytics",
        "=" * 65,
        "",
    ]

    for model_id, data in results.items():
        lines.append(f"{'─' * 65}")
        lines.append(f"  {model_id}")
        lines.append(f"{'─' * 65}")
        if isinstance(data, str):
            lines.append(f"  Status: {data}")
        elif isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, dict):
                    lines.append(f"  {key}:")
                    for k2, v2 in val.items():
                        lines.append(f"    {k2}: {v2}")
                else:
                    lines.append(f"  {key}: {val}")
        lines.append("")

    RESULTS_TXT.write_text("\n".join(lines), encoding="utf-8")
    log.info("  ✔ Results summary saved → reports/ml_results_summary.txt")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def run(skip_granger: bool = False):
    log.info("=" * 65)
    log.info("  ML MODELLING PIPELINE  |  Portfolio Risk Analytics")
    log.info("=" * 65)

    # ── Load data
    if not MASTER_DATA.exists():
        raise FileNotFoundError(f"Missing: {MASTER_DATA}\nRun: python pipeline/clean_data.py")

    df_raw = pd.read_csv(MASTER_DATA, parse_dates=["date"])
    df_raw = df_raw.set_index("date").sort_index()
    log.info("  Loaded master_data: %d rows × %d cols", *df_raw.shape)

    # ── Feature Engineering
    df = engineer_features(df_raw)
    df.reset_index().to_csv(FEATURES_OUT, index=False)
    log.info("  ML features saved → data/processed/ml_features.csv")

    # ── Train all models
    results = {}
    train_m1_shock_classifier(df, results)
    train_m2_recovery_predictor(df, results)
    train_m3_risk_scorer(df, results)
    train_m4_cross_domain_lag(df, results, skip=skip_granger)
    train_m5_sentiment_vol_regression(df, results)
    train_m6_portfolio_clustering(df, results)

    # ── Save summary
    save_results_summary(results)

    log.info("")
    log.info("=" * 65)
    log.info("  ALL MODELS COMPLETE")
    log.info("  Models → %s", MODELS_DIR.resolve())
    log.info("  Summary → %s", RESULTS_TXT.resolve())
    log.info("=" * 65)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train all 6 ML models for portfolio risk analytics."
    )
    parser.add_argument(
        "--skip-granger", action="store_true",
        help="Skip M4 Granger causality tests (faster runs during development)"
    )
    args = parser.parse_args()
    run(skip_granger=args.skip_granger)
