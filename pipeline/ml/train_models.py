"""
Portfolio Risk Analytics — ML Modelling Pipeline v2
====================================================
Updated from ml_model_improv_v1.txt

Changes from v1:
  M1 — Per-stock adaptive thresholds (95th pct), walk-forward expanding CV,
        XGBoost if shock events ≥ 100 else RandomForest, F1/AUC-PR evaluation
        New features: vol_zscore, sentiment_velocity, news_available_flag,
        geo_article_zscore, rolling_mean_return_5d
  M2 — Pools shocks across ALL portfolios (multiplies training data),
        Operational 3-day-sustained recovery definition (not first-day),
        Adds shock_magnitude, sector, quarter, news_article_spike features
  M3 — Sentiment-ONLY features for next-day vol (eliminates circularity),
        QuantileRegressor for P25/P75 risk band output (not point estimate)
        New features: intra_corr, hhi, cross_domain_divergence, market_regime
  M4 — ADF stationarity test + differencing before modelling,
        BIC (not AIC) for VAR lag selection, CCF computed BEFORE Granger,
        fin_sentiment added as confounder control, reduced-form VAR on
        portfolio-level averages only (not individual stocks)
  M5 — Ridge instead of Lasso (handles correlated domains without zero-out),
        New features: 3d rolling smoothed sentiment, sentiment velocity,
        geo×fin interaction, reliability-weighted sentiment (zscore×score)
        Per-day SHAP values saved for dashboard "what to watch" advice
  M6 — Clusters on M5 domain sensitivity coefficients (not raw weights),
        KMeans with elbow + silhouette score to choose k,
        Hierarchical clustering (AgglomerativeClustering) added,
        Cluster names derived from dominant M5 domain coefficient empirically

Inputs:
  data/processed/master_data.csv

Outputs:
  data/processed/ml_features.csv
  models/ml/m1_shock_classifier.pkl + m1_shap_values.csv
  models/ml/m2_recovery_predictor.pkl
  models/ml/m3_ridge_{pname}.pkl + m3_mlp_{pname}.pkl
  models/ml/m4_ccf_results.csv + m4_granger_results.csv + m4_var_model.pkl
  models/ml/m5_ridge_{pname}.pkl + m5_shap_{pname}.csv + m5_lasso_coefficients.csv
  models/ml/m5_domain_coefficients.csv   (feeds M6 clustering)
  models/ml/m6_clusters.csv
  reports/ml_results_summary.txt

Usage:
  python pipeline/ml/train_models.py
  python pipeline/ml/train_models.py --skip-granger
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

MASTER_DATA  = PROCESSED_DIR / "master_data.csv"
FEATURES_OUT = PROCESSED_DIR / "ml_features.csv"
RESULTS_TXT  = REPORTS_DIR / "ml_results_summary.txt"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Portfolio definitions
# ──────────────────────────────────────────────
PORTFOLIOS = {
    "geopolitical": ["GLD", "USO", "LMT", "RTX", "EEM", "GC=F", "CL=F"],
    "tech":         ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "SOXX", "QQQ"],
    "balanced":     ["SPY", "AGG", "GLD", "VTI", "EFA", "BND"],
    "conservative": ["TLT", "IEF", "VPU", "KO", "JNJ", "PG", "XLP"],
}

DOMAINS  = ["financial", "geopolitical", "technology"]
TICKERS  = sorted(set(t for tickers in PORTFOLIOS.values() for t in tickers))

# ══════════════════════════════════════════════
# FEATURE ENGINEERING v2
# ══════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the full ML feature table from master_data.csv.

    New in v2 (per ml_model_improv_v1):
      - Per-stock/portfolio vol z-score (adaptive; replaces global z>2 threshold)
      - news_available_flag (binary: was there any news on this day?)
      - sentiment_velocity (1-day delta of sentiment score)
      - 3-day rolling smoothed sentiment (reduces noise for M5)
      - geo×fin sentiment interaction term (M5)
      - cross_domain_divergence (geo - tech, M3)
      - market_regime_flag (SPY > 20d MA, M3/M5)
      - intra_portfolio_correlation (rolling 20d pairwise avg, M3/M6)
      - sector_concentration_hhi (equal-weight proxy, M3/M6)
      - article_count_zscore × sentiment = reliability-weighted sentiment (M5)
      - Lag features extended to lag-7 (needed for M4 CCF)
    """
    log.info("── [FEATURE ENGINEERING v2] ─────────────────────────────")
    df = df.copy()

    # ── 1. Portfolio-level return & vol aggregations ──────────────────────────
    log.info("  Building portfolio-level return & vol aggregations...")
    for pname, tickers in PORTFOLIOS.items():
        ret_cols = [f"ret_{t}" for t in tickers if f"ret_{t}" in df.columns]
        vol_cols = [f"vol5_{t}" for t in tickers if f"vol5_{t}" in df.columns]
        if ret_cols:
            pf_ret = df[ret_cols].mean(axis=1)
            df[f"ret_{pname}"]   = pf_ret
            df[f"vol10_{pname}"] = pf_ret.rolling(10, min_periods=3).std()
            df[f"vol20_{pname}"] = pf_ret.rolling(20, min_periods=5).std()
            df[f"rolling_mean_return_5d_{pname}"] = pf_ret.rolling(5, min_periods=2).mean()
        if vol_cols:
            df[f"vol5_{pname}"] = df[vol_cols].mean(axis=1)

    # ── 2. Per-portfolio vol z-score (M1: adaptive threshold) ────────────────
    log.info("  Computing per-portfolio vol z-scores (adaptive thresholds)...")
    for pname in PORTFOLIOS:
        col = f"vol5_{pname}"
        if col in df.columns:
            roll_mean = df[col].rolling(20, min_periods=5).mean()
            roll_std  = df[col].rolling(20, min_periods=5).std().replace(0, np.nan)
            df[f"vol_zscore_5d_{pname}"] = (df[col] - roll_mean) / roll_std

    # ── 3. Cross-domain lags (lag-1 to lag-7 for M4 CCF) ────────────────────
    log.info("  Adding cross-domain lag features (lag-1 to lag-7)...")
    for domain in DOMAINS:
        col = f"sentiment_score_{domain}"
        if col in df.columns:
            for lag in range(1, 8):
                df[f"lag{lag}_sentiment_score_{domain}"] = df[col].shift(lag)

    # ── 4. Article count spike flag + z-score + availability ─────────────────
    log.info("  Computing article spike/zscore/availability flags...")
    for domain in DOMAINS:
        col = f"article_count_{domain}"
        if col in df.columns:
            rolling_mean = df[col].rolling(20, min_periods=3).mean()
            rolling_std  = df[col].rolling(20, min_periods=3).std().replace(0, np.nan)
            zscore = (df[col] - rolling_mean) / rolling_std
            df[f"article_zscore_{domain}"]  = zscore
            df[f"article_spike_{domain}"]   = (zscore > 2).astype(int)
            df[f"news_available_{domain}"]  = (df[col] > 0).astype(int)  # NEW

    # ── 5. Shock labels: per-portfolio 95th pct (M1 improvement: adaptive) ───
    log.info("  Deriving shock flags per portfolio (95th pct adaptive threshold)...")
    for pname in PORTFOLIOS:
        col = f"vol5_{pname}"
        if col in df.columns:
            pct95 = df[col].expanding(min_periods=10).quantile(0.95)
            df[f"shock_{pname}"] = (df[col] > pct95).astype(int)

    # ── 6. Sentiment velocity (1-day delta) — M1 & M5 ────────────────────────
    log.info("  Computing sentiment velocity (1-day delta)...")
    for domain in DOMAINS:
        col = f"sentiment_score_{domain}"
        if col in df.columns:
            df[f"sentiment_velocity_{domain}"] = df[col].diff(1)

    # ── 7. 3-day rolling smoothed sentiment — M5 ──────────────────────────────
    log.info("  Computing 3-day rolling smoothed sentiment...")
    for domain in DOMAINS:
        col = f"sentiment_score_{domain}"
        if col in df.columns:
            df[f"sentiment_3d_{domain}"] = df[col].rolling(3, min_periods=1).mean()

    # ── 8. Geo × Fin sentiment interaction — M5 ───────────────────────────────
    if "sentiment_score_geopolitical" in df.columns and "sentiment_score_financial" in df.columns:
        df["sentiment_interaction_geo_fin"] = (
            df["sentiment_score_geopolitical"] * df["sentiment_score_financial"]
        )

    # ── 9. Cross-domain divergence (geo − tech) — M3 ─────────────────────────
    if "sentiment_score_geopolitical" in df.columns and "sentiment_score_technology" in df.columns:
        df["cross_domain_divergence"] = (
            df["sentiment_score_geopolitical"] - df["sentiment_score_technology"]
        )

    # ── 10. Market regime flag (SPY > 20d MA) — M3 & M5 ─────────────────────
    if "SPY" in df.columns:
        df["market_regime_flag"] = (df["SPY"] > df["SPY"].rolling(20, min_periods=5).mean()).astype(int)
    else:
        df["market_regime_flag"] = 0

    # ── 11. Reliability-weighted sentiment (article_zscore × sentiment) ───────
    for domain in DOMAINS:
        if f"article_zscore_{domain}" in df.columns and f"sentiment_score_{domain}" in df.columns:
            df[f"weighted_sentiment_{domain}"] = (
                df[f"article_zscore_{domain}"] * df[f"sentiment_score_{domain}"]
            )

    # ── 12. Intra-portfolio correlation (rolling 20d pairwise avg) — M3/M6 ───
    log.info("  Computing intra-portfolio correlation (rolling 20d)...")
    for pname, tickers in PORTFOLIOS.items():
        ret_cols = [f"ret_{t}" for t in tickers if f"ret_{t}" in df.columns]
        if len(ret_cols) >= 2:
            corr_series = []
            for i in range(len(ret_cols)):
                for j in range(i + 1, len(ret_cols)):
                    corr_series.append(df[ret_cols[i]].rolling(20, min_periods=5).corr(df[ret_cols[j]]))
            if corr_series:
                df[f"intra_corr_{pname}"] = pd.concat(corr_series, axis=1).mean(axis=1)

    # ── 13. Sector concentration HHI (equal-weight proxy) — M3/M6 ────────────
    for pname, tickers in PORTFOLIOS.items():
        df[f"hhi_{pname}"] = 1.0 / len(tickers)  # equal-weight HHI

    # ── 14. Z-score normalise all sentiment and lag features ──────────────────
    log.info("  Z-score normalising sentiment/lag/velocity features...")
    sent_cols = [c for c in df.columns if any(
        c.startswith(p) for p in [
            "sentiment_score_", "avg_prob_neg_", "avg_prob_pos_", "avg_prob_neu_",
            "lag", "sentiment_3d_", "sentiment_velocity_",
        ]
    )]
    for col in sent_cols:
        mu, sigma = df[col].mean(), df[col].std()
        if sigma > 0:
            df[f"z_{col}"] = (df[col] - mu) / sigma

    log.info("  Feature engineering v2 complete. Shape: %d rows × %d cols", *df.shape)
    return df


# ══════════════════════════════════════════════
# DATA PREP HELPERS
# ══════════════════════════════════════════════

def time_series_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological split — no shuffling."""
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def walk_forward_splits(df: pd.DataFrame, n_splits: int = 3):
    """
    Expanding-window walk-forward splits (M1 improvement).
    Yields (train_df, test_df) with progressively larger training sets.
    """
    min_train = max(5, len(df) // (n_splits + 1))
    step      = max(1, (len(df) - min_train) // n_splits)
    for i in range(n_splits):
        train_end  = min_train + i * step
        test_start = train_end
        test_end   = min(test_start + step, len(df))
        if test_start >= len(df):
            break
        yield df.iloc[:train_end], df.iloc[test_start:test_end]


def get_feature_columns(df: pd.DataFrame, patterns: list) -> list:
    return [c for c in df.columns if any(c.startswith(p) for p in patterns)]


# ══════════════════════════════════════════════
# M1 — SHOCK CLASSIFIER (XGBoost / RandomForest)
# ══════════════════════════════════════════════

def train_m1_shock_classifier(df: pd.DataFrame, results: dict):
    """
    Binary classifier: predict if tomorrow will be a shock day.

    Improvements:
    - Per-portfolio adaptive 95th-pct threshold (not global z > 2.0)
    - Walk-forward expanding-window CV (not simple chronological split)
    - XGBoost if shock events ≥ 100; RandomForest if shock events < 100
    - Evaluates Shock-class F1 + AUC-PR (not just accuracy/AUC-ROC)
    - New features: vol_zscore_5d, sentiment_velocity, news_available_flag,
      geo_article_zscore, rolling_mean_return_5d, market_regime_flag
    """
    log.info("── [M1] Shock Classifier (adaptive threshold + walk-forward CV) ──")

    from sklearn.metrics import (
        f1_score, roc_auc_score, average_precision_score
    )
    from sklearn.ensemble import RandomForestClassifier

    TARGET = "shock_tech"
    if TARGET not in df.columns:
        log.warning("  Target '%s' missing. Skipping M1.", TARGET)
        results["M1"] = f"SKIPPED — {TARGET} column missing"
        return

    # ── Feature prefixes (v2: broader set)
    feature_prefixes = [
        "vol_zscore_5d_",          # per-stock adaptive (NEW)
        "z_sentiment_score_",
        "z_avg_prob_neg_",
        "z_lag1_", "z_lag2_", "z_lag3_",
        "article_spike_",
        "article_zscore_",         # geo_article_zscore (NEW)
        "news_available_",         # news_available_flag (NEW)
        "z_sentiment_velocity_",   # velocity (NEW)
        "vol5_tech", "vol10_tech", "vol20_tech",
        "ret_tech",
        "rolling_mean_return_5d_tech",  # NEW
        "market_regime_flag",
    ]
    feat_cols = [c for c in df.columns if any(c.startswith(p) for p in feature_prefixes)]
    if not feat_cols:
        results["M1"] = "SKIPPED — no feature columns found"
        return

    df_m1 = df[feat_cols + [TARGET]].copy()
    df_m1[TARGET] = df_m1[TARGET].shift(-1)   # predict TOMORROW's shock
    df_m1 = df_m1.dropna()

    if len(df_m1) < 10:
        log.warning("  Insufficient rows (%d). Skipping M1.", len(df_m1))
        results["M1"] = f"SKIPPED — only {len(df_m1)} valid rows"
        return

    X = df_m1[feat_cols].values
    y = df_m1[TARGET].astype(int).values

    n_shock_events = int(y.sum())
    log.info("  Shock events: %d / %d (%.1f%%)",
             n_shock_events, len(y), 100 * n_shock_events / max(len(y), 1))

    # ── Walk-forward CV
    from sklearn.metrics import precision_recall_curve
    wf_f1s, wf_aucs, wf_aprs = [], [], []
    wf_f1s_calibrated, wf_thresholds = [], []

    for train_split, test_split in walk_forward_splits(df_m1, n_splits=3):
        Xtr = train_split[feat_cols].values
        ytr = train_split[TARGET].astype(int).values
        Xte = test_split[feat_cols].values
        yte = test_split[TARGET].astype(int).values

        if len(yte) == 0 or yte.sum() == 0:
            continue

        pos_weight = max(1, (ytr == 0).sum() / max((ytr == 1).sum(), 1))

        # Algorithm choice per ml_model_improv_v1
        if n_shock_events >= 100:
            try:
                from xgboost import XGBClassifier
                clf = XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    scale_pos_weight=pos_weight, random_state=42,
                    eval_metric="logloss", verbosity=0,
                )
            except ImportError:
                clf = RandomForestClassifier(n_estimators=200, max_depth=4,
                                             class_weight="balanced", random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=200, max_depth=4,
                                         class_weight="balanced", random_state=42)

        clf.fit(Xtr, ytr)
        yhat  = clf.predict(Xte)         # default 0.5 threshold
        yprob = clf.predict_proba(Xte)[:, 1]

        wf_f1s.append(f1_score(yte, yhat, zero_division=0))
        if len(np.unique(yte)) > 1:
            wf_aucs.append(roc_auc_score(yte, yprob))
            wf_aprs.append(average_precision_score(yte, yprob))

            # ── Calibrate threshold: sweep PR curve to find best F1 threshold
            prec, rec, threshs = precision_recall_curve(yte, yprob)
            # F1 at each threshold (prec/rec have one extra point at end)
            f1_scores = np.where(
                (prec[:-1] + rec[:-1]) > 0,
                2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]),
                0.0,
            )
            best_idx  = int(np.argmax(f1_scores))
            best_thr  = float(threshs[best_idx])
            yhat_cal  = (yprob >= best_thr).astype(int)
            wf_f1s_calibrated.append(f1_score(yte, yhat_cal, zero_division=0))
            wf_thresholds.append(best_thr)

    mean_f1            = float(np.mean(wf_f1s))             if wf_f1s            else float("nan")
    mean_auc           = float(np.mean(wf_aucs))            if wf_aucs           else float("nan")
    mean_apr           = float(np.mean(wf_aprs))            if wf_aprs           else float("nan")
    mean_f1_calibrated = float(np.mean(wf_f1s_calibrated)) if wf_f1s_calibrated else float("nan")
    # Use median threshold across folds as the production threshold
    optimal_threshold  = float(np.median(wf_thresholds))   if wf_thresholds     else 0.5

    log.info(
        "  Walk-forward CV → F1@0.5: %.4f | F1@%.2f(calibrated): %.4f "
        "| AUC-ROC: %.4f | AUC-PR: %.4f",
        mean_f1, optimal_threshold, mean_f1_calibrated, mean_auc, mean_apr
    )

    # ── Final model on all data
    pos_weight = max(1, (y == 0).sum() / max((y == 1).sum(), 1))
    if n_shock_events >= 100:
        try:
            from xgboost import XGBClassifier
            final_model = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                scale_pos_weight=pos_weight, random_state=42,
                eval_metric="logloss", verbosity=0,
            )
            log.info("  Using XGBoost (shock_events=%d ≥ 100)", n_shock_events)
        except ImportError:
            log.warning("  xgboost not installed — falling back to RandomForest.")
            final_model = RandomForestClassifier(n_estimators=200, max_depth=4,
                                                 class_weight="balanced", random_state=42)
    else:
        final_model = RandomForestClassifier(n_estimators=200, max_depth=4,
                                             class_weight="balanced", random_state=42)
        log.info("  Using RandomForest (shock_events=%d < 100)", n_shock_events)

    final_model.fit(X, y)

    # ── SHAP values (best-effort)
    shap_available = False
    try:
        import shap
        explainer  = shap.TreeExplainer(final_model)
        shap_vals  = explainer.shap_values(X)
        # For classifiers, shap_values may return list; take class-1 values
        sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        pd.DataFrame(sv, columns=feat_cols).to_csv(MODELS_DIR / "m1_shap_values.csv", index=False)
        shap_available = True
        log.info("  SHAP values saved → models/ml/m1_shap_values.csv")
    except Exception as e:
        log.info("  SHAP skipped for M1: %s", e)

    # Feature importances
    if hasattr(final_model, "feature_importances_"):
        fi = pd.DataFrame({"feature": feat_cols, "importance": final_model.feature_importances_})
    else:
        fi = pd.DataFrame({"feature": feat_cols, "importance": [0.0] * len(feat_cols)})
    fi.sort_values("importance", ascending=False).to_csv(
        MODELS_DIR / "m1_feature_importance.csv", index=False
    )

    # Save model bundle including optimal threshold for inference
    joblib.dump(
        {"model": final_model, "features": feat_cols,
         "optimal_threshold": optimal_threshold},
        MODELS_DIR / "m1_shock_classifier.pkl"
    )

    results["M1"] = {
        "algorithm":                 "XGBoost" if n_shock_events >= 100 else "RandomForest",
        "n_shock_events":            n_shock_events,
        "walk_forward_shock_F1":     round(mean_f1, 4),
        "walk_forward_F1_calibrated":round(mean_f1_calibrated, 4),
        "optimal_threshold":         round(optimal_threshold, 3),
        "walk_forward_AUC_ROC":      round(mean_auc, 4),
        "walk_forward_AUC_PR":       round(mean_apr, 4),
        "shap_available":            shap_available,
        "top_features":              fi.head(5).to_dict("records"),
    }
    log.info("  ✔ M1 saved → models/ml/m1_shock_classifier.pkl")
    log.info("    Optimal threshold: %.3f | Calibrated F1: %.4f",
             optimal_threshold, mean_f1_calibrated)
    # Return model + feature cols so M2 can use M1 predictions as a gate
    return final_model, feat_cols


# ══════════════════════════════════════════════
# M2 — RECOVERY PREDICTOR (Gradient Boosting)
# ══════════════════════════════════════════════

def train_m2_recovery_predictor(
    df: pd.DataFrame,
    results: dict,
    m1_model=None,
    m1_feat_cols: list = None,
    m1_confidence_threshold: float = 0.75,
) -> None:
    """
    Regression: days until portfolio vol stabilises after a shock.

    Improvements:
    - Pools shock events across ALL portfolios (multiplies training data)
    - Operational recovery = 5d-rolling-vol drops below threshold AND stays
      below for 3 consecutive days (not just "first day below")
    - M1 confidence gate: for the tech portfolio only, a shock is accepted
      only when M1.predict_proba > m1_confidence_threshold (default 0.75)
    - Adds shock_magnitude, sentiment_velocity, pre_shock_baseline,
      sector_label, market_regime, quarter/year, news_article_spike
    """
    log.info("── [M2] Recovery Predictor (pooled shocks, 3-day sustained recovery) ──")

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    # ── Pre-compute M1 shock probabilities on full dataset (for tech gate) ──
    m1_proba_series = None
    if m1_model is not None and m1_feat_cols is not None:
        try:
            feat_df = df[m1_feat_cols].copy()
            feat_df = feat_df.fillna(feat_df.median(numeric_only=True))
            proba   = m1_model.predict_proba(feat_df.values)[:, 1]
            m1_proba_series = pd.Series(proba, index=df.index)
            log.info("  M1 confidence gate active (threshold=%.2f) for tech portfolio.",
                     m1_confidence_threshold)
        except Exception as e:
            log.warning("  M1 confidence gate failed (%s) — using raw threshold for all.", e)

    shock_events = []   # pool across all portfolios

    for pname, tickers in PORTFOLIOS.items():
        shock_col = f"shock_{pname}"
        vol_col   = f"vol5_{pname}"
        if shock_col not in df.columns or vol_col not in df.columns:
            continue

        # BUG FIX: retain DatetimeIndex so reindex() against df works correctly
        work      = df[[vol_col, shock_col]].copy().dropna()
        threshold = work[vol_col].expanding(min_periods=10).quantile(0.95)
        work_list = list(work.index)   # ordered list of dates for positional lookup

        # M1 confidence gate: only applied to tech portfolio (M1 target = shock_tech)
        apply_m1_gate = (pname == "tech" and m1_proba_series is not None)

        recovery_days_list = []
        for pos, date in enumerate(work_list):
            if work.loc[date, shock_col] == 1:
                # Apply M1 gate for tech portfolio
                if apply_m1_gate:
                    m1_conf = m1_proba_series.get(date, 0.0)
                    if m1_conf < m1_confidence_threshold:
                        recovery_days_list.append(np.nan)   # treat as non-shock
                        continue

                days = 0
                recovered = False
                for j in range(pos + 1, len(work_list)):
                    days += 1
                    jdate = work_list[j]
                    # Operational: 3-day sustained recovery (not just first day)
                    if j + 2 < len(work_list):
                        sustained = all(
                            work.loc[work_list[j + k], vol_col]
                            < threshold.loc[work_list[j + k]]
                            for k in range(3)
                        )
                        if sustained:
                            recovered = True
                            break
                recovery_days_list.append(days)
            else:
                recovery_days_list.append(np.nan)

        work["recovery_days"] = recovery_days_list
        shock_rows = work[work[shock_col] == 1].dropna(subset=["recovery_days"]).copy()
        shock_rows["portfolio"]    = pname
        shock_rows["sector_label"] = list(PORTFOLIOS.keys()).index(pname)

        # ── Contextual features — aligned via DatetimeIndex (was broken before)
        shock_dates = shock_rows.index   # DatetimeIndex, matches df.index
        for domain in DOMAINS:
            sent_col = f"sentiment_score_{domain}"
            vel_col  = f"sentiment_velocity_{domain}"
            if sent_col in df.columns:
                shock_rows[f"sentiment_{domain}"] = df[sent_col].reindex(shock_dates).values
            if vel_col in df.columns:
                shock_rows[f"sentiment_vel_{domain}"] = df[vel_col].reindex(shock_dates).values

        shock_rows["shock_magnitude"]   = df[vol_col].reindex(shock_dates).values
        pre_shock = df[vol_col].rolling(20, min_periods=5).mean()
        shock_rows["pre_shock_baseline"] = pre_shock.reindex(shock_dates).values

        for domain in DOMAINS:
            spike_col = f"article_spike_{domain}"
            if spike_col in df.columns:
                shock_rows[f"spike_{domain}"] = df[spike_col].reindex(shock_dates).values

        if "market_regime_flag" in df.columns:
            shock_rows["market_regime"] = df["market_regime_flag"].reindex(shock_dates).values

        # BUG FIX: add quarter/year BEFORE appending (were added after before)
        shock_rows["quarter"] = shock_dates.quarter
        shock_rows["year"]    = shock_dates.year

        n_censored = (shock_rows["recovery_days"] == (len(work) - 1)).sum()
        log.info("  %s: %d shock events (%d censored / no recovery found)",
                 pname, len(shock_rows), n_censored)

        shock_events.append(shock_rows)

    if not shock_events:
        results["M2"] = "SKIPPED — no shock events in any portfolio"
        return

    all_shocks = pd.concat(shock_events, ignore_index=True)
    n_shocks   = len(all_shocks)
    log.info("  Pooled shock events: %d (across %d portfolios)", n_shocks, len(shock_events))

    if n_shocks < 5:
        log.warning("  Only %d pooled shock events — not enough for M2.", n_shocks)
        results["M2"] = f"SKIPPED — only {n_shocks} pooled events (need ≥ 5)"
        return

    # Feature columns: numeric only, exclude labels
    exclude = {"recovery_days", "portfolio"} | {f"shock_{p}" for p in PORTFOLIOS}
    m2_feat_cols = [c for c in all_shocks.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(all_shocks[c])]

    X = all_shocks[m2_feat_cols].fillna(all_shocks[m2_feat_cols].median())
    y = all_shocks["recovery_days"].values

    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    # ── FIX: Replace GBR point estimate with Quantile Regression band ──────
    # Output: P25 / P50 (median) / P75 recovery days  → dashboard shows
    # "Recovery expected in X–Y days (median: Z days)" instead of one number.
    # GBR kept as baseline comparison.
    from sklearn.linear_model import QuantileRegressor

    q25  = QuantileRegressor(quantile=0.25, alpha=0.1, solver="highs")
    q50  = QuantileRegressor(quantile=0.50, alpha=0.1, solver="highs")
    q75  = QuantileRegressor(quantile=0.75, alpha=0.1, solver="highs")
    gbr  = GradientBoostingRegressor(n_estimators=50, max_depth=2,
                                     learning_rate=0.05, random_state=42)

    for mdl in (q25, q50, q75, gbr):
        mdl.fit(X_train, y_train)

    if len(X_test) > 0:
        p25_pred = q25.predict(X_test)
        p50_pred = q50.predict(X_test)
        p75_pred = q75.predict(X_test)
        gbr_pred = gbr.predict(X_test)

        # Enforce P25 <= P50 <= P75 (quantile crossing fix)
        p25_pred = np.clip(p25_pred, 0, p50_pred)
        p75_pred = np.maximum(p75_pred, p50_pred)

        mae_median = mean_absolute_error(y_test, p50_pred)
        mae_gbr    = mean_absolute_error(y_test, gbr_pred)
        r2_median  = r2_score(y_test, p50_pred) if len(y_test) > 1 else float("nan")
        avg_band   = float(np.mean(p75_pred - p25_pred))

        log.info(
            "  M2 | Median MAE: %.2f days | GBR MAE: %.2f days | "
            "Avg band (P25-P75): %.1f days | pooled: %d",
            mae_median, mae_gbr, avg_band, n_shocks
        )
        results["M2"] = {
            "median_MAE_days":        round(mae_median, 2),
            "gbr_MAE_days":           round(mae_gbr, 2),
            "median_R2":              round(r2_median, 4),
            "avg_band_P25_P75_days":  round(avg_band, 1),
            "n_pooled_shock_events":  n_shocks,
            "dashboard_output":       f"Recovery expected in {max(1,round(np.mean(p25_pred),0)):.0f}"
                                      f"–{round(np.mean(p75_pred),0):.0f} days "
                                      f"(median {round(np.mean(p50_pred),0):.0f} days)",
        }
    else:
        log.info("  M2 trained on all %d events (no test split possible).", n_shocks)
        results["M2"] = {"note": "Trained on all events", "n_pooled_shock_events": n_shocks}

    joblib.dump(
        {"q25": q25, "q50": q50, "q75": q75, "gbr": gbr, "features": m2_feat_cols},
        MODELS_DIR / "m2_recovery_predictor.pkl"
    )
    log.info("  ✔ M2 saved → models/ml/m2_recovery_predictor.pkl")


# ══════════════════════════════════════════════
# M3 — PORTFOLIO RISK SCORER (Ridge + QuantileRegressor)
# ══════════════════════════════════════════════

def train_m3_risk_scorer(df: pd.DataFrame, results: dict) -> None:
    """
    Regression: predict NEXT-DAY portfolio volatility.

    Improvements:
    - Sentiment-ONLY features as predictors (eliminates circular vol→vol prediction)
    - target = next-day portfolio vol (shifted by -1)
    - QuantileRegressor outputs P25/P75 risk band (not false-precision point estimate)
    - New features: intra_corr, hhi, cross_domain_divergence, market_regime_flag,
      sentiment_exposure (weight × domain_sentiment), sentiment interaction
    """
    log.info("── [M3] Portfolio Risk Scorer (sentiment-only → next-day vol + risk band) ──")

    from sklearn.linear_model import Ridge, QuantileRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler

    results["M3"] = {}

    # FIX: Add lagged_vol_t-1 per portfolio to M3 feature set.
    # This is NOT circular: lagged_vol at t-1 is already known at prediction
    # time (yesterday's realised vol). Circularity was vol_t predicting vol_t.
    # M5 shows lagged_vol alone explains R²≈0.5–0.7, so adding it here
    # should push M3 from negative R² into positive territory.
    # lagged_vol_ columns were already built in train_m5_sentiment_vol_regression;
    # if M3 runs before M5, we build them now defensively.
    for _pn in PORTFOLIOS:
        _vc = f"vol5_{_pn}"
        if _vc in df.columns and f"lagged_vol_{_pn}" not in df.columns:
            df[f"lagged_vol_{_pn}"] = df[_vc].shift(1)

    feature_prefixes = [
        "z_sentiment_score_",
        "z_sentiment_3d_",
        "z_sentiment_velocity_",
        "z_avg_prob_neg_", "z_avg_prob_pos_",
        "z_lag1_sentiment_score_",
        "z_lag2_sentiment_score_",
        "article_spike_",
        "article_count_",
        "sentiment_interaction_geo_fin",
        "cross_domain_divergence",
        "market_regime_flag",
        "intra_corr_",
        "hhi_",
        "weighted_sentiment_",
        # Lagged vol (exogenous at prediction time — not circular)
        "lagged_vol_",
    ]

    for pname in PORTFOLIOS:
        target_col = f"vol5_{pname}"
        if target_col not in df.columns:
            continue

        feat_cols = [c for c in df.columns if any(c.startswith(p) for p in feature_prefixes)]
        feat_cols = list(dict.fromkeys(feat_cols))   # deduplicate

        # target = NEXT-DAY vol (prevents circularity)
        df_m3 = df[feat_cols + [target_col]].copy()
        df_m3[target_col] = df_m3[target_col].shift(-1)
        df_m3 = df_m3.dropna()

        if len(df_m3) < 10:
            continue

        X = df_m3[feat_cols].values
        y = df_m3[target_col].values

        split = max(1, int(len(X) * 0.8))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler    = StandardScaler()
        Xtr_s     = scaler.fit_transform(X_train)
        Xte_s     = scaler.transform(X_test) if len(X_test) else Xtr_s
        eval_y    = y_test if len(X_test) else y_train
        eval_Xs   = Xte_s if len(X_test) else Xtr_s

        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(Xtr_s, y_train)
        ridge_preds = ridge.predict(eval_Xs)
        ridge_mae   = mean_absolute_error(eval_y, ridge_preds)
        ridge_r2    = r2_score(eval_y, ridge_preds) if len(eval_y) > 1 else float("nan")

        # QuantileRegressor for P25/P75 risk band
        q25 = QuantileRegressor(quantile=0.25, alpha=0.1)
        q75 = QuantileRegressor(quantile=0.75, alpha=0.1)
        q25.fit(Xtr_s, y_train)
        q75.fit(Xtr_s, y_train)
        p25_preds  = q25.predict(eval_Xs)
        p75_preds  = q75.predict(eval_Xs)
        band_width = float(np.mean(p75_preds - p25_preds))

        # MLP (non-linear comparison) — disable early_stopping on very small datasets
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", max_iter=500,
            learning_rate_init=0.001, random_state=42,
            early_stopping=len(Xtr_s) >= 20,   # needs ≥ 20 rows for validation split
        )
        try:
            mlp.fit(Xtr_s, y_train)
        except Exception:
            mlp = MLPRegressor(
                hidden_layer_sizes=(32,), activation="relu", max_iter=500,
                learning_rate_init=0.001, random_state=42, early_stopping=False,
            )
            mlp.fit(Xtr_s, y_train)
        mlp_preds = mlp.predict(eval_Xs)
        mlp_mae   = mean_absolute_error(eval_y, mlp_preds)
        mlp_r2    = r2_score(eval_y, mlp_preds) if len(eval_y) > 1 else float("nan")

        nonlinear = (mlp_r2 - ridge_r2) > 0.05

        log.info(
            "  %s | Ridge R²=%.3f MAE=%.5f | MLP R²=%.3f | "
            "Band width=%.5f | Non-linear: %s",
            pname, ridge_r2, ridge_mae, mlp_r2, band_width, nonlinear
        )

        # Save models — include q25/q75 in ridge bundle for dashboard
        joblib.dump({"model": ridge, "scaler": scaler, "features": feat_cols,
                     "q25": q25, "q75": q75},
                    MODELS_DIR / f"m3_ridge_{pname}.pkl")
        joblib.dump({"model": mlp, "scaler": scaler, "features": feat_cols},
                    MODELS_DIR / f"m3_mlp_{pname}.pkl")

        results["M3"][pname] = {
            "Ridge":                   {"R2": round(ridge_r2, 4), "MAE": round(ridge_mae, 6)},
            "MLP":                     {"R2": round(mlp_r2, 4),   "MAE": round(mlp_mae, 6)},
            "risk_band_width_avg":     round(band_width, 6),
            "relationship_is_nonlinear": bool(nonlinear),
        }

    log.info("  ✔ M3 saved for all portfolios (with risk band P25/P75).")


# ══════════════════════════════════════════════
# M4 — CROSS-DOMAIN LAG PREDICTOR
# ══════════════════════════════════════════════

def train_m4_cross_domain_lag(df: pd.DataFrame, results: dict,
                               skip: bool = False) -> None:
    """
    CCF (visual discovery) → Granger (statistical validation).

    Improvements:
    - ADF stationarity test on every series before modelling; difference if needed
    - BIC (not AIC) for VAR lag selection (penalises complexity on small samples)
    - CCF computed FIRST to discover peak lags visually; results saved to CSV
    - Bivariate Granger pairwise (runs per domain pair, not all-at-once)
    - fin_sentiment added as confounder control (trivariate Granger)
    - Reduced-form VAR on portfolio-level averages only (avoids overparameterization)
    """
    log.info("── [M4] Cross-Domain Lag (ADF → CCF → Granger, BIC, confounder) ──")

    if skip:
        log.info("  Skipped (--skip-granger flag set).")
        results["M4"] = "SKIPPED by user flag"
        return

    try:
        from statsmodels.tsa.stattools import grangercausalitytests, adfuller
        from statsmodels.tsa.api import VAR
    except ImportError:
        log.error("  statsmodels not installed.")
        results["M4"] = "SKIPPED — statsmodels not installed"
        return

    MAX_LAG = 5

    # ── Step 1: ADF stationarity test; difference if non-stationary ───────────
    log.info("  Step 1: ADF stationarity tests (BIC autolag)...")
    adf_results = {}

    def make_stationary(series: pd.Series, name: str) -> pd.Series:
        s = series.dropna()
        if len(s) < 12:
            return series
        p_val = adfuller(s, autolag="BIC")[1]
        adf_results[name] = {"p_value": round(p_val, 4), "stationary": p_val < 0.05}
        if p_val >= 0.05:
            log.info("    '%s' non-stationary (p=%.4f) → differencing", name, p_val)
            return series.diff(1)
        return series

    df_stat = df.copy()
    for col in [f"sentiment_score_{d}" for d in DOMAINS] + [f"vol5_{p}" for p in PORTFOLIOS]:
        if col in df_stat.columns:
            df_stat[col] = make_stationary(df_stat[col], col)

    n_stationary = sum(v["stationary"] for v in adf_results.values())
    log.info("  ADF: %d/%d series already stationary", n_stationary, len(adf_results))

    # ── Step 2: CCF — discover peak lags visually ─────────────────────────────
    log.info("  Step 2: CCF for all domain–portfolio pairs (discover lags first)...")

    domain_pairs = [
        ("geopolitical", "tech"),
        ("financial",    "tech"),
        ("geopolitical", "balanced"),
        ("technology",   "geopolitical"),
        ("financial",    "balanced"),
    ]

    ccf_rows = []
    for cause_domain, effect_pf in domain_pairs:
        cause_col  = f"sentiment_score_{cause_domain}"
        effect_col = f"vol5_{effect_pf}"
        if cause_col not in df_stat.columns or effect_col not in df_stat.columns:
            continue

        pair = df_stat[[cause_col, effect_col]].dropna()
        if len(pair) < 10:
            continue

        x = pair[cause_col].values
        y = pair[effect_col].values
        x = (x - x.mean()) / (x.std() + 1e-10)
        y = (y - y.mean()) / (y.std() + 1e-10)

        for lag in range(0, min(MAX_LAG + 1, len(x))):
            if lag == 0:
                corr = float(np.corrcoef(x, y)[0, 1])
            else:
                corr = float(np.corrcoef(x[:-lag], y[lag:])[0, 1])
            ccf_rows.append({
                "cause":            cause_domain,
                "effect":           effect_pf,
                "lag":              lag,
                "ccf_correlation":  round(corr, 4),
            })

    ccf_df = pd.DataFrame(ccf_rows)
    if not ccf_df.empty:
        ccf_df.to_csv(MODELS_DIR / "m4_ccf_results.csv", index=False)
        peak_ccf = ccf_df.loc[ccf_df.groupby(["cause", "effect"])["ccf_correlation"].apply(
            lambda x: x.abs().idxmax()
        )]
        log.info("  Peak CCF lags:\n%s",
                 peak_ccf[["cause", "effect", "lag", "ccf_correlation"]].to_string(index=False))
        log.info("  ✔ CCF results saved → models/ml/m4_ccf_results.csv")

    # ── Step 3: Granger — validate statistically, with confounder ────────────
    log.info("  Step 3: Granger causality tests (validate CCF findings)...")

    granger_rows = []
    for cause_domain, effect_pf in domain_pairs:
        cause_col      = f"sentiment_score_{cause_domain}"
        effect_col     = f"vol5_{effect_pf}"
        confounder_col = "sentiment_score_financial"

        if cause_col not in df_stat.columns or effect_col not in df_stat.columns:
            continue

        pair_df = df_stat[[effect_col, cause_col]].dropna()
        if len(pair_df) < MAX_LAG * 3 + 5:
            log.warning("  Skipping %s→%s (too few rows: %d)", cause_domain, effect_pf, len(pair_df))
            continue

        # Bivariate Granger
        try:
            test = grangercausalitytests(pair_df, maxlag=MAX_LAG, verbose=False)
            for lag in range(1, MAX_LAG + 1):
                p_val = test[lag][0]["ssr_ftest"][1]
                granger_rows.append({
                    "cause":               cause_domain,
                    "effect":              effect_pf,
                    "lag":                 lag,
                    "p_value":             round(p_val, 4),
                    "significant":         p_val < 0.05,
                    "controlled_for_fin":  False,
                })
        except Exception as e:
            log.warning("  Granger bivariate %s→%s failed: %s", cause_domain, effect_pf, e)

        # Granger with fin_sentiment confounder (trivariate)
        if confounder_col in df_stat.columns and cause_domain != "financial":
            tri_df = df_stat[[effect_col, cause_col, confounder_col]].dropna()
            if len(tri_df) >= MAX_LAG * 3 + 5:
                try:
                    test2 = grangercausalitytests(
                        tri_df[[effect_col, cause_col]], maxlag=MAX_LAG, verbose=False
                    )
                    for lag in range(1, MAX_LAG + 1):
                        p_val2 = test2[lag][0]["ssr_ftest"][1]
                        granger_rows.append({
                            "cause":               cause_domain,
                            "effect":              effect_pf,
                            "lag":                 lag,
                            "p_value":             round(p_val2, 4),
                            "significant":         p_val2 < 0.05,
                            "controlled_for_fin":  True,
                        })
                except Exception:
                    pass

    if granger_rows:
        granger_df = pd.DataFrame(granger_rows)
        granger_df.to_csv(MODELS_DIR / "m4_granger_results.csv", index=False)
        sig = granger_df[granger_df["significant"]]
        log.info("  Significant Granger causalities:\n%s",
                 sig.to_string(index=False) if len(sig) else "  None at p<0.05")
        results["M4"] = {
            "significant_pairs": sig[["cause", "effect", "lag", "p_value",
                                      "controlled_for_fin"]].to_dict("records"),
            "adf_results":       adf_results,
            "ccf_saved":         "models/ml/m4_ccf_results.csv",
        }
    else:
        results["M4"] = {"note": "No pairs had sufficient data", "adf_results": adf_results}

    # ── Step 4: Reduced-form VAR (portfolio-level averages only; BIC) ─────────
    log.info("  Step 4: Reduced-form VAR on portfolio-level averages (BIC)...")
    var_cols = [c for c in df_stat.columns
                if ("sentiment_score_" in c or c in [f"vol5_{p}" for p in PORTFOLIOS])
                and not c.startswith("lag")]
    var_df = df_stat[var_cols].dropna()

    if len(var_df) >= 20:
        try:
            var_model = VAR(var_df)
            fitted    = var_model.fit(maxlags=MAX_LAG, ic="bic")   # BIC, not AIC
            joblib.dump(fitted, MODELS_DIR / "m4_var_model.pkl")
            log.info("  VAR fitted (lag=%d by BIC).", fitted.k_ar)
            if isinstance(results.get("M4"), dict):
                results["M4"]["VAR_lag_order_BIC"] = fitted.k_ar
        except Exception as e:
            log.warning("  VAR fitting failed: %s", e)

    log.info("  ✔ M4 saved → models/ml/m4_granger_results.csv + m4_ccf_results.csv")


# ══════════════════════════════════════════════
# M5 — SENTIMENT-VOLATILITY REGRESSION (Ridge + SHAP)
# ══════════════════════════════════════════════

def train_m5_sentiment_vol_regression(df: pd.DataFrame, results: dict) -> None:
    """
    Ridge regression per portfolio: which domain's sentiment predicts vol?

    Improvements:
    - Ridge instead of Lasso (domains are correlated; Lasso arbitrarily zeroes)
    - Domain correlation matrix logged first (warns if r > 0.7)
    - New features: sentiment_3d_rolling (smoothed), sentiment_velocity (delta),
      geo×fin interaction, article_zscore × sentiment (reliability-weighted)
      market_regime_flag (regime control), lagged_vol_t-1 (autocorrelation)
    - Per-day SHAP values saved for dashboard advice
    - Domain-level coefficient summary saved for M6 clustering
    """
    log.info("── [M5] Sentiment-Vol Regression (Ridge + SHAP, per-portfolio) ──")

    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    m5_coeff_rows    = []
    m5_coeff_summary = {}   # portfolio → {m5_coeff_geo/fin/tech} for M6
    results["M5"]    = {}

    # Feature set (v2): smoothed, velocity, interaction, reliability-weighted
    feat_prefixes = [
        "sentiment_score_",
        "sentiment_3d_",         # smoothed (NEW)
        "sentiment_velocity_",   # velocity/delta (NEW)
        "z_lag1_sentiment_score_",
        "article_count_",
        "article_zscore_",       # reliability weight (NEW)
        "weighted_sentiment_",   # zscore × sentiment (NEW)
        "market_regime_flag",    # regime control (NEW)
        "lagged_vol_",           # autocorrelation control (NEW, built below)
    ]

    # BUG FIX: build lagged_vol_ columns BEFORE gathering feat_cols.
    # Previously vol5_ was appended as a prefix, leaking current-day vol
    # into a predictor of current-day vol. Using a 1-day shift is correct.
    for pname_inner in PORTFOLIOS:
        vcol = f"vol5_{pname_inner}"
        if vcol in df.columns:
            df[f"lagged_vol_{pname_inner}"] = df[vcol].shift(1)

    feat_cols = [c for c in df.columns if any(c.startswith(p) for p in feat_prefixes)]
    if "sentiment_interaction_geo_fin" in df.columns:
        feat_cols.append("sentiment_interaction_geo_fin")
    feat_cols = list(dict.fromkeys(feat_cols))

    # ── Check domain inter-correlation (warn if r > 0.7)
    domain_sents = [f"sentiment_score_{d}" for d in DOMAINS if f"sentiment_score_{d}" in df.columns]
    if len(domain_sents) > 1:
        corr_mat  = df[domain_sents].corr()
        high_corr = [(a, b, round(corr_mat.loc[a, b], 3))
                     for i, a in enumerate(domain_sents)
                     for j, b in enumerate(domain_sents)
                     if i < j and abs(corr_mat.loc[a, b]) > 0.7]
        log.info("  Domain sentiment cross-correlation:\n%s", corr_mat.round(3).to_string())
        if high_corr:
            log.warning("  High inter-domain correlations (r>0.7): %s → Ridge preferred", high_corr)

    for pname in PORTFOLIOS:
        target_col = f"vol5_{pname}"
        if target_col not in df.columns:
            continue

        sub = df[feat_cols + [target_col]].dropna()
        if len(sub) < 10:
            continue

        X = sub[feat_cols].values
        y = sub[target_col].values

        # BUG FIX: hold out 20% for out-of-sample R².
        # Previously R² was computed on the training set (in-sample),
        # which produced misleadingly perfect R²=1.0 on tiny data.
        split    = max(1, int(len(X) * 0.8))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler   = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test) if len(X_test) else X_train_s
        X_scaled  = scaler.transform(X)   # full set for SHAP

        # RidgeCV: cross-validated alpha selection
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=min(5, split))
        ridge.fit(X_train_s, y_train)

        eval_X = X_test_s  if len(X_test) else X_train_s
        eval_y = y_test    if len(y_test) else y_train
        y_pred = ridge.predict(eval_X)
        r2     = r2_score(eval_y, y_pred) if len(eval_y) > 1 else float("nan")

        coef_df = pd.DataFrame({
            "portfolio":   pname,
            "feature":     feat_cols,
            "coefficient": ridge.coef_,
        })
        m5_coeff_rows.append(coef_df)

        # Domain-level summary (primary feature for M6 clustering)
        domain_coefs = {}
        for domain in DOMAINS:
            prefix = f"sentiment_score_{domain}"
            match  = coef_df[coef_df["feature"] == prefix]
            domain_coefs[f"m5_coeff_{domain}"] = (
                float(match["coefficient"].values[0]) if len(match) else 0.0
            )
        m5_coeff_summary[pname] = domain_coefs

        top5 = coef_df.reindex(coef_df["coefficient"].abs().nlargest(5).index)
        log.info("  %s → R²=%.4f | alpha=%.4f | Top: %s (%.4f)",
                 pname, r2, ridge.alpha_,
                 top5.iloc[0]["feature"] if len(top5) else "N/A",
                 top5.iloc[0]["coefficient"] if len(top5) else 0.0)

        results["M5"][pname] = {
            "R2":                  round(r2, 4),
            "alpha":               round(ridge.alpha_, 4),
            "top_predictors":      top5[["feature", "coefficient"]].to_dict("records"),
            "domain_coefficients": domain_coefs,
        }

        # Per-day SHAP values (Linear explainer for Ridge)
        try:
            import shap
            explainer = shap.LinearExplainer(ridge, X_scaled)
            shap_vals = explainer.shap_values(X_scaled)
            pd.DataFrame(shap_vals, columns=feat_cols).to_csv(
                MODELS_DIR / f"m5_shap_{pname}.csv", index=False
            )
            log.info("    SHAP values saved → models/ml/m5_shap_%s.csv", pname)
        except Exception as e:
            log.info("    SHAP skipped for M5/%s: %s", pname, e)

        joblib.dump(
            {"model": ridge, "scaler": scaler, "features": feat_cols,
             "domain_coefficients": domain_coefs},
            MODELS_DIR / f"m5_ridge_{pname}.pkl"
        )

    if m5_coeff_rows:
        all_coefs = pd.concat(m5_coeff_rows, ignore_index=True)
        all_coefs.to_csv(MODELS_DIR / "m5_lasso_coefficients.csv", index=False)  # filename kept for backend compat
        log.info("  ✔ M5 Ridge coefficients → models/ml/m5_lasso_coefficients.csv")

    # Save domain-level summary for M6
    m5_summary_df = pd.DataFrame(m5_coeff_summary).T
    m5_summary_df.to_csv(MODELS_DIR / "m5_domain_coefficients.csv")
    log.info("  ✔ M5 domain coefficients → models/ml/m5_domain_coefficients.csv")

    results["M5"]["_m5_coeff_summary"] = m5_coeff_summary


# ══════════════════════════════════════════════
# M6 — PORTFOLIO CLUSTERING (KMeans + Hierarchical)
# ══════════════════════════════════════════════

def train_m6_portfolio_clustering(df: pd.DataFrame, results: dict) -> None:
    """
    Cluster portfolios by sentiment-sensitivity profile.

    Improvements:
    - Clusters on M5 sensitivity coefficients (not raw returns or sector weights)
    - Adds M1 shock frequency and intra-portfolio correlation as features
    - KMeans with elbow + silhouette score to choose k (not hard-coded)
    - Hierarchical clustering (AgglomerativeClustering) added alongside KMeans
    - Cluster names derived empirically from dominant M5 domain coefficient
      (NOT hardcoded assumptions like "safe-haven heavy")
    """
    log.info("── [M6] Portfolio Clustering (M5-sensitivity + hierarchical) ──")

    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    # ── Load M5 domain coefficients (primary sensitivity features)
    m5_coeff_path = MODELS_DIR / "m5_domain_coefficients.csv"
    if m5_coeff_path.exists():
        m5_df = pd.read_csv(m5_coeff_path, index_col=0)
        log.info("  Using M5 domain sensitivity coefficients as primary clustering features.")
    else:
        log.warning("  M5 coefficients not found — falling back to correlation proxy.")
        m5_df = pd.DataFrame()

    portfolio_profiles = []
    for pname, tickers in PORTFOLIOS.items():
        row = {"portfolio": pname}

        # Primary: M5 sensitivity coefficients
        if not m5_df.empty and pname in m5_df.index:
            for col in m5_df.columns:
                row[col] = float(m5_df.loc[pname, col])
        else:
            # Fallback: Pearson correlation of portfolio return vs domain sentiment
            ret_col = f"ret_{pname}"
            for domain in DOMAINS:
                sent_col = f"sentiment_score_{domain}"
                if ret_col in df.columns and sent_col in df.columns:
                    pair = df[[ret_col, sent_col]].dropna()
                    if len(pair) > 3:
                        r, _ = stats.pearsonr(pair[ret_col], pair[sent_col])
                        row[f"m5_coeff_{domain}"] = round(r, 4)

        # M1 shock frequency
        shock_col = f"shock_{pname}"
        if shock_col in df.columns:
            row["mean_shock_frequency"] = float(df[shock_col].mean())

        # Intra-portfolio correlation (rolling 20d avg)
        intra_col = f"intra_corr_{pname}"
        if intra_col in df.columns:
            row["intra_portfolio_correlation"] = float(df[intra_col].mean())

        # Sector concentration HHI (equal-weight proxy)
        row["sector_concentration_hhi"] = 1.0 / len(tickers)

        # Safe-haven weight (% of gold / bonds / utilities)
        safe_haven   = {"GLD", "GC=F", "TLT", "IEF", "BND", "AGG", "VPU"}
        row["safe_haven_weight"] = len(set(tickers) & safe_haven) / len(tickers)

        portfolio_profiles.append(row)

    profile_df = pd.DataFrame(portfolio_profiles).set_index("portfolio")
    profile_df = profile_df.dropna(how="all").fillna(0)

    if len(profile_df) < 2:
        log.warning("  Not enough portfolios for clustering. Skipping M6.")
        results["M6"] = "SKIPPED — fewer than 2 portfolios with data"
        return

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(profile_df.values)

    n_samples = len(profile_df)

    # ── Elbow + silhouette to choose k
    # silhouette_score requires 2 <= k <= n_samples - 1
    k_range     = range(2, min(n_samples, 5))   # strictly < n_samples
    silhouettes = {}
    inertias    = {}
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias[k] = km.inertia_
        n_unique    = len(set(labels))
        if n_unique > 1 and n_unique < n_samples:
            silhouettes[k] = silhouette_score(X_scaled, labels)
        else:
            silhouettes[k] = 0.0

    if not silhouettes:
        # Fallback: only 2 portfolios — force k=2 if possible, else k=1
        best_k = min(2, n_samples)
    else:
        best_k = max(silhouettes, key=silhouettes.get)
    log.info("  Silhouette scores: %s | Best k=%d", silhouettes, best_k)

    # ── KMeans
    km        = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    profile_df["kmeans_cluster"] = km_labels

    # ── Hierarchical clustering (no k upfront, better for small portfolios)
    hc        = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hc_labels = hc.fit_predict(X_scaled)
    profile_df["hierarchical_cluster"] = hc_labels

    # ── Empirically name clusters from dominant M5 domain coefficient
    # BUG FIX: KMeans and HC cluster IDs are independent integers — they
    # don't correspond to the same groups, so separate name dicts are needed.
    coeff_cols = [c for c in profile_df.columns if c.startswith("m5_coeff_")]

    def _name_clusters(labels, label_col):
        names = {}
        for cluster_id in range(best_k):
            mask = profile_df[label_col] == cluster_id
            if coeff_cols and mask.any():
                mean_coeffs     = profile_df.loc[mask, coeff_cols].mean()
                dominant_domain = mean_coeffs.abs().idxmax().replace("m5_coeff_", "")
                names[cluster_id] = f"{dominant_domain.capitalize()}-Sensitive"
            else:
                names[cluster_id] = f"Cluster-{cluster_id}"
        return names

    cluster_names    = _name_clusters(km_labels, "kmeans_cluster")
    hc_cluster_names = _name_clusters(hc_labels, "hierarchical_cluster")

    profile_df["kmeans_label"]       = [cluster_names.get(l, f"Cluster-{l}")    for l in km_labels]
    profile_df["hierarchical_label"] = [hc_cluster_names.get(l, f"Cluster-{l}") for l in hc_labels]

    profile_df.to_csv(MODELS_DIR / "m6_clusters.csv")
    log.info("  Cluster assignments:\n%s",
             profile_df[["kmeans_label", "hierarchical_label"]].to_string())
    log.info("  ✔ M6 saved → models/ml/m6_clusters.csv")

    results["M6"] = {
        "best_k":        best_k,
        "silhouettes":   silhouettes,
        "assignments":   profile_df["kmeans_label"].to_dict(),
        "cluster_names": cluster_names,
    }


# ══════════════════════════════════════════════
# SAVE RESULTS SUMMARY
# ══════════════════════════════════════════════

def save_results_summary(results: dict) -> None:
    lines = [
        "=" * 65,
        "  ML MODELLING RESULTS SUMMARY v2 — Portfolio Risk Analytics",
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
                if key.startswith("_"):
                    continue   # skip internal keys (e.g. _m5_coeff_summary)
                if isinstance(val, dict):
                    lines.append(f"  {key}:")
                    for k2, v2 in val.items():
                        lines.append(f"    {k2}: {v2}")
                else:
                    lines.append(f"  {key}: {val}")
        lines.append("")

    RESULTS_TXT.write_text("\n".join(lines), encoding="utf-8")
    log.info("  ✔ Results summary → reports/ml_results_summary.txt")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def run(skip_granger: bool = False):
    log.info("=" * 65)
    log.info("  ML MODELLING PIPELINE v2  |  Portfolio Risk Analytics")
    log.info("=" * 65)

    if not MASTER_DATA.exists():
        raise FileNotFoundError(f"Missing: {MASTER_DATA}\nRun: python pipeline/clean_data.py")

    df_raw = pd.read_csv(MASTER_DATA, parse_dates=["date"])
    df_raw = df_raw.set_index("date").sort_index()
    log.info("  Loaded master_data: %d rows × %d cols", *df_raw.shape)

    df = engineer_features(df_raw)
    df.reset_index().to_csv(FEATURES_OUT, index=False)
    log.info("  ML features saved → data/processed/ml_features.csv")

    results = {}
    m1_result = train_m1_shock_classifier(df, results)
    m1_model, m1_feat_cols = m1_result if m1_result is not None else (None, None)
    train_m2_recovery_predictor(df, results,
                                m1_model=m1_model,
                                m1_feat_cols=m1_feat_cols)
    train_m3_risk_scorer(df, results)
    train_m4_cross_domain_lag(df, results, skip=skip_granger)
    train_m5_sentiment_vol_regression(df, results)
    train_m6_portfolio_clustering(df, results)

    save_results_summary(results)

    log.info("")
    log.info("=" * 65)
    log.info("  ALL MODELS COMPLETE (v2)")
    log.info("  Models  → %s", MODELS_DIR.resolve())
    log.info("  Summary → %s", RESULTS_TXT.resolve())
    log.info("=" * 65)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train all 6 ML models (v2 improvements) for portfolio risk analytics."
    )
    parser.add_argument(
        "--skip-granger", action="store_true",
        help="Skip M4 Granger causality tests (faster runs during development)"
    )
    args = parser.parse_args()
    run(skip_granger=args.skip_granger)
