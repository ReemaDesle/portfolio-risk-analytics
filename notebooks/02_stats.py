"""
Portfolio Risk Analytics — Statistical Validation (V3 Longitudinal)
===================================================================
Formalises the relationships identified in EDA using rigorous time-series
econometrics. Validates stationarity, causality, and cointegration.

Core Tests:
-----------
  T1. Stationarity Suite (ADF & KPSS)
  T2. Granger Causality (Proof of Lead-Lag)
  T3. Johansen Cointegration (Long-term Equilibrium)
  T4. Multi-Regime ANOVA (Yearly consistency)
  T5. ARIMAX Likelihood Ratio Test (Model Significance)

Outputs -> reports/plots/Stats/
-------
  reports/Stats_latest_results.txt
"""

import warnings
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Config & Paths
# ──────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT_DIR / "reports" / "plots" / "Stats"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MASTER_DATA = ROOT_DIR / "data" / "processed" / "master_data.csv"

# Styling
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "sans-serif",
})

RESULTS_LOG = []
def log_result(text):
    print(text)
    RESULTS_LOG.append(text)

def save_results_log():
    path = ROOT_DIR / "reports" / "Stats_latest_results.txt"
    path.write_text("\n".join(RESULTS_LOG), encoding="utf-8")
    print(f"  [+] Saved statistical inferences to {path}")

def savefig(name, fig):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [+] Saved {path.name}")

# Portfolio definitions
PORTFOLIOS = {
    "geopolitical": ["GLD", "USO", "LMT", "RTX", "EEM"],
    "tech":         ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA"],
    "balanced":     ["SPY", "AGG", "GLD", "VTI", "EFA"],
    "conservative": ["TLT", "IEF", "VPU", "KO", "JNJ", "PG", "XLP"],
}

DOMAINS = ["financial", "geopolitical", "technology"]

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────

def load_and_prep():
    df = pd.read_csv(MASTER_DATA, parse_dates=["date"]).set_index("date").sort_index()
    # Create portfolio return columns
    for name, tickers in PORTFOLIOS.items():
        cols = [f"ret_{t}" for t in tickers if f"ret_{t}" in df.columns]
        if cols:
            df[f"pf_{name}_ret"] = df[cols].mean(axis=1)
            
    # Z-Score the sentiment for numerical stability in tests
    sent_cols = [f"sentiment_score_{d}" for d in DOMAINS if f"sentiment_score_{d}" in df.columns]
    for col in sent_cols:
        df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()
        
    return df

# ══════════════════════════════════════════════
# TEST 1 — Stationarity Suite (ADF & KPSS)
# ══════════════════════════════════════════════
def test_stationarity(df):
    """Confirm data is I(0) before proceeding to causality/VAR."""
    log_result("\n--- T1: STATIONARITY SUITE ---")
    targets = ["pf_tech_ret", "pf_geopolitical_ret", "z_sentiment_score_technology", "z_sentiment_score_geopolitical"]
    
    results = []
    for col in targets:
        if col not in df.columns: continue
        series = df[col].dropna()
        
        # Augmented Dickey-Fuller
        adf_res = adfuller(series)
        adf_p = adf_res[1]
        
        # KPSS
        kpss_res = kpss(series)
        kpss_p = kpss_res[1]
        
        status = "STATIONARY" if adf_p < 0.05 and kpss_p > 0.05 else "MIXED/NON-STATIONARY"
        log_result(f"  {col:30} | ADF p={adf_p:.4f} | KPSS p={kpss_p:.4f} | Result: {status}")
        results.append({"Variable": col, "ADF_p": adf_p, "KPSS_p": kpss_p})
        
    return pd.DataFrame(results)

# ══════════════════════════════════════════════
# TEST 2 — Granger Causality (Lead-Lag Proof)
# ══════════════════════════════════════════════
def test_granger_causality(df):
    """Scientifically test if Sentiment leads Returns (Lag 1-3)."""
    log_result("\n--- T2: GRANGER CAUSALITY MATRIX ---")
    
    pairs = [
        ("z_sentiment_score_technology", "pf_tech_ret"),
        ("z_sentiment_score_geopolitical", "pf_geopolitical_ret"),
        ("z_sentiment_score_financial", "pf_balanced_ret")
    ]
    
    matrix_data = []
    for sent, ret in pairs:
        if sent not in df.columns or ret not in df.columns: continue
        combined = df[[ret, sent]].dropna()
        
        # Test max 5 days lag
        res = grangercausalitytests(combined, maxlag=3, verbose=False)
        p_vals = [res[lag][0]['ssr_ftest'][1] for lag in [1, 2, 3]]
        
        log_result(f"  {sent:30} -> {ret:20} | Lag1 p={p_vals[0]:.4f} | Lag2 p={p_vals[1]:.4f} | Lag3 p={p_vals[2]:.4f}")
        matrix_data.append([sent.split('_')[-1], p_vals[0], p_vals[1], p_vals[2]])

    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 5))
    m_df = pd.DataFrame(matrix_data, columns=["Domain", "Lag 1", "Lag 2", "Lag 3"]).set_index("Domain")
    sns.heatmap(m_df, annot=True, cmap="RdYlGn_r", vmin=0, vmax=0.1, center=0.05, ax=ax)
    ax.set_title("Granger Causality P-Values (Sent → Returns)\nLower is better (p < 0.05 = Cause Found)", color="white")
    savefig("granger_heatmap.png", fig)

# ══════════════════════════════════════════════
# TEST 3 — Johansen Cointegration
# ══════════════════════════════════════════════
def test_cointegration(df):
    """Test if News and Markets share a long-term equilibrium."""
    log_result("\n--- T3: JOHANSEN COINTEGRATION ---")
    
    # Check Gold vs Geo Sentiment long-term path
    geo_gold = df[["pf_geopolitical_ret", "z_sentiment_score_geopolitical"]].dropna()
    res = coint_johansen(geo_gold, det_order=0, k_ar_diff=1)
    
    trace_stat = res.lr1[0]
    critical_95 = res.cvt[0, 1]
    
    coint_found = "YES" if trace_stat > critical_95 else "NO"
    log_result(f"  Geo Portfolio <-> Geo Sentiment | Trace Stat: {trace_stat:.2f} | Critical(95%): {critical_95:.2f} | Cointegrated: {coint_found}")

# ══════════════════════════════════════════════
# TEST 4 — Multi-Regime ANOVA (Yearly consistency)
# ══════════════════════════════════════════════
def test_yearly_anova(df):
    """Does the Archetype effect hold across different years?"""
    log_result("\n--- T4: MULTI-YEAR BENCHMARK ANOVA ---")
    
    df["year"] = df.index.year
    years = sorted(df["year"].unique())
    pfs = ["pf_tech_ret", "pf_geopolitical_ret", "pf_conservative_ret"]
    
    anova_res = []
    for y in years:
        year_data = df[df["year"] == y]
        groups = [year_data[pf].dropna() for pf in pfs]
        f_stat, p_val = stats.f_oneway(*groups)
        log_result(f"  Year {y} | F-Stat: {f_stat:.4f} | p-value: {p_val:.4f}")
        anova_res.append({"year": y, "f": f_stat, "p": p_val})

    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 6))
    res_df = pd.DataFrame(anova_res)
    ax.bar(res_df["year"].astype(str), res_df["f"], color="#4a90d9", alpha=0.7)
    ax.axhline(3.0, color="#ff6b6b", linestyle="--", label="Critical F-approx")
    ax.set_title("ANOVA F-Statistics by Year (Portfolio Type Strength)", color="white")
    ax.set_ylabel("F-Statistic")
    ax.legend()
    ax.grid(True, alpha=0.2)
    savefig("yearly_anova_strength.png", fig)

# ══════════════════════════════════════════════
# TEST 5 — ARIMAX Likelihood Ratio Test (LRT)
# ══════════════════════════════════════════════
def test_arimax_significance(df):
    """Prove exogenous sentiment adds value over pure auto-regression."""
    log_result("\n--- T5: ARIMAX MODEL VALIDATION (LRT) ---")
    
    target = df["pf_tech_ret"].dropna()
    exog = df["z_sentiment_score_technology"].shift(1).loc[target.index].fillna(0)
    
    # 1. Base ARIMA(1,0,1)
    base_model = SARIMAX(target, order=(1,0,1)).fit(disp=False)
    # 2. ARIMAX(1,0,1) with Sentiment
    full_model = SARIMAX(target, exog=exog, order=(1,0,1)).fit(disp=False)
    
    lr_stat = 2 * (full_model.llf - base_model.llf)
    p_val = stats.chi2.sf(lr_stat, df=1) # 1 degree of freedom (the exog param)
    
    log_result(f"  ARIMA LLF: {base_model.llf:.2f}")
    log_result(f"  ARIMAX LLF: {full_model.llf:.2f}")
    log_result(f"  LR Statistic: {lr_stat:.4f} | P-value: {p_val:.4f}")
    
    if p_val < 0.05:
        log_result("  CONCLUSION: Sentiment provides statistically significant improvement to the forecast.")
    else:
        log_result("  CONCLUSION: Improvement is marginal at 95% confidence.")

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_stats_pipeline():
    print("=" * 60)
    print("  STATS  |  Portfolio Risk Analytics Validation (V3)")
    print("=" * 60)
    
    df = load_and_prep()
    log_result(f"Targeting Longitudinal Period: {df.index.min().date()} to {df.index.max().date()}")
    
    test_stationarity(df)
    test_granger_causality(df)
    test_cointegration(df)
    test_yearly_anova(df)
    test_arimax_significance(df)
    
    save_results_log()
    print("\n" + "=" * 60)
    print("  VALIDATION COMPLETE")
    print(f"  Outputs in: {FIGURES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    run_stats_pipeline()
