"""
Portfolio Risk Analytics — Exploratory Data Analysis (EDA)
==========================================================
Analyses baseline portfolio behaviour and the news cycle independently,
then investigates the interactions between them.

Research questions answered
----------------------------
  Q1. Baseline return distributions per portfolio archetype
  Q2. How often does panic occur in each news domain?
  Q3. Correlation between sentiment features and portfolio returns
  Q4. Does war-related (geopolitical) news significantly increase gold volatility?
  Q5. Which portfolio is most sensitive to geopolitical shocks?
  Q6. Do tech-heavy portfolios overreact to sentiment spikes?
  Q7. Is there a measurable lag between news intensity and volatility?
  Q8. Which information source has the highest predictive power?
  Q9. Which portfolio type takes longest to stabilize after a shock?

Outputs  →  reports/plots/EDA/
-------
  00_timeline_coverage.png
  01_return_distributions.png
  02_panic_detection.png
  03_correlation_heatmap.png
  04_shock_absorption_timeline.png
  05_rolling_vol_boxplots.png
  06_gold_vs_geopolitical.png
  07_geopolitical_sensitivity.png
  08_tech_overreaction.png
  09_lag_analysis.png
  10_predictive_power.png
  11_stabilisation_time.png
  12_rolling_correlation.png
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no interactive display — save only
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Config & Paths
# ──────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT_DIR / "reports" / "plots" / "EDA"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MASTER_DATA  = ROOT_DIR / "data" / "processed" / "master_data.csv"
MASTER_NEWS  = ROOT_DIR / "data" / "processed" / "master_news.csv"
SRI_SCORES   = ROOT_DIR / "data" / "processed" / "sri_scores.csv"

# Portfolio → constituent tickers
PORTFOLIOS = {
    "geopolitical": ["GLD", "USO", "LMT", "RTX", "EEM"],
    "tech":         ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA"],
    "balanced":     ["SPY", "AGG", "GLD", "VTI", "EFA"],
    "conservative": ["TLT", "IEF", "VPU", "KO", "JNJ", "PG", "XLP"],
}

PORTFOLIO_COLOURS = {
    "geopolitical": "#e07b39",
    "tech":         "#4a90d9",
    "balanced":     "#6abf69",
    "conservative": "#9b59b6",
}

DOMAINS = ["financial", "geopolitical", "technology"]
DOMAIN_COLOURS = {
    "financial":    "#2ecc71",
    "geopolitical": "#e74c3c",
    "technology":   "#3498db",
}

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
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
})

RESULTS_LOG = []
def log_result(text):
    print(text)
    RESULTS_LOG.append(text)

def save_results_log():
    path = ROOT_DIR / "reports" / "EDA_latest_results.txt"
    path.write_text("\n".join(RESULTS_LOG), encoding="utf-8")
    print(f"  [+] Saved results to {path}")

def savefig(name: str, fig):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [+] Saved  {path.name}")


# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────

def load_data():
    df = pd.read_csv(MASTER_DATA, parse_dates=["date"]).set_index("date").sort_index()
    news = pd.read_csv(MASTER_NEWS, parse_dates=["date"]) if MASTER_NEWS.exists() else None
    sri  = pd.read_csv(SRI_SCORES, parse_dates=["date"])  if SRI_SCORES.exists()  else None
    return df, news, sri


def portfolio_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight average of constituent log-returns per portfolio."""
    rows = {}
    for name, tickers in PORTFOLIOS.items():
        cols = [f"ret_{t}" for t in tickers if f"ret_{t}" in df.columns]
        if cols:
            rows[name] = df[cols].mean(axis=1)
    return pd.DataFrame(rows, index=df.index).dropna(how="all")


def portfolio_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight average of 5-day rolling vol per portfolio."""
    rows = {}
    for name, tickers in PORTFOLIOS.items():
        cols = [f"vol5_{t}" for t in tickers if f"vol5_{t}" in df.columns]
        if cols:
            rows[name] = df[cols].mean(axis=1)
    return pd.DataFrame(rows, index=df.index).dropna(how="all")

# ══════════════════════════════════════════════
# FIGURE 0 — Timeline Coverage
# ══════════════════════════════════════════════
def fig_timeline_coverage(df: pd.DataFrame):
    """Figure 00: News article volume over time."""
    count_cols = [f"article_count_{d}" for d in DOMAINS if f"article_count_{d}" in df.columns]
    if not count_cols:
        return
        
    monthly_counts = df[count_cols].resample("M").sum()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create stacked area
    labels = [c.replace("article_count_", "").capitalize() for c in count_cols]
    colors = [DOMAIN_COLOURS.get(c.replace("article_count_", ""), "#888") for c in count_cols]
    
    ax.stackplot(monthly_counts.index, monthly_counts.T.values, labels=labels, colors=colors, alpha=0.8)
    
    ax.set_title("Timeline Coverage: Monthly News Article Volume (2021-2026)", fontsize=14)
    ax.set_ylabel("Total Articles")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    
    plt.tight_layout()
    savefig("00_timeline_coverage.png", fig)
    log_result(f"Timeline Coverage: Captured {monthly_counts.sum().sum():.0f} total articles across {len(monthly_counts)} months.")

# ══════════════════════════════════════════════
# FIGURE 1 — Return distributions (histograms)
# ══════════════════════════════════════════════

def fig_return_distributions(pf_ret: pd.DataFrame):
    """Q1: Baseline daily return distributions per portfolio."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Daily Return Distributions by Portfolio Archetype", fontsize=15, y=1.01)

    for ax, (name, colour) in zip(axes.flat, PORTFOLIO_COLOURS.items()):
        if name not in pf_ret.columns:
            continue
        data = pf_ret[name].dropna()
        mean_, std_ = data.mean(), data.std()
        skew_       = data.skew()

        ax.hist(data, bins=30, color=colour, alpha=0.75, edgecolor="#0d1117", density=True)

        xmin, xmax = data.min(), data.max()
        xs = np.linspace(xmin, xmax, 200)
        kde = stats.gaussian_kde(data)
        ax.plot(xs, kde(xs), color="white", lw=1.5, linestyle="--", label="KDE")

        ax.axvline(mean_, color=colour, lw=1.5, linestyle="-", label=f"Mean {mean_:.4f}")
        ax.axvline(0, color="#8b949e", lw=1, linestyle=":")

        ax.set_title(name.capitalize())
        ax.set_xlabel("Log Return")
        ax.set_ylabel("Density")
        textstr = f"μ = {mean_:.4f}\nσ = {std_:.4f}\nskew = {skew_:.2f}"
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes, va="top", ha="right",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d", alpha=0.8))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("01_return_distributions.png", fig)


# ══════════════════════════════════════════════
# FIGURE 2 — Panic detection in sentiment
# ══════════════════════════════════════════════

def fig_panic_detection(df: pd.DataFrame):
    """Q2: How often panic occurs in each news domain using dynamic thresholding."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Sentiment Score Distributions & Panic Detection (Dynamic 5th Percentile)", fontsize=13)

    for ax, domain in zip(axes, DOMAINS):
        col = f"sentiment_score_{domain}"
        if col not in df.columns:
            ax.set_visible(False)
            continue
        data = df[col].dropna()
        colour = DOMAIN_COLOURS[domain]
        
        # Dynamic threshold (5th percentile)
        panic_threshold = data.quantile(0.05)

        ax.hist(data, bins=25, color=colour, alpha=0.7, edgecolor="#0d1117", density=True)
        xmin, xmax = data.min(), data.max()
        xs  = np.linspace(xmin, xmax, 200)
        kde = stats.gaussian_kde(data)
        ax.plot(xs, kde(xs), color="white", lw=1.5)

        ax.axvline(panic_threshold, color="#ff6b6b", lw=2, linestyle="--", label=f"Panic < {panic_threshold:.3f}")
        ax.axvline(0, color="#8b949e", lw=1, linestyle=":")

        panic_days  = (data < panic_threshold).sum()
        panic_pct   = panic_days / len(data) * 100
        ax.set_title(f"{domain.capitalize()} Domain\n"
                     f"Panic days: {panic_days}/{len(data)}  ({panic_pct:.1f}%)")
        ax.set_xlabel("Sentiment Score (prob_pos − prob_neg)")
        ax.set_ylabel("Density")

        ax.axvspan(xmin, panic_threshold, alpha=0.12, color="#ff6b6b")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        log_result(f"Panic Threshold ({domain}): {panic_threshold:.3f} | Instances: {panic_days}")

    plt.tight_layout()
    savefig("02_panic_detection.png", fig)


# ══════════════════════════════════════════════
# FIGURE 3 — Correlation heatmap
# ══════════════════════════════════════════════

def fig_correlation_heatmap(df: pd.DataFrame, pf_ret: pd.DataFrame):
    """Q3: Correlation between key features and portfolio returns."""
    feat_cols = (
        [f"sentiment_score_{d}" for d in DOMAINS if f"sentiment_score_{d}" in df.columns] +
        [f"avg_prob_neg_{d}"    for d in DOMAINS if f"avg_prob_neg_{d}"    in df.columns] +
        [f"lag1_sentiment_score_{d}" for d in DOMAINS
         if f"lag1_sentiment_score_{d}" in df.columns]
    )
    combined = pf_ret.join(df[feat_cols], how="inner").dropna()

    corr = combined.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.zeros_like(corr, dtype=bool)
    feature_start = len(PORTFOLIOS)
    mask[:feature_start, :feature_start] = np.triu(
        np.ones((feature_start, feature_start), dtype=bool), k=1
    )

    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    sns.heatmap(
        corr, ax=ax, mask=mask, cmap=cmap, center=0,
        vmin=-1, vmax=1, annot=True, fmt=".2f",
        annot_kws={"size": 7.5}, linewidths=0.3,
        linecolor="#21262d",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation: Portfolio Returns ↔ Sentiment Features", fontsize=13, pad=12)
    ax.tick_params(labelrotation=45, labelsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    savefig("03_correlation_heatmap.png", fig)


# ══════════════════════════════════════════════
# FIGURE 4 — Shock absorption timeline
# ══════════════════════════════════════════════

def fig_shock_timeline(df: pd.DataFrame, pf_ret: pd.DataFrame):
    """Q5/Q9: Visualize the prices around the top 3 biggest single-domain shock days."""
    geo_col = "sentiment_score_geopolitical"
    if geo_col not in df.columns or len(df[geo_col].dropna()) == 0:
        return
        
    # Get top 3 biggest negative shocks
    top_shocks = df[geo_col].nsmallest(3).index.tolist()
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), gridspec_kw={"height_ratios": [1, 1, 1]})
    fig.suptitle("Shock Absorption Timeline (Top 3 Geopolitical Shocks)", fontsize=15)
    
    for i, shock_date in enumerate(top_shocks):
        ax1 = axes[i, 0] # Returns
        ax2 = axes[i, 1] # Sentiment
        
        window = pd.date_range(shock_date - pd.Timedelta(days=7), shock_date + pd.Timedelta(days=10), freq="B")
        window = window[window.isin(pf_ret.index)]
        
        # Returns
        for name, colour in PORTFOLIO_COLOURS.items():
            if name not in pf_ret.columns: continue
            series = pf_ret.loc[pf_ret.index.isin(window), name]
            ax1.plot(series.index, series.values, marker="o", markersize=4, color=colour, label=name.capitalize(), linewidth=1.8)
        ax1.axvline(shock_date, color="#ff6b6b", lw=2, linestyle="--")
        ax1.axhline(0, color="#8b949e", lw=0.8, linestyle=":")
        ax1.set_ylabel("Return")
        ax1.set_title(f"Shock {i+1}: {shock_date.date()}")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        if i == 0: ax1.legend(fontsize=8)
        
        # Sentiment
        sent_window = df.loc[df.index.isin(window), geo_col]
        colors = [DOMAIN_COLOURS["geopolitical"] if v >= 0 else "#e74c3c" for v in sent_window.values]
        ax2.bar(sent_window.index, sent_window.values, color=colors, alpha=0.8, width=0.8)
        ax2.axvline(shock_date, color="#ff6b6b", lw=2, linestyle="--")
        ax2.axhline(0, color="#8b949e", lw=0.8, linestyle=":")
        ax2.set_ylabel("Geo Sentiment")
        ax2.set_title(f"Sentiment around {shock_date.date()}")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.tight_layout()
    savefig("04_shock_absorption_timeline.png", fig)
    log_result(f"Identified Top 3 Geo Shocks: {[d.date().isoformat() for d in top_shocks]}")


# ══════════════════════════════════════════════
# FIGURE 5 — Rolling volatility boxplots
# ══════════════════════════════════════════════

def fig_vol_boxplots(pf_vol: pd.DataFrame):
    """Boxplots of 5-day rolling vol per portfolio."""
    fig, ax = plt.subplots(figsize=(11, 6))

    plot_data  = [pf_vol[n].dropna().values for n in PORTFOLIOS if n in pf_vol.columns]
    plot_labels = [n.capitalize() for n in PORTFOLIOS if n in pf_vol.columns]
    colours    = [PORTFOLIO_COLOURS[n] for n in PORTFOLIOS if n in pf_vol.columns]

    bp = ax.boxplot(
        plot_data,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": "#8b949e"},
        capprops={"color": "#8b949e"},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
    )
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
    for flier, colour in zip(bp["fliers"], colours):
        flier.set_markerfacecolor(colour)

    ax.set_xticklabels(plot_labels)
    ax.set_title("5-Day Rolling Volatility Distribution by Portfolio Archetype")
    ax.set_ylabel("Rolling Volatility (σ of log returns)")
    ax.set_xlabel("Portfolio")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    savefig("05_rolling_vol_boxplots.png", fig)


# ══════════════════════════════════════════════
# FIGURE 6 — Gold volatility vs geopolitical news
# ══════════════════════════════════════════════

def fig_gold_vs_geopolitical(df: pd.DataFrame):
    """Q4: Does war-related news increase gold volatility?"""
    geo_col  = "avg_prob_neg_geopolitical"
    gold_col = "vol5_GLD"
    if geo_col not in df.columns or gold_col not in df.columns:
        return

    combined = df[[geo_col, gold_col]].dropna()
    if len(combined) < 3: return

    x = combined[geo_col].values
    y = combined[gold_col].values
    r, p_val = stats.pearsonr(x, y)

    fig, ax = plt.subplots(figsize=(10, 6))

    sc = ax.scatter(x, y, c=x, cmap="Reds", s=80, alpha=0.85, edgecolors="#0d1117", lw=0.5)
    plt.colorbar(sc, ax=ax, label="Geopolitical Negative Sentiment Prob")

    m, b = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, m * xline + b, color="#ff6b6b", lw=2,
            label=f"Regression  (r={r:.3f}, p={p_val:.3f})")

    significance = "SIGNIFICANT (p < 0.05)" if p_val < 0.05 else "NOT significant (p >= 0.05)"
    ax.set_title(f"Does war/geo news increase Gold volatility?\n"
                 f"Pearson r = {r:.3f}  —  {significance}", fontsize=12)
    ax.set_xlabel("Average Geopolitical Negative Sentiment Probability")
    ax.set_ylabel("Gold 5-Day Rolling Volatility")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("06_gold_vs_geopolitical.png", fig)
    log_result(f"Gold vs Geo-Sentiment (r={r:.3f}, p={p_val:.3f}) -> {significance}")


# ══════════════════════════════════════════════
# FIGURE 7 — Portfolio sensitivity to geo shocks
# ══════════════════════════════════════════════

def fig_geopolitical_sensitivity(df: pd.DataFrame, pf_ret: pd.DataFrame):
    """Q5: Which portfolio is most sensitive to geopolitical shocks?"""
    geo_col = "avg_prob_neg_geopolitical"
    if geo_col not in df.columns: return

    combined = pf_ret.join(df[[geo_col]], how="inner").dropna()
    correlations = {}
    for name in PORTFOLIOS:
        if name in combined.columns:
            r, p = stats.pearsonr(combined[geo_col], combined[name])
            correlations[name] = {"r": r, "p": p}

    names = list(correlations.keys())
    rs    = [correlations[n]["r"]   for n in names]
    ps    = [correlations[n]["p"]   for n in names]
    cols  = [PORTFOLIO_COLOURS[n]   for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, rs, color=cols, alpha=0.8, edgecolor="#0d1117")
    ax.axvline(0, color="#8b949e", lw=1, linestyle=":")

    for bar, p_val, r_val in zip(bars, ps, rs):
        sig = "★" if p_val < 0.05 else ""
        ax.text(r_val + (0.005 if r_val >= 0 else -0.005),
                bar.get_y() + bar.get_height() / 2,
                f"r={r_val:.3f} {sig}", va="center",
                ha="left" if r_val >= 0 else "right", fontsize=9)

    ax.set_title("Portfolio Sensitivity to Geopolitical Negative Sentiment\n"
                 "★ = statistically significant (p < 0.05)", fontsize=12)
    ax.set_xlabel("Pearson Correlation with Geopolitical Negative Sentiment")
    ax.set_ylabel("Portfolio")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_yticklabels([n.capitalize() for n in names])
    plt.tight_layout()
    savefig("07_geopolitical_sensitivity.png", fig)
    log_result("Portfolio Geopolitical Sensitivity:")
    for n, r_val, p_val in zip(names, rs, ps):
        log_result(f"  {n}: r={r_val:.3f} (p={p_val:.3f})")

# ══════════════════════════════════════════════
# FIGURE 8 — Tech overreaction to sentiment spikes
# ══════════════════════════════════════════════

def fig_tech_overreaction(df: pd.DataFrame, pf_ret: pd.DataFrame):
    """Q6: Do tech portfolios overreact to sentiment spikes?"""
    tech_sent_col = "sentiment_score_technology"
    if tech_sent_col not in df.columns or "tech" not in pf_ret.columns:
        return

    combined = pf_ret[["tech"]].join(df[[tech_sent_col]], how="inner").dropna()
    x = combined[tech_sent_col].values
    y = combined["tech"].values

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(x, y, c=["#e74c3c" if v < 0 else "#2ecc71" for v in x], s=80, alpha=0.8, edgecolors="#0d1117", lw=0.5)

    spike_threshold = np.percentile(np.abs(x), 95) # 95th percentile instead of 80th due to 5 year span
    for i, (xi, yi) in enumerate(zip(x, y)):
        if abs(xi) >= spike_threshold:
            ax.annotate(combined.index[i].strftime("%b %y"), (xi, yi), fontsize=7, color="white", xytext=(5, 5), textcoords="offset points")

    if len(x) > 2:
        m, b = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        r, p_val = stats.pearsonr(x, y)
        ax.plot(xline, m * xline + b, color="#4a90d9", lw=2, label=f"Regression  (r={r:.3f}, p={p_val:.3f})")
        log_result(f"Tech overreaction correlation: r={r:.3f}, p={p_val:.3f}")

    ax.axvline(0, color="#8b949e", lw=0.8, linestyle=":")
    ax.axhline(0, color="#8b949e", lw=0.8, linestyle=":")
    ax.set_title("Tech Portfolio: Returns vs Tech Sentiment\n[ - ] Negative sentiment  [ + ] Positive sentiment", fontsize=12)
    ax.set_xlabel("Technology Sentiment Score")
    ax.set_ylabel("Tech Portfolio Log Return")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("08_tech_overreaction.png", fig)


# ══════════════════════════════════════════════
# FIGURE 9 — Lag analysis
# ══════════════════════════════════════════════

def fig_lag_analysis(df: pd.DataFrame, pf_vol: pd.DataFrame):
    """Q7: Is there a measurable lag between news intensity and volatility?"""
    intensity_cols = {d: f"article_count_{d}" for d in DOMAINS if f"article_count_{d}" in df.columns}
    max_lag = min(5, len(df) - 2)
    fig, axes = plt.subplots(len(PORTFOLIOS), 1, figsize=(12, 4 * len(PORTFOLIOS)), sharex=True)
    fig.suptitle("Cross-Correlation: News Intensity → Portfolio Volatility\n(lag in trading days)", fontsize=13)

    for ax, (pf_name, colour) in zip(axes, PORTFOLIO_COLOURS.items()):
        if pf_name not in pf_vol.columns:
            ax.set_visible(False)
            continue
        vol_series = pf_vol[pf_name]
        lags = range(0, max_lag + 1)
        for domain, col in intensity_cols.items():
            corrs = []
            for lag in lags:
                combined = pd.concat([df[col].shift(lag), vol_series], axis=1).dropna()
                if len(combined) < 3:
                    corrs.append(np.nan)
                    continue
                r, _ = stats.pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])
                corrs.append(r)
            ax.plot(list(lags), corrs, marker="o", markersize=5, color=DOMAIN_COLOURS[domain], label=domain.capitalize(), lw=1.8)
        ax.axhline(0, color="#8b949e", lw=0.8, linestyle=":")
        ax.axvline(0, color="#8b949e", lw=0.8, linestyle=":")
        ax.set_title(f"{pf_name.capitalize()} Portfolio")
        ax.set_ylabel("Pearson r")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Lag (trading days news precedes volatility)")
    ax.set_xticks(list(lags))
    plt.tight_layout()
    savefig("09_lag_analysis.png", fig)


# ══════════════════════════════════════════════
# FIGURE 10 — Predictive power of each source
# ══════════════════════════════════════════════

def fig_predictive_power(df: pd.DataFrame, pf_ret: pd.DataFrame):
    """Q8: Which information source has the highest predictive power?"""
    lag_cols = {d: f"lag1_sentiment_score_{d}" for d in DOMAINS if f"lag1_sentiment_score_{d}" in df.columns}
    results = []
    for pf_name in PORTFOLIOS:
        if pf_name not in pf_ret.columns: continue
        for domain, col in lag_cols.items():
            combined = pf_ret[[pf_name]].join(df[[col]], how="inner").dropna()
            if len(combined) < 3: continue
            r, p_val = stats.pearsonr(combined[col], combined[pf_name])
            results.append({"portfolio": pf_name, "domain": domain, "r": r, "p": p_val, "abs_r": abs(r)})

    if not results: return
    res_df = pd.DataFrame(results)
    pivot  = res_df.pivot(index="portfolio", columns="domain", values="abs_r").fillna(0)
    pivot.index = [n.capitalize() for n in pivot.index]
    pivot.columns = [c.capitalize() for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(11, 6))
    x, n, w = np.arange(len(pivot)), len(pivot.columns), 0.8 / len(pivot.columns)

    for i, (col, dom_colour) in enumerate(zip(pivot.columns, [DOMAIN_COLOURS.get(c.lower(), "#888") for c in pivot.columns])):
        ax.bar(x + i * w - 0.4 + w / 2, pivot[col], width=w, color=dom_colour, alpha=0.8, label=col, edgecolor="#0d1117")

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("|Pearson r| (lag-1 sentiment → next-day return)")
    ax.set_xlabel("Portfolio")
    ax.set_title("Predictive Power of Each News Domain", fontsize=12)
    ax.legend(title="News Domain", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    savefig("10_predictive_power.png", fig)

# ══════════════════════════════════════════════
# FIGURE 11 — Time to stabilise after shock
# ══════════════════════════════════════════════

def fig_stabilisation_time(df: pd.DataFrame, pf_vol: pd.DataFrame):
    """Q9: Which portfolio takes longest to stabilise after a shock?"""
    geo_col = "sentiment_score_geopolitical"
    if geo_col not in df.columns or len(pf_vol.dropna()) < 5: return

    shock_date = df[geo_col].idxmin()
    post_shock = pf_vol.loc[pf_vol.index >= shock_date].head(15)
    if len(post_shock) < 3: return

    shock_vols = post_shock.iloc[0]
    norm       = post_shock.div(shock_vols).replace([np.inf, -np.inf], np.nan)

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, colour in PORTFOLIO_COLOURS.items():
        if name not in norm.columns: continue
        series = norm[name].dropna()
        ax.plot(range(len(series)), series.values, marker="o", markersize=5, color=colour, label=name.capitalize(), linewidth=2)

    ax.axhline(1.0, color="#ff6b6b", lw=1.5, linestyle="--", label="Shock level (t=0)")
    ax.axhline(1.0, color="#8b949e", lw=0.8, linestyle=":")
    ax.set_title(f"Time to Stabilise After Shock  (shock: {shock_date.date()})\nNormalised Rolling Volatility", fontsize=12)
    ax.set_xlabel("Trading Days After Shock")
    ax.set_ylabel("Normalised 5-Day Rolling Volatility")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("11_stabilisation_time.png", fig)

# ══════════════════════════════════════════════
# FIGURE 12 — Rolling Correlation (Tech & Tech Sentiment)
# ══════════════════════════════════════════════
def fig_rolling_correlation(df: pd.DataFrame, pf_ret: pd.DataFrame):
    """Track exactly how the relationship macro-shifts over the 5 years via 60-day rolling correlation."""
    tech_sent_col = "sentiment_score_technology"
    if tech_sent_col not in df.columns or "tech" not in pf_ret.columns:
        return
        
    combined = pf_ret[["tech"]].join(df[[tech_sent_col]], how="inner").dropna()
    # 60 day rolling correlation
    rolling_r = combined["tech"].rolling(window=60).corr(combined[tech_sent_col])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(rolling_r.index, rolling_r.values, color=PORTFOLIO_COLOURS["tech"], lw=2, label="60-Day Rolling Pearson 'r'")
    
    ax.axhline(0, color="#8b949e", lw=1, linestyle=":")
    # Highlight significant positive and negative regions conceptually
    ax.fill_between(rolling_r.index, 0, rolling_r.values, where=rolling_r.values >= 0, color="#2ecc71", alpha=0.2, interpolate=True)
    ax.fill_between(rolling_r.index, 0, rolling_r.values, where=rolling_r.values < 0, color="#e74c3c", alpha=0.2, interpolate=True)
    
    ax.set_title("Longitudinal Macro-Trend: 60-Day Rolling Correlation (Tech Returns vs. Tech Sentiment)", fontsize=13)
    ax.set_ylabel("Pearson Correlation 'r'")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    
    plt.tight_layout()
    savefig("12_rolling_correlation.png", fig)
    log_result(f"Rolling Correlation (Tech/Tech Sentiment): Peak={rolling_r.max():.3f}, Trough={rolling_r.min():.3f}")


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def run_eda_pipeline():
    print("=" * 60)
    print("  EDA  |  Portfolio Risk Analytics (V3 Longitudinal Update)")
    print("=" * 60)
    print(f"  Figures: {FIGURES_DIR.resolve()}")
    print()

    df, news, sri = load_data()
    pf_ret = portfolio_returns(df)
    pf_vol = portfolio_volatility(df)

    print(f"  Loaded master_data: {len(df)} trading days")
    print(f"  Portfolios tracked: {list(pf_ret.columns)}")
    print()

    log_result("--- EDA PIPELINE RESULTS ---")
    log_result(f"Data Date Range: {df.index.min().date()} to {df.index.max().date()}")

    print("  Generating figures...")
    fig_timeline_coverage(df)
    fig_return_distributions(pf_ret)
    fig_panic_detection(df)
    fig_correlation_heatmap(df, pf_ret)
    fig_shock_timeline(df, pf_ret)
    fig_vol_boxplots(pf_vol)
    fig_gold_vs_geopolitical(df)
    fig_geopolitical_sensitivity(df, pf_ret)
    fig_tech_overreaction(df, pf_ret)
    fig_lag_analysis(df, pf_vol)
    fig_predictive_power(df, pf_ret)
    fig_stabilisation_time(df, pf_vol)
    fig_rolling_correlation(df, pf_ret)

    save_results_log()

    print()
    print("=" * 60)
    print("  EDA COMPLETE")
    print(f"  Figures saved to:  {FIGURES_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    run_eda_pipeline()
