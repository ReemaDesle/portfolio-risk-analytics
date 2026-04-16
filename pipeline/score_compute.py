"""
Portfolio Risk Analytics — Load / Compute Scoring
=================================================
Load stage (ETL): reads the cleaned `master_data.csv`, computes the
final mathematical risk formulas per portfolio, and saves the SRI scores.

Formulas
--------
  Risk Score (0-100)   = Weighted avg of negative sentiment across domains
  Safety Score (0-100) = Weighted avg of positive sentiment across domains
  SRI                  = Risk Score − Safety Score

Inputs
------
  data/processed/master_data.csv

Outputs
-------
  data/processed/sri_scores.csv

Usage
-----
  python pipeline/score_compute.py
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MASTER_DATA   = PROCESSED_DIR / "master_data.csv"
SRI_SCORES    = PROCESSED_DIR / "sri_scores.csv"

# ──────────────────────────────────────────────
# Portfolio definitions & Weights
# ──────────────────────────────────────────────
PORTFOLIOS = {
    "geopolitical": ["GLD", "USO", "LMT", "RTX", "EEM", "GC=F", "CL=F"],
    "tech":         ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "SOXX", "QQQ"],
    "balanced":     ["SPY", "AGG", "GLD", "VTI", "EFA", "BND"],
    "conservative": ["TLT", "IEF", "VPU", "KO", "JNJ", "PG", "XLP"],
}

PORTFOLIO_DOMAIN_WEIGHTS = {
    "geopolitical": {"geopolitical": 0.6, "financial": 0.3, "technology": 0.1},
    "tech":         {"technology":   0.6, "financial": 0.3, "geopolitical": 0.1},
    "balanced":     {"financial":    0.4, "geopolitical": 0.3, "technology": 0.3},
    "conservative": {"financial":    0.5, "geopolitical": 0.4, "technology": 0.1},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# COMPUTE FORMULAS
# ══════════════════════════════════════════════

def compute_risk_scores(master: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Risk Score, Safety Score, and SRI.
    """
    log.info("── Computing Risk / Safety / SRI ───────────")
    rows = []

    for date_val, row in master.iterrows():
        # Ensure we have a string date for output formatting
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d") if "date" in master.columns else pd.to_datetime(date_val).strftime("%Y-%m-%d")

        for portfolio, domain_weights in PORTFOLIO_DOMAIN_WEIGHTS.items():
            risk_score   = 0.0
            safety_score = 0.0
            total_weight = 0.0

            for domain, weight in domain_weights.items():
                neg_val = row.get(f"avg_prob_neg_{domain}", np.nan)
                pos_val = row.get(f"avg_prob_pos_{domain}", np.nan)

                if pd.notna(neg_val) and pd.notna(pos_val):
                    risk_score   += weight * neg_val
                    safety_score += weight * pos_val
                    total_weight += weight

            if total_weight > 0:
                risk_score   = round((risk_score   / total_weight) * 100, 4)
                safety_score = round((safety_score / total_weight) * 100, 4)
            else:
                risk_score = safety_score = np.nan

            sri = round(risk_score - safety_score, 4) if pd.notna(risk_score) else np.nan

            rows.append({
                "date":         date_str,
                "portfolio":    portfolio,
                "risk_score":   risk_score,
                "safety_score": safety_score,
                "sri":          sri,
            })

    sri_df = pd.DataFrame(rows).sort_values(["date", "portfolio"]).reset_index(drop=True)
    log.info("  SRI rows generated: %d", len(sri_df))
    return sri_df


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def run():
    log.info("═" * 60)
    log.info("  LOAD (COMPUTE)  |  Portfolio Risk Analytics")
    log.info("═" * 60)

    if not MASTER_DATA.exists():
        raise FileNotFoundError(f"Cannot find {MASTER_DATA}. Run clean_data.py first!")

    master = pd.read_csv(MASTER_DATA)
    log.info("  Loaded master_data: %d rows", len(master))

    sri_df = compute_risk_scores(master)

    # Save
    sri_df.to_csv(SRI_SCORES, index=False)

    log.info("")
    log.info("═" * 60)
    log.info("  COMPLETE")
    log.info("  SRI summary (mean per portfolio):")
    summary = sri_df.groupby("portfolio")[["risk_score", "safety_score", "sri"]].mean().round(2)
    log.info("\n%s", summary.to_string())
    log.info("")
    log.info("  Outputs → %s", SRI_SCORES.resolve())
    log.info("═" * 60)

    return sri_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute final Risk metrics from cleaned master data."
    )
    args = parser.parse_args()
    run()
