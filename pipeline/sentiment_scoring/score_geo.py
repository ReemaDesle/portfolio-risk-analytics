"""
score_geo.py

Loads geo_news.csv, removes existing tone_score, and scores headlines with FinBERT.
Outputs: geo_news_scored.csv (adds columns: tone_label, tone_score, prob_pos, prob_neg, prob_neu)

Usage:
    python pipeline/score_geo.py
"""

import os
import ssl
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Fix for SSL Certificate Errors ──────────────────────────────────────────
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# ── 0. Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = ROOT_DIR / "data" / "raw" / "news" / "geo_news.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "raw" / "news" / "geo_news_scored.csv"
MODEL_PATH     = ROOT_DIR / "models" / "finbert"

# ── 1. FinBERT Scorer ─────────────────────────────────────────────────────────

LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

def load_finbert(device: str):
    print(f"Loading local model from {MODEL_PATH} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.to(device)
    model.eval()
    return tokenizer, model


def score_batch(
    headlines: list[str],
    tokenizer,
    model,
    device: str,
) -> list[dict]:
    inputs = tokenizer(
        headlines,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = F.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for p in probs:
        label_idx = int(p.argmax())
        results.append({
            "tone_label":  LABEL_MAP[label_idx],
            "tone_score":  round(float(p[label_idx]), 4),
            "prob_pos":    round(float(p[0]), 4),
            "prob_neg":    round(float(p[1]), 4),
            "prob_neu":    round(float(p[2]), 4),
        })
    return results


# ── 2. Run Pipeline ───────────────────────────────────────────────────────────

def run(input_path: str, output_path: str, batch_size: int = 64):
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows from {input_path}")

    # Remove existing tone_score column if present
    if "tone_score" in df.columns:
        df = df.drop(columns=["tone_score"])
        print("Removed existing 'tone_score' column.")

    # Prepare model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_finbert(device)
    
    # Use float16 on GPU for ~2x speed
    if device == "cuda":
        model = model.half()

    # Score headlines
    texts   = df["headline"].astype(str).tolist()
    records = []

    print(f"Scoring {len(texts)} headlines...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
        batch = texts[i : i + batch_size]
        records.extend(score_batch(batch, tokenizer, model, device))

    # Combine results
    score_df = pd.DataFrame(records)
    df = pd.concat([df.reset_index(drop=True), score_df], axis=1)

    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved scored data to {output_path}")

    # Summary
    print(f"\nSentiment distribution:")
    print(df["tone_label"].value_counts().to_string())
    print(f"\nMean confidence by label:")
    print(df.groupby("tone_label")["tone_score"].mean().round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=str(DEFAULT_INPUT))
    parser.add_argument("--output",     default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    run(args.input, args.output, args.batch_size)
