"""
score_finance.py

Loads financial news headlines and scores them with FinBERT.
Assumes input data is already preprocessed (cleaned and filtered) by the scraper.
Outputs: financial_news_scored.csv (adds columns: tone_label, tone_score, prob_pos, prob_neg, prob_neu)

Usage:
    python pipeline/score_finance.py --input data/raw/news/financial_news_final.csv
"""

import argparse
import os
import ssl
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ── Fix for SSL Certificate Errors ──────────────────────────────────────────
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# ── 0. Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT_DIR / "data" / "raw" / "news" / "financial_news_final.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "raw" / "news" / "financial_news_scored.csv"

# ── 1. FinBERT scorer ─────────────────────────────────────────────────────────

LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"} # ProsusAI/finbert label order

def load_finbert(device: str):
    # Use the local model path
    model_name = str(ROOT_DIR / "models" / "finbert")
    print(f"Loading local model from {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model

def score_batch(
    headlines: list[str],
    tokenizer,
    model,
    device: str,
) -> list[dict]:
    """
    Returns list of {tone_label, tone_score, prob_pos, prob_neg, prob_neu} per headline.
    """
    inputs = tokenizer(
        headlines,
        padding=True,
        truncation=True,
        max_length=128, # headlines never exceed this
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for p in probs:
        label_idx = int(p.argmax())
        results.append({
            "tone_label": LABEL_MAP[label_idx],
            "tone_score": round(float(p[label_idx]), 4),
            "prob_pos": round(float(p[0]), 4),
            "prob_neg": round(float(p[1]), 4),
            "prob_neu": round(float(p[2]), 4),
        })
    return results

# ── 2. Main ───────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str, batch_size: int = 64):
    if not Path(input_path).exists():
        print(f"Error: Input file {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows from {input_path}")

    # Deduplicate just in case
    before = len(df)
    df = df.drop_duplicates(subset=["headline", "date"]).reset_index(drop=True)
    if len(df) < before:
        print(f"Dropped {before - len(df):,} duplicates")

    # ── FinBERT scoring setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_finbert(device)
   
    # Use float16 on GPU for ~2x speed
    if device == "cuda":
        model = model.half()

    # We score the 'headline' column (which is now cleaned by the scraper)
    texts = df["headline"].astype(str).tolist()
    records = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
        batch = texts[i : i + batch_size]
        records.extend(score_batch(batch, tokenizer, model, device))

    # Combine original df with newly computed scores
    score_df = pd.DataFrame(records)
    df = pd.concat([df.reset_index(drop=True), score_df], axis=1)

    # ── Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Summary
    print(f"\nSentiment distribution:")
    print(df["tone_label"].value_counts().to_string())
    print(f"\nMean scores by label:")
    print(df.groupby("tone_label")["tone_score"].mean().round(4).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Increase to 128/256 on GPU, keep 32-64 on CPU")
    args = parser.parse_args()
    run(args.input, args.output, args.batch_size)
