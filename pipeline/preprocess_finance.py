"""
preprocess_and_score.py

Cleans financial_news_4.csv and scores headlines with FinBERT.
Outputs: financial_news_scored.csv  (adds columns: clean_headline, tone_label, tone_score)

Usage:
    pip install transformers torch pandas tqdm
    python3 preprocess_and_score.py --input financial_news_4.csv
"""

import re
import argparse
import os
import ssl
from pathlib import Path
import pandas as pd

# ── Fix for SSL Certificate Errors ──────────────────────────────────────────
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ── 0. Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT_DIR / "data" / "raw" / "news" / "financial_news_final.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "raw" / "news" / "financial_news_scored.csv"

# ── 1. Noise filters ──────────────────────────────────────────────────────────

# Regex patterns for junk rows (case-insensitive via compiler flag)
JUNK_PATTERNS = [
    r"^(live updates?|live blog)",
    r"(results? (live|announced)|live result)",
    r"^(word of the day|quote of the day)",
    r"(wishes|greetings|messages|images quotes)",
    r"^(sensex today|gold rate today|nifty today)",
    r"(election \d{4} result)",
    r"(board results \d{4})",
    r"^oscars? (winner|results?)",
]

JUNK_RE = re.compile("|".join(JUNK_PATTERNS), re.IGNORECASE)

def is_junk(headline: str) -> bool:
    if not isinstance(headline, str): return True
    return bool(JUNK_RE.search(headline))


# ── 2. Text cleaner ───────────────────────────────────────────────────────────

def clean_headline(text: str) -> str:
    if not isinstance(text, str): return ""
    # Decode HTML entities
    text = text.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    # Remove content after pipe/dash separators (source branding artifacts)
    # e.g. "Nifty falls 200 points | ET Markets"
    text = re.sub(r"\s*[|\u2014\u2013]\s*(ET Markets?|Moneycontrol|Livemint|BSE|NSE).*$", "", text, flags=re.I)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── 3. FinBERT scorer ─────────────────────────────────────────────────────────

LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}   # ProsusAI/finbert label order

def load_finbert(device: str):
    # Use the local model path you provided
    model_name = str(ROOT_DIR / "models" / "finbert")
    print(f"Loading local model from {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)
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
    Returns list of {tone_label, positive, negative, neutral} per headline.
    FinBERT max token length is 512 — headlines are well under that.
    """
    inputs = tokenizer(
        headlines,
        padding=True,
        truncation=True,
        max_length=128,   # headlines never exceed this; saves compute vs 512
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
            "tone_score":  round(float(p[label_idx]), 4),   # confidence of winning label
            "prob_pos":    round(float(p[0]), 4),
            "prob_neg":    round(float(p[1]), 4),
            "prob_neu":    round(float(p[2]), 4),
        })
    return results


# ── 4. Main ───────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str, batch_size: int = 64):
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows")

    # ── Step 1: Drop junk rows
    before = len(df)
    df = df[~df["headline"].apply(is_junk)].copy()
    print(f"Dropped {before - len(df):,} junk rows → {len(df):,} remain")

    # ── Step 2: Drop rows below minimum length (after junk filter)
    df = df[df["headline"].str.len() >= 40].copy()
    print(f"After length filter: {len(df):,} rows")

    # ── Step 3: Clean headlines
    df["clean_headline"] = df["headline"].apply(clean_headline)

    # ── Step 4: Drop exact duplicates on cleaned text
    before = len(df)
    df = df.drop_duplicates(subset="clean_headline").reset_index(drop=True)
    print(f"Dropped {before - len(df):,} duplicate clean headlines → {len(df):,} remain")

    # ── Step 5: FinBERT scoring
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float16 on GPU for ~2x speed, stays float32 on CPU
    tokenizer, model = load_finbert(device)
    if device == "cuda":
        model = model.half()

    texts   = df["clean_headline"].tolist()
    records = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
        batch   = texts[i : i + batch_size]
        records.extend(score_batch(batch, tokenizer, model, device))

    score_df = pd.DataFrame(records)
    df = pd.concat([df.reset_index(drop=True), score_df], axis=1)

    # ── Step 6: Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Summary
    print(f"\nSentiment distribution:")
    print(df["tone_label"].value_counts().to_string())
    print(f"\nMean scores by label:")
    print(df.groupby("tone_label")["tone_score"].mean().round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=str(DEFAULT_INPUT))
    parser.add_argument("--output",     default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Increase to 128/256 on GPU, keep 32-64 on CPU")
    args = parser.parse_args()
    run(args.input, args.output, args.batch_size)