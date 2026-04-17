"""
score_geo.py (v3 - CPU Accelerated)
===================================
Loads geo_news_v3.csv, scores headlines with FinBERT using CPU batching.
Includes a resume feature to continue from the last processed article.

Outputs: geo_news_scored_v3.csv
"""

import os
import ssl
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Fix for SSL ─────────────────────────────────────────────────────────────
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR  = ROOT_DIR / "data" / "raw" / "news"
MODEL_DIR = ROOT_DIR / "models" / "finbert"
MODEL_ID  = "ProsusAI/finbert"

DEFAULT_INPUT  = RAW_DIR / "geo_news_v3.csv"
DEFAULT_OUTPUT = RAW_DIR / "geo_news_scored_v3.csv"

# FinBERT labels: 0=pos, 1=neg, 2=neu
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

# ── CPU Optimization ────────────────────────────────────────────────────────
torch.set_num_threads(os.cpu_count() or 4)

def load_finbert(device: str):
    # Check if directory exists AND contains weight files
    weights_exist = any((MODEL_DIR / f).exists() for f in ["pytorch_model.bin", "model.safetensors"])
    
    if MODEL_DIR.exists() and weights_exist:
        print(f"Loading local FinBERT model from {MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    else:
        if MODEL_DIR.exists():
            print(f"Local model directory found but weights are missing. Re-downloading...")
        else:
            print(f"Downloading FinBERT model ({MODEL_ID})...")
            
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        
        # Save locally for future use
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(MODEL_DIR))
        model.save_pretrained(str(MODEL_DIR))
        print(f"Model saved locally to {MODEL_DIR}")
    
    model.to(device)
    model.eval()
    return tokenizer, model

def score_batch(headlines, tokenizer, model, device):
    inputs = tokenizer(
        headlines, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = F.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for p in probs:
        idx = int(p.argmax())
        results.append({
            "tone_label": LABEL_MAP[idx],
            "tone_score": round(float(p[idx]), 4),
            "prob_pos":   round(float(p[0]), 4),
            "prob_neg":   round(float(p[1]), 4),
            "prob_neu":   round(float(p[2]), 4),
        })
    return results

def robust_run(in_p, out_p, bs, lim):
    if not os.path.exists(in_p):
        print(f"Error: {in_p} not found.")
        return

    df_all = pd.read_csv(in_p)
    if lim: df_all = df_all.head(lim)
    
    # Determine what's missing
    if os.path.exists(out_p):
        df_scored = pd.read_csv(out_p)
        scored_urls = set(df_scored['url'].unique())
        df_todo = df_all[~df_all['url'].isin(scored_urls)].copy()
        print(f"Resuming: Found {len(scored_urls):,} already scored articles.")
    else:
        df_todo = df_all.copy()
        
    if len(df_todo) == 0:
        print("Scoring complete. Output is up to date.")
        return

    # Filter out empty or extremely short headlines that can crash the tokenizer
    df_todo['headline'] = df_todo['headline'].fillna("").astype(str).str.strip()
    df_todo = df_todo[df_todo['headline'].str.len() > 1].copy()
    
    if len(df_todo) == 0:
        print("No articles with valid content found to score.")
        return

    print(f"Scoring {len(df_todo):,} articles on CPU (Batch size: {bs})...")
    tokenizer, model = load_finbert("cpu")
    texts = df_todo["headline"].astype(str).tolist()
    
    all_results = []
    
    try:
        for i in tqdm(range(0, len(texts), bs), desc="Progress"):
            batch = texts[i : i + bs]
            res = score_batch(batch, tokenizer, model, "cpu")
            all_results.extend(res)
            
            # Save every 512 for safety
            if len(all_results) >= 512 or (i + bs) >= len(texts):
                chunk_df = df_todo.iloc[:len(all_results)].copy()
                scored_df = pd.DataFrame(all_results)
                final_chunk = pd.concat([chunk_df.reset_index(drop=True), scored_df], axis=1)
                
                # Append to file
                final_chunk.to_csv(out_p, mode='a', index=False, header=not os.path.exists(out_p))
                
                # Advance remaining
                df_todo = df_todo.iloc[len(all_results):]
                texts = texts[len(all_results):]
                all_results = []
    except KeyboardInterrupt:
        print("\nInterrupted! Current progress saved.")

    print(f"\nFinal results available at: {out_p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--limit",  type=int, default=None)
    args = parser.parse_args()
    
    robust_run(args.input, args.output, args.batch, args.limit)
