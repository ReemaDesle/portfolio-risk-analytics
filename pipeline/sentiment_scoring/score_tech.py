"""
score_tech.py (v3 - CPU Accelerated)
====================================
Loads tech_news_v3.csv, scores headlines with RoBERTa using CPU batching.
Includes a resume feature to continue from the last processed article.

Outputs: tech_news_scored_v3.csv
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
MODEL_DIR = ROOT_DIR / "models" / "roberta"
MODEL_ID  = "cardiffnlp/twitter-roberta-base-sentiment-latest"

DEFAULT_INPUT  = RAW_DIR / "tech_news_v3.csv"
DEFAULT_OUTPUT = RAW_DIR / "tech_news_scored_v3.csv"

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# ── CPU Optimization ────────────────────────────────────────────────────────
# Using more threads for transformer operations on CPU
torch.set_num_threads(os.cpu_count() or 4)

def load_roberta(device: str):
    # Check if directory exists AND contains weight files
    weights_exist = any((MODEL_DIR / f).exists() for f in ["pytorch_model.bin", "model.safetensors"])
    
    if MODEL_DIR.exists() and weights_exist:
        print(f"Loading local RoBERTa model from {MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    else:
        if MODEL_DIR.exists():
            print(f"Local model directory found but weights are missing. Re-downloading...")
        else:
            print(f"Downloading RoBERTa model ({MODEL_ID})...")
            
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
            "prob_neg":   round(float(p[0]), 4),
            "prob_neu":   round(float(p[1]), 4),
            "prob_pos":   round(float(p[2]), 4),
        })
    return results

def run(input_path, output_path, batch_size=16, limit=None):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # 1. Load Data
    df_raw = pd.read_csv(input_path)
    if limit:
        df_raw = df_raw.head(limit)
    
    # 2. Resume Logic
    processed_urls = set()
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path)
        processed_urls = set(df_existing['url'].tolist())
        print(f"Resuming: Found {len(processed_urls)} already scored articles.")
    
    df_to_score = df_raw[~df_raw['url'].isin(processed_urls)].copy()
    
    if len(df_to_score) == 0:
        print("All articles already scored. Nothing to do.")
        return

    print(f"Total to score: {len(df_to_score):,} (Batch Size: {batch_size})")

    # 3. Model Setup
    device = "cpu" # Explicitly CPU as requested
    tokenizer, model = load_roberta(device)

    # 4. Scoring Loop
    texts = df_to_score["headline"].astype(str).tolist()
    urls  = df_to_score["url"].tolist()
    results = []

    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
            batch_texts = texts[i : i + batch_size]
            batch_results = score_batch(batch_texts, tokenizer, model, device)
            results.extend(batch_results)
            
            # Periodic save (every 1000 items) to avoid data loss
            if (i + batch_size) % 1024 == 0 or (i + batch_size) >= len(texts):
                temp_df = df_to_score.iloc[:len(results)].copy()
                scored_slice = pd.DataFrame(results)
                combined = pd.concat([temp_df.reset_index(drop=True), scored_slice], axis=1)
                
                # Append to existing or create new
                if os.path.exists(output_path):
                    combined.to_csv(output_path, mode='a', header=False, index=False)
                else:
                    combined.to_csv(output_path, index=False)
                
                # Clear results to avoid double-saving
                results = []
                df_to_score = df_to_score.iloc[len(batch_texts):] # This logic is tricky, better to just write at end or use a more robust append
                # Actually, simple append is safer if we reset results.
    except KeyboardInterrupt:
        print("\nInterrupted! Results saved up to last checkpoint.")
    
    print(f"\nScoring complete. Output saved to {output_path}")

if __name__ == "__main__":
    # Simplified main for the user request
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--limit",  type=int, default=None)
    args = parser.parse_args()
    
    # Overriding with a more robust run function for the user
    # I'll just rewrite the actual run to handle the dataframe better
    
    def robust_run(in_p, out_p, bs, lim):
        df_all = pd.read_csv(in_p)
        if lim: df_all = df_all.head(lim)
        
        # Determine what's missing
        if os.path.exists(out_p):
            df_scored = pd.read_csv(out_p)
            scored_urls = set(df_scored['url'].unique())
            df_todo = df_all[~df_all['url'].isin(scored_urls)].copy()
        else:
            df_todo = df_all.copy()
            
        if len(df_todo) == 0:
            print("Already complete.")
            return

        tokenizer, model = load_roberta("cpu")
        texts = df_todo["headline"].astype(str).tolist()
        
        print(f"Scoring {len(df_todo):,} items on CPU...")
        all_results = []
        
        for i in tqdm(range(0, len(texts), bs), desc="Progress"):
            batch = texts[i : i + bs]
            res = score_batch(batch, tokenizer, model, "cpu")
            all_results.extend(res)
            
            # Save every 512 for safety
            if len(all_results) >= 512:
                chunk_df = df_todo.iloc[:len(all_results)].copy()
                scored_df = pd.DataFrame(all_results)
                final_chunk = pd.concat([chunk_df.reset_index(drop=True), scored_df], axis=1)
                final_chunk.to_csv(out_p, mode='a', index=False, header=not os.path.exists(out_p))
                
                df_todo = df_todo.iloc[len(all_results):]
                texts = texts[len(all_results):]
                all_results = []
        
        if all_results:
            chunk_df = df_todo.iloc[:len(all_results)].copy()
            scored_df = pd.DataFrame(all_results)
            final_chunk = pd.concat([chunk_df.reset_index(drop=True), scored_df], axis=1)
            final_chunk.to_csv(out_p, mode='a', index=False, header=not os.path.exists(out_p))

    robust_run(args.input, args.output, args.batch, args.limit)
