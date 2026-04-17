import pandas as pd
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = ROOT_DIR / "data" / "raw" / "news" / "financial_news_final.csv"

def clean_csv():
    if not FILE_PATH.exists():
        print(f"Error: File not found at {FILE_PATH}")
        return

    print(f"Reading {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)
    initial_count = len(df)
    
    # Deduplicate based on headline and date
    print("Removing duplicates...")
    df_clean = df.drop_duplicates(subset=["headline", "date"], keep="first")
    final_count = len(df_clean)
    
    # Save back
    df_clean.to_csv(FILE_PATH, index=False)
    
    print("-" * 50)
    print(f"Initial rows: {initial_count}")
    print(f"Final rows:   {final_count}")
    print(f"Removed:       {initial_count - final_count}")
    print("-" * 50)
    
    print("\nSource Distribution:")
    print(df_clean["source"].value_counts().to_string())
    print("-" * 50)
    
    if final_count > 0:
        print("\nDate Range:")
        print(f"Start: {df_clean['date'].min()}")
        print(f"End:   {df_clean['date'].max()}")

    # Filter and save Moneycontrol specifically
    #mc_only = df_clean[df_clean['source'] == 'Moneycontrol']
    #mc_path = ROOT_DIR / "data" / "raw" / "news" / "moneycontrol_cleaned.csv"
    #mc_only.to_csv(mc_path, index=False)
    #print(f"Saved {len(mc_only)} Moneycontrol rows to {mc_path.name}")


if __name__ == "__main__":
    clean_csv()
