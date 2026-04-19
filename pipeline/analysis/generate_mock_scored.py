import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_mock_scored(output_path, num_years=5):
    """
    Generates a mock 'financial_news_v3_scored.csv' for pipeline testing.
    """
    start_date = datetime(2021, 1, 1)
    end_date = start_date + timedelta(days=365 * num_years)
    
    dates = []
    headlines = []
    labels = []
    
    current_date = start_date
    while current_date <= end_date:
        # Generate 5-15 headlines per day
        num_headlines = np.random.randint(5, 15)
        for _ in range(num_headlines):
            dates.append(current_date.strftime('%Y-%m-%d'))
            headlines.append(f"Mock headline for {current_date.strftime('%Y-%m-%d')}")
            # Randomly assign sentiments with a slight 'optimism bias'
            labels.append(np.random.choice(['positive', 'neutral', 'negative'], p=[0.4, 0.4, 0.2]))
        
        current_date += timedelta(days=1)
    
    df = pd.DataFrame({
        'date': dates,
        'headline': headlines,
        'sentiment_label': labels
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} mock rows at {output_path}")

if __name__ == "__main__":
    MOCK_PATH = "../../data/raw/news/financial_news_v3_scored_MOCK.csv"
    generate_mock_scored(MOCK_PATH)
