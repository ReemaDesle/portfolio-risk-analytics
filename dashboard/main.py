import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data paths
DATA_DIR = os.path.join(os.getcwd(), "data", "processed")
MASTER_DATA_PATH = os.path.join(DATA_DIR, "master_data.csv")
SRI_SCORES_PATH = os.path.join(DATA_DIR, "sri_scores.csv")

def load_data():
    try:
        master_df = pd.read_csv(MASTER_DATA_PATH).fillna(0)
        sri_df = pd.read_csv(SRI_SCORES_PATH).fillna(0)
        return master_df, sri_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

@app.get("/")
def read_root():
    return {"message": "Portfolio Risk Analytics API"}

@app.get("/portfolios")
def get_portfolios():
    _, sri_df = load_data()
    if sri_df is None:
        raise HTTPException(status_code=500, detail="Data not found")
    
    portfolios = sri_df['portfolio'].unique().tolist()
    return {"portfolios": portfolios}

@app.get("/analytics/{portfolio}")
def get_analytics(portfolio: str):
    master_df, sri_df = load_data()
    if master_df is None or sri_df is None:
        raise HTTPException(status_code=500, detail="Data not found")
    
    # Filter SRI scores for the selected portfolio
    p_sri = sri_df[sri_df['portfolio'] == portfolio].to_dict(orient='records')
    
    # Since master_data is global, we return it as is (or filter by date if needed)
    # Reducing size for the demo
    master_samples = master_df.tail(30).to_dict(orient='records')
    
    return {
        "sri_history": p_sri,
        "market_data": master_samples
    }

@app.get("/suggestions/{portfolio}")
def get_suggestions(portfolio: str):
    # This will eventually run M1, M2, M3 models
    # For now, we return mock/baseline suggestions based on recent SRI
    _, sri_df = load_data()
    if sri_df is None:
        raise HTTPException(status_code=500, detail="Data not found")
    
    recent_sri = sri_df[sri_df['portfolio'] == portfolio].iloc[-1]
    sri_val = recent_sri['sri']
    
    status = "Stable"
    action = "Hold"
    reasoning = "Sentiment and volatility are within normal ranges."
    
    if sri_val > 10:
        status = "High Risk"
        action = "Sell / Hedge"
        reasoning = "Significant geopolitical sentiment spikes detected. Portfolio sensitivity is high."
    elif sri_val < 0:
        status = "Oversold"
        action = "Buy"
        reasoning = "Market has overreacted to negative tech news; recovery predicted within 3 days."
        
    return {
        "status": status,
        "action": action,
        "reasoning": reasoning,
        "sri_value": sri_val,
        "category": "Tech-Reactive" if portfolio == "tech" else "Balanced"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
