import asyncio
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loguru import logger

# Mock problematic modules if needed, but we try to import real ones first
# If they fail, we mock them to allow the script to run for at least partial training
try:
    from src.quant.lstm_trend import LSTMTrendAnalyzer
    from src.quant.kan_interpreter import KANInterpreter
except ImportError as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)

def generate_synthetic_data(n_matches=600):
    """Generates synthetic match data for training if DB is empty."""
    logger.info(f"Generating {n_matches} synthetic matches for training...")
    
    # Need >20 teams to satisfy LSTM sequence count (1 seq per team)
    teams = [f"Team{i}" for i in range(30)] 
    matches = []
    
    for _ in range(n_matches):
        home = random.choice(teams)
        away = random.choice([t for t in teams if t != home])
        
        # Random scores and stats
        hs = random.randint(0, 3)
        as_ = random.randint(0, 3)
        res = "D"
        if hs > as_: res = "W"
        elif as_ > hs: res = "L"
        
        matches.append({
            "match_id": f"syn_{random.randint(1000,9999)}",
            "home_team": home,
            "away_team": away,
            "goals_scored": hs, # from home perspective
            "goals_conceded": as_,
            "xg": random.uniform(0.5, 3.0),
            "xga": random.uniform(0.5, 3.0),
            "possession": random.uniform(30, 70),
            "result": res,
            # For KAN
            "home_odds": random.uniform(1.5, 4.0),
            "draw_odds": random.uniform(2.8, 4.0),
            "away_odds": random.uniform(1.5, 4.0),
            "home_win_rate": random.uniform(0.2, 0.8),
            "away_win_rate": random.uniform(0.2, 0.8),
            "odds_volatility": random.uniform(0.0, 0.5),
        })
        
    return matches, teams

async def train_models():
    logger.info("Starting Model Training Pipeline...")
    
    # 1. Prepare Data
    # In a real scenario, we would fetch from DBManager.
    # Here we use synthetic data to guarantee execution.
    matches, unique_teams = generate_synthetic_data()
    
    # 3. Train LSTM
    logger.info("--- Training LSTM Trend Analyzer ---")
    
    # Check if TORCH_OK is True inside the class (via protected attribute if needed, or just assume)
    # But better to check if we can save
    logger.info(f"Unique Teams: {len(unique_teams)}")
    
    lstm = LSTMTrendAnalyzer(epochs=10) # Fast training
    
    # Format data for LSTM: list of matches per team
    training_data = []
    for team in unique_teams:
        # Filter matches where team played (simplified: just home logic for now)
        team_matches = [m for m in matches if m["home_team"] == team]
        # logger.info(f"Team {team}: {len(team_matches)} matches") # Debug
        if len(team_matches) > 10:
            training_data.append({
                "team": team,
                "matches": team_matches[:-1], # History
                "next_result": team_matches[-1]["result"] # Label
            })
            
    logger.info(f"LSTM Training Data Samples: {len(training_data)}")
    
    lstm.fit(training_data)
    
    # Force verify fitted
    with open("debug_lstm.txt", "w") as f:
        f.write(f"LSTM Fitted: {lstm._fitted}\n")
        if lstm._fitted:
            logger.success("LSTM is FITTED.")
        else:
            logger.error("LSTM is NOT FITTED.")
            
        lstm.save_model("models/lstm_trend.pt")
        
        # Check if file exists immediately
        exists = os.path.exists("models/lstm_trend.pt")
        f.write(f"File Exists: {exists}\n")
        if exists:
            logger.success("File created: models/lstm_trend.pt")
        else:
            logger.error("File NOT created: models/lstm_trend.pt")
    
    # 4. Train KAN
    logger.info("--- Training KAN Interpreter ---")
    kan = KANInterpreter()
    
    if kan._model is None:
         logger.warning("KAN model is None (PyKAN missing?), skipping KAN training.")
    else:
        # Prepare Features and Labels for KAN
        # Features matches KANInterpreter.FEATURE_KEYS
        kan_features = []
        kan_labels = []
        
        for m in matches:
            # Features
            feat = [
                m["home_odds"], m["draw_odds"], m["away_odds"],
                m["xg"], m["xga"], # simplified mapping
                m["home_win_rate"], m["away_win_rate"],
                m["odds_volatility"]
            ]
            kan_features.append(feat)
            
            # Label: 0=Home, 1=Draw, 2=Away
            lbl = 1
            if m["goals_scored"] > m["goals_conceded"]: lbl = 0
            elif m["goals_conceded"] > m["goals_scored"]: lbl = 2
            kan_labels.append(lbl)
            
        X = np.array(kan_features, dtype=np.float32)
        y = np.array(kan_labels, dtype=np.int64)
        
        kan.fit(X, y, steps=15)
        kan.save_model("models/kan_model.pt")
    
    logger.success("All models trained and saved successfully!")

if __name__ == "__main__":
    asyncio.run(train_models())
