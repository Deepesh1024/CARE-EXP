import os
import json
import numpy as np
import pandas as pd
import joblib
import sys

# Make sure we can import from exp4
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiment4'))
from config import XGBOOST_PARAMS, LOCAL_FEATURES
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler

from data_loader import load_raw_features

def main():
    os.makedirs("results/exp5", exist_ok=True)
    
    print("Loading Exp 1 data via Exp 4 data_loader...")
    try:
        df, _ = load_raw_features()
    except Exception as e:
        print(f"ERROR loading features: {e}")
        return
        
    print(f"Found {len(df)} pairs for middle layer.")
    
    # 1. Train CARE_COM Predictor
    # To deploy CARE_COM cleanly, it should use the 11 local features.
    # Geometry distance relies on Oracle KL embedding in Exp 4, which is not deployable 
    # without Oracle KL. To make it strictly deployable (zero Oracle dependency), 
    # we train CARE_COM exclusively on the 11 functional features here.
    
    X = df[LOCAL_FEATURES].values.astype(np.float64)
    y = df["Oracle_KL"].values.astype(np.float64)
    
    print("Fitting RobustScaler...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training XGBoost Regressor...")
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_scaled, y)
    
    # Save the model and scaler
    joblib.dump(scaler, "results/exp5/care_com_scaler.joblib")
    model.save_model("results/exp5/care_com_model.json")
    
    print("Saved CARE_COM predictor to results/exp5/care_com_model.json")
    
    # Also save the usage statistics of the middle layer for the original 64-expert model
    # We can extract this from expert_specialization
    exp1_data_path = "results/exp1/output.json"
    if os.path.exists(exp1_data_path):
        with open(exp1_data_path, "r") as f:
            raw_data = json.load(f)
        usage_stats = raw_data.get("expert_specialization", {}).get("middle_S512", {})
        if usage_stats:
            with open("results/exp5/original_usage_stats.json", "w") as f:
                json.dump(usage_stats, f, indent=4)
            print("Saved original usage stats.")
        
if __name__ == "__main__":
    main()
