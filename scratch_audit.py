import sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

sys.path.insert(0, '/Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/experiments/experiment4')
from data_loader import load_all
from model_a import train_model_a, predict_model_a
from config import RANDOM_SEED

def main():
    print("--- Model-A Sanity Check (Naive Pair-Level Split) ---")
    data = load_all()
    X = data["X_unscaled"]
    y = data["y"]
    
    # 80/20 train/test split at the pair level (no expert disjointness constraint)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    model, scaler = train_model_a(X_train, y_train)
    y_pred = predict_model_a(model, scaler, X_test)
    
    rho, _ = spearmanr(y_test, y_pred)
    print(f"Model-A Spearman ρ (Naive Pair-Level Split): {rho:.4f}")

    print("\n--- Top-K Audit ---")
    df = pd.read_csv("/Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/results/exp4/fold_metrics.csv")
    
    for k in [10, 25, 50]:
        col_b = f"prec_at_{k}_B"
        col_c = f"prec_at_{k}_C"
        
        b_wins = (df[col_b] > df[col_c]).sum()
        c_wins = (df[col_c] > df[col_b]).sum()
        ties = (df[col_b] == df[col_c]).sum()
        
        mean_b = df[col_b].mean()
        mean_c = df[col_c].mean()
        
        print(f"K={k}:")
        print(f"  Mean Prec@B: {mean_b:.4f}")
        print(f"  Mean Prec@C: {mean_c:.4f}")
        print(f"  B wins: {b_wins}, C wins: {c_wins}, Ties: {ties}")
        print(f"  Is B > C consistently? {'Yes' if mean_b > mean_c and b_wins > c_wins else 'No'}")

if __name__ == "__main__":
    main()
