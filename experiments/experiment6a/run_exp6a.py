import os
import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

EXP3C_DIR = "results/exp3c"
EXP6_DIR = "results/exp6"
os.makedirs(os.path.join(EXP6_DIR, "predictions"), exist_ok=True)
os.makedirs(os.path.join(EXP6_DIR, "metrics"), exist_ok=True)

# 1. Load checkpoints metadata
with open(os.path.join(EXP3C_DIR, "checkpoint_metadata.json"), "r") as f:
    metadata = json.load(f)
checkpoints = sorted(metadata.values(), key=lambda x: x["actual_pct"])
t_steps = [c["actual_step"] for c in checkpoints]
names = [c["checkpoint_name"] for c in checkpoints]

layers = ["first", "middle", "last"]

def get_pairs_dict(path):
    df = pd.read_csv(path, header=None).values
    pairs = {}
    for i in range(df.shape[0]):
        for j in range(i+1, df.shape[1]):
            if not np.isnan(df[i, j]):
                pairs[(i, j)] = df[i, j]
    return pairs

# Collect data
data = {layer: [] for layer in layers}
for layer in layers:
    for name in names:
        path = os.path.join(EXP3C_DIR, name, layer, "oracle_distance.csv")
        data[layer].append(get_pairs_dict(path))

# Intersection of pairs
common_pairs = {}
for layer in layers:
    s = set(data[layer][0].keys())
    for d in data[layer][1:]:
        s = s.intersection(d.keys())
    common_pairs[layer] = sorted(list(s))

def eval_metrics(y_true, y_pred):
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    k = max(1, int(len(y_true) * 0.1))
    top_true = set(np.argsort(y_true)[:k])
    top_pred = set(np.argsort(y_pred)[:k])
    overlap = len(top_true.intersection(top_pred)) / k
    
    return {"spearman": float(sr), "pearson": float(pr), "rmse": float(rmse), "mae": float(mae), "r2": float(r2), "top10_overlap": float(overlap)}

results = []

# --- PREDICTION TASKS ---
for task_idx in range(1, len(checkpoints)):
    target_idx = task_idx
    current_idx = task_idx - 1
    past_idx = task_idx - 2 if task_idx >= 2 else None
    
    target_name = names[target_idx]
    current_name = names[current_idx]
    
    delta_t_future = t_steps[target_idx] - t_steps[current_idx]
    
    for layer in layers:
        pairs = common_pairs[layer]
        y_true = np.array([data[layer][target_idx][p] for p in pairs])
        current_vals = np.array([data[layer][current_idx][p] for p in pairs])
        
        # 1. Mean Baseline
        mean_val = np.mean(current_vals)
        pred_mean = np.full_like(y_true, mean_val)
        res_mean = eval_metrics(y_true, pred_mean)
        # Note: correlation for constant is NaN, so we handle it gracefully below by overriding 0.
        res_mean["spearman"] = 0.0
        res_mean["pearson"] = 0.0
        
        results.append({
            "task": f"{current_name} -> {target_name}",
            "layer": layer,
            "method": "Mean",
            **res_mean
        })
        
        # 2. Persistence Baseline
        res_pers = eval_metrics(y_true, current_vals)
        results.append({
            "task": f"{current_name} -> {target_name}",
            "layer": layer,
            "method": "Persistence",
            **res_pers
        })
        
        # 3. Linear Extrapolation
        if past_idx is not None:
            past_vals = np.array([data[layer][past_idx][p] for p in pairs])
            delta_t_past = t_steps[current_idx] - t_steps[past_idx]
            scale = delta_t_future / delta_t_past
            pred_lin = current_vals + scale * (current_vals - past_vals)
            res_lin = eval_metrics(y_true, pred_lin)
            results.append({
                "task": f"{current_name} -> {target_name}",
                "layer": layer,
                "method": "Linear Extrapolation",
                **res_lin
            })
            
    # 4. Learned Predictor
    # To predict target_idx, we can train on all transitions up to current_idx
    # Transitions available for training: (i -> i+1) for i+1 <= current_idx
    if current_idx >= 1: # We need at least one past transition to train on
        train_X = []
        train_y = []
        
        # Build training set from all past valid transitions
        for train_target_idx in range(1, current_idx + 1):
            train_current_idx = train_target_idx - 1
            train_past_idx = train_target_idx - 2 if train_target_idx >= 2 else None
            
            dt_fut = t_steps[train_target_idx] - t_steps[train_current_idx]
            
            for l_idx, layer in enumerate(layers):
                pairs = common_pairs[layer]
                tr_y = np.array([data[layer][train_target_idx][p] for p in pairs])
                tr_c = np.array([data[layer][train_current_idx][p] for p in pairs])
                
                # If we have a past past, include delta D, otherwise 0
                if train_past_idx is not None:
                    tr_p = np.array([data[layer][train_past_idx][p] for p in pairs])
                    tr_d = tr_c - tr_p
                else:
                    tr_d = np.zeros_like(tr_c)
                    
                for i in range(len(pairs)):
                    train_X.append([tr_c[i], tr_d[i], dt_fut, 1 if l_idx==0 else 0, 1 if l_idx==1 else 0])
                    train_y.append(tr_y[i])
                    
        model = Ridge(alpha=1.0)
        model.fit(train_X, train_y)
        
        # Evaluate on current -> target
        for l_idx, layer in enumerate(layers):
            pairs = common_pairs[layer]
            y_true = np.array([data[layer][target_idx][p] for p in pairs])
            current_vals = np.array([data[layer][current_idx][p] for p in pairs])
            
            if past_idx is not None:
                past_vals = np.array([data[layer][past_idx][p] for p in pairs])
                delta_d = current_vals - past_vals
            else:
                delta_d = np.zeros_like(current_vals)
                
            test_X = []
            for i in range(len(pairs)):
                test_X.append([current_vals[i], delta_d[i], delta_t_future, 1 if l_idx==0 else 0, 1 if l_idx==1 else 0])
                
            pred_learned = model.predict(test_X)
            res_learn = eval_metrics(y_true, pred_learned)
            results.append({
                "task": f"{current_name} -> {target_name}",
                "layer": layer,
                "method": "Ridge Regression",
                **res_learn
            })

df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(EXP6_DIR, "metrics", "prediction_results.csv"), index=False)

# Format markdown table
with open(os.path.join(EXP6_DIR, "metrics", "results_summary.md"), "w") as f:
    f.write("# Prediction Results\n\n")
    for task in df_res['task'].unique():
        f.write(f"## Task: {task}\n")
        f.write(df_res[df_res['task'] == task].drop(columns=['task']).to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")

print("PREDICTION TASKS COMPLETE")
