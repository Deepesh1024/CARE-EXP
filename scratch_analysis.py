import json
import numpy as np
from scipy.stats import spearmanr, pearsonr

tau_path = "results/exp6b/telemetry/tau_database.json"
with open(tau_path, "r") as f:
    tau_db = json.load(f)
    
layers = ["first", "middle", "last"]
checkpoints = ["checkpoint_10", "checkpoint_40", "checkpoint_70"]
next_checkpoints = {"checkpoint_10": "checkpoint_40", "checkpoint_40": "checkpoint_70", "checkpoint_70": "checkpoint_100"}

print("Correlation between TopK Routing Frequency (Tau) and Displacement Magnitude ||Delta C||")
print("-" * 80)

for layer in layers:
    for ckpt in checkpoints:
        next_ckpt = next_checkpoints[ckpt]
        
        # Load tau
        try:
            tau_topk = np.array(tau_db[layer][ckpt]["macro"]["tau_topk"])
        except KeyError:
            continue
            
        # Load DeltaC
        deltaC_path = f"results/exp6b/embeddings/q4/deltaC_{layer}_{ckpt}_to_{next_ckpt}.npy"
        try:
            deltaC = np.load(deltaC_path)
            magnitudes = np.linalg.norm(deltaC, axis=1)
            
            # Compute correlation
            rho, p1 = spearmanr(tau_topk, magnitudes)
            pr, p2 = pearsonr(tau_topk, magnitudes)
            
            print(f"Layer: {layer:6s} | {ckpt:13s} -> {next_ckpt:14s} | Spearman: {rho:6.3f} | Pearson: {pr:6.3f}")
        except FileNotFoundError:
            pass
