import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 1. Define deterministic paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
input_file = os.path.join(_PROJECT_ROOT, "results", "exp1", "output.json")
# 2. Load and parse the raw checkpoint
if not os.path.exists(input_file):
    raise FileNotFoundError(f"CRITICAL ERROR: {input_file} not found. Verify pipeline output.")

with open(input_file, "r") as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError("CRITICAL ERROR: JSON read collision. Wait 5 seconds and re-run.")

df = pd.DataFrame(data.get("results", []))

if df.empty:
    raise ValueError("CRITICAL ERROR: JSON results array is empty.")

# 3. Define target parameters strictly for completed layers
layers = ["first", "middle", "last"]
proxies = [
    "Weight_Distance", 
    "Weight_Cosine", 
    "Activation_Similarity", 
    "Output_Similarity", 
    "Routing_Similarity",
    "Usage_Frequency",
    "Jaccard_Overlap"
]

# 4. Execute segmented plotting across all available sequence lengths
for seq_len in sorted(df["Seq_Len"].unique()):
    df_seq = df[df["Seq_Len"] == seq_len]
    if df_seq.empty:
        continue
        
    base_output_dir = os.path.join(_PROJECT_ROOT, "results", "exp1", f"{int(seq_len)}_segmented")
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"\n==========================================")
    print(f"Executing plotting for N={int(seq_len)}")
    print(f"==========================================")
    
    for layer in layers:
        # Slice dataframe by layer
        df_layer = df_seq[df_seq["Layer"] == layer]
        
        if df_layer.empty:
            print(f"WARNING: No data found for layer '{layer}' at N={seq_len}. Skipping.")
            continue
            
        # Create isolated output directory for this layer
        layer_dir = os.path.join(base_output_dir, layer)
        os.makedirs(layer_dir, exist_ok=True)
        
        print(f"Processing Layer: {layer.upper()} | Pairs: {len(df_layer)}")
        
        for proxy in proxies:
            if proxy in df_layer.columns:
                # Drop NaNs to prevent scipy errors
                clean_df = df_layer.dropna(subset=[proxy, "Oracle_KL"])
                
                if len(clean_df) < 2:
                    print(f"  -> WARNING: Not enough data points to compute correlation for '{proxy}'.")
                    continue
                
                # Compute Spearman rank correlation
                rho, p_value = spearmanr(clean_df[proxy], clean_df["Oracle_KL"])
                
                plt.figure(figsize=(8, 6))
                plt.scatter(clean_df[proxy], clean_df["Oracle_KL"], alpha=0.3, c="crimson", s=12)
                
                plt.xlabel(proxy, fontweight='bold')
                plt.ylabel("Oracle KL Divergence", fontweight='bold')
                plt.title(f"N={int(seq_len)} | Layer: {layer.capitalize()} | {proxy} vs Oracle_KL\nSpearman $\\rho$ = {rho:.4f} (p={p_value:.2e})", pad=10)
                plt.grid(True, linestyle="--", alpha=0.4)
                
                save_path = os.path.join(layer_dir, f"scatter_{layer}_{proxy}.png")
                plt.tight_layout()
                plt.savefig(save_path, dpi=200)
                plt.close()
                
                print(f"  -> Rendered: {proxy:<25} | Spearman rho: {rho:+.4f}")
            else:
                print(f"  -> WARNING: Metric '{proxy}' not found.")

print("\nExecution complete. Inspect the isolated diagnostics in results/exp1/.")