"""
Phase 13: Aggregate Multi-Capability Results & Plot

Parses the outputs from Phase 12 (WikiText-2 custom eval + lm-eval JSONs),
calculates capability retention, generates CSVs, plots, and a final Markdown report.
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

RESULTS_DIR = "benchmark_results/JetMoE-8B/multicapability_results"
PLOTS_DIR = "benchmark_results/JetMoE-8B/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

METHODS = ["Random", "Sub-MoE", "REAP", "CARE"]
COLORS = ["#95a5a6", "#3498db", "#e67e22", "#2ecc71"]
MARKERS = ["o", "s", "D", "^"]
EXPERTS = [7, 6, 4]

def parse_lmeval_results(output_dir):
    # lm-eval saves to results.json or a timestamped json file
    json_files = glob.glob(os.path.join(output_dir, "**/*.json"), recursive=True)
    if not json_files:
        return None
    
    # Try to find a file containing 'results'
    for file in json_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            if "results" in data:
                res = data["results"]
                
                # Extract MMLU
                mmlu_acc = None
                if "mmlu" in res:
                    mmlu_acc = res["mmlu"].get("acc,none", res["mmlu"].get("acc"))
                
                # Extract GSM8K
                gsm8k_acc = None
                if "gsm8k" in res:
                    gsm8k_acc = res["gsm8k"].get("exact_match,strict-match", res["gsm8k"].get("exact_match"))
                
                # Extract HumanEval
                humaneval_pass = None
                if "humaneval" in res:
                    humaneval_pass = res["humaneval"].get("pass@1", res["humaneval"].get("pass@1,none"))
                    
                return {
                    "mmlu": mmlu_acc,
                    "gsm8k": gsm8k_acc,
                    "humaneval": humaneval_pass,
                    "raw": data
                }
        except Exception:
            pass
    return None

def main():
    print("=" * 60)
    print("PHASE 13: AGGREGATING MULTI-CAPABILITY RESULTS")
    print("=" * 60)
    
    wiki_path = os.path.join(RESULTS_DIR, "results_wikitext2.json")
    if not os.path.exists(wiki_path):
        print(f"Error: {wiki_path} not found. Run Phase 12 first.")
        return
        
    with open(wiki_path, "r") as f:
        wiki_results = json.load(f)
        
    all_results = []
    
    # 1. Process Base Model First
    base_name = "jetmoe_base"
    base_wiki = wiki_results.get(base_name, {}).get("ppl", None)
    base_lmeval = parse_lmeval_results(os.path.join(RESULTS_DIR, f"{base_name}_lmeval"))
    
    base_mmlu = base_lmeval["mmlu"] if base_lmeval else None
    base_gsm8k = base_lmeval["gsm8k"] if base_lmeval else None
    base_he = base_lmeval["humaneval"] if base_lmeval else None
    
    # If base metrics are missing, we can't compute retention properly, but we'll try our best.
    # We will assume some values if they are missing just to allow plotting logic to work if testing.
    # In a real run, lm_eval output should provide these.
    
    all_results.append({
        "Method": "Base",
        "Experts": 8,
        "Compression": "0%",
        "WikiText_PPL": base_wiki,
        "MMLU": base_mmlu,
        "GSM8K": base_gsm8k,
        "HumanEval_pass1": base_he
    })
    
    # 2. Process Compressed Models
    for method in METHODS:
        for exp in EXPERTS:
            model_name = f"jetmoe_{method.lower().replace(' ', '')}_{exp}"
            if method == "CARE":
                model_name = f"jetmoe_care_{exp}" # assuming the name is jetmoe_care_X
            if method == "Sub-MoE":
                model_name = f"jetmoe_submoe_{exp}"
                
            wiki_val = wiki_results.get(model_name, {}).get("ppl", None)
            lmeval_val = parse_lmeval_results(os.path.join(RESULTS_DIR, f"{model_name}_lmeval"))
            
            mmlu_val = lmeval_val["mmlu"] if lmeval_val else None
            gsm8k_val = lmeval_val["gsm8k"] if lmeval_val else None
            he_val = lmeval_val["humaneval"] if lmeval_val else None
            
            comp_pct = f"{(8 - exp) / 8 * 100:.1f}%"
            
            all_results.append({
                "Method": method,
                "Experts": exp,
                "Compression": comp_pct,
                "WikiText_PPL": wiki_val,
                "MMLU": mmlu_val,
                "GSM8K": gsm8k_val,
                "HumanEval_pass1": he_val
            })
            
    df = pd.DataFrame(all_results)
    
    # 3. Calculate Retention
    df_retention = df.copy()
    
    if base_mmlu:
        df_retention["MMLU_Retention"] = (df["MMLU"] / base_mmlu) * 100
    if base_gsm8k:
        df_retention["GSM8K_Retention"] = (df["GSM8K"] / base_gsm8k) * 100
    if base_he:
        df_retention["HumanEval_Retention"] = (df["HumanEval_pass1"] / base_he) * 100
        
    df_retention = df_retention[["Method", "Experts", "Compression", "MMLU_Retention", "GSM8K_Retention", "HumanEval_Retention"]]
    
    # 4. Save CSVs
    df.to_csv(os.path.join(RESULTS_DIR, "jetmoe_multicapability_results.csv"), index=False)
    df_retention.to_csv(os.path.join(RESULTS_DIR, "jetmoe_capability_retention.csv"), index=False)
    
    print("Saved CSVs.")
    
    # 5. Plotting Functions
    def plot_metric(metric, ylabel, title, filename, data_df=None, is_ppl=False, hline_val=None):
        if data_df is None:
            data_df = df
        fig, ax = plt.subplots(figsize=(10, 6))
        for method, color, marker in zip(METHODS, COLORS, MARKERS):
            subset = data_df[data_df["Method"] == method]
            if subset.empty or subset[metric].isnull().all():
                continue
            x = subset["Experts"].tolist()
            y = subset[metric].tolist()
            
            # Add base point if it exists
            if hline_val is not None:
                x = [8] + x
                y = [hline_val] + y
                
            ax.plot(x, y, label=method, color=color, marker=marker, linewidth=2.5, markersize=8)
            
        ax.set_xlabel("MLP Experts per Layer", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xticks([8, 7, 6, 4])
        ax.invert_xaxis()
        
        if hline_val is not None:
            ax.axhline(hline_val, color="red", linestyle=":", linewidth=1.5, label=f"Base ({hline_val:.2f})")
            
        if is_ppl:
            ax.set_yscale("log")
            
        ax.legend(fontsize=11)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches="tight")
        plt.close()
        
    # Generate Plots
    if base_wiki:
        plot_metric("WikiText_PPL", "Perplexity (log scale, lower is better)", "WikiText-2 PPL vs Experts Remaining", "plot1_wikitext_ppl.png", is_ppl=True, hline_val=base_wiki)
        
    if "MMLU_Retention" in df_retention:
        plot_metric("MMLU_Retention", "MMLU Retention (%)", "MMLU Retention vs Experts Remaining", "plot2_mmlu_retention.png", data_df=df_retention, hline_val=100)
    
    if "GSM8K_Retention" in df_retention:
        plot_metric("GSM8K_Retention", "GSM8K Retention (%)", "GSM8K Retention vs Experts Remaining", "plot3_gsm8k_retention.png", data_df=df_retention, hline_val=100)
        
    if "HumanEval_Retention" in df_retention:
        plot_metric("HumanEval_Retention", "HumanEval Retention (%)", "HumanEval Retention vs Experts Remaining", "plot4_humaneval_retention.png", data_df=df_retention, hline_val=100)
        
    print(f"Saved plots to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
