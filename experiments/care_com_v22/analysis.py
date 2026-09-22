import os
import json
import glob
import pandas as pd

def generate_analysis(config):
    """
    Parses the generated JSON traces and creates tables and summary Markdown report.
    """
    data = {}
    
    # 1. Load Data
    adaptive_path = os.path.join(config.trajectories_dir, "adaptive_v21.json")
    if os.path.exists(adaptive_path):
        with open(adaptive_path, "r") as f:
            data["Adaptive"] = json.load(f)
            
    static_path = os.path.join(config.trajectories_dir, "static_v1.json")
    if os.path.exists(static_path):
        with open(static_path, "r") as f:
            data["Static"] = json.load(f)
            
    random_files = glob.glob(os.path.join(config.trajectories_dir, "random_seed_*.json"))
    data["Random"] = []
    for f_path in random_files:
        with open(f_path, "r") as f:
            data["Random"].append(json.load(f))
            
    if not data.get("Adaptive") or not data.get("Static") or not data.get("Random"):
        print("Missing trajectory files, skipping analysis.")
        return
        
    # 2. Extract PPL results
    ppl_data = []
    for m in ["Adaptive", "Static"]:
        ppl_dict = data[m]["ppl"]
        ppl_data.append({
            "Method": m,
            "PPL@64": ppl_dict.get(str(64), ppl_dict.get(64, None)),
            "PPL@60": ppl_dict.get(str(60), ppl_dict.get(60, None)),
            "PPL@56": ppl_dict.get(str(56), ppl_dict.get(56, None))
        })
        
    # Aggregate random
    rnd_64 = []
    rnd_60 = []
    rnd_56 = []
    for rnd in data["Random"]:
        ppl = rnd["ppl"]
        rnd_64.append(ppl.get(str(64), ppl.get(64, None)))
        rnd_60.append(ppl.get(str(60), ppl.get(60, None)))
        rnd_56.append(ppl.get(str(56), ppl.get(56, None)))
        
    import numpy as np
    ppl_data.append({
        "Method": f"Random (mean {len(data['Random'])} seeds)",
        "PPL@64": f"{np.mean(rnd_64):.4f} \u00B1 {np.std(rnd_64):.4f}",
        "PPL@60": f"{np.mean(rnd_60):.4f} \u00B1 {np.std(rnd_60):.4f}",
        "PPL@56": f"{np.mean(rnd_56):.4f} \u00B1 {np.std(rnd_56):.4f}",
    })
    
    df_ppl = pd.DataFrame(ppl_data)
    df_ppl.to_csv(os.path.join(config.tables_dir, "compression_results.csv"), index=False)
    
    # 3. Extract Marginal KL
    kl_data = []
    for m in ["Adaptive", "Static"]:
        trace = data[m]["trace"]
        damages = [step["functional_damage"] for step in trace]
        kl_data.append({
            "Method": m,
            "Mean Marginal KL": np.mean(damages),
            "Cumulative KL": np.sum(damages)
        })
        
    rnd_damages_mean = []
    rnd_damages_sum = []
    for rnd in data["Random"]:
        damages = [step["functional_damage"] for step in rnd["trace"]]
        rnd_damages_mean.append(np.mean(damages))
        rnd_damages_sum.append(np.sum(damages))
        
    kl_data.append({
        "Method": f"Random (mean {len(data['Random'])} seeds)",
        "Mean Marginal KL": f"{np.mean(rnd_damages_mean):.4f} \u00B1 {np.std(rnd_damages_mean):.4f}",
        "Cumulative KL": f"{np.mean(rnd_damages_sum):.4f} \u00B1 {np.std(rnd_damages_sum):.4f}"
    })
    
    df_kl = pd.DataFrame(kl_data)
    df_kl.to_csv(os.path.join(config.tables_dir, "marginal_damage.csv"), index=False)
    
    # 4. Pair Selection Comparison
    pair_comparison = []
    adaptive_trace = data["Adaptive"]["trace"]
    static_trace = data["Static"]["trace"]
    
    state_diverged = False
    
    for i in range(len(adaptive_trace)):
        a_step = adaptive_trace[i]
        s_step = static_trace[i]
        
        a_pair = tuple(sorted(a_step["selected_pair"]))
        s_pair = tuple(sorted(s_step["selected_pair"]))
        
        if a_pair != s_pair and not state_diverged:
            pair_comparison.append({
                "Step": a_step["experts_before"],
                "Static Pair": s_pair,
                "Adaptive Pair": a_pair,
                "Static Capability Dist": s_step["capability_distance"],
                "Adaptive Capability Dist": a_step["capability_distance"],
                "Static Marginal KL": s_step["functional_damage"],
                "Adaptive Marginal KL": a_step["functional_damage"],
                "Delta KL (Static - Adaptive)": s_step["functional_damage"] - a_step["functional_damage"],
                "Note": "Direct comparison at identical state M_t"
            })
            state_diverged = True
        elif a_pair != s_pair:
            pair_comparison.append({
                "Step": a_step["experts_before"],
                "Static Pair": s_pair,
                "Adaptive Pair": a_pair,
                "Static Capability Dist": s_step["capability_distance"],
                "Adaptive Capability Dist": a_step["capability_distance"],
                "Static Marginal KL": s_step["functional_damage"],
                "Adaptive Marginal KL": a_step["functional_damage"],
                "Delta KL (Static - Adaptive)": s_step["functional_damage"] - a_step["functional_damage"],
                "Note": "State diverged (KL measured at different states)"
            })
            
    df_pairs = pd.DataFrame(pair_comparison)
    df_pairs.to_csv(os.path.join(config.tables_dir, "pair_comparison.csv"), index=False)
    
    # Generate summary markdown
    with open(os.path.join(config.output_dir, "v22_summary.md"), "w") as f:
        f.write("# CARE-COM v2.2 Controlled Adaptive Compression Validation\n\n")
        f.write(f"Random Seeds Evaluated: {len(data['Random'])}\n\n")
        f.write("## Perplexity Results\n")
        f.write(df_ppl.to_markdown(index=False) + "\n\n")
        f.write("## Marginal Damage Results\n")
        f.write(df_kl.to_markdown(index=False) + "\n\n")
        f.write("## Pair Selection Divergence\n")
        f.write(df_pairs.to_markdown(index=False) + "\n\n")
        
if __name__ == "__main__":
    from experiments.care_com_v22.config import CareComV22Config
    generate_analysis(CareComV22Config())
