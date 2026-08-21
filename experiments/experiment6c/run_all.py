import os
import sys
import subprocess
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIRS, ensure_dirs, TRANSITIONS

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"Executing {script_name}...")
    print(f"{'='*60}")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    result = subprocess.run([sys.executable, script_path])
    
    
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
        
def generate_final_report():
    print("\nGenerating EXP6C_FINAL_REPORT.md...")
    report_path = os.path.join(DIRS["root"], "EXP6C_FINAL_REPORT.md")
    
    with open(os.path.join(DIRS["models"], "EXP6C_MODEL_RESULTS.json"), "r") as f:
        models = json.load(f)
    with open(os.path.join(DIRS["nulls"], "EXP6C_NULL_RESULTS.json"), "r") as f:
        nulls = json.load(f)
        
    report = f"""# EXPERIMENT 6C FINAL REPORT
**Generated:** {datetime.now().isoformat()}

## Capability-Space Vector Movement and Expert Interaction

### 1. Research Question
How does an MoE expert's functional state move through an empirical capability space as it encounters different token/task environments, and how is that movement related to the expert's existing functional state and its neighboring experts?
Specifically, does the incoming environment $\\tau$ induce a directional angular shift determined by the capability-conditioned interaction vector $I_i = C_i \\odot \\tau_i$?

### 2. Common Vector-Space Construction
We constructed a 10-dimensional empirical capability space using audited ARC and MMLU subset categories.
The functional vectors $C_i$ were extracted as the capability-probe response strength (the mean output activation norm of expert $i$ when fed tokens from axis $k$), bypassing the router entirely.

### 3. Directional Alignments and Interaction (10D)
Does the interaction vector $C \\odot \\tau$ better explain the functional displacement $\\Delta C$ than $\\tau$ alone?
"""

    for trans in TRANSITIONS:
        t = f"{trans[0]}->{trans[1]}"
        if t in models:
            m = models[t]
            report += f"\n#### Transition {t}\n"
            report += f"- **Mean 10D $\\cos(\\Delta C, \\tau)$**: {m['Mean_Cos_DeltaC_Tau']:.4f}\n"
            report += f"- **Mean 10D $\\cos(\\Delta C, C)$**: {m['Mean_Cos_DeltaC_C']:.4f}\n"
            report += f"- **Mean 10D $\\cos(\\Delta C, I)$**: {m['Mean_Cos_DeltaC_I']:.4f}\n"
            report += f"- **Mean 10D $\\cos(\\Delta C_\\perp, I_\\perp)$**: {m['Mean_Cos_DeltaC_Perp_I_Perp']:.4f}\n"
            
            report += f"\n**Predictive Models ($R^2$)**\n"
            report += f"- $\\tau$ only: {m['R2_Tau_Only']:.4f}\n"
            report += f"- $C$ only: {m['R2_C_Only']:.4f}\n"
            report += f"- $C + \\tau$: {m['R2_C_plus_Tau']:.4f}\n"
            report += f"- $I$ only: {m['R2_I_Only']:.4f}\n"
            report += f"- Full ($C + \\tau + I$): {m['R2_C_plus_Tau_plus_I']:.4f}\n"
            report += f"- **$I_\\perp$ predicting $\\Delta C_\\perp$**: {m['R2_I_Perp_predicting_DeltaC_Perp']:.4f}\n"

            report += f"\n**Angular Susceptibility ($S_\\theta$) Sensitivity Analysis**\n"
            for pct, s in m['Angular_Susceptibility_Sensitivity'].items():
                report += f"- **Threshold**: Exclude bottom {s['threshold_used']} $||C||$ (Retained {s['experts_retained']}). "
                report += f"Layer Model $R^2$: {s['r2']:.4f}\n"

    report += "\n### 4. Null-Model Analysis\n"
    report += "Are the directional alignments and spatial convergences statistically significant against randomized nulls?\n"
    for trans in TRANSITIONS:
        t = f"{trans[0]}->{trans[1]}"
        if t in nulls:
            n = nulls[t]
            report += f"\n#### Transition {t}\n"
            
            if "Observed_Mean_Cos_I" in n:
                report += f"- **Observed $\\cos(\\Delta C, I)$**: {n['Observed_Mean_Cos_I']:.4f}\n"
                report += f"  - Random Direction Null Z-Score: {n['Random_Null_Z_Score']:.2f} (Significant: {n['Significant_vs_Random']})\n"
                report += f"  - $\\tau$-Permutation Null Z-Score: {n['Permutation_Null_Z_Score']:.2f} (Significant: {n['Significant_vs_Permutation']})\n"
                
                report += f"- **Observed $\\cos(\\Delta C_\\perp, I_\\perp)$**: {n.get('Observed_Mean_Cos_I_Perp', 0.0):.4f}\n"
                report += f"  - Random Direction Null Z-Score: {n.get('Random_Null_Z_Score_Perp', 0.0):.2f} (Significant: {n.get('Perp_Significant_vs_Random', False)})\n"
                report += f"  - $\\tau$-Permutation Null Z-Score: {n.get('Permutation_Null_Z_Score_Perp', 0.0):.2f} (Significant: {n.get('Perp_Significant_vs_Permutation', False)})\n"
            
            if "Observed_Overlap_DeltaD_Rho" in n:
                report += f"\n- **Observed Task-Overlap vs $\\Delta D$ (Spearman $\\rho$)**: {n['Observed_Overlap_DeltaD_Rho']:.4f}\n"
                report += f"  - Pair-Matched Null Z-Score: {n['Z_Score']:.2f} (Significant: {n['Significant']})\n"
            
    report += """
### 5. Conclusions
(This section to be filled by researcher after reviewing the generated outputs and plots).
"""
    
    with open(report_path, "w") as f:
        f.write(report)
        
    print(f"Saved {report_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run a small pilot version (adjust config max_samples)")
    args = parser.parse_args()
    
    ensure_dirs()
    
    if not os.path.exists(os.path.join(DIRS["token_vectors"], "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet")):
        run_script("phase1_token_environments.py")
    else:
        print("[SKIP] Phase 1 (Token Environments) already completed.")
        
    if not os.path.exists(os.path.join(DIRS["routing"], "EXP6C_ROUTING_ENVIRONMENT.parquet")):
        run_script("phase2_routing_extraction.py")
    else:
        print("[SKIP] Phase 2 (Routing Extraction) already completed.")
        
    if not os.path.exists(os.path.join(DIRS["expert_vectors"], "EXP6C_EXPERT_CAPABILITY_VECTORS.parquet")):
        run_script("phase3_expert_probing.py")
    else:
        print("[SKIP] Phase 3 (Expert Probing) already completed.")
        
    if not os.path.exists(os.path.join(DIRS["pair_metrics"], "EXP6C_PAIR_METRICS.parquet")):
        run_script("phase4_movement_and_geometry.py")
    else:
        print("[SKIP] Phase 4 (Movement and Geometry) already completed.")
        
    if not os.path.exists(os.path.join(DIRS["models"], "EXP6C_MODEL_RESULTS.json")):
        run_script("phase5_dynamics_modeling.py")
    else:
        print("[SKIP] Phase 5 (Dynamics Modeling) already completed.")
        
    if not os.path.exists(os.path.join(DIRS["nulls"], "EXP6C_NULL_RESULTS.json")):
        run_script("phase6_null_controls.py")
    else:
        print("[SKIP] Phase 6 (Null Controls) already completed.")
        
    if not os.path.exists(os.path.join(DIRS["plots"], "10D_Alignment_Comparison.png")):
        run_script("phase7_plotting.py")
    else:
        print("[SKIP] Phase 7 (Plotting) already completed.")
    
    generate_final_report()
    
    print("\n[SUCCESS] Experiment 6C Pipeline Completed Successfully.")

if __name__ == "__main__":
    main()
