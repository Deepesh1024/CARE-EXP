"""
CARE-MoE Experiment 3A — Export Tables
======================================================
Converts all JSON statistics into formal CSV tables in the tables/ directory.
"""

import os
import pandas as pd

from config import (
    BASELINES_DIR,
    COMMUNITIES_DIR,
    VALIDATION_DIR,
    TABLES_DIR,
    LAYERS,
    K_PRIMARY
)
from utils import load_json, save_csv

def export_tables():
    # 1. Baseline Statistics
    sig_path = os.path.join(BASELINES_DIR, "significance_tests.json")
    if os.path.exists(sig_path):
        sig_data = load_json(sig_path)
        rows = []
        for layer, layer_data in sig_data.items():
            if f"k{K_PRIMARY}" in layer_data:
                metrics = layer_data[f"k{K_PRIMARY}"]
                for metric_name, m_data in metrics.items():
                    rows.append({
                        "Layer": layer,
                        "Metric": metric_name,
                        "CARE_Value": m_data.get("CARE_Value"),
                        "Random_Mean": m_data.get("Random_Mean"),
                        "Z_Score": m_data.get("Z_Score"),
                        "P_Value": m_data.get("P_Value"),
                        "Significant": m_data.get("Significant")
                    })
        save_csv(pd.DataFrame(rows), os.path.join(TABLES_DIR, "significance_tests_table.csv"))

    # 2. Validation Statistics (Within vs Between KL)
    val_path = os.path.join(VALIDATION_DIR, "validation_statistics.json")
    if os.path.exists(val_path):
        val_data = load_json(val_path)
        rows = []
        for layer, d in val_data.items():
            rows.append({
                "Layer": layer,
                "N_Within": d.get("N_Within"),
                "N_Between": d.get("N_Between"),
                "Mean_Within_KL": d.get("Mean_Within"),
                "Mean_Between_KL": d.get("Mean_Between"),
                "CI_Within_Lower": d.get("CI_Within")[0] if "CI_Within" in d else None,
                "CI_Within_Upper": d.get("CI_Within")[1] if "CI_Within" in d else None,
                "CI_Between_Lower": d.get("CI_Between")[0] if "CI_Between" in d else None,
                "CI_Between_Upper": d.get("CI_Between")[1] if "CI_Between" in d else None,
                "MannWhitney_P_Value": d.get("MannWhitney_p"),
                "Cohens_D": d.get("Cohens_d"),
                "Significant": d.get("Significant")
            })
        save_csv(pd.DataFrame(rows), os.path.join(TABLES_DIR, "validation_statistics_table.csv"))

    # 3. Community Summary
    comm_path = os.path.join(COMMUNITIES_DIR, "community_summary.json")
    if os.path.exists(comm_path):
        comm_data = load_json(comm_path)
        rows = []
        for layer, layer_data in comm_data.items():
            if f"k{K_PRIMARY}" in layer_data:
                k_data = layer_data[f"k{K_PRIMARY}"]
                for algo, a_data in k_data.items():
                    rows.append({
                        "Layer": layer,
                        "Algorithm": algo,
                        "Num_Communities": a_data.get("Num_Communities"),
                        "Modularity": a_data.get("Modularity")
                    })
        save_csv(pd.DataFrame(rows), os.path.join(TABLES_DIR, "community_summary_table.csv"))

    # 4. Robustness
    rob_path = os.path.join(VALIDATION_DIR, "robustness_ari_nmi.json")
    if os.path.exists(rob_path):
        rob_data = load_json(rob_path)
        rows = []
        for layer, layer_data in rob_data.items():
            for comp, scores in layer_data.items():
                rows.append({
                    "Layer": layer,
                    "Comparison": comp,
                    "ARI": scores.get("ARI"),
                    "NMI": scores.get("NMI")
                })
        save_csv(pd.DataFrame(rows), os.path.join(TABLES_DIR, "robustness_table.csv"))
        
    # 5. Centrality Correlations
    for layer in LAYERS + ["aggregated"]:
        corr_path = os.path.join(VALIDATION_DIR, f"{layer}_centrality_correlations.json")
        if os.path.exists(corr_path):
            corr_data = load_json(corr_path)
            rows = []
            for metric, algos in corr_data.items():
                for algo, scores in algos.items():
                    rows.append({
                        "Layer": layer,
                        "Centrality_Metric": metric,
                        "Correlation_Algorithm": algo,
                        "Coefficient": scores.get("coef"),
                        "P_Value": scores.get("p")
                    })
            save_csv(pd.DataFrame(rows), os.path.join(TABLES_DIR, f"{layer}_centrality_correlations_table.csv"))

if __name__ == "__main__":
    export_tables()
