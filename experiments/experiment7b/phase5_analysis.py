"""
EXPERIMENT 7B - PHASE 5: STATISTICAL ANALYSIS & DECISION GATES
==============================================================
Executes the primary hypothesis test, Gate 1 & Gate 2 evaluations,
Spearman rank correlations with bootstrap confidence intervals,
exploratory degeneracy classification, cheap proxy comparison,
publication figures, and final report generation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ensure_dirs, mark_task, is_task_completed,
    MERGES_DIR, INTERVENTION_DIR, ANALYSIS_DIR, PLOTS_DIR,
    BOOTSTRAP_N_SAMPLES, BOOTSTRAP_CI_LEVEL,
    GATE1_MIN_RHO, GATE1_ALPHA, GATE2_MIN_RHO, GATE2_ALPHA
)
from utils.statistics import (
    compute_spearman, compute_pearson,
    compute_partial_spearman, bootstrap_correlation_ci
)

def run_analysis():
    ensure_dirs()
    print("=" * 70)
    print("EXPERIMENT 7B — PHASE 5: STATISTICAL ANALYSIS & DECISION GATES")
    print("=" * 70)

    merges_csv = os.path.join(MERGES_DIR, "actual_merge_results.csv")
    joint_csv = os.path.join(INTERVENTION_DIR, "joint_interventions.csv")

    if not os.path.exists(merges_csv) or not os.path.exists(joint_csv):
        raise FileNotFoundError(
            f"Missing required data:\n  Merges: {merges_csv}\n  Joint: {joint_csv}\n"
            "Run phase2 and phase4 on the GPU VM before running analysis."
        )

    df_merges = pd.read_csv(merges_csv)
    df_joint = pd.read_csv(joint_csv)

    # Merge on pair_id
    df = pd.merge(df_merges, df_joint[["pair_id", "I_KL", "I_loss", "I_acc", "D_sum_KL"]], on="pair_id")
    print(f"[Phase 5] Consolidated {len(df)} pairs for final statistical analysis.")

    # ══════════════════════════════════════════════════════════
    # Primary Hypothesis Tests
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("PRIMARY STATISTICAL HYPOTHESIS TESTING")
    print("=" * 50)

    # 1. Baseline CARE-COM vs Actual Damage
    rho_pred_actual, p_pred_actual = compute_spearman(df["D_pred"], df["D_actual_KL"])
    ci_pred_actual = bootstrap_correlation_ci(df["D_pred"], df["D_actual_KL"], n_boot=BOOTSTRAP_N_SAMPLES)

    # 2. Joint Interaction vs Prediction Error E = D_actual - D_pred
    rho_I_E, p_I_E = compute_spearman(df["I_KL"], df["Error_KL"])
    ci_I_E = bootstrap_correlation_ci(df["I_KL"], df["Error_KL"], n_boot=BOOTSTRAP_N_SAMPLES)

    # 3. Joint Interaction vs Actual Damage D_actual
    rho_I_actual, p_I_actual = compute_spearman(df["I_KL"], df["D_actual_KL"])
    ci_I_actual = bootstrap_correlation_ci(df["I_KL"], df["D_actual_KL"], n_boot=BOOTSTRAP_N_SAMPLES)

    # 4. Partial Correlation rho(I, D_actual | D_pred)
    partial_rho, partial_p = compute_partial_spearman(df["I_KL"], df["D_actual_KL"], df["D_pred"])

    print(f"1. CARE-COM Baseline vs Actual Damage:  rho = {rho_pred_actual:+.4f} (p = {p_pred_actual:.4e}, 95% CI: [{ci_pred_actual[0]:+.4f}, {ci_pred_actual[1]:+.4f}])")
    print(f"2. Interaction I vs Prediction Error E: rho = {rho_I_E:+.4f} (p = {p_I_E:.4e}, 95% CI: [{ci_I_E[0]:+.4f}, {ci_I_E[1]:+.4f}])")
    print(f"3. Interaction I vs Actual Damage:      rho = {rho_I_actual:+.4f} (p = {p_I_actual:.4e}, 95% CI: [{ci_I_actual[0]:+.4f}, {ci_I_actual[1]:+.4f}])")
    print(f"4. Partial rho(I, D_actual | D_pred):   rho = {partial_rho:+.4f} (p = {partial_p:.4e})")

    # ══════════════════════════════════════════════════════════
    # Pre-Registered Decision Gates
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("PRE-REGISTERED DECISION GATES")
    print("=" * 50)

    gate1_passed = (rho_pred_actual >= GATE1_MIN_RHO) and (p_pred_actual < GATE1_ALPHA)
    gate2_passed = (rho_I_E >= GATE2_MIN_RHO) and (ci_I_E[0] > 0.0) and (p_I_E < GATE2_ALPHA)

    decision = "BUILD" if (gate1_passed and gate2_passed) else "KILL"

    print(f"GATE 1 (Baseline Predictive Utility): {'PASS (Proceed)' if gate1_passed else 'FAIL (Critical Baseline Flaw)'}")
    print(f"  Target: rho >= {GATE1_MIN_RHO:.2f}, p < {GATE1_ALPHA} -> Observed: rho = {rho_pred_actual:.4f}, p = {p_pred_actual:.4e}")

    print(f"\nGATE 2 (Interaction Predictive Power): {'PASS' if gate2_passed else 'FAIL'}")
    print(f"  Target: rho >= {GATE2_MIN_RHO:.2f}, CI strictly > 0 -> Observed: rho = {rho_I_E:.4f}, CI = [{ci_I_E[0]:.4f}, {ci_I_E[1]:.4f}]")

    print(f"\nFINAL VERDICT: [{decision}]")
    if decision == "BUILD":
        print("  -> Non-additive joint functional interaction explains CARE-COM merge errors.")
        print("  -> Proceed to research cheap observable proxies for interaction.")
    else:
        print("  -> Falsification confirmed. Joint interaction does not meaningfully predict merge failures.")
        print("  -> Terminate interaction-based additions to CARE-COM.")

    # ══════════════════════════════════════════════════════════
    # Secondary Analysis: Exploratory Degeneracy Classification
    # ══════════════════════════════════════════════════════════
    # Candidate non-additive pairs: Low individual damage + High non-additive interaction
    d_sum_median = df["D_sum_KL"].median()
    i_median = df["I_KL"].median()

    def classify_pair(r):
        if r["D_sum_KL"] <= d_sum_median and r["I_KL"] > i_median:
            return "candidate non-additive"
        elif r["D_sum_KL"] <= d_sum_median and r["I_KL"] <= i_median:
            return "additive safe"
        elif r["D_sum_KL"] > d_sum_median and r["I_KL"] > i_median:
            return "synergistic destructive"
        else:
            return "additive destructive"

    df["classification"] = df.apply(classify_pair, axis=1)

    # ══════════════════════════════════════════════════════════
    # Publication Figures
    # ══════════════════════════════════════════════════════════
    print("\n[Phase 5] Generating publication-grade figures...")

    # Figure 1: Predicted vs Actual Damage
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"Group_A": "forestgreen", "Group_B": "royalblue", "Group_C": "crimson"}
    for stratum, group in df.groupby("selection_stratum"):
        ax.scatter(group["D_pred"], group["D_actual_KL"], label=stratum, color=colors.get(stratum, "gray"), s=70, alpha=0.85)
    
    # Trendline
    m, b = np.polyfit(df["D_pred"], df["D_actual_KL"], 1)
    x_vals = np.linspace(df["D_pred"].min(), df["D_pred"].max(), 100)
    ax.plot(x_vals, m * x_vals + b, "k--", alpha=0.6, label=f"Fit (rho = {rho_pred_actual:+.3f})")

    ax.set_xlabel("CARE-COM Predicted Damage (D_pred)", fontsize=11)
    ax.set_ylabel("Actual Merge Damage (D_actual_KL)", fontsize=11)
    ax.set_title("CARE-COM Predicted vs Actual Merge Damage (Gate 1)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig1_path = os.path.join(PLOTS_DIR, "scatter_pred_vs_actual.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # Figure 2: Interaction vs Prediction Error (Gate 2)
    fig, ax = plt.subplots(figsize=(7, 6))
    for stratum, group in df.groupby("selection_stratum"):
        ax.scatter(group["I_KL"], group["Error_KL"], label=stratum, color=colors.get(stratum, "gray"), s=70, alpha=0.85)
    
    m, b = np.polyfit(df["I_KL"], df["Error_KL"], 1)
    x_vals = np.linspace(df["I_KL"].min(), df["I_KL"].max(), 100)
    ax.plot(x_vals, m * x_vals + b, "k--", alpha=0.6, label=f"Fit (rho = {rho_I_E:+.3f})")

    ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Non-Additive Joint Interaction I(i, j)", fontsize=11)
    ax.set_ylabel("CARE-COM Merge Prediction Error E(i, j)", fontsize=11)
    ax.set_title("Joint Interaction vs Merge Prediction Error (Gate 2)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig2_path = os.path.join(PLOTS_DIR, "scatter_interaction_vs_error.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # Figure 3: Exploratory 2D Degeneracy / Interaction Map
    fig, ax = plt.subplots(figsize=(8, 6))
    class_markers = {
        "candidate non-additive": ("*", "purple", 140),
        "additive safe": ("o", "forestgreen", 60),
        "synergistic destructive": ("^", "crimson", 70),
        "additive destructive": ("s", "darkorange", 60)
    }
    for cls, group in df.groupby("classification"):
        marker, color, size = class_markers.get(cls, ("o", "gray", 50))
        scatter = ax.scatter(group["D_sum_KL"], group["I_KL"], label=cls, marker=marker, c=color, s=size, alpha=0.85)

    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("Individual Damage Sum D(i) + D(j)", fontsize=11)
    ax.set_ylabel("Joint Interaction I(i, j) = D(i,j) - D(i) - D(j)", fontsize=11)
    ax.set_title("Exploratory Degeneracy Landscape", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig3_path = os.path.join(PLOTS_DIR, "degeneracy_classification.png")
    plt.savefig(fig3_path, dpi=200)
    plt.close()

    # ══════════════════════════════════════════════════════════
    # Save Reports
    # ══════════════════════════════════════════════════════════
    summary_data = {
        "gate1": {
            "passed": bool(gate1_passed),
            "spearman_rho": float(rho_pred_actual),
            "p_value": float(p_pred_actual),
            "ci_95": [float(ci_pred_actual[0]), float(ci_pred_actual[1])]
        },
        "gate2": {
            "passed": bool(gate2_passed),
            "spearman_rho": float(rho_I_E),
            "p_value": float(p_I_E),
            "ci_95": [float(ci_I_E[0]), float(ci_I_E[1])],
            "rho_I_actual": float(rho_I_actual),
            "partial_rho": float(partial_rho)
        },
        "decision": decision,
        "sample_size": len(df)
    }
    summary_json_path = os.path.join(ANALYSIS_DIR, "statistical_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Save detailed markdown report
    report_md_path = os.path.join(ANALYSIS_DIR, "EXPERIMENT_7B_FINAL_REPORT.md")
    with open(report_md_path, "w") as f:
        f.write(f"""# Experiment 7B: Final Diagnostic Report
## CARE-COM Merge Failure Analysis and Joint Functional Interaction

**Generated:** {pd.Timestamp.now().isoformat()}  
**Sample Size:** N = {len(df)} pairs (Stratified across Group A, B, C)  
**Decision Verdict:** **[{decision}]**

---

### Pre-Registered Decision Gates

| Decision Gate | Metric / Target | Observed Value | 95% Bootstrap CI | Status |
|---|---|---|---|---|
| **GATE 1 (Baseline Predictive Utility)** | Spearman rho >= {GATE1_MIN_RHO:.2f}, p < {GATE1_ALPHA} | rho = {rho_pred_actual:+.4f} (p = {p_pred_actual:.4e}) | [{ci_pred_actual[0]:+.4f}, {ci_pred_actual[1]:+.4f}] | **{'PASS' if gate1_passed else 'FAIL'}** |
| **GATE 2 (Interaction vs Prediction Error)** | Spearman rho >= {GATE2_MIN_RHO:.2f}, CI > 0 | rho = {rho_I_E:+.4f} (p = {p_I_E:.4e}) | [{ci_I_E[0]:+.4f}, {ci_I_E[1]:+.4f}] | **{'PASS' if gate2_passed else 'FAIL'}** |

---

### Primary Hypothesis Outcomes
- **H0:** Joint functional interaction does not meaningfully explain CARE-COM merge errors.
- **H1:** Pairs with stronger non-additive joint functional interaction exhibit larger CARE-COM merge prediction errors.
- **Conclusion:** **{'H1 Supported' if decision == 'BUILD' else 'H0 Retained (Falsification Confirmed)'}**.

---

### Key Figures Generated
1. `plots/scatter_pred_vs_actual.png`: Predicted vs actual merge damage.
2. `plots/scatter_interaction_vs_error.png`: Non-additive interaction vs CARE-COM prediction error.
3. `plots/degeneracy_classification.png`: 2D exploratory interaction landscape.
""")

    print(f"\n[Phase 5] Summary saved to {summary_json_path}")
    print(f"[Phase 5] Final report written to {report_md_path}")
    mark_task("phase5_analysis", "completed", details=f"Decision: {decision}")

if __name__ == "__main__":
    run_analysis()
