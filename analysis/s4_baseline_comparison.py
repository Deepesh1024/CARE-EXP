"""
Script 4: Representation Baselines Comparison
==============================================
Compares all available descriptors against ground-truth merge cost D_actual_KL.
Reports Spearman, Pearson, RMSE, and rank accuracy with bootstrap CIs.

Inputs:  results/exp7b/predictions_18_pairs.csv  (18 pairs × 16 cols)
         results/exp7b/merges/actual_merge_results.csv (18 pairs × 15 cols)
Outputs: analysis_outputs/baseline_comparison.csv
         analysis_outputs/fig_baseline_bars.png
         analysis_outputs/fig_baseline_scatter.png

Statistical note: n=18 pairs. Bootstrap CIs (10,000 resamples). Low power —
all CIs will be wide. Spearman p-values reported but treat as indicative only.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from scipy.stats import spearmanr, pearsonr
import os

np.random.seed(42)
OUT = "analysis_outputs"
os.makedirs(OUT, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
pred   = pd.read_csv("results/exp7b/predictions_18_pairs.csv")
pred   = pred.rename(columns={"pair": "pair_id"})
merge  = pd.read_csv("results/exp7b/merges/actual_merge_results.csv")
df     = pred.merge(merge[["pair_id", "D_actual_KL", "delta_loss", "delta_acc"]], on="pair_id")

n = len(df)
D_act = df["D_actual_KL"].values
print(f"n = {n} pairs")
print(f"D_actual_KL: mean={D_act.mean():.5f}, std={D_act.std():.5f}, range={D_act.min():.5f}–{D_act.max():.5f}")

DESCRIPTORS = {
    "Functional Geometry (ours)": "D_pred",
    "Weight Distance (L2)":       "Weight_Distance",
    "Weight Cosine":               "Weight_Cosine",
    "Activation Similarity":       "Activation_Similarity",
    "Output Similarity":           "Output_Similarity",
    "Routing Similarity":          "Routing_Similarity",
    "Usage Frequency":             "Usage_Frequency",
    "Jaccard Overlap":             "Jaccard_Overlap",
    "Cross-Entropy Delta":         "CrossEntropy_Delta",
    "Hidden L2 Drift":             "Hidden_L2_Drift",
    "Random Baseline":             "Random_Baseline",
}

# ── Bootstrap helpers ─────────────────────────────────────────────────────────
def bootstrap_stats(x, y, n_boot=10_000):
    """Return (spearman_r, spearman_p, pearson_r, pearson_p, rmse,
               spearman_lo, spearman_hi, pearson_lo, pearson_hi)"""
    # Point estimates
    sp_r, sp_p = spearmanr(x, y)
    pe_r, pe_p = pearsonr(x, y)
    rmse = np.sqrt(np.mean((x - y)**2))

    sp_boots, pe_boots = [], []
    for _ in range(n_boot):
        idx = np.random.randint(0, len(x), len(x))
        if len(set(idx)) < 3:
            continue
        sr, _ = spearmanr(x[idx], y[idx])
        pr, _ = pearsonr(x[idx], y[idx])
        sp_boots.append(sr)
        pe_boots.append(pr)

    sp_boots = np.array(sp_boots)
    pe_boots = np.array(pe_boots)
    return (sp_r, sp_p, pe_r, pe_p, rmse,
            np.percentile(sp_boots, 2.5), np.percentile(sp_boots, 97.5),
            np.percentile(pe_boots, 2.5), np.percentile(pe_boots, 97.5))

# ── Compute all descriptors ───────────────────────────────────────────────────
rows = []
print(f"\n{'Descriptor':<30} {'Spearman r':<28} {'Pearson r':<28} {'RMSE':<10}")
print("-" * 100)

for name, col in DESCRIPTORS.items():
    if col not in df.columns:
        print(f"  {name:<30} COLUMN MISSING")
        continue
    x = df[col].values.astype(float)

    # Handle NaN
    valid = ~np.isnan(x)
    if valid.sum() < 5:
        print(f"  {name:<30} INSUFFICIENT DATA (n={valid.sum()})")
        continue

    x_v  = x[valid]
    y_v  = D_act[valid]
    n_v  = valid.sum()

    sp_r, sp_p, pe_r, pe_p, rmse, sp_lo, sp_hi, pe_lo, pe_hi = bootstrap_stats(x_v, y_v)

    rows.append({
        "descriptor": name, "column": col, "n_valid": n_v,
        "spearman_r": sp_r, "spearman_p": sp_p,
        "spearman_lo95": sp_lo, "spearman_hi95": sp_hi,
        "pearson_r": pe_r, "pearson_p": pe_p,
        "pearson_lo95": pe_lo, "pearson_hi95": pe_hi,
        "rmse": rmse,
    })

    print(f"  {name:<30} r={sp_r:+.3f} [{sp_lo:+.3f} {sp_hi:+.3f}] p={sp_p:.3f}  "
          f"r={pe_r:+.3f} [{pe_lo:+.3f} {pe_hi:+.3f}] p={pe_p:.3f}  "
          f"RMSE={rmse:.5f}")

result = pd.DataFrame(rows).sort_values("spearman_r", ascending=False)
result.to_csv(f"{OUT}/baseline_comparison.csv", index=False)
print(f"\nSaved: {OUT}/baseline_comparison.csv")

# ── Statistical limitation warning ────────────────────────────────────────────
print(f"\n=== STATISTICAL LIMITATIONS (n={n}) ===")
print(f"  Bootstrap CIs are wide due to n=18. Spearman p-values at n=18:")
print(f"    p<0.05 threshold: |r| ≥ ~0.47 (Spearman, two-tailed)")
print(f"    p<0.10 threshold: |r| ≥ ~0.40")
print(f"  Report 95% CI ranges, not just point estimates.")
print(f"  Descriptors with CI spanning 0 are not reliably different from null.")

# ── Figure 1: Ranked bar chart of Spearman r with bootstrap CI ────────────────
fig, ax = plt.subplots(figsize=(12, 6))

ours_mask = result["descriptor"] == "Functional Geometry (ours)"
colours = ["#2ecc71" if m else "#95a5a6" for m in ours_mask]

bars = ax.barh(result["descriptor"], result["spearman_r"],
               color=colours, edgecolor="k", linewidth=0.7,
               xerr=[result["spearman_r"] - result["spearman_lo95"],
                     result["spearman_hi95"] - result["spearman_r"]],
               capsize=4, error_kw={"elinewidth": 1.5})

ax.axvline(0, color="k", linewidth=1.0)

# p=0.05 threshold for n=18
ax.axvline(0.470, color="red",  linewidth=1.2, linestyle="--", alpha=0.7,
           label="p=0.05 threshold (n=18)")
ax.axvline(-0.470, color="red", linewidth=1.2, linestyle="--", alpha=0.7)

ax.set_xlabel("Spearman r vs D_actual_KL", fontsize=11)
ax.set_title(f"Descriptor Comparison — Merge Cost Prediction\n"
             f"n={n} pairs, 95% bootstrap CI, CRITICAL: wide CIs due to n=18",
             fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis="x", linestyle="--", alpha=0.4)

# Add r values as text
for _, row in result.iterrows():
    x_pos = row["spearman_r"] + (0.02 if row["spearman_r"] >= 0 else -0.02)
    ha = "left" if row["spearman_r"] >= 0 else "right"
    ax.text(x_pos, row["descriptor"], f"{row['spearman_r']:+.3f}", va="center",
            ha=ha, fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/fig_baseline_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}/fig_baseline_bars.png")

# ── Figure 2: Scatter plots for top-3 and ours ────────────────────────────────
top_cols = result.head(4)[["descriptor", "column"]].values
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, (name, col) in zip(axes, top_cols):
    x = df[col].values.astype(float)
    valid = ~np.isnan(x)
    xv, yv = x[valid], D_act[valid]
    ax.scatter(xv, yv, color="#3498db", s=60, edgecolors="k", linewidths=0.5, zorder=3)

    # Fit line
    if len(xv) > 2:
        m, b = np.polyfit(xv, yv, 1)
        xl = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(xl, m*xl+b, "r--", linewidth=1.5)

    sp_r, sp_p = spearmanr(xv, yv)
    ax.set_title(f"{name}\nr={sp_r:.3f}, p={sp_p:.3f}", fontsize=9)
    ax.set_xlabel(col, fontsize=8)
    ax.set_ylabel("D_actual_KL", fontsize=8) if ax == axes[0] else None

    # Label pairs
    for _, row in df[valid].iterrows():
        ax.annotate(row["pair_id"], (row[col], row["D_actual_KL"]),
                    fontsize=6, alpha=0.6, textcoords="offset points", xytext=(3,3))

plt.suptitle(f"Top-4 Descriptors vs Actual Merge Cost (n={n} pairs)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_baseline_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}/fig_baseline_scatter.png")
print("\n[Script 4 COMPLETE]")
