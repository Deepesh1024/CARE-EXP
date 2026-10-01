"""
Script 5: Intervention Prediction
=================================
Calculates correlation (Spearman, Pearson, RMSE, and R^2) between
our predicted capability geometry distance (D_pred) and actual
empirical merge cost (D_actual_KL).

Inputs:  results/exp7b/predictions_18_pairs.csv
         results/exp7b/merges/actual_merge_results.csv
Outputs: analysis_outputs/intervention_prediction.csv
         analysis_outputs/fig_intervention_prediction.png

Statistical note: n=18 pairs. Bootstrap CIs are wide.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score
import os

np.random.seed(42)
OUT = "analysis_outputs"
os.makedirs(OUT, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
pred   = pd.read_csv("results/exp7b/predictions_18_pairs.csv")
pred   = pred.rename(columns={"pair": "pair_id"})
merge  = pd.read_csv("results/exp7b/merges/actual_merge_results.csv")
df     = pred.merge(merge[["pair_id", "D_actual_KL"]], on="pair_id")

n = len(df)
print(f"n = {n} intervention pairs loaded.")

x = df["D_pred"].values
y = df["D_actual_KL"].values

# ── Compute statistics ────────────────────────────────────────────────────────
sp_r, sp_p = spearmanr(x, y)
pe_r, pe_p = pearsonr(x, y)
rmse = np.sqrt(np.mean((x - y)**2))
r2 = r2_score(y, x)  # Note: R2 uses actual vs pred, order matters. Here y is true, x is pred.

# Bootstrap CI for Spearman
n_boot = 10_000
boots = []
for _ in range(n_boot):
    idx = np.random.randint(0, n, n)
    if len(set(idx)) < 3:
        continue
    r, _ = spearmanr(x[idx], y[idx])
    boots.append(r)
boots = np.array(boots)
sp_lo = np.percentile(boots, 2.5)
sp_hi = np.percentile(boots, 97.5)

print("\n=== INTERVENTION PREDICTION ===")
print(f"  Method: Functional Geometry (D_pred)")
print(f"  Spearman r: {sp_r:.3f} (95% CI: {sp_lo:.3f} – {sp_hi:.3f}), p={sp_p:.4f}")
print(f"  Pearson r:  {pe_r:.3f}, p={pe_p:.4f}")
print(f"  RMSE:       {rmse:.5f}")
print(f"  R^2 score:  {r2:.3f}")
print(f"  * Note: n=18 pairs means low statistical power.")

# Save to CSV
res = pd.DataFrame([{
    "n_pairs": n,
    "spearman_r": sp_r, "spearman_p": sp_p, "spearman_ci_lo": sp_lo, "spearman_ci_hi": sp_hi,
    "pearson_r": pe_r, "pearson_p": pe_p,
    "rmse": rmse,
    "r2_score": r2
}])
res.to_csv(f"{OUT}/intervention_prediction.csv", index=False)
print(f"\nSaved: {OUT}/intervention_prediction.csv")

# ── Figure: Scatter plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))

ax.scatter(x, y, color="#2ecc71", s=80, edgecolors="k", linewidths=1.0, zorder=3)

# Identity line
min_val = min(x.min(), y.min()) * 0.9
max_val = max(x.max(), y.max()) * 1.1
ax.plot([min_val, max_val], [min_val, max_val], "k:", alpha=0.5, label="Identity (y=x)")

# Fit line
m, b = np.polyfit(x, y, 1)
xl = np.linspace(min_val, max_val, 100)
ax.plot(xl, m*xl+b, "r--", linewidth=1.5, label=f"Fit (slope={m:.2f})")

ax.set_title("Intervention Prediction\nFunctional Geometry vs Actual Merge Cost", fontsize=12)
ax.set_xlabel("Predicted Cost ($D_{pred}$)", fontsize=11)
ax.set_ylabel("Actual Cost ($D_{actual\\_KL}$)", fontsize=11)

# Annotate points
for i, row in df.iterrows():
    ax.annotate(row["pair_id"], (row["D_pred"], row["D_actual_KL"]),
                fontsize=7, alpha=0.7, textcoords="offset points", xytext=(4,4))

textstr = "\n".join((
    f"$n={n}$ pairs",
    f"Spearman $r={sp_r:.3f}$",
    f"Pearson $r={pe_r:.3f}$",
    f"$R^2={r2:.3f}$"
))
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

ax.grid(linestyle="--", alpha=0.5)
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(f"{OUT}/fig_intervention_prediction.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}/fig_intervention_prediction.png")
print("\n[Script 5 COMPLETE]")
