"""
Script 2: Intrinsic Dimensionality Analysis
============================================
Estimates the intrinsic dimensionality of the functional representation space
using Levina-Bickel kNN estimator and PCA explained-variance ratio.

Inputs:  results/exp6b/telemetry/checkpoint_{10,40,70,100}/{first,middle,last}/embedding_centroids.npy
Outputs: analysis_outputs/intrinsic_dim_results.csv
         analysis_outputs/fig_intrinsic_dim_heatmap.png
         analysis_outputs/fig_pca_scree.png

Statistical note: n=64 experts. For Levina-Bickel, reliable range is k ≤ n/5 = 12.
At k≥20 the estimator saturates and underestimates. Results at k=5,8,10 are reported.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import os

np.random.seed(42)
OUT = "analysis_outputs"
os.makedirs(OUT, exist_ok=True)

CHECKPOINTS = [10, 40, 70, 100]
LAYERS = ["first", "middle", "last"]
K_VALUES = [5, 8, 10, 15, 20]

# ── Levina-Bickel ID estimator ─────────────────────────────────────────────────
def levina_bickel_id(X, k):
    """
    Levina & Bickel (2004) maximum likelihood intrinsic dimensionality estimator.
    For each point x_i, uses k nearest neighbors.
    ID_i = (1/(k-1)) * sum_{j=1}^{k-1} log(r_k / r_j)^{-1}
    where r_j is distance to j-th neighbor.
    Returns: mean ID, std, per-point IDs.
    """
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)  # +1 to exclude self
    dists, _ = nbrs.kneighbors(X)
    dists = dists[:, 1:]  # remove self (distance=0), shape (n, k)

    # Replace zeros with small epsilon to avoid log(0)
    dists = np.maximum(dists, 1e-12)

    ids = []
    for i in range(n):
        r_k = dists[i, -1]  # distance to k-th neighbor
        if r_k <= 0:
            continue
        log_ratios = np.log(r_k / dists[i, :-1])  # k-1 terms
        valid = log_ratios[log_ratios > 0]
        if len(valid) == 0:
            continue
        id_i = (len(valid)) / np.sum(1.0 / log_ratios[log_ratios > 0])
        # Wait, correct formula: id_i = (k-1) / sum(log(r_k/r_j))
        id_i = (k - 1) / np.sum(np.log(r_k / dists[i, :-1]))
        ids.append(id_i)

    ids = np.array(ids)
    return ids.mean(), ids.std(), ids

# ── PCA explained variance dimensionality ─────────────────────────────────────
def pca_dim(X, threshold=0.90):
    """Number of PCs needed to explain `threshold` fraction of variance."""
    pca = PCA().fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    d90 = int(np.searchsorted(cumvar, 0.90)) + 1
    d95 = int(np.searchsorted(cumvar, 0.95)) + 1
    d99 = int(np.searchsorted(cumvar, 0.99)) + 1
    return pca.explained_variance_ratio_, d90, d95, d99

# ── Main loop ──────────────────────────────────────────────────────────────────
rows = []
pca_results = {}

print("Levina-Bickel ID estimates (n=64 experts):")
print(f"{'Checkpoint':<12} {'Layer':<8} " + " ".join(f"{'k='+str(k):<10}" for k in K_VALUES) +
      f"  {'PCA-90':<8} {'PCA-95':<8} {'PCA-99':<8}")
print("-" * 100)

for ckpt in CHECKPOINTS:
    for layer in LAYERS:
        path = f"results/exp6b/telemetry/checkpoint_{ckpt}/{layer}/embedding_centroids.npy"
        if not os.path.exists(path):
            continue
        C = np.load(path).astype(np.float64)  # (64, 2048)
        C_norm = C / np.linalg.norm(C, axis=1, keepdims=True)

        evr, d90, d95, d99 = pca_dim(C)
        pca_results[(ckpt, layer)] = evr

        row = {"checkpoint": ckpt, "layer": layer,
               "mean_norm": np.linalg.norm(C, axis=1).mean(),
               "pca_dim_90": d90, "pca_dim_95": d95, "pca_dim_99": d99}

        lb_vals = []
        for k in K_VALUES:
            if k >= C.shape[0] - 1:
                id_mean, id_std = np.nan, np.nan
            else:
                id_mean, id_std, _ = levina_bickel_id(C, k)
            row[f"LB_k{k}_mean"] = id_mean
            row[f"LB_k{k}_std"]  = id_std
            lb_vals.append(f"{id_mean:.2f}±{id_std:.2f}")

        # Also on normalized C
        evr_n, d90n, d95n, d99n = pca_dim(C_norm)
        row["pca_dim_90_norm"] = d90n
        row["pca_dim_95_norm"] = d95n
        row["pca_dim_99_norm"] = d99n
        for k in K_VALUES:
            if k >= C.shape[0] - 1:
                row[f"LB_norm_k{k}_mean"] = np.nan
            else:
                id_mean_n, id_std_n, _ = levina_bickel_id(C_norm, k)
                row[f"LB_norm_k{k}_mean"] = id_mean_n
                row[f"LB_norm_k{k}_std"]  = id_std_n

        rows.append(row)
        print(f"  ckpt_{ckpt:<6} {layer:<8} " + "  ".join(f"{v:<10}" for v in lb_vals) +
              f"  {d90:<8} {d95:<8} {d99:<8}")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/intrinsic_dim_results.csv", index=False)
print(f"\nSaved: {OUT}/intrinsic_dim_results.csv")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n=== INTRINSIC DIMENSIONALITY SUMMARY (checkpoint_100) ===")
for layer in LAYERS:
    sub = df[(df.checkpoint == 100) & (df.layer == layer)]
    if sub.empty:
        continue
    r = sub.iloc[0]
    print(f"\n  [{layer}]")
    print(f"    Raw C:    LB(k=5)={r.LB_k5_mean:.2f}±{r.LB_k5_std:.2f}  "
          f"LB(k=8)={r.LB_k8_mean:.2f}±{r.LB_k8_std:.2f}  "
          f"LB(k=10)={r.LB_k10_mean:.2f}±{r.LB_k10_std:.2f}  "
          f"PCA-90={r.pca_dim_90}  PCA-95={r.pca_dim_95}")
    print(f"    Norm C:   LB(k=5)={r.LB_norm_k5_mean:.2f}±{r.LB_norm_k5_std:.2f}  "
          f"PCA-90={r.pca_dim_90_norm}  PCA-95={r.pca_dim_95_norm}")
    print(f"    LIMITATION: n=64, reliable for k≤12 only. Treat k≥15 as indicative.")

# ── Figure 1: Heatmap of LB ID (k=8) across layers and checkpoints ────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

for ax_idx, (col_prefix, title_suffix) in enumerate([("LB_k8_mean", "Raw C"), ("LB_norm_k8_mean", "L2-Normalized C")]):
    heat_data = np.zeros((len(CHECKPOINTS), len(LAYERS)))
    for i, ckpt in enumerate(CHECKPOINTS):
        for j, layer in enumerate(LAYERS):
            sub = df[(df.checkpoint == ckpt) & (df.layer == layer)]
            if not sub.empty:
                heat_data[i, j] = sub.iloc[0][col_prefix]

    ax = axes[ax_idx]
    im = ax.imshow(heat_data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels(LAYERS)
    ax.set_yticks(range(len(CHECKPOINTS)))
    ax.set_yticklabels([f"ckpt_{c}" for c in CHECKPOINTS])
    ax.set_title(f"Levina-Bickel ID (k=8)\n{title_suffix}", fontsize=11)
    plt.colorbar(im, ax=ax, label="Estimated ID")
    for i in range(len(CHECKPOINTS)):
        for j in range(len(LAYERS)):
            ax.text(j, i, f"{heat_data[i,j]:.1f}", ha="center", va="center",
                    fontsize=9, color="black" if heat_data[i,j] < heat_data.max()*0.7 else "white")

plt.suptitle(f"Intrinsic Dimensionality (n=64 experts)\nNote: n=64 limits reliability; k≤12 recommended",
             fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_intrinsic_dim_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}/fig_intrinsic_dim_heatmap.png")

# ── Figure 2: PCA scree for checkpoint_100, all 3 layers ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, layer in zip(axes, LAYERS):
    evr = pca_results.get((100, layer))
    if evr is None:
        continue
    cumvar = np.cumsum(evr)
    ax.bar(range(1, min(31, len(evr)+1)), evr[:30]*100, color="#3498db", alpha=0.7, label="Indiv. EVR")
    ax2 = ax.twinx()
    ax2.plot(range(1, min(31, len(evr)+1)), cumvar[:30]*100, color="#e74c3c",
             linewidth=2, marker="o", markersize=3, label="Cumulative")
    ax2.axhline(90, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.axhline(95, color="gray", linestyle=":",  linewidth=1, alpha=0.7)
    ax2.set_ylabel("Cumulative EVR (%)", fontsize=9, color="#e74c3c")
    ax2.set_ylim(0, 105)
    ax.set_xlabel("PC index", fontsize=9)
    ax.set_ylabel("EVR (%)", fontsize=9, color="#3498db")
    r = df[(df.checkpoint == 100) & (df.layer == layer)].iloc[0]
    ax.set_title(f"ckpt_100 / {layer}\nPCA-90={r.pca_dim_90}  PCA-95={r.pca_dim_95}  PCA-99={r.pca_dim_99}",
                 fontsize=10)

plt.suptitle("PCA Scree — Functional Representations (checkpoint_100, raw C)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_pca_scree.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}/fig_pca_scree.png")
print("\n[Script 2 COMPLETE]")
