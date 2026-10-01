"""
Script 1: Magnitude vs Direction Analysis
==========================================
Tests whether functional geometry is driven by output magnitude or direction.

Inputs:  results/exp6b/telemetry/checkpoint_100/{first,middle,last}/embedding_centroids.npy
         results/exp7b/predictions_18_pairs.csv
         results/exp7b/merges/actual_merge_results.csv
Outputs: analysis_outputs/magnitude_vs_direction.csv
         analysis_outputs/fig_mag_vs_dir_scatter.png
         analysis_outputs/fig_mag_vs_dir_corr.png

Statistical note: n=64 experts for geometry correlations; n=18 pairs for
D_actual_KL correlation. Bootstrap CIs (10,000 resamples) reported for n=18.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
import os, json

np.random.seed(42)
OUT = "analysis_outputs"
os.makedirs(OUT, exist_ok=True)

CKPT = "results/exp6b/telemetry/checkpoint_100"
LAYERS = ["first", "middle", "last"]

# ── Load data ─────────────────────────────────────────────────────────────────
centroids = {}
for layer in LAYERS:
    path = os.path.join(CKPT, layer, "embedding_centroids.npy")
    centroids[layer] = np.load(path).astype(np.float64)  # (64, 2048)

pred = pd.read_csv("results/exp7b/predictions_18_pairs.csv")
pred = pred.rename(columns={"pair": "pair_id"})
merge = pd.read_csv("results/exp7b/merges/actual_merge_results.csv")
df = pred.merge(merge[["pair_id","D_actual_KL"]], on="pair_id")

# ── Bootstrap CI helper ───────────────────────────────────────────────────────
def bootstrap_spearman_ci(x, y, n_boot=10_000, ci=0.95):
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        if len(set(idx)) < 3:
            continue
        r, _ = spearmanr(x[idx], y[idx])
        boots.append(r)
    boots = np.array(boots)
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    r, p = spearmanr(x, y)
    return r, p, lo, hi

# ── Main analysis ─────────────────────────────────────────────────────────────
rows = []
for layer in LAYERS:
    C = centroids[layer]  # (64, 2048)
    norms = np.linalg.norm(C, axis=1)  # (64,)

    # Construct representations
    C_norm = C / norms[:, None]                        # L2-normalized (direction only)
    C_mag  = norms[:, None] * np.ones((64, 2048))      # magnitude-only (scalar broadcast)

    # Pairwise distance matrices
    D_full    = cdist(C,      C,      metric="euclidean")
    D_cosine  = cdist(C_norm, C_norm, metric="euclidean")  # equiv to sqrt(2(1-cos)) but monotone w/ cosine
    D_cosine2 = cdist(C,      C,      metric="cosine")     # proper 1-cosine
    D_mag     = cdist(norms[:, None], norms[:, None], metric="euclidean")  # |‖c_i‖ - ‖c_j‖|

    # Upper triangle indices (n*(n-1)/2 = 2016 pairs for n=64)
    ii, jj = np.triu_indices(64, k=1)

    full_tri   = D_full[ii, jj]
    cosine_tri = D_cosine2[ii, jj]
    mag_tri    = D_mag[ii, jj]

    # Correlation: full vs direction
    r_full_dir, p_full_dir = spearmanr(full_tri, cosine_tri)
    # Correlation: full vs magnitude
    r_full_mag, p_full_mag = spearmanr(full_tri, mag_tri)
    # Correlation: direction vs magnitude (orthogonality check)
    r_dir_mag, p_dir_mag   = spearmanr(cosine_tri, mag_tri)

    # For n=2016 pairs, CIs via Fisher-z
    def fisher_ci(r, n, z=1.96):
        z_r = np.arctanh(r)
        se  = 1.0 / np.sqrt(n - 3)
        return np.tanh(z_r - z * se), np.tanh(z_r + z * se)

    n_tri = len(full_tri)
    fd_lo, fd_hi = fisher_ci(r_full_dir, n_tri)
    fm_lo, fm_hi = fisher_ci(r_full_mag, n_tri)
    dm_lo, dm_hi = fisher_ci(r_dir_mag,  n_tri)

    # Mean norms
    mean_norm = norms.mean()
    std_norm  = norms.std()

    rows.append({
        "layer": layer,
        "n_pairs": n_tri,
        "mean_norm": mean_norm,
        "std_norm":  std_norm,
        "r_full_vs_direction": r_full_dir, "p_full_vs_direction": p_full_dir,
        "r_full_vs_direction_lo95": fd_lo,  "r_full_vs_direction_hi95": fd_hi,
        "r_full_vs_magnitude": r_full_mag, "p_full_vs_magnitude": p_full_mag,
        "r_full_vs_magnitude_lo95": fm_lo,  "r_full_vs_magnitude_hi95": fm_hi,
        "r_direction_vs_magnitude": r_dir_mag, "p_direction_vs_magnitude": p_dir_mag,
        "r_dir_vs_mag_lo95": dm_lo, "r_dir_vs_mag_hi95": dm_hi,
    })

    # ── Per-pair D_actual_KL correlation (n=18) ───────────────────────────────
    ei = df["expert_i"].values
    ej = df["expert_j"].values
    D_act = df["D_actual_KL"].values

    d_full_18    = D_full[ei, ej]
    d_cosine_18  = D_cosine2[ei, ej]
    d_mag_18     = D_mag[ei, ej]

    r_act_full,  p_act_full,  lo_af, hi_af = bootstrap_spearman_ci(d_full_18,   D_act)
    r_act_cos,   p_act_cos,   lo_ac, hi_ac = bootstrap_spearman_ci(d_cosine_18, D_act)
    r_act_mag,   p_act_mag,   lo_am, hi_am = bootstrap_spearman_ci(d_mag_18,    D_act)

    rows[-1].update({
        "n18_r_full_vs_D_actual":    r_act_full,  "n18_p_full": p_act_full,
        "n18_r_full_lo95": lo_af, "n18_r_full_hi95": hi_af,
        "n18_r_cosine_vs_D_actual":  r_act_cos,   "n18_p_cosine": p_act_cos,
        "n18_r_cosine_lo95": lo_ac, "n18_r_cosine_hi95": hi_ac,
        "n18_r_magnitude_vs_D_actual": r_act_mag, "n18_p_magnitude": p_act_mag,
        "n18_r_mag_lo95": lo_am, "n18_r_mag_hi95": hi_am,
    })

    print(f"\n[{layer}]")
    print(f"  Mean expert norm: {mean_norm:.4f} ± {std_norm:.4f}")
    print(f"  Full vs Direction: r={r_full_dir:.4f} (95%CI {fd_lo:.3f}–{fd_hi:.3f}), p={p_full_dir:.2e}")
    print(f"  Full vs Magnitude: r={r_full_mag:.4f} (95%CI {fm_lo:.3f}–{fm_hi:.3f}), p={p_full_mag:.2e}")
    print(f"  Dir vs Magnitude:  r={r_dir_mag:.4f}  (orthogonality), p={p_dir_mag:.2e}")
    print(f"  ── Against D_actual_KL (n=18, bootstrap 95%CI): ──")
    print(f"  Full:      r={r_act_full:.3f} [{lo_af:.3f}–{hi_af:.3f}]")
    print(f"  Direction: r={r_act_cos:.3f}  [{lo_ac:.3f}–{hi_ac:.3f}]")
    print(f"  Magnitude: r={r_act_mag:.3f}  [{lo_am:.3f}–{hi_am:.3f}]")

result_df = pd.DataFrame(rows)
result_df.to_csv(f"{OUT}/magnitude_vs_direction.csv", index=False)
print(f"\nSaved: {OUT}/magnitude_vs_direction.csv")

# ── Figure 1: Scatter D_full vs D_cosine colored by norm-diff ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, layer in zip(axes, LAYERS):
    C = centroids[layer]
    norms = np.linalg.norm(C, axis=1)
    C_norm = C / norms[:, None]
    ii, jj = np.triu_indices(64, k=1)
    D_full   = cdist(C, C, "euclidean")[ii, jj]
    D_cosine = cdist(C, C, "cosine")[ii, jj]
    D_mag    = np.abs(norms[ii] - norms[jj])

    sc = ax.scatter(D_full, D_cosine, c=D_mag, cmap="plasma",
                    alpha=0.15, s=4, rasterized=True)
    r, _ = spearmanr(D_full, D_cosine)
    ax.set_title(f"{layer} layer\nSpearman(full, dir) = {r:.3f}", fontsize=11)
    ax.set_xlabel("L2 distance (full C)", fontsize=10)
    ax.set_ylabel("Cosine distance (direction)", fontsize=10)
    plt.colorbar(sc, ax=ax, label="|‖cᵢ‖ − ‖cⱼ‖|")

plt.suptitle("Magnitude vs Direction: Full L2 vs Cosine distance\n(n=2016 expert pairs, colour=magnitude difference)",
             fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_mag_vs_dir_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}/fig_mag_vs_dir_scatter.png")

# ── Figure 2: Bar chart of correlations with D_actual_KL ──────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(3)
width = 0.25
rep_labels = ["Full C (L2)", "Direction (cosine)", "Magnitude"]
colours    = ["#2ecc71", "#3498db", "#e74c3c"]

for i, (layer, row) in enumerate(zip(LAYERS, rows)):
    rs  = [row["n18_r_full_vs_D_actual"], row["n18_r_cosine_vs_D_actual"], row["n18_r_magnitude_vs_D_actual"]]
    los = [row["n18_r_full_lo95"],        row["n18_r_cosine_lo95"],        row["n18_r_mag_lo95"]]
    his = [row["n18_r_full_hi95"],        row["n18_r_cosine_hi95"],        row["n18_r_mag_hi95"]]
    errs = [[r - l for r, l in zip(rs, los)], [h - r for h, r in zip(his, rs)]]

for j, (label, colour) in enumerate(zip(rep_labels, colours)):
    vals = [rows[i][["n18_r_full_vs_D_actual","n18_r_cosine_vs_D_actual","n18_r_magnitude_vs_D_actual"][j]]
            for i in range(3)]
    los  = [rows[i][["n18_r_full_lo95","n18_r_cosine_lo95","n18_r_mag_lo95"][j]] for i in range(3)]
    his  = [rows[i][["n18_r_full_hi95","n18_r_cosine_hi95","n18_r_mag_hi95"][j]] for i in range(3)]
    yerr = [[v - l for v, l in zip(vals, los)], [h - v for h, v in zip(his, vals)]]
    ax.bar(x + j * width, vals, width, label=label, color=colour, edgecolor="k",
           linewidth=0.6, yerr=yerr, capsize=4, error_kw={"elinewidth": 1.2})

ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
ax.set_xticks(x + width)
ax.set_xticklabels(LAYERS, fontsize=11)
ax.set_ylabel("Spearman r vs D_actual_KL", fontsize=11)
ax.set_title("Representation vs Merge Cost (n=18 pairs)\n95% bootstrap CI", fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(-0.6, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_mag_vs_dir_corr.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}/fig_mag_vs_dir_corr.png")
print("\n[Script 1 COMPLETE]")
