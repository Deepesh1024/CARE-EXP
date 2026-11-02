import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

out_dirs = [
    "/Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/reports/CROSS_MODEL_CAPABILITY/run_20261026_master/figures",
    "/Users/deepeshkumarjha/.gemini/antigravity-ide/brain/e680cf5c-105e-4348-8855-7b9923ce377b"
]
for d in out_dirs:
    os.makedirs(d, exist_ok=True)

def save_fig(fig, base_name):
    for d in out_dirs:
        path = os.path.join(d, base_name)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {base_name}")

# Load results data
results_csv = "/Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/reports/CROSS_MODEL_CAPABILITY/run_20261026_master/aggregates/results.csv"
df = pd.read_csv(results_csv)

retention_csv = "/Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/reports/CROSS_MODEL_CAPABILITY/run_20261026_master/aggregates/capability_retention.csv"
df_ret = pd.read_csv(retention_csv)

jaccard_csv = "/Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/reports/CROSS_MODEL_CAPABILITY/run_20261026_master/aggregates/candidate_pool_jaccard_matrix.csv"
df_jac = pd.read_csv(jaccard_csv, index_col=0)

base_ppl = 10.1479
base_gsm8k = 7.0
base_mmlu = 54.0

# -------------------------------------------------------------
# FIG 1: WikiText-2 PPL vs Compression Target across methods
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
targets = [64, 60, 48]

# Method mappings
plot_methods = [
    ("REAP-Prune", "#2ca02c", "o-", [base_ppl, 10.3992, 12.8404]),
    ("CARE-Static (Generic)", "#1f77b4", "s--", [base_ppl, 11.1774, 17.6951]),
    ("CARE-Adaptive (Generic)", "#ff7f0e", "D-", [base_ppl, 11.8156, 20.3228]),
    ("CARE-Adaptive (Multi)", "#9467bd", "^-", [base_ppl, 12.1290, 23.0231]),
    ("Random-K5", "#7f7f7f", "x:", [base_ppl, 12.1646, 22.5990]),
    ("MC-SMoE", "#8c564b", "v-.", [base_ppl, 11.7983, 26.5846]),
    ("HC-SMoE", "#d62728", "*--", [base_ppl, 12.0226, 30.9081]),
    ("Sub-MoE", "#e377c2", "p-.", [base_ppl, 12.1974, 34.1186]),
]

for label, color, style, vals in plot_methods:
    ax.plot(targets, vals, style, label=label, color=color, linewidth=2, markersize=8)

ax.set_xlabel("Active Experts per Layer (Budget)")
ax.set_ylabel("WikiText-2 Test Perplexity (PPL ↓)")
ax.set_title("WikiText-2 Language Modeling PPL vs Compression Target (OLMoE-1B-7B)", fontweight='bold')
ax.set_xticks([64, 60, 48])
ax.invert_xaxis()
ax.legend(loc="upper left", frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)
save_fig(fig, "fig1_ppl_vs_compression.png")

# -------------------------------------------------------------
# FIG 2: Held-out Causal KL Divergence vs Compression Target
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
kl_methods = [
    ("REAP-Prune", "#2ca02c", "o-", [0.0, 0.0328, 0.3453]),
    ("CARE-Static (Generic)", "#1f77b4", "s--", [0.0, 0.1068, 0.5493]),
    ("CARE-Adaptive (Generic)", "#ff7f0e", "D-", [0.0, 0.1460, 0.7342]),
    ("CARE-Adaptive (Multi)", "#9467bd", "^-", [0.0, 0.1862, 0.7448]),
    ("Random-K5", "#7f7f7f", "x:", [0.0, 0.1839, 0.8707]),
    ("MC-SMoE", "#8c564b", "v-.", [0.0, 0.1631, 0.9822]),
    ("HC-SMoE", "#d62728", "*--", [0.0, 0.1803, 1.1231]),
    ("Sub-MoE", "#e377c2", "p-.", [0.0, 0.1959, 1.1945]),
]

for label, color, style, vals in kl_methods:
    ax.plot(targets, vals, style, label=label, color=color, linewidth=2, markersize=8)

ax.set_xlabel("Active Experts per Layer (Budget)")
ax.set_ylabel("Held-out Causal KL Divergence (KL ↓)")
ax.set_title("Held-out Causal KL Divergence vs Compression Target (OLMoE-1B-7B)", fontweight='bold')
ax.set_xticks([64, 60, 48])
ax.invert_xaxis()
ax.legend(loc="upper left", frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)
save_fig(fig, "fig2_causal_kl_vs_compression.png")

# -------------------------------------------------------------
# FIG 3: GSM8K Reasoning Retention across Methods & Probes
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

df_60 = df[df['target_experts'] == 60].sort_values(by='gsm8k_accuracy', ascending=False)
df_48 = df[(df['target_experts'] == 48) & (df['seed'] == 42)].sort_values(by='gsm8k_accuracy', ascending=False)

def clean_label(row):
    meth = row['method']
    pr = row['probe_strategy']
    if pr != 'none':
        return f"{meth.replace('_', ' ').title()} ({pr.title()})"
    return meth.replace('_', ' ').title()

labels_60 = [clean_label(r) for _, r in df_60.iterrows()]
labels_48 = [clean_label(r) for _, r in df_48.iterrows()]

sns.barplot(x=df_60['gsm8k_accuracy'], y=labels_60, ax=ax1, palette="Blues_r")
ax1.axvline(base_gsm8k, color='red', linestyle='--', label=f'Baseline 64E ({base_gsm8k}%)')
ax1.set_xlabel("GSM8K Accuracy (%) ↑")
ax1.set_title("60 Experts / Layer (~6.25% Compression)", fontweight='bold')
ax1.legend(loc='lower right')

sns.barplot(x=df_48['gsm8k_accuracy'], y=labels_48, ax=ax2, palette="Oranges_r")
ax2.axvline(base_gsm8k, color='red', linestyle='--', label=f'Baseline 64E ({base_gsm8k}%)')
ax2.set_xlabel("GSM8K Accuracy (%) ↑")
ax2.set_title("48 Experts / Layer (25% Deep Compression)", fontweight='bold')
ax2.legend(loc='lower right')

plt.suptitle("GSM8K Mathematical Reasoning Retention Across Methods & Probes", fontweight='bold', y=1.02)
save_fig(fig, "fig3_gsm8k_reasoning_retention.png")

# -------------------------------------------------------------
# FIG 4: MMLU General Knowledge Retention across Methods & Probes
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

df_60_mmlu = df[df['target_experts'] == 60].sort_values(by='mmlu_accuracy', ascending=False)
df_48_mmlu = df[(df['target_experts'] == 48) & (df['seed'] == 42)].sort_values(by='mmlu_accuracy', ascending=False)

labels_60_m = [clean_label(r) for _, r in df_60_mmlu.iterrows()]
labels_48_m = [clean_label(r) for _, r in df_48_mmlu.iterrows()]

sns.barplot(x=df_60_mmlu['mmlu_accuracy'], y=labels_60_m, ax=ax1, palette="crest_r")
ax1.axvline(base_mmlu, color='red', linestyle='--', label=f'Baseline 64E ({base_mmlu}%)')
ax1.set_xlabel("MMLU Accuracy (%) ↑")
ax1.set_title("60 Experts / Layer", fontweight='bold')
ax1.legend(loc='lower right')

sns.barplot(x=df_48_mmlu['mmlu_accuracy'], y=labels_48_m, ax=ax2, palette="flare_r")
ax2.axvline(base_mmlu, color='red', linestyle='--', label=f'Baseline 64E ({base_mmlu}%)')
ax2.set_xlabel("MMLU Accuracy (%) ↑")
ax2.set_title("48 Experts / Layer", fontweight='bold')
ax2.legend(loc='lower right')

plt.suptitle("MMLU General Knowledge Retention Across Methods & Probes", fontweight='bold', y=1.02)
save_fig(fig, "fig4_mmlu_knowledge_retention.png")

# -------------------------------------------------------------
# FIG 5: Multi-Capability Retention Heatmap
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 8))

# Filter rows for 48E comparison
df_48_ret = df_ret[(df_ret['target_experts'] == 48) & (df_ret['seed'] == 42)].copy()
df_48_ret['Label'] = df_48_ret.apply(lambda r: f"{r['method'].replace('_', ' ').title()} [{r['probe_strategy']}]", axis=1)

heatmap_data = df_48_ret.set_index('Label')[['ppl_retention', 'gsm8k_retention', 'mmlu_retention', 'mean_capability_retention']]
heatmap_data.columns = ['LM Retention (1/PPL)', 'GSM8K Retention', 'MMLU Retention', 'Overall Mean Retention']
heatmap_data = heatmap_data.sort_values(by='Overall Mean Retention', ascending=False)

sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Retention Ratio (1.0 = Base)'}, ax=ax)
ax.set_title("Comprehensive Capability Retention Matrix @ 48 Experts (Deep Compression)", fontweight='bold')
ax.set_ylabel("")
save_fig(fig, "fig5_capability_retention_heatmap.png")

# -------------------------------------------------------------
# FIG 6: Candidate Pool Jaccard Overlap Matrix Heatmap
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(df_jac, annot=True, fmt=".3f", cmap="magma", vmin=0.0, vmax=1.0, square=True, ax=ax)
ax.set_title("Candidate Merge Pool Jaccard Similarity (Section 18.H)", fontweight='bold')
ax.set_xlabel("Probe Strategy")
ax.set_ylabel("Probe Strategy")
save_fig(fig, "fig6_candidate_pool_jaccard.png")

# -------------------------------------------------------------
# FIG 7: CARE-Adaptive vs CARE-Static Retention
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# PPL comparison
categories = ['60 Experts', '48 Experts']
adaptive_ppl = [11.8156, 20.3228]
static_ppl = [11.1774, 17.6951]
x = np.arange(len(categories))
width = 0.35

ax1.bar(x - width/2, adaptive_ppl, width, label='CARE-Adaptive (Generic)', color='#ff7f0e')
ax1.bar(x + width/2, static_ppl, width, label='CARE-Static (Generic)', color='#1f77b4')
ax1.set_ylabel('WikiText-2 PPL (↓)')
ax1.set_title('Perplexity Degradation: Adaptive vs Static', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5)

# Task retention comparison
adaptive_m = [48.0, 25.0]
static_m = [50.0, 25.0]
ax2.bar(x - width/2, adaptive_m, width, label='CARE-Adaptive MMLU', color='#ffbb78')
ax2.bar(x + width/2, static_m, width, label='CARE-Static MMLU', color='#aec7e8')
ax2.set_ylabel('MMLU Accuracy % (↑)')
ax2.set_title('MMLU Knowledge Retention: Adaptive vs Static', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Static Geometry vs Dynamic Adaptive Recalibration (OLMoE-1B-7B)", fontweight='bold', y=1.02)
save_fig(fig, "fig7_adaptive_vs_static.png")

# -------------------------------------------------------------
# FIG 8: CARE-Adaptive vs Random-K5 Control
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

random_seeds_48_ppl = [25.95, 34.90, 28.43, 31.40, 28.39] # from audit
mean_rand = np.mean(random_seeds_48_ppl)
std_rand = np.std(random_seeds_48_ppl)

methods = ['CARE-Adaptive\n(Generic)', 'CARE-Static\n(Generic)', 'Random-K=5\n(Mean ± 1σ)', 'Random-1\n(Historical)']
ppl_vals = [20.32, 17.70, mean_rand, 54.28]
yerrs = [0, 0, std_rand, 0]
colors = ['#ff7f0e', '#1f77b4', '#7f7f7f', '#d62728']

bars = ax.bar(methods, ppl_vals, yerr=yerrs, capsize=6, color=colors, alpha=0.9, width=0.5)
ax.axhline(base_ppl, color='green', linestyle='--', label=f'Base 64E ({base_ppl})')
ax.set_ylabel("WikiText-2 Test PPL (↓)")
ax.set_title("CARE vs Controlled Random-K=5 Baseline @ 48 Experts (OLMoE)", fontweight='bold')
ax.legend()
for bar, val in zip(bars, ppl_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.2, f"{val:.2f}", ha='center', fontweight='bold')
save_fig(fig, "fig8_care_vs_random_control.png")

# -------------------------------------------------------------
# FIG 9: CARE-Adaptive vs External Baselines (Sub-MoE, HC-SMoE, MC-SMoE)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))

ext_methods = ['CARE-Adaptive\n(Generic)', 'CARE-Static\n(Generic)', 'MC-SMoE', 'HC-SMoE', 'Sub-MoE']
ext_ppl = [20.32, 17.70, 26.58, 30.91, 34.12]
ext_kl = [0.734, 0.549, 0.982, 1.123, 1.194]

x = np.arange(len(ext_methods))
width = 0.35

ax1 = ax
ax2 = ax.twinx()

bars1 = ax1.bar(x - width/2, ext_ppl, width, label='WikiText-2 PPL (↓)', color='#1f77b4')
bars2 = ax2.bar(x + width/2, ext_kl, width, label='Heldout Causal KL (↓)', color='#ff7f0e')

ax1.set_ylabel('Perplexity (PPL)', color='#1f77b4')
ax2.set_ylabel('Causal KL Divergence', color='#ff7f0e')
ax1.set_xticks(x)
ax1.set_xticklabels(ext_methods)
ax1.set_title("Deep Compression (48E): CARE vs State-of-the-Art Merging Baselines", fontweight='bold')

for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{bar.get_height():.1f}", ha='center', color='#1f77b4', fontsize=9, fontweight='bold')
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f"{bar.get_height():.2f}", ha='center', color='#ff7f0e', fontsize=9, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
save_fig(fig, "fig9_care_vs_external_baselines.png")

# -------------------------------------------------------------
# FIG 10: Pruning (REAP) vs Merging Separation
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter([12.84], [43.0], s=250, color='#2ca02c', label='REAP-Prune (Pruning)', marker='s', edgecolors='black', linewidth=1.5)
ax.scatter([20.32], [25.0], s=200, color='#ff7f0e', label='CARE-Adaptive Generic (Merge)', marker='D', edgecolors='black', linewidth=1.5)
ax.scatter([17.70], [25.0], s=200, color='#1f77b4', label='CARE-Static Generic (Merge)', marker='o', edgecolors='black', linewidth=1.5)
ax.scatter([20.90], [28.0], s=200, color='#d62728', label='CARE-Adaptive Coding (Merge)', marker='^', edgecolors='black', linewidth=1.5)
ax.scatter([26.58], [28.0], s=180, color='#8c564b', label='MC-SMoE (Merge)', marker='v', edgecolors='black')
ax.scatter([30.91], [28.0], s=180, color='#9467bd', label='HC-SMoE (Merge)', marker='*', edgecolors='black')
ax.scatter([34.12], [25.0], s=180, color='#e377c2', label='Sub-MoE (Merge)', marker='p', edgecolors='black')
ax.scatter([22.60], [21.0], s=180, color='#7f7f7f', label='Random-K5 (Merge)', marker='x', linewidth=2)

ax.set_xlabel("WikiText-2 Test Perplexity (PPL ↓)")
ax.set_ylabel("MMLU General Knowledge Accuracy (%) ↑")
ax.set_title("Pruning vs Merging Duality @ 48 Experts (OLMoE-1B-7B)", fontweight='bold')
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
save_fig(fig, "fig10_pruning_vs_merging.png")

# -------------------------------------------------------------
# FIG 11: Compression Runtime vs Capability Retention Trade-off
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))

df_plot = df[(df['target_experts'] == 48) & (df['seed'] == 42)].copy()
for _, r in df_plot.iterrows():
    lbl = clean_label(r)
    time_s = r['compression_time_s']
    ppl = r['wikitext2_ppl']
    ax.scatter(time_s, ppl, s=160, edgecolors='black', alpha=0.85)
    ax.annotate(lbl, (time_s, ppl), textcoords="offset points", xytext=(5, 5), fontsize=8.5)

ax.set_xscale('log')
ax.set_xlabel("Compression Wall-Clock Runtime (seconds, log scale) ↓")
ax.set_ylabel("WikiText-2 Test Perplexity (PPL ↓)")
ax.set_title("Pareto Trade-off: Compression Runtime vs LM Quality (48 Experts)", fontweight='bold')
ax.grid(True, which="both", ls="--", alpha=0.5)
save_fig(fig, "fig11_runtime_vs_retention.png")

# -------------------------------------------------------------
# FIG 12: Cross-Model Architecture Summary
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Model 1: OLMoE-1B-7B (64 -> 60 -> 48)
olmoe_pct = [0, 6.25, 25.0]
olmoe_care = [10.15, 11.82, 20.32]
olmoe_reap = [10.15, 10.40, 12.84]
olmoe_sub = [10.15, 12.20, 34.12]

ax1.plot(olmoe_pct, olmoe_care, 'D-', label='CARE-Adaptive', color='#ff7f0e', linewidth=2)
ax1.plot(olmoe_pct, olmoe_reap, 'o-', label='REAP-Prune', color='#2ca02c', linewidth=2)
ax1.plot(olmoe_pct, olmoe_sub, 'p-.', label='Sub-MoE', color='#e377c2', linewidth=2)
ax1.set_xlabel("Experts Consolidated (%)")
ax1.set_ylabel("WikiText-2 PPL (↓)")
ax1.set_title("OLMoE-1B-7B (64 Experts Base)", fontweight='bold')
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

# Model 2: Qwen3-30B-A3B (128 -> 112 -> 96 -> 80)
qwen_pct = [0, 12.5, 25.0, 37.5]
qwen_care = [9.57, 11.19, 14.81, 19.93]
qwen_reap = [9.57, 11.20, 13.17, 16.14]
qwen_sub = [9.57, 11.88, 14.38, 90.72]
qwen_rand = [9.57, 12.74, 26.35, 47.62]

ax2.plot(qwen_pct, qwen_care, 'D-', label='CARE-Adaptive', color='#ff7f0e', linewidth=2)
ax2.plot(qwen_pct, qwen_reap, 'o-', label='REAP-Prune', color='#2ca02c', linewidth=2)
ax2.plot(qwen_pct, qwen_rand, 'x:', label='Random Control', color='#7f7f7f', linewidth=2)
ax2.plot(qwen_pct, qwen_sub, 'p-.', label='Sub-MoE', color='#e377c2', linewidth=2)
ax2.set_xlabel("Experts Consolidated (%)")
ax2.set_ylabel("WikiText-2 PPL (↓)")
ax2.set_title("Qwen3-30B-A3B (128 Experts Base, 48 Layers)", fontweight='bold')
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()

plt.suptitle("Cross-Model MoE Compression Scaling: OLMoE-1B-7B vs Qwen3-30B-A3B", fontweight='bold', y=1.02)
save_fig(fig, "fig12_cross_model_comparison.png")

print("All 12 publication figures successfully created!")
