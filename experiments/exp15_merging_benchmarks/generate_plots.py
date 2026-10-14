import os
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 15

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
BM_PLOTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "benchmark_results", "exp15_merging"))
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(BM_PLOTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Definition
# -----------------------------------------------------------------------------

# JetMoE-8B (Base PPL = 7.1258)
jet_experts = [8, 7, 6, 4]
jet_care = [7.13, 10.17, 15.69, 87.19]
jet_submoe = [7.13, 27.06, 83.49, 1149.09]
jet_hcsmoe = [7.13, 11.22, 73.93, 3859.96]
jet_mcsmoe = [7.13, 10.91, 3116.29, 17964.92]

# Phi-3.5-MoE (Base PPL = 4.57)
phi_experts = [16, 14, 12, 10, 8]
phi_care = [4.57, 6.88, 9.24, 13.57, 32.75]
phi_submoe = [4.57, 8.64, 8.41, 11.50, 26.68]
phi_hcsmoe = [4.57, 8.73, 14.29, 95.09, 231.56]
phi_mcsmoe = [4.57, 8.44, 13.97, 40.75, 2543.96]

colors = {
    'CARE-Adaptive': '#1f77b4',  # Vibrant blue
    'Sub-MoE': '#ff7f0e',        # Amber orange
    'HC-SMoE': '#2ca02c',        # Forest green
    'MC-SMoE': '#d62728',        # Crimson red
}

markers = {
    'CARE-Adaptive': 'o',
    'Sub-MoE': 's',
    'HC-SMoE': '^',
    'MC-SMoE': 'D',
}

# -----------------------------------------------------------------------------
# Plot 1: JetMoE-8B Perplexity Curve
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

ax.plot(jet_experts, jet_care, label='CARE-Adaptive (Ours)', color=colors['CARE-Adaptive'], 
        marker=markers['CARE-Adaptive'], linewidth=2.5, markersize=8)
ax.plot(jet_experts, jet_submoe, label='Sub-MoE (CA Clustering)', color=colors['Sub-MoE'], 
        marker=markers['Sub-MoE'], linewidth=2, markersize=7, linestyle='--')
ax.plot(jet_experts, jet_hcsmoe, label='HC-SMoE (ICML 2025)', color=colors['HC-SMoE'], 
        marker=markers['HC-SMoE'], linewidth=2, markersize=7, linestyle='-.')
ax.plot(jet_experts, jet_mcsmoe, label='MC-SMoE (arXiv:2310.01334)', color=colors['MC-SMoE'], 
        marker=markers['MC-SMoE'], linewidth=2, markersize=7, linestyle=':')

ax.set_yscale('log')
ax.set_xlabel('Retained Experts per Layer (Baseline = 8)', fontweight='bold')
ax.set_ylabel('WikiText-2 Test Perplexity (Log Scale)', fontweight='bold')
ax.set_title('JetMoE-8B: Expert Merging Perplexity Degradation', fontweight='bold', pad=15)
ax.set_xticks(jet_experts)
ax.set_xticklabels(['8 (Base)', '7 (-12.5%)', '6 (-25%)', '4 (-50%)'])
ax.invert_xaxis()  # Compression increases to the right

# Annotation for 4 experts
ax.annotate('CARE-Adaptive: 87.19\n(44× lower than HC-SMoE)', 
            xy=(4, 87.19), xytext=(4.3, 20),
            arrowprops=dict(facecolor=colors['CARE-Adaptive'], arrowstyle='->', lw=1.5),
            fontsize=10, fontweight='bold', color=colors['CARE-Adaptive'])

ax.annotate('MC-SMoE: 17,964.92\nHC-SMoE: 3,859.96', 
            xy=(4, 3859.96), xytext=(4.7, 7000),
            arrowprops=dict(facecolor=colors['MC-SMoE'], arrowstyle='->', lw=1.5),
            fontsize=10, fontweight='bold', color='#990000')

ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
plt.tight_layout()

p1_path = os.path.join(PLOTS_DIR, "jetmoe_merging_benchmark.png")
p1_bm = os.path.join(BM_PLOTS_DIR, "jetmoe_merging_benchmark.png")
plt.savefig(p1_path, bbox_inches='tight')
plt.savefig(p1_bm, bbox_inches='tight')
plt.close()
print(f"Saved: {p1_path}")

# -----------------------------------------------------------------------------
# Plot 2: Phi-3.5-MoE Perplexity Curve
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

ax.plot(phi_experts, phi_care, label='CARE-Adaptive (Ours)', color=colors['CARE-Adaptive'], 
        marker=markers['CARE-Adaptive'], linewidth=2.5, markersize=8)
ax.plot(phi_experts, phi_submoe, label='Sub-MoE (Historical Baseline)', color=colors['Sub-MoE'], 
        marker=markers['Sub-MoE'], linewidth=2, markersize=7, linestyle='--')
ax.plot(phi_experts, phi_hcsmoe, label='HC-SMoE (ICML 2025)', color=colors['HC-SMoE'], 
        marker=markers['HC-SMoE'], linewidth=2, markersize=7, linestyle='-.')
ax.plot(phi_experts, phi_mcsmoe, label='MC-SMoE (arXiv:2310.01334)', color=colors['MC-SMoE'], 
        marker=markers['MC-SMoE'], linewidth=2, markersize=7, linestyle=':')

ax.set_yscale('log')
ax.set_xlabel('Retained Experts per Layer (Baseline = 16)', fontweight='bold')
ax.set_ylabel('WikiText-2 Test Perplexity (Log Scale)', fontweight='bold')
ax.set_title('Phi-3.5-MoE: Expert Merging Perplexity Degradation', fontweight='bold', pad=15)
ax.set_xticks(phi_experts)
ax.set_xticklabels(['16 (Base)', '14 (-12.5%)', '12 (-25%)', '10 (-37.5%)', '8 (-50%)'])
ax.invert_xaxis()

# Annotations
ax.annotate('CARE-Adaptive: 32.75', 
            xy=(8, 32.75), xytext=(8.8, 12),
            arrowprops=dict(facecolor=colors['CARE-Adaptive'], arrowstyle='->', lw=1.5),
            fontsize=10, fontweight='bold', color=colors['CARE-Adaptive'])

ax.annotate('MC-SMoE: 2,543.96\nHC-SMoE: 231.56', 
            xy=(8, 2543.96), xytext=(9.8, 1200),
            arrowprops=dict(facecolor=colors['MC-SMoE'], arrowstyle='->', lw=1.5),
            fontsize=10, fontweight='bold', color='#990000')

ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
plt.tight_layout()

p2_path = os.path.join(PLOTS_DIR, "phi_merging_benchmark.png")
p2_bm = os.path.join(BM_PLOTS_DIR, "phi_merging_benchmark.png")
plt.savefig(p2_path, bbox_inches='tight')
plt.savefig(p2_bm, bbox_inches='tight')
plt.close()
print(f"Saved: {p2_path}")

# -----------------------------------------------------------------------------
# Plot 3: Dual-Panel Comparison (Cross-Architecture Overview)
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# Panel 1: JetMoE-8B
ax1.plot(jet_experts, jet_care, label='CARE-Adaptive', color=colors['CARE-Adaptive'], 
         marker=markers['CARE-Adaptive'], linewidth=2.5, markersize=8)
ax1.plot(jet_experts, jet_submoe, label='Sub-MoE', color=colors['Sub-MoE'], 
         marker=markers['Sub-MoE'], linewidth=2, markersize=7, linestyle='--')
ax1.plot(jet_experts, jet_hcsmoe, label='HC-SMoE', color=colors['HC-SMoE'], 
         marker=markers['HC-SMoE'], linewidth=2, markersize=7, linestyle='-.')
ax1.plot(jet_experts, jet_mcsmoe, label='MC-SMoE', color=colors['MC-SMoE'], 
         marker=markers['MC-SMoE'], linewidth=2, markersize=7, linestyle=':')
ax1.set_yscale('log')
ax1.set_xlabel('Retained Experts (Base = 8)', fontweight='bold')
ax1.set_ylabel('WikiText-2 Perplexity (Log Scale)', fontweight='bold')
ax1.set_title('(A) JetMoE-8B (8 Experts)', fontweight='bold', pad=12)
ax1.set_xticks(jet_experts)
ax1.set_xticklabels(['8 (Base)', '7', '6', '4 (-50%)'])
ax1.invert_xaxis()
ax1.legend(frameon=True, loc='upper left')

# Panel 2: Phi-3.5-MoE
ax2.plot(phi_experts, phi_care, label='CARE-Adaptive', color=colors['CARE-Adaptive'], 
         marker=markers['CARE-Adaptive'], linewidth=2.5, markersize=8)
ax2.plot(phi_experts, phi_submoe, label='Sub-MoE', color=colors['Sub-MoE'], 
         marker=markers['Sub-MoE'], linewidth=2, markersize=7, linestyle='--')
ax2.plot(phi_experts, phi_hcsmoe, label='HC-SMoE', color=colors['HC-SMoE'], 
         marker=markers['HC-SMoE'], linewidth=2, markersize=7, linestyle='-.')
ax2.plot(phi_experts, phi_mcsmoe, label='MC-SMoE', color=colors['MC-SMoE'], 
         marker=markers['MC-SMoE'], linewidth=2, markersize=7, linestyle=':')
ax2.set_yscale('log')
ax2.set_xlabel('Retained Experts (Base = 16)', fontweight='bold')
ax2.set_ylabel('WikiText-2 Perplexity (Log Scale)', fontweight='bold')
ax2.set_title('(B) Phi-3.5-MoE (16 Experts)', fontweight='bold', pad=12)
ax2.set_xticks(phi_experts)
ax2.set_xticklabels(['16 (Base)', '14', '12', '10', '8 (-50%)'])
ax2.invert_xaxis()
ax2.legend(frameon=True, loc='upper left')

plt.suptitle('Cross-Architecture Expert Merging Benchmark: CARE-Adaptive vs HC-SMoE, MC-SMoE & Sub-MoE', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

p3_path = os.path.join(PLOTS_DIR, "cross_arch_merging_comparison.png")
p3_bm = os.path.join(BM_PLOTS_DIR, "cross_arch_merging_comparison.png")
plt.savefig(p3_path, bbox_inches='tight')
plt.savefig(p3_bm, bbox_inches='tight')
plt.close()
print(f"Saved: {p3_path}")
