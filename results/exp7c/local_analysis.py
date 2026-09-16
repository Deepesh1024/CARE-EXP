"""
Standalone local analysis of 7C results.
Run: python results/exp7c/local_analysis.py
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(OUT_DIR, "analysis/7c_final_residual_analysis.csv"))

r = df['residual'].values
C  = df['C_mutual'].values

rho, p = spearmanr(C, r)
pr, pp = pearsonr(C, r)

np.random.seed(42)
boot_rhos = [spearmanr(C[np.random.randint(0,18,18)], r[np.random.randint(0,18,18)])[0] for _ in range(5000)]
boot_rhos = [x for x in boot_rhos if not np.isnan(x)]
ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

print(f"N = {len(df)}")
print(f"Spearman rho = {rho:+.4f}   p = {p:.4e}")
print(f"Pearson  r   = {pr:+.4f}   p = {pp:.4e}")
print(f"95% Bootstrap CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
print()
print("--- Sorted by C_mutual ---")
print(df[['pair_id','expert_i','expert_j','D_pred','D_actual_KL','residual','C_mutual']].sort_values('C_mutual', ascending=False).to_string(index=False))
print()

# Flag the outlier pair_10 (7,32)
outlier = df[df['pair_id'] == 'pair_10']
print(f"Notable outlier pair_10 (expert 7,32):")
print(f"  C_mutual = {outlier['C_mutual'].values[0]:.4f}  (highest by far)")
print(f"  residual = {outlier['residual'].values[0]:+.6f}")

# --- Scatter plot ---
fig, ax = plt.subplots(figsize=(7, 5))
colors = ['#E74C3C' if abs(r_) > 0.01 else '#3498DB' for r_ in r]

for i, row in df.iterrows():
    ax.scatter(row['C_mutual'], row['residual'], color=colors[i], s=60, zorder=3)
    ax.annotate(f"({int(row['expert_i'])},{int(row['expert_j'])})",
                (row['C_mutual'], row['residual']),
                fontsize=7, xytext=(3, 3), textcoords='offset points', color='#555')

# Regression line
m, b = np.polyfit(C, r, 1)
x_line = np.linspace(C.min(), C.max(), 100)
ax.plot(x_line, m * x_line + b, '--', color='grey', alpha=0.7, label='OLS fit')
ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)

ax.set_xlabel("C_mutual  (Functional Neuron Mutual Coverage)", fontsize=11)
ax.set_ylabel("Residual  R = D_actual − D_pred", fontsize=11)
ax.set_title(f"7C: Functional Neuron Coverage vs CARE-COM Residual\n"
             f"Spearman ρ = {rho:+.3f}  p = {p:.3f}  N = {len(df)}", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(OUT_DIR, "analysis/scatter_coverage_vs_residual.png")
plt.savefig(plot_path, dpi=150)
print(f"\nPlot saved to {plot_path}")
