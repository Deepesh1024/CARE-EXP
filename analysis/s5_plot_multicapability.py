import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
OUT_DIR = "analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Data extracted from Phase 12 execution logs
base_ppl = 7.1258
base_mmlu = 0.2655

data_ppl = {
    "Random": {4: 5128.3320, 6: 106.4914, 7: 25.9373, 8: base_ppl},
    "Sub-MoE": {4: 1149.0869, 6: 83.4945, 7: 27.0604, 8: base_ppl},
    "REAP": {4: 17.48, 6: 9.81, 7: 8.10, 8: base_ppl},
    "CARE": {4: 87.1913, 6: 15.6933, 7: 10.1729, 8: base_ppl},
}

data_mmlu = {
    "REAP": {4: 0.2305, 6: 0.2308, 7: 0.2308, 8: base_mmlu},
    "CARE": {4: 0.2365, 6: 0.2520, 7: 0.2541, 8: base_mmlu},
}

colors = {"Random": "#95a5a6", "Sub-MoE": "#3498db", "REAP": "#e74c3c", "CARE": "#2ecc71"}
markers = {"Random": "o", "Sub-MoE": "s", "REAP": "D", "CARE": "^"}
checkpoints = [8, 7, 6, 4]

# 1. PPL Plot
fig, ax = plt.subplots(figsize=(10, 6))
for method in ["Random", "Sub-MoE", "REAP", "CARE"]:
    ppls = [data_ppl[method][c] for c in checkpoints]
    ax.plot(checkpoints, ppls, label=method, color=colors[method], marker=markers[method], linewidth=2.5, markersize=8)

ax.set_yscale("log")
ax.set_xlabel("MLP Experts per Layer", fontsize=13)
ax.set_ylabel("Perplexity (log scale, lower is better)", fontsize=13)
ax.set_title("JetMoE-8B Compression - WikiText-2 Perplexity", fontsize=14, pad=15)
ax.set_xticks(checkpoints)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.invert_xaxis()  # 8 -> 7 -> 6 -> 4
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "final_ppl_curve.png"), dpi=300)
plt.close()

# 2. MMLU Plot (only CARE and REAP)
fig, ax = plt.subplots(figsize=(10, 6))
for method in ["REAP", "CARE"]:
    mmlus = [data_mmlu[method][c] * 100 for c in checkpoints] # convert to %
    ax.plot(checkpoints, mmlus, label=method, color=colors[method], marker=markers[method], linewidth=2.5, markersize=8)

ax.set_xlabel("MLP Experts per Layer", fontsize=13)
ax.set_ylabel("MMLU Accuracy (%)", fontsize=13)
ax.set_title("JetMoE-8B Compression - MMLU Benchmark", fontsize=14, pad=15)
ax.set_xticks(checkpoints)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.invert_xaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "final_mmlu_curve.png"), dpi=300)
plt.close()

print(f"Saved plots to {OUT_DIR}")
