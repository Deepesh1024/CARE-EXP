python3 -c '
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=300)

targets = [64, 60, 48]
base_ppl = 10.1479

# Verified points
lines = [
    ("REAP-Prune (Pruning)", "#16a34a", "o--", [base_ppl, 10.3992, 12.8404], 2.0, 6),
    ("CARE-Static (Ours)", "#0284c7", "s-", [base_ppl, 11.1774, 17.6951], 2.4, 7),
    ("CARE-Adaptive (Ours)", "#0369a1", "D-", [base_ppl, 11.8156, 20.3228], 2.0, 6),
    ("Random-K5 Control", "#94a3b8", "x:", [base_ppl, 12.1646, 22.5990], 1.5, 6),
    ("MC-SMoE", "#64748b", "v-.", [base_ppl, 11.7983, 26.5846], 1.5, 5),
    ("HC-SMoE", "#dc2626", "*--", [base_ppl, 12.0226, 30.9081], 1.5, 6),
    ("Sub-MoE", "#d97706", "p-.", [base_ppl, 12.1974, 34.1186], 1.5, 5),
]

for label, color, style, vals, lw, ms in lines:
    ax.plot(targets, vals, style, label=label, color=color, linewidth=lw, markersize=ms, alpha=0.95)

ax.set_xlabel("Active Experts per Layer (OLMoE-1B-7B)", fontsize=10, fontweight="bold", labelpad=5)
ax.set_ylabel("WikiText-2 Test PPL (↓)", fontsize=10, fontweight="bold", labelpad=5)
ax.set_xticks([64, 60, 48])
ax.set_xticklabels(["64 (Base)", "60 (-6%)", "48 (-25%)"], fontsize=9)
ax.tick_params(axis="both", which="major", labelsize=9)
ax.invert_xaxis()
ax.grid(True, linestyle="--", alpha=0.5)

# Annotation for deep compression divergence
ax.annotate("Sub-MoE (34.1)", xy=(48, 34.12), xytext=(52.5, 33.5),
            fontsize=8, color="#d97706", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#d97706", lw=1.0))
ax.annotate("CARE-Static (17.7)", xy=(48, 17.70), xytext=(53.5, 17.2),
            fontsize=8, color="#0284c7", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.0))
ax.annotate("REAP-Prune (12.8)", xy=(48, 12.84), xytext=(53.5, 12.0),
            fontsize=8, color="#16a34a", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.0))

ax.legend(loc="upper left", fontsize=7.8, frameon=True, framealpha=0.9, facecolor="white", edgecolor="#e2e8f0", ncol=2)

plt.tight_layout()
out_png = "/tmp/hero_ppl_chart.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Hero chart generated successfully at", out_png)
'