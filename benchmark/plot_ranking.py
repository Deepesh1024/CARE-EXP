import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

csv_path = "benchmark_results/summary/results.csv"
out_path = "benchmark_results/summary/plots/fig8_ranking.png"

df = pd.read_csv(csv_path)

df_compressed = df[df["experts"] < 64]
method_deltas = df_compressed.groupby("method")["delta_ppl"].sum()

if "random" in method_deltas:
    random_delta = method_deltas["random"]
else:
    random_delta = method_deltas.max()

scores = {}
for method, delta in method_deltas.items():
    if method != "random":
        score = 100 * (random_delta - delta) / random_delta
        scores[method] = score

scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

methods = list(scores.keys())
values = list(scores.values())

pretty_names = {
    "care_adaptive": "CARE (Adaptive)",
    "care_static": "CARE (Static)",
    "sub_moe": "Sub-MoE (REAP)",
    "m_smoe": "M-SMoE (REAP)",
    "hc_smoe": "HC-SMoE (REAP)"
}
labels = [pretty_names.get(m, m) for m in methods]

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

colors = sns.color_palette("viridis", len(methods))
bars = plt.bar(labels, values, color=colors)

plt.title("Overall Compression Quality Ranking", fontsize=16, pad=20)
plt.ylabel("Degradation Prevented vs Random (%)", fontsize=12)
plt.xlabel("Compression Algorithm", fontsize=12)
plt.ylim(0, max(values) * 1.1)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(out_path, dpi=300)
print(f"Saved ranking plot to {out_path}")
