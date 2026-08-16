import os
import json
import matplotlib.pyplot as plt

base_dir = "results/exp5"
trajectories = ["Random_one_shot", "Parameter_iterative", "CARE_COM_one_shot"]
labels = ["Random Baseline", "Iterative Parameter Baseline", "CARE-COM (Ours)"]

plt.figure(figsize=(10, 6))

md_lines = ["# Experiment 5 Compression Results\n"]
md_lines.append("| Experts | " + " | ".join(labels) + " |")
md_lines.append("|---" * (len(labels) + 1) + "|")

all_levels = [64, 56, 48, 40, 32, 24, 16]
data_table = {level: [] for level in all_levels}

for traj, label in zip(trajectories, labels):
    path = os.path.join(base_dir, traj, "metrics.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            x = []
            y = []
            for step in data:
                experts = step["level"]
                ppl = step["tier1"]["wikitext_2_ppl"]
                x.append(experts)
                y.append(ppl)
                data_table[experts].append(ppl)
                
            plt.plot(x, y, marker='o', label=label)

for level in sorted(all_levels, reverse=True):
    row = [f"{level}"]
    for val in data_table[level]:
        row.append(f"{val:.2f}")
    md_lines.append("| " + " | ".join(row) + " |")

plt.gca().invert_xaxis()  # 64 down to 16
plt.xlabel("Number of Experts")
plt.ylabel("Wikitext-2 Perplexity (Lower is Better)")
plt.title("Perplexity Degradation During Compression")
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig("results/exp5/compression_ppl.png")

with open("results/exp5/compression_summary.md", "w") as f:
    f.write("\n".join(md_lines))

print("Results compiled.")
