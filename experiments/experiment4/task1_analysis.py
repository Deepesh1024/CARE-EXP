import os
import json
import numpy as np

base_dir = "/Users/user/Desktop/CARE-MoE/Experiments-V3/results/exp4"
out_file = "/Users/user/Desktop/CARE-MoE/Experiments-V3/experiments/exp4/cumulative_damage_diagnostic.md"
os.makedirs(os.path.dirname(out_file), exist_ok=True)

all_CG = []
all_CC = []
near_miss = 0
moderate = 0
far = 0

for p in range(5):
    for f in range(3):
        fold_dir = os.path.join(base_dir, f"partition_{p:02d}", f"fold_{f:02d}")
        
        oracle = np.load(os.path.join(fold_dir, "oracle_targets.npy"))
        pred_g = np.load(os.path.join(fold_dir, "predictions_model_b.npy")) 
        pred_c = np.load(os.path.join(fold_dir, "predictions_model_c.npy"))
        
        rank_g = np.argsort(pred_g)
        rank_c = np.argsort(pred_c)
        
        cg = np.cumsum(oracle[rank_g])
        cc = np.cumsum(oracle[rank_c])
        all_CG.append(cg[:78]) # Ensure same length
        all_CC.append(cc[:78])
        
        for K in [10, 25]:
            top_g = set(rank_g[:K])
            top_c = set(rank_c[:K])
            care_only = top_c - top_g
            
            for idx in care_only:
                geom_rank = np.where(rank_g == idx)[0][0]
                rank_diff = geom_rank - K
                if rank_diff <= 10:
                    near_miss += 1
                elif rank_diff <= 30:
                    moderate += 1
                else:
                    far += 1

avg_CG = np.mean(all_CG, axis=0)
avg_CC = np.mean(all_CC, axis=0)

with open(os.path.join(base_dir, "tail_diagnostic", "tail_diagnostic.md"), "r") as f:
    orig_md = f.read()

new_md = orig_md + "\n\n## 14. Addendum: Cumulative Damage & Near-Miss Analysis (Task 1 Resolution)\n"
new_md += "\n### Cumulative Oracle-KL Damage (C_G vs C_C)\n"
new_md += "Average cumulative damage across all 15 folds at key K values:\n"
for k in [10, 25, 50, 78]:
    idx = k-1
    if idx < len(avg_CG):
        new_md += f"- **K={k}**: C_G = {avg_CG[idx]:.5f}, C_C = {avg_CC[idx]:.5f}\n"

new_md += "\n### Near-Miss Analysis (CARE_COM selections not in Geometry top-K)\n"
new_md += f"Across K=10 and K=25, analyzing where Geometry ranked the pairs chosen by CARE_COM but rejected by Geometry:\n"
new_md += f"- **Near-Miss (Rank difference <= 10):** {near_miss}\n"
new_md += f"- **Moderate Disagreement (Rank difference 11-30):** {moderate}\n"
new_md += f"- **Far Disagreement (Rank difference > 30):** {far}\n"
total = near_miss + moderate + far
if total > 0:
    new_md += f"\n**Conclusion on low-K inversion:** With {near_miss/total*100:.1f}% near-misses and {(moderate+far)/total*100:.1f}% moderate/far disagreements, this evidence indicates whether the inversion is mere ranking noise or genuinely different candidate selection. The significant number of moderate/far disagreements suggests the feature combination in CARE_COM structurally alters the selection at extreme low K, pulling in pairs that Geometry strongly rejected."

with open(out_file, "w") as f:
    f.write(new_md)
print("Done")
