import json
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    with open("experiments/experiment10/results/frc_results.json", "r") as f:
        results = json.load(f)
        
    epsilons = ["0.05", "0.01", "0.001"]
    conditions = ["raw", "norm"]
    
    # We will generate a markdown table
    md = "# Experiment 10: Functional Compressibility Results\n\n"
    
    for condition in conditions:
        md += f"## Condition: {condition.upper()}\n\n"
        
        # Plotting Setup
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Functional Representation Cost ({condition.upper()})")
        
        for e_idx, eps in enumerate(epsilons):
            ax = axes[e_idx]
            
            md += f"### Fidelity: {100 - float(eps)*100}%\n"
            md += "| Group | Mean Overlap | Mean FRC (Indep) | Mean FRC (Shared) | Compression Ratio | Shared Fraction |\n"
            md += "|-------|--------------|------------------|-------------------|-------------------|-----------------|\n"
            
            box_data = []
            labels = []
            
            for group_name, pairs in results.items():
                overlaps = []
                indep_costs = []
                shared_costs = []
                comp_ratios = []
                shared_fracs = []
                
                for p in pairs:
                    data = p[condition][eps]
                    
                    k_A = data["k_A"]
                    k_B = data["k_B"]
                    opt_cost = data["optimal_cost"]
                    k_s = data["k_shared"]
                    overlap = data["overlap"]
                    
                    if opt_cost == 0:
                        continue
                        
                    indep_cost = k_A + k_B
                    comp_ratio = indep_cost / opt_cost
                    shared_frac = k_s / opt_cost
                    
                    overlaps.append(overlap)
                    indep_costs.append(indep_cost)
                    shared_costs.append(opt_cost)
                    comp_ratios.append(comp_ratio)
                    shared_fracs.append(shared_frac)
                    
                if len(indep_costs) == 0:
                    continue
                    
                m_overlap = np.mean(overlaps)
                m_indep = np.mean(indep_costs)
                m_shared = np.mean(shared_costs)
                m_ratio = np.mean(comp_ratios)
                m_frac = np.mean(shared_fracs)
                
                md += f"| {group_name} | {m_overlap:.4f} | {m_indep:.1f} | {m_shared:.1f} | {m_ratio:.3f} | {m_frac:.3f} |\n"
                
                box_data.append(shared_fracs)
                labels.append(group_name)
                
            md += "\n"
            
            ax.boxplot(box_data, labels=labels)
            ax.set_title(f"Fidelity: {100 - float(eps)*100}%")
            ax.set_ylabel("Shared Fraction")
            
        plt.tight_layout()
        plt.savefig(f"experiments/experiment10/results/frc_plot_{condition}.png", dpi=300)
        plt.close()
        
    with open("experiments/experiment10/results/report.md", "w") as f:
        f.write(md)
        
    print("Report generated at experiments/experiment10/results/report.md")

if __name__ == "__main__":
    main()
