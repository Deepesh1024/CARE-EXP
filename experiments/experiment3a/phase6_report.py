"""
CARE-MoE Experiment 3A — Phase 6: Report Generation
======================================================
1. Generate Figures (Capability graph, True empirical baselines,
   Within vs Between KL boxplots, Community robustness heatmap,
   Community Adjacency Heatmap).
2. Compile results into the final experiment3a_report.md.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from config import (
    GRAPHS_DIR,
    COMMUNITIES_DIR,
    BASELINES_DIR,
    VALIDATION_DIR,
    REPORT_PATH,
    PRE_REGISTRATION_PATH,
    LAYERS,
    K_PRIMARY,
    N_EXPERTS
)
from utils import (
    set_global_seed,
    ensure_dirs,
    set_pub_style,
    save_fig,
    load_pickle,
    load_json
)

def plot_capability_graph(layer="aggregated", k=K_PRIMARY):
    graph_path = os.path.join(GRAPHS_DIR, f"{layer}_k{k}_graph.pkl")
    if not os.path.exists(graph_path): return None
    G = load_pickle(graph_path)
    
    comm_path = os.path.join(COMMUNITIES_DIR, "community_assignments.csv")
    if not os.path.exists(comm_path): return None
    assignments = pd.read_csv(comm_path)
    
    layer_assign = assignments[(assignments["Layer"] == layer) & (assignments["k"] == k)]
    if layer_assign.empty: return None
        
    comm_map = dict(zip(layer_assign["Expert"], layer_assign["Louvain_Community"]))
    
    node_colors = [comm_map.get(node, 0) for node in G.nodes()]
        
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42, k=0.15)
    
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, cmap=plt.cm.Set3, edgecolors='gray', ax=ax)
    
    edges = G.edges(data=True)
    weights = [d['weight'] for u, v, d in edges]
    if weights:
        max_w = max(weights)
        norm_weights = [w/max_w for w in weights]
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color=norm_weights, edge_cmap=plt.cm.Blues, ax=ax)
        
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    ax.set_title(f"CARE Capability Graph ({layer.capitalize()} Layer, k={k})", pad=20)
    ax.axis("off")
    return save_fig(fig, f"01_capability_graph_{layer}")

def plot_baseline_comparison(layer="aggregated", k=K_PRIMARY):
    dist_path = os.path.join(BASELINES_DIR, "empirical_null_distributions.json")
    if not os.path.exists(dist_path): return None
    dist_data = load_json(dist_path)
    
    sig_path = os.path.join(BASELINES_DIR, "significance_tests.json")
    sig_data = load_json(sig_path)
    
    if layer not in dist_data or f"k{k}" not in dist_data[layer]: return None
        
    dists = dist_data[layer][f"k{k}"]
    stats = sig_data[layer][f"k{k}"]
    
    metrics = ["Unweighted_Modularity", "Weighted_Modularity", "Global_Efficiency", "Transitivity"]
    metrics = [m for m in metrics if m in dists]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1: axes = [axes]
        
    for i, m in enumerate(metrics):
        care_val = stats[m]["CARE_Value"]
        empirical_vals = dists[m]
        
        sns.histplot(empirical_vals, ax=axes[i], color='lightgray', stat='density', kde=True, label='Empirical Null')
        axes[i].axvline(care_val, color='red', linestyle='--', linewidth=2, label='CARE Graph')
        axes[i].set_title(m.replace('_', ' '))
        if i == 0: axes[i].legend()
            
    plt.tight_layout()
    return save_fig(fig, f"02_graph_statistics_{layer}")

def plot_validation(layer="aggregated"):
    val_path = os.path.join(VALIDATION_DIR, "within_vs_between_kl.csv")
    if not os.path.exists(val_path): return None
    df = pd.read_csv(val_path)
    
    layer_df = df[df["Layer"] == layer]
    if layer_df.empty: return None
        
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=layer_df, x="Is_Within", y="Oracle_KL", 
        hue="Is_Within", palette=["#e74c3c", "#3498db"], inner="quartile", legend=False, ax=ax
    )
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Between Communities", "Within Community"])
    ax.set_xlabel("")
    ax.set_ylabel("Oracle KL Divergence")
    ax.set_title(f"Merge Affinity ({layer.capitalize()} Layer)")
    
    stats_path = os.path.join(VALIDATION_DIR, "validation_statistics.json")
    if os.path.exists(stats_path):
        stats = load_json(stats_path).get(layer, {})
        p_val = stats.get("MannWhitney_p", 1.0)
        d = stats.get("Cohens_d", 0.0)
        sil = stats.get("Silhouette_Score", 0.0)
        ax.text(0.5, 0.95, f"p = {p_val:.2e} | d = {d:.2f} | Silhouette = {sil:.3f}", 
                transform=ax.transAxes, ha='center', va='top', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                
    return save_fig(fig, f"03_within_vs_between_kl_{layer}")

def plot_robustness():
    rob_path = os.path.join(VALIDATION_DIR, "robustness_ari_nmi.json")
    if not os.path.exists(rob_path): return None
    rob_data = load_json(rob_path)
    
    layers = list(rob_data.keys())
    if not layers: return None
    comparisons = ["k5_vs_k8", "k8_vs_k10", "k5_vs_k10"]
    
    ari_matrix = np.zeros((len(layers), len(comparisons)))
    nmi_matrix = np.zeros((len(layers), len(comparisons)))
    
    for i, l in enumerate(layers):
        for j, c in enumerate(comparisons):
            ari_matrix[i, j] = rob_data[l].get(c, {}).get("ARI", 0.0)
            nmi_matrix[i, j] = rob_data[l].get(c, {}).get("NMI", 0.0)
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(ari_matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1,
                xticklabels=[c.replace('_', ' ') for c in comparisons],
                yticklabels=[l.capitalize() for l in layers], ax=ax1)
    ax1.set_title("Adjusted Rand Index (ARI)")
    
    sns.heatmap(nmi_matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1,
                xticklabels=[c.replace('_', ' ') for c in comparisons],
                yticklabels=[l.capitalize() for l in layers], ax=ax2)
    ax2.set_title("Normalized Mutual Information (NMI)")
    
    plt.tight_layout()
    return save_fig(fig, "04_community_robustness")

def plot_community_adjacency(layer="aggregated", k=K_PRIMARY):
    graph_path = os.path.join(GRAPHS_DIR, f"{layer}_k{k}_graph.pkl")
    if not os.path.exists(graph_path): return None
    G = load_pickle(graph_path)
    
    comm_path = os.path.join(COMMUNITIES_DIR, "community_assignments.csv")
    assignments = pd.read_csv(comm_path)
    layer_assign = assignments[(assignments["Layer"] == layer) & (assignments["k"] == k)]
    if layer_assign.empty: return None
    
    # Sort experts by community
    sorted_experts = layer_assign.sort_values(by="Louvain_Community")["Expert"].tolist()
    
    adj_matrix = nx.to_numpy_array(G, nodelist=sorted_experts, weight='weight')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(adj_matrix, cmap="Blues", square=True, cbar=False, xticklabels=False, yticklabels=False, ax=ax)
    
    # Draw boundary lines for communities
    comm_counts = layer_assign["Louvain_Community"].value_counts().sort_index()
    curr_idx = 0
    for count in comm_counts:
        curr_idx += count
        ax.axhline(curr_idx, color='red', linewidth=1)
        ax.axvline(curr_idx, color='red', linewidth=1)
        
    ax.set_title(f"Block Adjacency Heatmap ({layer.capitalize()} Layer)")
    return save_fig(fig, f"05_block_adjacency_{layer}")

def generate_markdown_report():
    print("[Phase 6] Generating Markdown report...")
    
    pre_reg_content = ""
    if os.path.exists(PRE_REGISTRATION_PATH):
        with open(PRE_REGISTRATION_PATH, "r") as f:
            pre_reg_content = f.read()

    sig_data = load_json(os.path.join(BASELINES_DIR, "significance_tests.json"))
    comm_data = load_json(os.path.join(COMMUNITIES_DIR, "community_summary.json"))
    val_data = load_json(os.path.join(VALIDATION_DIR, "validation_statistics.json"))
    corr_data = load_json(os.path.join(VALIDATION_DIR, f"aggregated_centrality_correlations.json"))

    report = [
        "# CARE Experiment 3A: Capability Graph Discovery",
        "\n> **Note:** This experiment strictly preserves $k=5$ as the primary analysis and compares against true empirical nulls.",
        "\n---\n", pre_reg_content, "\n---\n",
        "## Results\n"
    ]
    
    # H1
    report.append("### H1: Graph Organization (Global Topology)")
    report.append("We compared the capability graphs against 1000 Erdős-Rényi random baselines. Both unweighted metrics (binary structure) and weighted metrics (random empirical weight assignments) were computed.\n")
    
    if "aggregated" in sig_data and f"k{K_PRIMARY}" in sig_data["aggregated"]:
        d = sig_data["aggregated"][f"k{K_PRIMARY}"]
        w_mod = d.get("Weighted_Modularity", {})
        u_mod = d.get("Unweighted_Modularity", {})
        
        report.append(f"- **Weighted Modularity:** CARE = {w_mod.get('CARE_Value',0):.4f}, Random = {w_mod.get('Random_Mean',0):.4f} ($p = {w_mod.get('P_Value',1):.4e}$)")
        report.append(f"- **Unweighted Modularity:** CARE = {u_mod.get('CARE_Value',0):.4f}, Random = {u_mod.get('Random_Mean',0):.4f} ($p = {u_mod.get('P_Value',1):.4e}$)")
        
        # Fingerprint Table
        report.append("\n#### Graph Fingerprints (Aggregated Layer)\n")
        report.append("| Metric | CARE Graph | Random ER Graph (Mean) |")
        report.append("|--------|------------|------------------------|")
        report.append(f"| Nodes | {N_EXPERTS} | {N_EXPERTS} |")
        report.append(f"| Edges | {int(d.get('Avg_Degree',{}).get('CARE_Value',0)*N_EXPERTS/2)} | {int(d.get('Avg_Degree',{}).get('Random_Mean',0)*N_EXPERTS/2)} |")
        report.append(f"| Connected Components | {d.get('Connected_Components',{}).get('CARE_Value',0)} | {d.get('Connected_Components',{}).get('Random_Mean',0):.1f} |")
        report.append(f"| LCC Size | {d.get('LCC_Size',{}).get('CARE_Value',0)} | {d.get('LCC_Size',{}).get('Random_Mean',0):.1f} |")
        report.append(f"| Density | {d.get('Density',{}).get('CARE_Value',0):.4f} | {d.get('Density',{}).get('Random_Mean',0):.4f} |")
        report.append(f"| Global Efficiency | {d.get('Global_Efficiency',{}).get('CARE_Value',0):.4f} | {d.get('Global_Efficiency',{}).get('Random_Mean',0):.4f} |")
        report.append(f"| Transitivity | {d.get('Transitivity',{}).get('CARE_Value',0):.4f} | {d.get('Transitivity',{}).get('Random_Mean',0):.4f} |\n")

    report.append("\n![Graph Statistics](figures/02_graph_statistics_aggregated.png)\n")

    # H2
    report.append("### H2: Functional Organization (Local Community Validation)")
    report.append("The topological communities were rigorously validated against the ground-truth Oracle KL data.\n")
    
    if "aggregated" in val_data:
        d = val_data["aggregated"]
        report.append(f"- **Within-Community Mean KL:** {d.get('Mean_Within',0):.6f}")
        report.append(f"- **Between-Community Mean KL:** {d.get('Mean_Between',0):.6f}")
        report.append(f"- **Mann-Whitney U Test:** $p = {d.get('MannWhitney_p',1):.4e}$")
        report.append(f"- **Cohen's d:** {d.get('Cohens_d',0):.4f}")
        report.append(f"- **Oracle KL Silhouette Score:** {d.get('Silhouette_Score',0):.4f}")
        
    report.append("\n![Within vs Between KL](figures/03_within_vs_between_kl_aggregated.png)\n")
    report.append("\n![Block Adjacency](figures/05_block_adjacency_aggregated.png)\n")

    report.append("### Topological Correlates (Centrality vs Merge Sensitivity)")
    report.append("We correlated graph centrality with the actual Oracle Merge Loss (Merge Sensitivity).")
    if corr_data:
        deg = corr_data.get("Degree", {})
        bet = corr_data.get("Betweenness", {})
        report.append(f"- **Degree (Spearman):** $\\rho = {deg.get('Spearman',{}).get('coef',0):.4f}$ ($p = {deg.get('Spearman',{}).get('p',1):.4e}$)")
        report.append(f"- **Degree (Kendall):** $\\tau = {deg.get('Kendall',{}).get('coef',0):.4f}$ ($p = {deg.get('Kendall',{}).get('p',1):.4e}$)")

    report.append("\n### Robustness")
    report.append("![Robustness](figures/04_community_robustness.png)\n")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report))

def main():
    set_global_seed()
    set_pub_style()
    ensure_dirs()
    print("=" * 70)
    print("PHASE 6 — REPORT GENERATION")
    print("=" * 70)
    
    for layer in LAYERS + ["aggregated"]: plot_capability_graph(layer)
    for layer in LAYERS + ["aggregated"]: plot_baseline_comparison(layer)
    for layer in LAYERS + ["aggregated"]: plot_validation(layer)
    for layer in LAYERS + ["aggregated"]: plot_community_adjacency(layer)
    plot_robustness()
    generate_markdown_report()

    print("\n" + "=" * 60)
    print("PHASE 6 — REPORT GENERATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
