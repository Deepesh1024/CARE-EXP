"""
CARE-MoE Experiment 3A — Phase 7: Community Characterization
======================================================
1. Profile each community (size, density, conductance, KL, frequencies).
2. Compute 6 centrality metrics (Degree, Weighted Degree, Betweenness,
   Closeness, Eigenvector, PageRank).
3. Correlate centralities with Merge Sensitivity (Oracle & Predicted KL),
   Routing Similarity, and Usage Frequency.
4. Output scatter plots with regression lines.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau

from config import (
    GRAPHS_DIR,
    COMMUNITIES_DIR,
    VALIDATION_DIR,
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
    save_csv,
    save_json
)

def get_node_aggregates(layer_df, expert_id):
    """Aggregate pairwise properties to node-level properties."""
    mask = (layer_df["Expert_A"] == expert_id) | (layer_df["Expert_B"] == expert_id)
    subset = layer_df[mask]
    if subset.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    avg_oracle_kl = subset["Oracle_KL"].mean()
    avg_pred_kl = subset["Predicted_KL"].mean()
    avg_usage = subset["Usage_Frequency"].mean()
    avg_routing = subset["Routing_Similarity"].mean()
    avg_spec = subset["Jaccard_Overlap"].mean() # Proxy for specialization
    
    return avg_oracle_kl, avg_pred_kl, avg_usage, avg_routing, avg_spec

def compute_conductance(G, nodes, comm_map, c):
    """Computes conductance of a community (bridge edges / total volume)."""
    if len(nodes) == 0: return 0.0
    
    internal_volume = 0
    bridge_edges = 0
    
    for n in nodes:
        internal_volume += G.degree(n)
        for neighbor in G.neighbors(n):
            if comm_map.get(neighbor) != c:
                bridge_edges += 1
                
    if internal_volume == 0: return 0.0
    return bridge_edges / internal_volume

def plot_scatter_regression(x, y, xlabel, ylabel, title, layer, metric_name):
    """Generates a scatter plot with regression line and CI."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    
    spearman_c, spearman_p = spearmanr(x, y)
    ax.text(0.05, 0.95, f"Spearman $\\rho$ = {spearman_c:.3f}\n$p$ = {spearman_p:.2e}", 
            transform=ax.transAxes, ha='left', va='top', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    save_fig(fig, f"06_{layer}_scatter_{xlabel.replace(' ', '')}_vs_{ylabel.replace(' ', '')}")

def main():
    set_global_seed()
    set_pub_style()
    ensure_dirs()
    print("=" * 70)
    print("PHASE 7 — COMMUNITY CHARACTERIZATION & TOPOLOGY")
    print("=" * 70)

    comm_path = os.path.join(COMMUNITIES_DIR, "community_assignments.csv")
    assignments = pd.read_csv(comm_path)
    assignments_k = assignments[assignments["k"] == K_PRIMARY]

    pred_path = os.path.join(GRAPHS_DIR, "predictions_df.pkl")
    df = load_pickle(pred_path)

    layers = LAYERS + ["aggregated"]

    for layer in layers:
        print(f"\n[Phase 7] Analyzing {layer} layer (k={K_PRIMARY})...")
        
        layer_assign = assignments_k[assignments_k["Layer"] == layer]
        if layer_assign.empty: continue
            
        comm_map = dict(zip(layer_assign["Expert"], layer_assign["Louvain_Community"]))
        
        graph_path = os.path.join(GRAPHS_DIR, f"{layer}_k{K_PRIMARY}_graph.pkl")
        G = load_pickle(graph_path)
        
        if layer == "aggregated":
            layer_df = df.groupby(["Expert_A", "Expert_B"], as_index=False)[["Oracle_KL", "Predicted_KL", "Usage_Frequency", "Routing_Similarity", "Jaccard_Overlap"]].mean()
        else:
            layer_df = df[df["Layer"] == layer]
            
        # Extract node properties
        node_props = {i: get_node_aggregates(layer_df, i) for i in range(N_EXPERTS)}
        
        # Compute Centralities
        deg_cen = nx.degree_centrality(G)
        wdeg_cen = {n: sum(d['weight'] for u, v, d in G.edges(n, data=True)) for n in G.nodes()}
        # Normalize weighted degree
        max_wdeg = max(wdeg_cen.values()) if wdeg_cen else 1
        if max_wdeg == 0: max_wdeg = 1
        wdeg_cen = {k: v / max_wdeg for k, v in wdeg_cen.items()}
        
        bet_cen = nx.betweenness_centrality(G)
        clo_cen = nx.closeness_centrality(G)
        try:
            eig_cen = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            eig_cen = {n: 0.0 for n in G.nodes()}
        try:
            pr_cen = nx.pagerank(G)
        except:
            pr_cen = {n: 0.0 for n in G.nodes()}
            
        # 1. Community Profiling
        communities = set(comm_map.values())
        profiles = []
        
        for c in communities:
            if c < 0: continue
            
            nodes = [n for n, comm in comm_map.items() if comm == c]
            subG = G.subgraph(nodes)
            
            internal_edges = subG.number_of_edges()
            possible_internal = len(nodes) * (len(nodes) - 1) / 2
            internal_density = internal_edges / possible_internal if possible_internal > 0 else 0
            
            bridge_edges = sum(1 for n in nodes for neighbor in G.neighbors(n) if comm_map.get(neighbor) != c)
            conductance = compute_conductance(G, nodes, comm_map, c)
            
            if len(nodes) > 0:
                hub = max(nodes, key=lambda x: deg_cen.get(x, 0))
                boundary = max(nodes, key=lambda x: bet_cen.get(x, 0))
            else:
                hub = -1; boundary = -1
                
            kl_vals = []
            pred_vals = []
            usage_vals = []
            routing_vals = []
            spec_vals = []
            for n in nodes:
                kl, pred, usage, routing, spec = node_props[n]
                kl_vals.append(kl)
                pred_vals.append(pred)
                usage_vals.append(usage)
                routing_vals.append(routing)
                spec_vals.append(spec)
            
            profiles.append({
                "Layer": layer,
                "Community": c,
                "Size": len(nodes),
                "Avg_Oracle_KL": np.mean(kl_vals) if kl_vals else 0,
                "Avg_Predicted_KL": np.mean(pred_vals) if pred_vals else 0,
                "Internal_Density": internal_density,
                "Conductance": conductance,
                "Bridge_Edges": bridge_edges,
                "Avg_Routing_Freq": np.mean(routing_vals) if routing_vals else 0,
                "Avg_Usage_Freq": np.mean(usage_vals) if usage_vals else 0,
                "Avg_Specialization": np.mean(spec_vals) if spec_vals else 0,
                "Hub_Expert": hub,
                "Boundary_Expert": boundary
            })
            
        profiles_df = pd.DataFrame(profiles)
        save_csv(profiles_df, os.path.join(VALIDATION_DIR, f"{layer}_community_profiles.csv"))
        if layer == "aggregated":
            save_csv(profiles_df, os.path.join(VALIDATION_DIR, "community_summary.csv"))
        
        # 2. Centrality Correlates
        metrics = {
            "Degree": [deg_cen.get(i, 0) for i in range(N_EXPERTS)],
            "Weighted_Degree": [wdeg_cen.get(i, 0) for i in range(N_EXPERTS)],
            "Betweenness": [bet_cen.get(i, 0) for i in range(N_EXPERTS)],
            "Closeness": [clo_cen.get(i, 0) for i in range(N_EXPERTS)],
            "Eigenvector": [eig_cen.get(i, 0) for i in range(N_EXPERTS)],
            "PageRank": [pr_cen.get(i, 0) for i in range(N_EXPERTS)]
        }
        
        targets = {
            "Oracle_KL": [node_props[i][0] for i in range(N_EXPERTS)],
            "Predicted_KL": [node_props[i][1] for i in range(N_EXPERTS)],
            "Usage_Frequency": [node_props[i][2] for i in range(N_EXPERTS)],
            "Routing_Similarity": [node_props[i][3] for i in range(N_EXPERTS)],
            "Specialization": [node_props[i][4] for i in range(N_EXPERTS)]
        }
        
        corrs = {}
        for c_name, c_vals in metrics.items():
            corrs[c_name] = {}
            for t_name, t_vals in targets.items():
                p_c, p_p = pearsonr(c_vals, t_vals)
                s_c, s_p = spearmanr(c_vals, t_vals)
                k_c, k_p = kendalltau(c_vals, t_vals)
                
                corrs[c_name][t_name] = {
                    "Pearson": {"coef": float(p_c), "p": float(p_p)},
                    "Spearman": {"coef": float(s_c), "p": float(s_p)},
                    "Kendall": {"coef": float(k_c), "p": float(k_p)}
                }
                
                # Plot for key combinations on aggregated layer
                if layer == "aggregated" and c_name in ["Degree", "Betweenness"] and t_name in ["Oracle_KL", "Usage_Frequency"]:
                    plot_scatter_regression(c_vals, t_vals, c_name, t_name, f"{c_name} vs {t_name}", layer, c_name)
                    
        save_json(corrs, os.path.join(VALIDATION_DIR, f"{layer}_centrality_correlations.json"))

    print("\n" + "=" * 60)
    print("PHASE 7 — CHARACTERIZATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
