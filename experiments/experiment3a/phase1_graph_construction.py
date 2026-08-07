"""
CARE-MoE Experiment 3A — Phase 1: Graph Construction
======================================================
1. Predict Oracle KL for all C(64,2) expert pairs using the frozen Exp 2 surrogate.
2. Transform predicted KL into affinity.
3. Construct Mutual-kNN graphs (k=3, 5, 8) for each layer.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

from config import (
    DATA_PATH,
    FROZEN_SURROGATE_PATH,
    FROZEN_SCALER_PATH,
    GRAPHS_DIR,
    SEQ_LEN_FILTER,
    N_EXPERTS,
    LAYERS,
    K_VALUES,
    ALL_FEATURES,
    LAYER_DEPTH_MAP,
    EPSILON
)
from utils import (
    set_global_seed,
    ensure_dirs,
    load_raw_data,
    load_pickle,
    save_csv,
    save_pickle,
    save_json
)

# Reuse the descriptor logic from Exp 2
def compute_per_expert_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer in sorted(df["Layer"].unique()):
        layer_df = df[df["Layer"] == layer]
        all_experts = sorted(set(layer_df["Expert_A"].unique()) | set(layer_df["Expert_B"].unique()))
        for exp_id in all_experts:
            mask = (layer_df["Expert_A"] == exp_id) | (layer_df["Expert_B"] == exp_id)
            pairs = layer_df[mask]
            if len(pairs) == 0:
                continue
            rows.append({
                "Layer": layer,
                "Expert": exp_id,
                "Usage_Mean": pairs["Usage_Frequency"].mean(),
                "Usage_Std": pairs["Usage_Frequency"].std(),
            })
    return pd.DataFrame(rows)

def compute_usage_asymmetry(df: pd.DataFrame, expert_stats: pd.DataFrame) -> np.ndarray:
    usage_lookup = {(row["Layer"], row["Expert"]): row["Usage_Mean"] for _, row in expert_stats.iterrows()}
    values = np.zeros(len(df))
    for idx, row in df.iterrows():
        usage_a = usage_lookup.get((row["Layer"], row["Expert_A"]), 0.0)
        usage_b = usage_lookup.get((row["Layer"], row["Expert_B"]), 0.0)
        values[df.index.get_loc(idx)] = abs(usage_a - usage_b)
    return values

def compute_routing_jsd_proxy(df: pd.DataFrame) -> np.ndarray:
    rs = np.clip(df["Routing_Similarity"].values, -1.0, 1.0)
    jo = np.clip(df["Jaccard_Overlap"].values, 0.0, 1.0)
    return (1.0 - rs) * (1.0 - jo)

def compute_routing_npmi_proxy(df: pd.DataFrame, expert_stats: pd.DataFrame) -> np.ndarray:
    usage_lookup = {(row["Layer"], row["Expert"]): row["Usage_Mean"] for _, row in expert_stats.iterrows()}
    layer_means = expert_stats.groupby("Layer")["Usage_Mean"].mean().to_dict()
    values = np.zeros(len(df))
    for idx, row in df.iterrows():
        loc = df.index.get_loc(idx)
        layer = row["Layer"]
        global_mean = layer_means.get(layer, 1.0)
        p_i = usage_lookup.get((layer, row["Expert_A"]), global_mean) / max(global_mean * 3, EPSILON)
        p_j = usage_lookup.get((layer, row["Expert_B"]), global_mean) / max(global_mean * 3, EPSILON)
        p_ij = max(row["Jaccard_Overlap"] * row["Usage_Frequency"], EPSILON)
        pmi = np.log(p_ij / max(p_i * p_j, EPSILON))
        neg_log_pij = -np.log(max(p_ij, EPSILON))
        if neg_log_pij > EPSILON:
            values[loc] = pmi / neg_log_pij
        else:
            values[loc] = 0.0
    return np.clip(values, -1.0, 1.0)

def compute_specialization_diff(df: pd.DataFrame, expert_stats: pd.DataFrame) -> np.ndarray:
    spec_lookup = {(row["Layer"], row["Expert"]): 1.0 / (row["Usage_Mean"] + EPSILON) for _, row in expert_stats.iterrows()}
    values = np.zeros(len(df))
    for idx, row in df.iterrows():
        loc = df.index.get_loc(idx)
        spec_a = spec_lookup.get((row["Layer"], row["Expert_A"]), 0.0)
        spec_b = spec_lookup.get((row["Layer"], row["Expert_B"]), 0.0)
        values[loc] = abs(spec_a - spec_b)
    return values

def construct_mutual_knn_graph(df_layer: pd.DataFrame, k: int, layer_name: str):
    """Constructs a Mutual k-NN graph from the pairwise affinity data."""
    # Build full dense affinity matrix
    affinity_matrix = np.zeros((N_EXPERTS, N_EXPERTS))
    for _, row in df_layer.iterrows():
        i, j = int(row["Expert_A"]), int(row["Expert_B"])
        affinity_matrix[i, j] = row["Affinity"]
        affinity_matrix[j, i] = row["Affinity"] # symmetric

    # Find top-k for each node
    # Note: argsort sorts ascending, so we take the last k elements (excluding self)
    top_k_selections = {}
    for i in range(N_EXPERTS):
        # Set self-affinity to -inf to avoid selecting self
        aff_copy = affinity_matrix[i, :].copy()
        aff_copy[i] = -np.inf
        # Get indices of top k elements
        top_k_idx = np.argsort(aff_copy)[-k:]
        top_k_selections[i] = set(top_k_idx)
    
    # Mutual selection
    G = nx.Graph()
    G.add_nodes_from(range(N_EXPERTS))
    
    edges = []
    for i in range(N_EXPERTS):
        for j in range(i + 1, N_EXPERTS):
            if j in top_k_selections[i] and i in top_k_selections[j]:
                weight = affinity_matrix[i, j]
                G.add_edge(i, j, weight=weight)
                edges.append({"source": i, "target": j, "weight": weight})
    
    edge_df = pd.DataFrame(edges)
    
    # Convert adjacency to DataFrame
    adj_matrix = nx.to_numpy_array(G, nodelist=range(N_EXPERTS))
    adj_df = pd.DataFrame(adj_matrix, columns=[f"E{i}" for i in range(N_EXPERTS)], index=[f"E{i}" for i in range(N_EXPERTS)])
    aff_df = pd.DataFrame(affinity_matrix, columns=[f"E{i}" for i in range(N_EXPERTS)], index=[f"E{i}" for i in range(N_EXPERTS)])

    # Save artifacts
    save_csv(aff_df, os.path.join(GRAPHS_DIR, f"{layer_name}_affinity_matrix.csv"))
    save_csv(adj_df, os.path.join(GRAPHS_DIR, f"{layer_name}_k{k}_adjacency.csv"))
    if not edge_df.empty:
        save_csv(edge_df, os.path.join(GRAPHS_DIR, f"{layer_name}_k{k}_edgelist.csv"))
    save_pickle(G, os.path.join(GRAPHS_DIR, f"{layer_name}_k{k}_graph.pkl"))

    return G

def main():
    set_global_seed()
    ensure_dirs()
    print("=" * 70)
    print("PHASE 1 — GRAPH CONSTRUCTION")
    print("=" * 70)

    # 1. Load Data
    raw_df = load_raw_data()
    df = raw_df[raw_df["Seq_Len"] == SEQ_LEN_FILTER].copy()
    print(f"[Phase 1] Filtered to Seq_Len={SEQ_LEN_FILTER}: {len(df):,} rows")

    # 2. Compute Descriptors
    expert_stats = compute_per_expert_stats(df)
    df["Usage_Asymmetry"] = compute_usage_asymmetry(df, expert_stats)
    df["Routing_JSD_Proxy"] = compute_routing_jsd_proxy(df)
    df["Routing_NPMI_Proxy"] = compute_routing_npmi_proxy(df, expert_stats)
    df["Specialization_Diff"] = compute_specialization_diff(df, expert_stats)
    df["Relative_Depth"] = df["Layer"].map(LAYER_DEPTH_MAP)

    # 3. Load Frozen Surrogate & Scaler
    scaler = load_pickle(FROZEN_SCALER_PATH)
    model = load_pickle(FROZEN_SURROGATE_PATH)

    # 4. Prepare Features (Variant C)
    # Scale the 11 base features
    X_base = scaler.transform(df[ALL_FEATURES].values)
    depth = df["Relative_Depth"].values.reshape(-1, 1)
    # Add depth and interactions
    X_b = np.hstack([X_base, depth])
    interactions = X_base * depth
    X_c = np.hstack([X_b, interactions])
    
    # 5. Predict Oracle KL
    predicted_kl = model.predict(X_c)
    df["Predicted_KL"] = predicted_kl
    
    # 6. Affinity Transformation
    # affinity(i,j) = exp(-predicted_KL / median(predicted_KL))
    median_kl = df["Predicted_KL"].median()
    df["Affinity"] = np.exp(-df["Predicted_KL"] / median_kl)

    # Save the predictions dataframe for later phases (validation)
    save_pickle(df, os.path.join(GRAPHS_DIR, "predictions_df.pkl"))
    save_csv(df, os.path.join(GRAPHS_DIR, "predictions_df.csv"))
    
    # 7. Graph Construction
    for layer in LAYERS:
        layer_df = df[df["Layer"] == layer]
        print(f"\n[Phase 1] Constructing graphs for Layer: {layer}")
        for k in K_VALUES:
            G = construct_mutual_knn_graph(layer_df, k, layer)
            print(f"  k={k}: Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Also create an aggregated graph (average affinity across layers)
    print(f"\n[Phase 1] Constructing graphs for Aggregated layers")
    agg_df = df.groupby(["Expert_A", "Expert_B"], as_index=False)["Affinity"].mean()
    for k in K_VALUES:
        G = construct_mutual_knn_graph(agg_df, k, "aggregated")
        print(f"  k={k}: Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
    print("\n" + "=" * 60)
    print("PHASE 1 — GRAPH CONSTRUCTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
