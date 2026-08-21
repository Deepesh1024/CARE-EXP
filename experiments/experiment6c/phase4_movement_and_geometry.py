"""
EXPERIMENT 6C - PHASE 4: MOVEMENT AND PAIRWISE GEOMETRY
============================================================
Calculates functional displacement DeltaC_i, velocity, 
10D angular directions, 2D angles for visualization, and pairwise geometry.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIRS, TRANSITIONS, NUM_LAYERS, NUM_EXPERTS, ensure_dirs

STEP_COUNTS = {
    "checkpoint_10": 120000,
    "checkpoint_40": 490000,
    "checkpoint_70": 795000,
    "checkpoint_100": 1220000
}

def circular_difference(angle_1, angle_2):
    diff = angle_1 - angle_2
    return (diff + np.pi) % (2 * np.pi) - np.pi

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    val = np.dot(u, v) / (norm_u * norm_v)
    return np.clip(val, -1.0, 1.0)

def process_expert_movement():
    print("Processing expert movement (DeltaC_i) and directional angles...")
    df_c = pd.read_parquet(os.path.join(DIRS["expert_vectors"], "EXP6C_EXPERT_CAPABILITY_VECTORS.parquet"))
    df_env = pd.read_parquet(os.path.join(DIRS["routing"], "EXP6C_ROUTING_ENVIRONMENT.parquet"))
    
    movement_records = []
    
    for t_idx, (ckpt_a, ckpt_b) in enumerate(TRANSITIONS):
        dt = STEP_COUNTS[ckpt_b] - STEP_COUNTS[ckpt_a]
        
        for l in range(NUM_LAYERS):
            for e in range(NUM_EXPERTS):
                # C_i at t and t+1
                row_a = df_c[(df_c["checkpoint"] == ckpt_a) & (df_c["layer_idx"] == l) & (df_c["expert_idx"] == e)].iloc[0]
                row_b = df_c[(df_c["checkpoint"] == ckpt_b) & (df_c["layer_idx"] == l) & (df_c["expert_idx"] == e)].iloc[0]
                
                C_i_t = row_a["C_raw"]
                C_i_t1 = row_b["C_raw"]
                DeltaC_i = C_i_t1 - C_i_t
                mag_DeltaC_i = np.linalg.norm(DeltaC_i)
                mag_C_t = np.linalg.norm(C_i_t)
                mag_C_t1 = np.linalg.norm(C_i_t1)
                
                # 10D Angular Displacement
                cos_10d = cosine_similarity(C_i_t, C_i_t1)
                theta_10d = np.arccos(cos_10d)
                
                # tau_i
                env_a = df_env[(df_env["checkpoint"] == ckpt_a) & (df_env["layer_idx"] == l) & (df_env["expert_idx"] == e)].iloc[0]
                tau_i = env_a["tau_weighted"]
                mag_tau = np.linalg.norm(tau_i)
                
                # Decompositions (Parallel/Perpendicular)
                if mag_C_t > 0:
                    c_hat = C_i_t / mag_C_t
                    dc_par_mag = np.dot(DeltaC_i, c_hat)
                    dc_par = dc_par_mag * c_hat
                    dc_perp = DeltaC_i - dc_par
                    mag_dc_perp = np.linalg.norm(dc_perp)
                    
                    interaction = C_i_t * tau_i
                    i_par_mag = np.dot(interaction, c_hat)
                    i_par = i_par_mag * c_hat
                    i_perp = interaction - i_par
                    mag_i_perp = np.linalg.norm(i_perp)
                else:
                    dc_par = np.zeros(10)
                    dc_perp = np.zeros(10)
                    dc_par_mag = 0.0
                    mag_dc_perp = 0.0
                    interaction = np.zeros(10)
                    i_par = np.zeros(10)
                    i_perp = np.zeros(10)
                    i_par_mag = 0.0
                    mag_i_perp = 0.0
                
                # 2D Angles for interpretability (e.g., picking the two axes with highest variance in this layer)
                # For simplicity in this general framework, we just take axes 0 and 1, 
                # but these are specifically marked as interpretability-only metrics.
                theta_C_2d = np.arctan2(C_i_t[1], C_i_t[0])
                theta_tau_2d = np.arctan2(tau_i[1], tau_i[0])
                theta_C_next_2d = np.arctan2(C_i_t1[1], C_i_t1[0])
                Delta_theta_C_2d = circular_difference(theta_C_next_2d, theta_C_2d)
                Delta_theta_tau_C_2d = circular_difference(theta_tau_2d, theta_C_2d)
                
                movement_records.append({
                    "transition": f"{ckpt_a}->{ckpt_b}",
                    "layer_idx": int(l),
                    "expert_idx": int(e),
                    "C_i_t": C_i_t.tolist(),
                    "C_i_t1": C_i_t1.tolist(),
                    "tau_i": tau_i.tolist(),
                    "DeltaC_i": DeltaC_i.tolist(),
                    "DeltaC_par": dc_par.tolist(),
                    "DeltaC_perp": dc_perp.tolist(),
                    "I_i": interaction.tolist(),
                    "I_par": i_par.tolist(),
                    "I_perp": i_perp.tolist(),
                    "mag_C": float(mag_C_t),
                    "mag_DeltaC": float(mag_DeltaC_i),
                    "mag_DeltaC_par": float(np.abs(dc_par_mag)),
                    "mag_DeltaC_perp": float(mag_dc_perp),
                    "mag_tau": float(mag_tau),
                    "theta_10D": float(theta_10d),
                    "theta_C_2d": float(theta_C_2d),
                    "theta_tau_2d": float(theta_tau_2d),
                    "Delta_theta_C_2d": float(Delta_theta_C_2d),
                    "Delta_theta_tau_C_2d": float(Delta_theta_tau_C_2d)
                })
                
    df_mov = pd.DataFrame(movement_records)
    out_mov = os.path.join(DIRS["trajectories"], "EXP6C_EXPERT_MOVEMENT.parquet")
    df_mov.to_parquet(out_mov, engine="pyarrow")
    print(f"Saved {out_mov}")
    return df_mov

def process_pairwise_geometry(df_mov):
    print("Processing pairwise geometry (DeltaD_ij, radial, tangential)...")
    pair_records = []
    
    transitions = df_mov["transition"].unique()
    
    for trans in transitions:
        for l in range(NUM_LAYERS):
            layer_data = df_mov[(df_mov["transition"] == trans) & (df_mov["layer_idx"] == l)]
            
            experts = np.arange(NUM_EXPERTS)
            C_t = np.array(layer_data["C_i_t"].tolist())
            C_t1 = np.array(layer_data["C_i_t1"].tolist())
            DeltaC = np.array(layer_data["DeltaC_i"].tolist())
            tau = np.array(layer_data["tau_i"].tolist())
            
            for i in experts:
                for j in range(i + 1, NUM_EXPERTS):
                    C_i = C_t[i]
                    C_j = C_t[j]
                    
                    r_ij = C_j - C_i
                    D_ij = np.linalg.norm(r_ij)
                    
                    C_i1 = C_t1[i]
                    C_j1 = C_t1[j]
                    D_ij1 = np.linalg.norm(C_j1 - C_i1)
                    
                    DeltaD_ij = D_ij1 - D_ij
                    task_overlap = cosine_similarity(tau[i], tau[j])
                    
                    pair_records.append({
                        "transition": trans,
                        "layer_idx": l,
                        "expert_i": i,
                        "expert_j": j,
                        "D_ij_t": D_ij,
                        "D_ij_t1": D_ij1,
                        "DeltaD_ij": DeltaD_ij,
                        "task_overlap": task_overlap
                    })
                    
    df_pair = pd.DataFrame(pair_records)
    out_pair = os.path.join(DIRS["pair_metrics"], "EXP6C_PAIR_METRICS.parquet")
    df_pair.to_parquet(out_pair, engine="pyarrow")
    print(f"Saved {out_pair}")

def main():
    ensure_dirs()
    df_mov = process_expert_movement()
    process_pairwise_geometry(df_mov)

if __name__ == "__main__":
    main()
