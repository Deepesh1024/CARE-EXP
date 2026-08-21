"""
EXPERIMENT 6D - GEOMETRY
============================================================
Decomposes the raw functional displacement (DeltaC) and the
environmental capability vector (tau) into parallel and
perpendicular components relative to the baseline state (C_before).
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIRS, ensure_dirs

def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))

def main():
    ensure_dirs()
    print("Running Geometry Decomposition...")
    
    in_path = os.path.join(DIRS["results"], "EXP6D_PILOT_RESULTS.parquet")
    if not os.path.exists(in_path):
        in_path = os.path.join(DIRS["results"], "EXP6D_RAW_RESULTS.parquet")
    if not os.path.exists(in_path):
        print(f"Skipping geometry: results not found.")
        return
        
    df = pd.read_parquet(in_path)
    geom_results = []
    
    for idx, row in df.iterrows():
        c_before = np.array(row["C_before"])
        c_after = np.array(row["C_after"])
        delta_c = np.array(row["DeltaC"])
        tau = np.array(row["tau_actual"])
        
        mag_c = np.linalg.norm(c_before)
        c_hat = c_before / mag_c if mag_c > 0 else np.zeros(10)
            
        # Delta C Decomposition
        dc_par = np.dot(delta_c, c_hat) * c_hat
        dc_perp = delta_c - dc_par
        
        # Tau Decomposition
        tau_par = np.dot(tau, c_hat) * c_hat
        tau_perp = tau - tau_par
        
        theta_after = angle_between(c_before, c_after)
        delta_theta = theta_after # Baseline is 0
        
        tau_perp_mag = np.linalg.norm(tau_perp)
        susceptibility_ratio = tau_perp_mag / (mag_c + 1e-12)
        
        rec = row.to_dict()
        rec.update({
            "mag_C": mag_c,
            "mag_DeltaC": np.linalg.norm(delta_c),
            "mag_tau": np.linalg.norm(tau),
            "DeltaC_par": dc_par.tolist(),
            "DeltaC_perp": dc_perp.tolist(),
            "mag_DeltaC_par": np.linalg.norm(dc_par),
            "mag_DeltaC_perp": np.linalg.norm(dc_perp),
            "tau_par": tau_par.tolist(),
            "tau_perp": tau_perp.tolist(),
            "mag_tau_par": np.linalg.norm(tau_par),
            "mag_tau_perp": tau_perp_mag,
            "susceptibility_ratio": susceptibility_ratio,
            "delta_theta": delta_theta,
            "angle_c_tau": angle_between(c_before, tau),
            "angle_c_dc": angle_between(c_before, delta_c),
            "angle_tau_dc": angle_between(tau, delta_c)
        })
        geom_results.append(rec)
        
    df_geom = pd.DataFrame(geom_results)
    out_path = os.path.join(DIRS["results"], "EXP6D_GEOMETRY.parquet")
    df_geom.to_parquet(out_path)
    print(f"Saved {len(geom_results)} geometry records to {out_path}")

if __name__ == "__main__":
    main()
