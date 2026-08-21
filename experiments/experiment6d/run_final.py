import os
import sys
import subprocess
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIRS, ensure_dirs, CALIBRATION_CONFIGS

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"Executing {script_name}...")
    print(f"{'='*60}")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    cmd = [sys.executable, script_path] + sys.argv[1:]
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def generate_calibration_report():
    print("Generating EXP6D_INTERVENTION_CALIBRATION_REPORT.md...")
    out_path = os.path.join(DIRS["results"], "EXP6D_INTERVENTION_CALIBRATION_REPORT.md")
    
    # 1. Load Noise Floor
    df_noise = pd.read_parquet(os.path.join(DIRS["results"], "EXP6D_NOISE_FLOOR.parquet"))
    sigma_dc = df_noise["mag_DeltaC"].std()
    sigma_theta = df_noise["delta_theta"].std()
    
    # 2. Load Calibration Results
    df_cal = pd.read_parquet(os.path.join(DIRS["results"], "EXP6D_CALIBRATION_RESULTS.parquet"))
    
    with open(out_path, "w") as f:
        f.write("# EXPERIMENT 6D: INTERVENTION CALIBRATION REPORT\n\n")
        f.write("## 1. ZERO-UPDATE NOISE FLOOR\n")
        f.write(f"- $\\sigma(||\\Delta C||)$: {sigma_dc:.6f}\n")
        f.write(f"- $\\sigma(\\Delta \\theta)$: {sigma_theta:.6f} degrees\n\n")
        
        f.write("## 2. INTERVENTION CONFIGURATION EVALUATION\n\n")
        
        valid_configs = []
        
        for c_dict in CALIBRATION_CONFIGS:
            c_name = c_dict["name"]
            df_c = df_cal[df_cal["config_name"] == c_name]
            
            if len(df_c) == 0:
                continue
                
            mean_dc = df_c["mag_DeltaC"].mean()
            mean_theta = df_c["delta_theta"].mean()
            snr_dc = mean_dc / (sigma_dc + 1e-12)
            snr_theta = mean_theta / (sigma_theta + 1e-12)
            
            # Tau Fidelity
            err_tau = df_c["err_tau"].mean()
            cos_tau = df_c["cos_tau"].mean()
            
            # Stability
            # Calculate proportion where ||DeltaC|| > ||C_before|| / 2
            stability_violations = (df_c["mag_DeltaC"] > (df_c["mag_C_before"] / 2.0)).sum()
            
            is_valid = (snr_dc > 5.0) and (snr_theta > 5.0) and (stability_violations == 0)
            if is_valid:
                valid_configs.append(c_name)
                
            f.write(f"### CONFIG {c_name} (Steps: {c_dict['steps']}, LR: {c_dict['lr']})\n")
            f.write(f"- Mean $||\\Delta C||$: {mean_dc:.6f} (SNR: **{snr_dc:.2f}**)\n")
            f.write(f"- Mean $\\Delta \\theta$: {mean_theta:.6f} degrees (SNR: **{snr_theta:.2f}**)\n")
            f.write(f"- Mean Absolute Tau Error $||\\tau_{{target}} - \\tau_{{actual}}||$: {err_tau:.6f}\n")
            f.write(f"- Mean Cosine$(\\tau_{{target}}, \\tau_{{actual}})$: {cos_tau:.6f}\n")
            f.write(f"- Stability Exclusions ($||\\Delta C|| > ||C_{{before}}|| / 2$): {stability_violations} / {len(df_c)}\n")
            f.write(f"- Valid Criterion Passed: **{'YES' if is_valid else 'NO'}**\n\n")
            
        f.write("## 3. RECOMMENDATION\n")
        if len(valid_configs) > 0:
            recommended = valid_configs[0]
            f.write(f"**RECOMMENDED CONFIGURATION: {recommended}**\n\n")
            f.write(f"Config {recommended} is the minimum intervention strength that satisfies both the $5\\times\\sigma$ independent measurement resolution criteria for functional magnitude and directional movement, while maintaining complete numerical stability within the predeclared bounds.\n")
        else:
            f.write("**NO VALID CONFIGURATION FOUND.**\n\n")
            f.write("None of the tested configurations surpassed the instrument noise floor without violating stability boundaries. Do NOT launch the full sweep. Re-evaluate measurement sensitivity or config bounds.\n")

    print(f"Saved calibration report to {out_path}")

def generate_pilot_report():
    print("Generating EXP6D_GPU_PILOT_REPORT.md...")
    out_path = os.path.join(DIRS["results"], "EXP6D_GPU_PILOT_REPORT.md")
    
    df = pd.read_parquet(os.path.join(DIRS["results"], "EXP6D_PILOT_RESULTS.parquet"))
    
    with open(out_path, "w") as f:
        f.write("# EXPERIMENT 6D: GPU PILOT REPORT\n\n")
        f.write("This pilot verifies the feasibility, stability, and observability of the PyTorch intervention pipeline before launching the full 1,980-condition sweep.\n\n")
        
        f.write("## 1. EXPERIMENTAL CONFIGURATION\n")
        f.write(f"- **Total Pilot Conditions Executed**: {len(df)}\n")
        
        first_row = df.iloc[0]
        f.write(f"- **Target Expert Trainable Parameters**: {first_row['trainable_params']:,}\n")
        f.write(f"- **Frozen Parameters (Rest of 7B Model)**: {first_row['frozen_params']:,}\n")
        f.write("  *(Only the target expert's gate_proj, up_proj, down_proj receive gradients)*\n")
        f.write("- **Optimizer**: Plain SGD (momentum=0)\n\n")
        
        f.write("## 2. TAU TARGET VS ACTUAL COMPOSITION\n")
        f.write("Does the randomly sampled discrete batch actually match the continuous tau_target capability vector?\n\n")
        err_tau = df["err_tau"].mean()
        cos_tau = df["cos_tau"].mean()
        f.write(f"- Mean ||tau_target - tau_actual||: {err_tau:.6f}\n")
        f.write(f"- Mean Cosine Similarity(tau_target, tau_actual): {cos_tau:.6f}\n")
        if cos_tau > 0.99:
            f.write("*(SUCCESS: The sampled discrete batches flawlessly reconstruct the intended environments.)*\n\n")
        else:
            f.write("*(WARNING: High divergence in batch sampling.)*\n\n")
            
        f.write("## 3. OBSERVABILITY OF RESPONSE\n")
        f.write("Did the intervention produce a measurable change in functional state?\n\n")
        
        mean_delta_c = df["mag_DeltaC_perp"].mean() + df["mag_DeltaC_par"].mean()
        f.write(f"- Mean ||Delta C||: {mean_delta_c:.6f}\n")
        f.write(f"- Mean ||Delta C_parallel||: {df['mag_DeltaC_par'].mean():.6f}\n")
        f.write(f"- Mean ||Delta C_perpendicular||: {df['mag_DeltaC_perp'].mean():.6f}\n")
        f.write(f"- Mean Delta_theta: {df['delta_theta'].mean():.4f} degrees\n\n")
        
        if mean_delta_c < 1e-5:
            f.write("*(FAILURE: The functional response is indistinguishable from numerical noise. Increase LR or Update Steps.)*\n\n")
        else:
            f.write("*(SUCCESS: The functional displacement is clearly measurable and significantly above float32 noise floors.)*\n\n")
            
        f.write("## 4. SEED VARIANCE (REPRODUCIBILITY)\n")
        seed_var = df.groupby(["layer_idx", "expert_idx", "alpha", "target_angle_deg"])["delta_theta"].std().mean()
        f.write(f"- Mean StdDev across Seeds (Delta_theta): {seed_var:.6f} degrees\n")
        if pd.isna(seed_var):
            f.write("*(No replications run in pilot)*\n")
        elif seed_var < df['delta_theta'].mean() * 0.5:
            f.write("*(SUCCESS: The primary signal is substantially larger than random batch/SGD noise.)*\n\n")
        else:
            f.write("*(WARNING: Response is highly unstable across seeds.)*\n\n")
            
        f.write("## 5. CONCLUSION\n")
        f.write("If the above checks pass, the pipeline is structurally sound. You may now execute:\n")
        f.write("`python3 run_final.py --full`\n")

    print(f"Saved pilot report to {out_path}")

def generate_full_report():
    # Implementation deferred
    pass

def main():
    ensure_dirs()
    
    if "--calibrate" in sys.argv:
        print("\n=== RUNNING 6D GPU CALIBRATION MATRIX ===")
        run_script("intervention.py")
        generate_calibration_report()
        print("\n[SUCCESS] Calibration Matrix Completed. Check EXP6D_INTERVENTION_CALIBRATION_REPORT.md.")
    elif "--pilot" in sys.argv:
        print("\n=== RUNNING 6D GPU PILOT ===")
        run_script("intervention.py")
        generate_pilot_report()
        print("\n[SUCCESS] GPU Pilot Completed. Check EXP6D_GPU_PILOT_REPORT.md.")
    elif "--full" in sys.argv:
        print("\n=== RUNNING FINAL 6D GPU SWEEP ===")
        run_script("intervention.py")
        run_script("geometry.py")
        run_script("controls.py")
        run_script("analysis.py")
        run_script("plotting.py")
        print("\n[SUCCESS] Final 6D Intervention Completed.")
    else:
        print("\n=== GENERATING 6D TARGETS ===")
        run_script("config.py")
        run_script("controlled_tau.py")
        print("\n[SUCCESS] Phase 1 Targets generated.")
        print("Run `python3 run_final.py --calibrate` to execute the GPU calibration matrix.")

if __name__ == "__main__":
    main()
