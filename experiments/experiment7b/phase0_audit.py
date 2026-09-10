"""
EXPERIMENT 7B - PHASE 0: PRE-EXECUTION AUDIT
============================================
Audits all external dependencies, pre-existing models, feature datasets,
and runtime execution environment before any experiment stages run.
"""

import os
import sys
import json
import joblib
from xgboost import XGBRegressor

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ensure_dirs, mark_task, is_task_completed,
    EXP1_OUTPUT_JSON, EXP5_MODEL_PATH, EXP5_SCALER_PATH,
    RESULTS_DIR, DEVICE, DTYPE, BASE_MODEL_ID, TARGET_LAYER_IDX,
    LOCAL_FEATURES
)

def run_audit():
    print("=" * 70)
    print("EXPERIMENT 7B — PHASE 0: PRE-EXECUTION AUDIT")
    print("=" * 70)
    
    ensure_dirs()
    audit_report = {
        "device": DEVICE,
        "dtype": str(DTYPE),
        "target_model": BASE_MODEL_ID,
        "target_layer": TARGET_LAYER_IDX,
        "checks": {}
    }
    
    # 1. Check Exp 1 output.json
    print(f"\n[1/4] Checking Feature Provenance Dataset: {EXP1_OUTPUT_JSON}...")
    if not os.path.exists(EXP1_OUTPUT_JSON):
        raise FileNotFoundError(f"Missing Exp 1 data at {EXP1_OUTPUT_JSON}")
    with open(EXP1_OUTPUT_JSON, "r") as f:
        exp1_data = json.load(f)
    middle_records = [
        r for r in exp1_data.get("results", [])
        if r.get("Layer") == "middle" and r.get("Seq_Len") == 512
    ]
    print(f"      Found {len(middle_records)} pairs for Layer 'middle' (S=512).")
    if len(middle_records) != 2016:
        raise ValueError(f"Expected 2016 middle layer pairs, found {len(middle_records)}.")
    audit_report["checks"]["exp1_middle_pairs"] = len(middle_records)
    
    # 2. Check CARE-COM model and scaler
    print(f"\n[2/4] Checking CARE-COM Model & Scaler...")
    if not os.path.exists(EXP5_MODEL_PATH):
        raise FileNotFoundError(f"Missing CARE-COM model at {EXP5_MODEL_PATH}")
    if not os.path.exists(EXP5_SCALER_PATH):
        raise FileNotFoundError(f"Missing CARE-COM scaler at {EXP5_SCALER_PATH}")
        
    scaler = joblib.load(EXP5_SCALER_PATH)
    xgb = XGBRegressor()
    xgb.load_model(EXP5_MODEL_PATH)
    print(f"      Scaler loaded: {scaler.__class__.__name__}")
    print(f"      XGBoost Model loaded: {xgb.n_features_in_} features expected.")
    if xgb.n_features_in_ != len(LOCAL_FEATURES):
        raise ValueError(f"Feature count mismatch: expected {len(LOCAL_FEATURES)}, got {xgb.n_features_in_}")
    audit_report["checks"]["care_com_model_loaded"] = True
    audit_report["checks"]["care_com_features"] = len(LOCAL_FEATURES)
    
    # 3. Test Sample Prediction
    print(f"\n[3/4] Testing Sample CARE-COM Prediction...")
    sample_feat = [middle_records[0].get(k, 0.0) for k in LOCAL_FEATURES]
    sample_scaled = scaler.transform([sample_feat])
    sample_pred = float(xgb.predict(sample_scaled)[0])
    sample_actual = float(middle_records[0].get("Oracle_KL", 0.0))
    print(f"      Pair E{middle_records[0]['Expert_A']}-E{middle_records[0]['Expert_B']}:")
    print(f"      Predicted Damage: {sample_pred:.6f} nats KL")
    print(f"      Historical Actual: {sample_actual:.6f} nats KL")
    audit_report["checks"]["sample_prediction"] = {
        "pred": sample_pred,
        "actual": sample_actual,
        "diff": abs(sample_pred - sample_actual)
    }
    
    # 4. Hardware & Environment Audit
    print(f"\n[4/4] Auditing Execution Environment...")
    if torch is not None:
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None (CPU Execution)"
        torch_ver = torch.__version__
    else:
        cuda_avail = False
        gpu_name = "None (Torch not installed on local development machine)"
        torch_ver = "Not installed (GPU VM execution required for Stage A/B)"
        
    print(f"      Device: {DEVICE} ({gpu_name})")
    print(f"      PyTorch Version: {torch_ver}")
    audit_report["checks"]["cuda_available"] = cuda_avail
    audit_report["checks"]["gpu_name"] = gpu_name
    audit_report["checks"]["torch_version"] = torch_ver
    
    out_audit_path = os.path.join(RESULTS_DIR, "audit_summary.json")
    with open(out_audit_path, "w") as f:
        json.dump(audit_report, f, indent=2)
    print(f"\n[Audit] Successfully completed! Report saved to {out_audit_path}")
    
    mark_task("phase0_audit", "completed", details="All dependencies and artifacts verified.")

if __name__ == "__main__":
    run_audit()
