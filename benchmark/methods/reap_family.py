import os
import argparse
import traceback
import json
import torch
import sys

from benchmark.run_benchmark import load_config
from transformers import AutoModelForCausalLM

def run_reap_method(method, config):
    print(f"[{method.upper()}] Attempting to apply method to {config['model']['name']}...")
    
    # Try to import REAP
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "reap", "src")))
        from reap.model_util import MODEL_ATTRS
    except ImportError as e:
        print(f"[{method.upper()}] Failed to import REAP repository: {e}")
        sys.exit(1)
        
    # Check if OLMoE is natively supported in REAP
    model_type = "OLMoEForCausalLM" # HF class name for OLMoE
    if model_type not in MODEL_ATTRS:
        error_msg = (
            f"Architecture Incompatibility: REAP repository (which provides {method}) "
            f"does not support {model_type} out of the box. "
            f"Available models in MODEL_ATTRS: {list(MODEL_ATTRS.keys())}."
        )
        print(f"\n[ERROR] {error_msg}")
        
        # Log failure as required by spec
        out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "benchmark_results", method)
        os.makedirs(out_dir, exist_ok=True)
        
        with open(os.path.join(out_dir, "error.log"), "w") as f:
            f.write(error_msg + "\n\n")
            f.write(traceback.format_exc())
            
        with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
            json.dump({"error": "not reproduced on OLMoE", "message": error_msg}, f, indent=2)
            
        return False
        
    # If somehow supported, we would execute the REAP logic here...
    print(f"[{method.upper()}] Method natively supports OLMoE! (Proceeding...)")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()
    
    config = load_config()
    run_reap_method(args.method, config)
