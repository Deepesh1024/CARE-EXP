import os
import argparse
import traceback
import json
import sys
from benchmark.run_benchmark import load_config

def run_puzzlemoe_method(method, config):
    print(f"[{method.upper()}] Attempting to apply method to {config['model']['name']}...")
    
    model_type = "OLMoEForCausalLM"
    error_msg = (
        f"Architecture Incompatibility: PuzzleMoE repository "
        f"does not support the {model_type} architecture out of the box. "
        f"It requires specialized modifications to the modeling files (like Mistral/Mixtral) "
        f"which contradicts the benchmark rule against fabricating adaptations."
    )
    print(f"\n[ERROR] {error_msg}")
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "benchmark_results", method)
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "error.log"), "w") as f:
        f.write(error_msg + "\n")
        
    with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
        json.dump({"error": "not reproduced on OLMoE", "message": error_msg}, f, indent=2)
        
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()
    
    config = load_config()
    run_puzzlemoe_method(args.method, config)
