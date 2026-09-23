import os
import argparse
import yaml
import subprocess
import sys

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "benchmark_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_method(method, seed=None):
    """
    Spawns a subprocess to run the requested method. 
    Using subprocess isolation prevents VRAM fragmentation when running multiple methods.
    """
    print(f"\n{'='*80}")
    if seed:
        print(f"Starting Benchmark: {method.upper()} (Seed: {seed})")
    else:
        print(f"Starting Benchmark: {method.upper()}")
    print(f"{'='*80}")
    
    # We will map method names to their respective runner scripts inside `methods/`
    script_map = {
        "random": "benchmark.methods.random_merge",
        "care_static": "benchmark.methods.care_com",
        "care_adaptive": "benchmark.methods.care_com",
        "hc_smoe": "benchmark.methods.reap_family",
        "m_smoe": "benchmark.methods.reap_family",
        "sub_moe": "benchmark.methods.reap_family",
        "reap": "benchmark.methods.reap_family",
        "ream": "benchmark.methods.ream",
        "puzzlemoe": "benchmark.methods.puzzlemoe"
    }
    
    if method not in script_map:
        print(f"Error: Unknown method '{method}'")
        sys.exit(1)
        
    cmd = [
        sys.executable, "-m", script_map[method],
        "--method", method
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
        
    # Execute the method in an isolated process
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Method {method} failed with code {result.returncode}")
        # We don't exit here to allow the benchmark to continue to other methods, as requested by the spec
    else:
        print(f"\n[SUCCESS] Method {method} completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="CARE-COM External Benchmark Orchestrator")
    parser.add_argument("--method", type=str, required=True, 
                        help="Method to run (e.g., 'all', 'care_adaptive', 'random')")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Specific random seed to use (only for random baseline)")
    
    parser.add_argument("--resume", action="store_true", 
                        help="Skip methods that have already completed successfully")
    
    args = parser.parse_args()
    config = load_config()
    
    if args.method == "all":
        methods = config["baselines"]["methods_to_run"]
    else:
        methods = [args.method]
        
    for method in methods:
        if args.resume:
            # Check if this method already completed
            traj_file = os.path.join(os.path.dirname(__file__), "..", "benchmark_results", method, "trajectory.json")
            if method == "random":
                traj_file = os.path.join(os.path.dirname(__file__), "..", "benchmark_results", "random", "random_seed_42", "trajectory.json")
                
            if os.path.exists(traj_file):
                try:
                    import json
                    with open(traj_file, "r") as f:
                        data = json.load(f)
                    if "error" in data:
                        print(f"Skipping {method} (previously recorded as error/incompatible).")
                        continue
                    elif "steps" in data and len(data["steps"]) >= len(config["compression"]["checkpoints"]):
                        print(f"Skipping {method} (already completed).")
                        continue
                except Exception:
                    pass

        if method == "random":
            seeds = [args.seed] if args.seed is not None else config["baselines"]["random_seeds"]
            for seed in seeds:
                run_method(method, seed)
        else:
            run_method(method)
            
    print("\nBenchmark Execution Complete. Run aggregation scripts for final CSVs.")

if __name__ == "__main__":
    main()
