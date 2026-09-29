import subprocess
import sys
import os

def run_script(script_path):
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {script_path}")
    print(f"{'='*60}\n")
    
    try:
        # We use sys.executable to ensure it runs with the exact same python environment
        result = subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: {script_path} failed with exit code {e.returncode}.")
        print("Halting the pipeline.")
        sys.exit(1)
        
    print(f"\n✅ COMPLETED: {script_path}\n")

def main():
    # Ensure we are running from the root of the repo (or adjust paths dynamically)
    base_dir = "benchmark/JetMoE-8B"
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' not found.")
        print("Please run this script from the root of the CARE-EXP repository.")
        sys.exit(1)

    scripts_to_run = [
        f"{base_dir}/phase1_2_inspect.py",
        f"{base_dir}/phase3_4_routing.py",
        f"{base_dir}/phase5_baseline.py",
        f"{base_dir}/phase6_7_ablation_merge.py",
        f"{base_dir}/phase8_benchmark.py",
        f"{base_dir}/phase9_plot.py"
    ]

    print("Starting JetMoE-8B Full Inspection and Benchmarking Pipeline...")
    
    for script in scripts_to_run:
        run_script(script)
        
    print("🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
    print(f"Please check the JSON outputs and the generated compression_plot.png in 'benchmark_results/JetMoE-8B/'.")

if __name__ == "__main__":
    main()
