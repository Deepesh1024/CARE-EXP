import os
import subprocess
import sys

def main():
    print("=" * 60)
    print("JETMOE-8B MULTI-CAPABILITY COMPRESSION BENCHMARK PIPELINE")
    print("=" * 60)

    # Base directory for the scripts
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # The scripts to run in order
    phases = [
        {
            "name": "Phase 11: Save Checkpoints",
            "script": os.path.join(script_dir, "phase11_save_checkpoints.py"),
            "description": "Generates and saves all 13 compressed checkpoints to disk."
        },
        {
            "name": "Phase 12: Evaluate Multicapability",
            "script": os.path.join(script_dir, "phase12_evaluate_multicapability.py"),
            "description": "Evaluates checkpoints on WikiText-2, MMLU, GSM8K, and HumanEval."
        },
        {
            "name": "Phase 13: Aggregate and Plot",
            "script": os.path.join(script_dir, "phase13_aggregate_and_plot.py"),
            "description": "Aggregates JSON results, computes retention, exports CSVs, and plots."
        }
    ]

    for phase in phases:
        print(f"\n\n{'=' * 60}")
        print(f"Starting {phase['name']}")
        print(f"{phase['description']}")
        print(f"{'=' * 60}\n")
        
        try:
            # Run the script using the same Python executable
            result = subprocess.run([sys.executable, phase["script"]], check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] {phase['name']} failed with exit code {e.returncode}.")
            print("Pipeline aborted.")
            sys.exit(1)
            
    print("\n\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("All checkpoints saved, evaluated, and results aggregated.")
    print("Check benchmark_results/JetMoE-8B/multicapability_results for raw data.")
    print("Check benchmark_results/JetMoE-8B/plots for final visualizations.")
    print("=" * 60)

if __name__ == "__main__":
    main()
