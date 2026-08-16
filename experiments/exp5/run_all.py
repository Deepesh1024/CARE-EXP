import os
import subprocess
import sys
import json

# Define absolute paths dynamically based on this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Change working directory to the project root (Experiments-V3)
# This ensures all relative paths to results/ and models resolve correctly.
os.chdir(PROJECT_ROOT)

def run_command(command, description):
    print(f"\n{'='*80}")
    print(f"🚀 STARTING: {description}")
    print(f"💻 COMMAND: {command}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with return code {result.returncode}")
        sys.exit(1)
    else:
        print(f"\n✅ SUCCESS: {description} completed successfully!\n")

def main():
    os.makedirs("results/exp5", exist_ok=True)
    
    print("Welcome to Experiment 5 Automated Pipeline")
    print("All statistics and metrics will be saved into the 'results/exp5/' directory.")
    print("Starting execution...\n")

    script_evaluate = os.path.join(SCRIPT_DIR, "evaluate_noise_floor.py")
    script_train = os.path.join(SCRIPT_DIR, "train_care_predictors.py")
    script_run = os.path.join(SCRIPT_DIR, "run_compression_trajectories.py")

    # Step 1: Evaluate Noise Floor (Original 64-expert model)
    if not os.path.exists("results/exp5/evaluation_noise_floor.json"):
        run_command(f"python3 {script_evaluate}", "Evaluate Noise Floor")
    else:
        print("⏭️  SKIPPING: Noise Floor (already completed)")

    # Step 2: Train the CARE_COM Predictor
    if not os.path.exists("results/exp5/care_com_model.json"):
        run_command(f"python3 {script_train}", "Train CARE_COM Predictor")
    else:
        print("⏭️  SKIPPING: CARE_COM Predictor (already trained)")

    def check_trajectory_done(strategy, mode):
        path = f"results/exp5/{strategy}_{mode}/metrics.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    data = json.load(f)
                    if any(d.get("level") == 16 for d in data):
                        return True
                except:
                    pass
        return False

    # Step 3: Run CARE_COM (One-Shot) Compression
    if not check_trajectory_done("CARE_COM", "one_shot"):
        run_command(f"python3 {script_run} --strategy CARE_COM --mode one_shot", "CARE_COM Compression Trajectory (One-Shot)")
    else:
        print("⏭️  SKIPPING: CARE_COM Trajectory (already completed)")

    # Step 4: Run Parameter Similarity (Iterative) Baseline Compression
    if not check_trajectory_done("Parameter", "iterative"):
        run_command(f"python3 {script_run} --strategy Parameter --mode iterative", "Parameter Similarity Compression Trajectory (Iterative)")
    else:
        print("⏭️  SKIPPING: Parameter Trajectory (already completed)")

    # Step 5: Run Random (One-Shot) Baseline Compression
    if not check_trajectory_done("Random", "one_shot"):
        run_command(f"python3 {script_run} --strategy Random --mode one_shot", "Random Compression Trajectory (One-Shot)")
    else:
        print("⏭️  SKIPPING: Random Trajectory (already completed)")

    print(f"\n{'='*80}")
    print("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY! 🎉")
    print("You can find all the generated metrics, JSON files, and models in: results/exp5/")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
