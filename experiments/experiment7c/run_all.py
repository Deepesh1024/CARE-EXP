import os
import subprocess
import sys

def run_pipeline(scripts, exp_dir, token_count, phase_name):
    print("\n" + "=" * 70)
    print(f"STARTING EXPERIMENT 7C PIPELINE: {phase_name} ({token_count} tokens)")
    print("=" * 70)
    
    # Set the token count as an environment variable for phase 6 to pick up
    os.environ["CARE_7C_TOKENS"] = str(token_count)
    
    for script in scripts:
        script_path = os.path.join(exp_dir, script)
        print(f"\n>>> Running {script}...")
        
        # Run the script using the current Python interpreter
        result = subprocess.run([sys.executable, script_path], check=False)
        
        if result.returncode != 0:
            print(f"\n[ERROR] Pipeline aborted because {script} failed with exit code {result.returncode} during {phase_name}.")
            return False
            
    print("\n" + "=" * 70)
    print(f"{phase_name.upper()} COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return True

def main():
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts = [
        "phase6_extract_activations.py",
        "phase7_cloud_analysis.py",
        "phase8_residual_analysis.py"
    ]
    
    # 1. Run Sanity Check First
    success = run_pipeline(scripts, exp_dir, token_count=512, phase_name="Sanity Check")
    
    if not success:
        print("\n[FATAL] Sanity check failed. Aborting full extraction.")
        sys.exit(1)
        
    print("\n[INFO] Sanity check passed! Automatically proceeding to the FULL GPU extraction...\n")
    
    # 2. Run Full Extraction if Sanity Check passes
    success = run_pipeline(scripts, exp_dir, token_count=262144, phase_name="Full Extraction")
    
    if not success:
        sys.exit(1)
        
    print("\n[SUCCESS] Entire Experiment 7C completed perfectly!")

if __name__ == "__main__":
    main()
