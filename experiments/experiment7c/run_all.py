import os
import subprocess
import sys

def run_pipeline(scripts, exp_dir, phase_name):
    print("\n" + "=" * 70)
    print(f"STARTING REVISED 7C PIPELINE: {phase_name}")
    print("=" * 70)
    
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
    
    # We skip Phase 6 since we're using the existing 29GB memmap from the initial run!
    scripts = [
        "phase7_cloud_analysis.py",
        "phase8_residual_analysis.py",
        "generate_revised_report.py"
    ]
    
    success = run_pipeline(scripts, exp_dir, phase_name="Revised Mutual Support Analysis")
    
    if not success:
        sys.exit(1)
        
    print("\n[SUCCESS] Entire Revised Experiment 7C completed perfectly!")

if __name__ == "__main__":
    main()
