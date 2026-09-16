import os
import subprocess
import sys

def main():
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts = [
        "phase6_extract_activations.py",
        "phase7_cloud_analysis.py",
        "phase8_residual_analysis.py"
    ]
    
    print("=" * 70)
    print("STARTING EXPERIMENT 7C PIPELINE")
    print("=" * 70)
    
    for script in scripts:
        script_path = os.path.join(exp_dir, script)
        print(f"\n>>> Running {script}...")
        
        # Run the script using the current Python interpreter
        result = subprocess.run([sys.executable, script_path], check=False)
        
        if result.returncode != 0:
            print(f"\n[ERROR] Pipeline aborted because {script} failed with exit code {result.returncode}.")
            sys.exit(result.returncode)
            
    print("\n" + "=" * 70)
    print("EXPERIMENT 7C PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
