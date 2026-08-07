"""
CARE-MoE Experiment 3A — Master Runner
========================================
Executes all phases of Experiment 3A sequentially.
"""

import sys
import subprocess

def run_phase(script_name: str):
    print(f"\n{'='*80}")
    print(f"Executing: {script_name}")
    print(f"{'='*80}")
    
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with return code {result.returncode}.")
        sys.exit(result.returncode)

def main():
    phases = [
        "phase1_graph_construction.py",
        "phase2_random_baselines.py",
        "phase3_community_detection.py",
        "phase4_validation.py",
        "phase5_robustness.py",
        "phase6_report.py"
    ]
    
    for phase in phases:
        run_phase(phase)
        
    print("\n" + "=" * 80)
    print("EXPERIMENT 3A COMPLETELY FINISHED")
    print("=" * 80)

if __name__ == "__main__":
    main()
