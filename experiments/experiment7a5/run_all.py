"""
EXPERIMENT 7A - ORCHESTRATOR
==============================
Runs the full pipeline for Neuron-Level Capability Localization.
"""

import sys
from phase1_dataset import generate_partitions
from phase2_sensitivity import run_sensitivity_computation
from phase3_sampling import run_sampling
from phase4_intervention import run_interventions
from phase5_analysis import run_analysis

def main():
    print("======================================================================")
    print("EXPERIMENT 7A: NEURON-LEVEL CAPABILITY LOCALIZATION")
    print("PHASE A: CAPABILITY-CONDITIONED SENSITIVITY VALIDATION")
    print("======================================================================")
    
    try:
        print("\n--- STEP 1: DATASET PARTITIONING ---")
        generate_partitions()
        
        print("\n--- STEP 2: SENSITIVITY COMPUTATION ---")
        run_sensitivity_computation()
        
        print("\n--- STEP 3: STRATIFIED NEURON SAMPLING ---")
        run_sampling()
        
        print("\n--- STEP 4: NEURON INTERVENTION ---")
        run_interventions()
        
        print("\n--- STEP 5: STATISTICAL ANALYSIS ---")
        run_analysis()
        
        print("\n======================================================================")
        print("EXPERIMENT 7A PIPELINE COMPLETE.")
        print("See results/exp7a/analysis/statistics.json for the final Go/No-Go report.")
        print("======================================================================")
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
