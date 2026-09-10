"""
EXPERIMENT 7B - ORCHESTRATOR
=============================
Runs the complete sequential execution pipeline for Experiment 7B:
CARE-COM Merge Failure Analysis and Joint Functional Interaction.
"""

import sys
from phase0_audit import run_audit
from phase1_candidate_selection import run_candidate_selection
from phase2_actual_merges import run_actual_merges
from phase3_individual_interventions import run_individual_interventions
from phase4_joint_interventions import run_joint_interventions
from phase5_analysis import run_analysis

def main():
    print("=" * 75)
    print("EXPERIMENT 7B: CARE-COM MERGE FAILURE ANALYSIS & JOINT INTERACTION")
    print("=" * 75)
    
    try:
        print("\n--- STEP 0: PRE-EXECUTION AUDIT ---")
        run_audit()

        print("\n--- STEP 1: STRATIFIED CANDIDATE SELECTION ---")
        run_candidate_selection()

        print("\n--- STEP 2: STAGE A — ACTUAL EXPERT MERGES ---")
        run_actual_merges()

        print("\n--- STEP 3: STAGE B1 — INDIVIDUAL EXPERT ABLATIONS ---")
        run_individual_interventions()

        print("\n--- STEP 4: STAGE B2 — JOINT EXPERT ABLATIONS & INTERACTION ---")
        run_joint_interventions()

        print("\n--- STEP 5: STATISTICAL ANALYSIS & DECISION GATES ---")
        run_analysis()

        print("\n" + "=" * 75)
        print("EXPERIMENT 7B PIPELINE COMPLETED SUCCESSFULLY.")
        print("See results/exp7b/analysis/EXPERIMENT_7B_FINAL_REPORT.md for details.")
        print("=" * 75)

    except Exception as e:
        print(f"\n[FATAL ERROR] Experiment 7B pipeline encountered an error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
