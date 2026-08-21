"""
EXPERIMENT 6B — TASK 18: PIPELINE ORCHESTRATOR & FINAL REPORT
============================================================================
Orchestrates all phases of Experiment 6B:
  Phase 1: Telemetry Extraction
  Phase 2: Functional Alignment
  Phase 3: Fine Windows
  Phase 4: Predictive Models
  Phase 5: Advanced Analysis
  
After successful execution, compiles all generated markdown reports into 
the final EXP6B_FINAL_REPORT.md (TASK 18).
"""

import os
import sys
import subprocess
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, ensure_dirs


def run_script(script_name):
    """Run a python script and return success."""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    print(f"\n{'#' * 80}")
    print(f"RUNNING: {script_name}")
    print(f"{'#' * 80}\n")
    
    # We use subprocess to isolate memory and ensure garbage collection between phases
    result = subprocess.run([sys.executable, script_path])
    
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with exit code {result.returncode}")
        return False
    return True


def compile_final_report():
    """TASK 18: Compile EXP6B_FINAL_REPORT.md from all sub-reports."""
    print("\n" + "=" * 70)
    print("TASK 18: FINAL REPORT GENERATION")
    print("=" * 70)
    
    report_parts = [
        "data_audit.md",
        "functional_alignment_report.md",
        "checkpoint_trajectory_analysis.md",
        "exposure_analysis.md",
        "empirical_law_analysis.md",
    ]
    
    final_md = [
        "# EXPERIMENT 6B — FINAL REPORT\n",
        f"**Compiled:** {datetime.datetime.now().isoformat()}\n\n",
        "## Token/Routing Environment → Functional Evolution of MoE Experts\n\n",
        "This report aggregates the findings from all phases of Experiment 6B, "
        "tracing how the routing environment (exposure) dictates the movement "
        "of experts through functional space across the training lifecycle.\n\n"
    ]
    
    for part in report_parts:
        part_path = os.path.join(RESULTS_DIR, part)
        if os.path.exists(part_path):
            with open(part_path, "r") as f:
                content = f.read()
                # Demote headers by one level so they nest correctly
                content = content.replace("\n# ", "\n## ")
                final_md.append(f"\n---\n{content}\n")
        else:
            final_md.append(f"\n---\n## Missing Section: {part}\n\n")
            
    final_path = os.path.join(RESULTS_DIR, "EXP6B_FINAL_REPORT.md")
    with open(final_path, "w") as f:
        f.write("".join(final_md))
        
    print(f"[TASK 18] Final report saved to {final_path}")


def ensure_calibration_data():
    """Ensure the calibration dataset exists, generate it if not."""
    from config import CALIBRATION_CACHE_FILE
    if not os.path.exists(CALIBRATION_CACHE_FILE):
        print("\n" + "=" * 70)
        print("PRE-REQUISITE: GENERATING CALIBRATION DATASET")
        print("=" * 70)
        
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "..", "experiment3c", "phase1_calibration.py"
        )
        
        result = subprocess.run([sys.executable, script_path])
        if result.returncode != 0:
            print(f"\n[ERROR] Calibration generation failed.")
            sys.exit(1)
        print("[SUCCESS] Calibration dataset generated.")


def main():
    ensure_dirs()
    ensure_calibration_data()
    
    phases = [
        "phase1_telemetry.py",
        "phase2_alignment.py",
        "phase3_fine_windows.py",
        "phase4_models.py",
        "phase5_analysis.py",
    ]
    
    for phase in phases:
        success = run_script(phase)
        if not success:
            print(f"\nPipeline halted at {phase}.")
            sys.exit(1)
            
    compile_final_report()
    print("\n[SUCCESS] Experiment 6B Pipeline Complete.")


if __name__ == "__main__":
    main()
