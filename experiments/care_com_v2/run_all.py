import os
import sys

def main():
    print("=" * 60)
    print("CARE-COM v2.0 - FULL PIPELINE EXECUTION")
    print("=" * 60)
    
    # Add parent directory to path so imports work
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, parent_dir)
    
    try:
        from experiments.care_com_v2.driver import run_care_com_v2_experiment
        run_care_com_v2_experiment()
        print("\n[SUCCESS] CARE-COM v2.0 pipeline completed perfectly.")
    except Exception as e:
        print(f"\n[ERROR] Pipeline aborted due to error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
