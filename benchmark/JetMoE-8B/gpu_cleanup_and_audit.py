"""
GPU Cleanup, REAP Config Patcher & Memory Audit
================================================
Run this on the VM before re-launching phase12/phase13.

What this does:
  1. Audits GPU state -- shows which PIDs are holding VRAM
  2. Kills zombie GPU processes (interactive prompt for safety)
  3. Patches REAP checkpoint config.json files (moe_num_experts mismatch fix)
  4. Audits all pipeline scripts for known memory-leak patterns
  5. Optionally clears any failed/partial lm-eval output directories
"""

import os
import re
import json
import subprocess
import shutil
import sys
import time

CKPT_DIR    = "benchmark_results/JetMoE-8B/checkpoints"
RESULTS_DIR = "benchmark_results/JetMoE-8B/multicapability_results"
SCRIPTS_DIR = "benchmark/JetMoE-8B"

REAP_TARGETS = {
    "jetmoe_reap_7": 7,
    "jetmoe_reap_6": 6,
    "jetmoe_reap_4": 4,
}

# Sub-MoE checkpoints also use batched tensors and may have same mismatch
SUBMOE_TARGETS = {
    "jetmoe_submoe_7": 7,
    "jetmoe_submoe_6": 6,
    "jetmoe_submoe_4": 4,
}

DIVIDER = "=" * 70


# ------------------------------------------------------------------------------
# 1. GPU AUDIT
# ------------------------------------------------------------------------------

def audit_gpu():
    print(f"\n{DIVIDER}")
    print("  STEP 1/4 -- GPU STATE AUDIT")
    print(DIVIDER)
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            print("  OK  No active processes holding GPU memory.")
            return []

        zombie_pids = []
        print(f"  {'PID':<10} {'VRAM Used':>12}  Process Name")
        print("  " + "-" * 50)
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            pid  = parts[0]
            mem  = parts[1] if len(parts) > 1 else "?"
            name = parts[2] if len(parts) > 2 else "?"
            is_ours = "python" in name.lower() or "lm_eval" in name.lower()
            flag = "  <-- ZOMBIE?" if is_ours else ""
            print(f"  {pid:<10} {mem:>12}  {name}{flag}")
            if is_ours:
                zombie_pids.append(pid)

        return zombie_pids

    except FileNotFoundError:
        print("  [WARNING] nvidia-smi not found. Are you on the GPU VM?")
        return []
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] nvidia-smi failed: {e}")
        return []


# ------------------------------------------------------------------------------
# 2. KILL ZOMBIE PROCESSES
# ------------------------------------------------------------------------------

def kill_zombies(pids):
    print(f"\n{DIVIDER}")
    print("  STEP 2/4 -- KILL ZOMBIE GPU PROCESSES")
    print(DIVIDER)
    if not pids:
        print("  OK  No zombie Python/lm_eval processes found. GPU is clean.")
        return

    print(f"\n  Found {len(pids)} Python/lm_eval process(es) holding VRAM: {pids}")
    answer = input("  Kill ALL of them? [y/N]: ").strip().lower()
    if answer != "y":
        print("  Skipping kill. Memory will NOT be freed.")
        return

    for pid in pids:
        try:
            subprocess.run(["kill", "-9", pid], check=True)
            print(f"  OK  Killed PID {pid}")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Failed to kill PID {pid}: {e}")

    time.sleep(1)
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        remaining = result.stdout.strip()
        if remaining:
            print(f"\n  [WARNING] Some processes still alive:\n{remaining}")
        else:
            print("\n  OK  GPU memory fully released!")
    except Exception:
        pass


# ------------------------------------------------------------------------------
# 3. PATCH REAP + Sub-MoE CHECKPOINT CONFIGS
# ------------------------------------------------------------------------------

def patch_configs():
    print(f"\n{DIVIDER}")
    print("  STEP 3/4 -- PATCH CHECKPOINT CONFIGS (moe_num_experts mismatch)")
    print(DIVIDER)

    all_targets = {**REAP_TARGETS, **SUBMOE_TARGETS}

    for ckpt_name, n_experts in all_targets.items():
        config_path = os.path.join(CKPT_DIR, ckpt_name, "config.json")
        if not os.path.exists(config_path):
            print(f"  [SKIP] {ckpt_name}/config.json -- not found (checkpoint not yet generated)")
            continue

        with open(config_path, "r") as f:
            config = json.load(f)

        current = config.get("moe_num_experts", config.get("num_experts", "?"))

        if current == n_experts:
            print(f"  OK  {ckpt_name} -- already correct (moe_num_experts={n_experts})")
            continue

        patched = False
        if "moe_num_experts" in config:
            config["moe_num_experts"] = n_experts
            patched = True
        if "num_experts" in config:
            config["num_experts"] = n_experts
            patched = True

        if patched:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"  PATCHED  {ckpt_name}: {current} -> {n_experts}")
        else:
            print(f"  [WARNING] {ckpt_name}: no 'moe_num_experts' or 'num_experts' key in config!")


# ------------------------------------------------------------------------------
# 4. MEMORY LEAK AUDIT OF PIPELINE SCRIPTS
# ------------------------------------------------------------------------------

LEAK_PATTERNS = [
    # (regex, description, severity)
    # NOTE: batch_size=1 is correct; flag only 'auto'
    (r"--batch_size['",\s]+auto",
     "batch_size=auto in lm_eval -- will OOM on 8B models, use batch_size=1",
     "WARN"),
    (r"except\s+Exception\s*:\s*\n\s*pass",
     "bare 'except Exception: pass' -- silently swallows OOM errors",
     "WARN"),
    # float32 is unavoidable for numpy; only flag explicit model dtype casts
    (r"dtype\s*=\s*torch\.float32",
     "float32 model dtype -- uses 2x VRAM vs bfloat16, consider switching",
     "WARN"),
    # Correct pattern: .detach() must come BEFORE .clone()
    (r"(?<!detach\(\))\.clone\(\)",
     ".clone() without preceding .detach() -- can hold gradient graphs in memory",
     "INFO"),
    (r"device_map\s*=\s*['\"]auto['\"]",
     "device_map='auto' -- ensure del model + empty_cache() after each eval",
     "INFO"),
    (r"del\s+model|del\s+tokenizer",
     "explicit del model/tokenizer -- good practice",
     "OK"),
    (r"torch\.cuda\.empty_cache\(\)",
     "explicit empty_cache() -- good practice",
     "OK"),
    (r"torch\.bfloat16",
     "bfloat16 dtype -- good for memory efficiency",
     "OK"),
    (r"torch\.inference_mode\(\)|torch\.no_grad\(\)",
     "no_grad/inference_mode context -- good practice",
     "OK"),
]

def audit_scripts():
    print(f"\n{DIVIDER}")
    print("  STEP 4/4 -- PIPELINE SCRIPT MEMORY AUDIT")
    print(DIVIDER)

    THIS_SCRIPT = os.path.basename(__file__)  # exclude self from audit
    scripts = []
    for root, _, files in os.walk(SCRIPTS_DIR):
        for f in sorted(files):
            if f.endswith(".py") and f != THIS_SCRIPT:
                scripts.append(os.path.join(root, f))

    if not scripts:
        print(f"  [ERROR] No Python scripts found in {SCRIPTS_DIR}")
        return

    total_warns = 0
    for script in scripts:
        with open(script, "r", errors="ignore") as f:
            lines = f.read().splitlines()

        hits = []
        for pattern, desc, severity in LEAK_PATTERNS:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    hits.append((i, severity, desc, line.strip()))

        warns = [h for h in hits if h[1] == "WARN"]
        infos = [h for h in hits if h[1] == "INFO"]
        total_warns += len(warns)

        if warns or infos:
            print(f"\n  {script}")
            for lineno, severity, desc, code in hits:
                if severity == "WARN":
                    print(f"    L{lineno:<5} WARN  {desc}")
                    print(f"           > {code[:85]}")
                elif severity == "INFO":
                    print(f"    L{lineno:<5} INFO  {desc}")

    print(f"\n  -- AUDIT SUMMARY --")
    print(f"  Scanned {len(scripts)} scripts in '{SCRIPTS_DIR}'")
    if total_warns == 0:
        print("  OK  No critical memory issues found in scripts.")
    else:
        print(f"  {total_warns} WARN(s) found -- fix before running the pipeline!")


# ------------------------------------------------------------------------------
# BONUS: Clean failed/partial lm-eval output dirs
# ------------------------------------------------------------------------------

def clean_failed_lmeval():
    print(f"\n{DIVIDER}")
    print("  BONUS -- CLEAN PARTIAL/FAILED lm-eval OUTPUT DIRS")
    print(DIVIDER)

    if not os.path.exists(RESULTS_DIR):
        print(f"  {RESULTS_DIR} doesn't exist yet, nothing to clean.")
        return

    lmeval_dirs = [
        d for d in os.listdir(RESULTS_DIR)
        if d.endswith("_lmeval") and os.path.isdir(os.path.join(RESULTS_DIR, d))
    ]

    if not lmeval_dirs:
        print("  No lm-eval output directories found.")
        return

    failed, ok = [], []
    for d in sorted(lmeval_dirs):
        full = os.path.join(RESULTS_DIR, d)
        has_json = any(f.endswith(".json") for f in os.listdir(full))
        if has_json:
            ok.append(d)
        else:
            failed.append((d, full))

    print(f"\n  Complete lm-eval outputs ({len(ok)}):")
    for d in ok:
        print(f"    OK  {d}")

    if not failed:
        print("\n  OK  No failed/partial lm-eval directories found.")
        return

    print(f"\n  Partial/empty lm-eval dirs ({len(failed)}) -- pipeline will skip")
    print("  re-evaluation of these unless they are deleted:")
    for d, _ in failed:
        print(f"    PARTIAL  {d}")

    answer = input("\n  Delete partial dirs so they get re-evaluated? [y/N]: ").strip().lower()
    if answer == "y":
        for d, full in failed:
            shutil.rmtree(full)
            print(f"  Deleted  {d}")
        print("  Done! Re-run phase12 -- failed models will be re-evaluated.")
    else:
        print("  Skipping deletion.")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{DIVIDER}")
    print("  GPU CLEANUP + REAP CONFIG PATCH + MEMORY AUDIT")
    print("  Run from the root of your Experiments-V3 directory on the VM")
    print(DIVIDER)

    pids = audit_gpu()
    kill_zombies(pids)
    patch_configs()
    audit_scripts()
    clean_failed_lmeval()

    print(f"\n{DIVIDER}")
    print("  All cleanup steps complete. Now run:")
    print("      python benchmark/JetMoE-8B/phase12_evaluate_multicapability.py")
    print(DIVIDER + "\n")
