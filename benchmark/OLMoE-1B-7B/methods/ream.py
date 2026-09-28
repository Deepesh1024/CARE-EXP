import os
import sys
import argparse
import traceback
import json
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add REAM to path
REAM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "moe-expert-compress", "src"))
sys.path.insert(0, REAM_PATH)

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from benchmark.run_benchmark import load_config
from benchmark.core.evaluate import prepare_wikitext_eval_batches, compute_ppl
from benchmark.core.logging import BenchmarkLogger

def run_ream_method(method, config):
    """
    Run REAM on OLMoE.
    
    Strategy:
    1. Load the model
    2. Run REAM using its high-level API
    3. Evaluate PPL on the merged model
    """
    print(f"\n[{method.upper()}] Attempting to apply method to {config['model']['name']}...")

    out_dir = os.path.join(PROJECT_ROOT, "benchmark_results", method)
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    print(f"[{method.upper()}] Loading model...")
    dtype = torch.bfloat16 if config['model']['precision'] == "bfloat16" else torch.float32
    
    # REAM mutates model in-place. We evaluate baseline first on a fresh copy, or just load and evaluate.
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=config['model']['device'],
        offload_folder="offload"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_chunks = prepare_wikitext_eval_batches(tokenizer, config)
    ppl_64 = compute_ppl(model, eval_chunks, batch_size=config['evaluation']['batch_size'])
    print(f"[{method.upper()}] Baseline PPL @ 64: {ppl_64:.4f}")

    logger = BenchmarkLogger(os.path.join(out_dir, "trajectory.json"), config)
    logger.log_ppl(64, ppl_64)

    try:
        from moe_compress.api import compress
        from moe_compress.configs import CalibrationConfig
        
        # Load calibration data config
        calib_cfg = CalibrationConfig(
            dataset_name="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
            split="train",
            max_length=512,
            batch_size=4,
            num_batches=32
        )

        for target in config['compression']['checkpoints']:
            if target >= 64:
                continue

            print(f"\n[{method.upper()}] Compressing to {target} experts per layer...")
            start_time = time.time()
            
            # Since REAM edits in place, we reload model to start from 64 experts
            del model
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                config['model']['name'],
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map=config['model']['device'],
                offload_folder="offload"
            )
            model.eval()

            # REAM expects num_kept_experts, or compression_ratio
            num_kept_experts = target
            
            # Run compression
            model = compress(
                model=model,
                method=method, # 'ream'
                tokenizer=tokenizer,
                calibration_data=calib_cfg,
                num_kept_experts=num_kept_experts,
                # REAM default saliency/group size
                group_size=16,
                saliency="freq",
            )
            
            step_time = time.time() - start_time
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

            # Evaluate
            ppl = compute_ppl(model, eval_chunks, batch_size=config['evaluation']['batch_size'])
            logger.log_ppl(target, ppl)

            logger.log_step({
                "method": method,
                "seed": None,
                "step": 64 - target,
                "experts_before": 64,
                "experts_after": target,
                "candidate_pairs": None,
                "selected_pair": None,
                "selection_score": None,
                "capability_distance": None,
                "marginal_kl": None,
                "cumulative_kl": None,
                "wall_time_sec": float(step_time),
                "peak_memory_mb": float(peak_mem)
            })

            print(f"[{method.upper()}] PPL @ {target}: {ppl:.4f} (Time: {step_time:.2f}s)")

        print(f"\n[{method.upper()}] Completed successfully!")
        return True

    except Exception as e:
        error_msg = f"Runtime error running {method}: {e}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        with open(os.path.join(out_dir, "error.log"), "w") as f:
            f.write(error_msg + "\n" + traceback.format_exc())
        with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
            json.dump({"error": str(e), "message": error_msg}, f, indent=2)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()

    config = load_config()
    run_ream_method(args.method, config)
