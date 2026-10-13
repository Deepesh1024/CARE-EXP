import os
import sys
import time
import copy
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from hc_smoe_engine import apply_hcsmoe_jetmoe_layer
from mc_smoe_engine import apply_mcsmoe_jetmoe_layer
from evaluate_ppl import evaluate_wikitext_ppl

MODEL_ID = "jetmoe/jetmoe-8b"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "jetmoe")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

JETMOE_TARGETS = [7, 6, 4]

# Reference results from prior experiments (do not recompute)
REFERENCE_RESULTS = {
    "Original": {"8": 7.1258},
    "CARE-Adaptive": {"7": 10.17, "6": 15.69, "4": 87.19},
    "REAP": {"7": 8.10, "6": 9.81, "4": 17.48}
}

def validate_tensor_invariants(model, target_k: int) -> Dict[str, bool]:
    shapes_ok = True
    no_nan_or_inf = True
    
    for l_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        in_w = mlp.input_linear.weight.data
        out_w = mlp.output_linear.weight.data
        r_w = mlp.router.layer.weight.data
        
        if in_w.shape[0] != target_k or out_w.shape[0] != target_k or r_w.shape[0] != target_k:
            shapes_ok = False
            print(f"  [Validation Error] Layer {l_idx} shapes: in={in_w.shape}, out={out_w.shape}, r={r_w.shape} != target {target_k}")
            
        if torch.isnan(in_w).any() or torch.isinf(in_w).any() or \
           torch.isnan(out_w).any() or torch.isinf(out_w).any() or \
           torch.isnan(r_w).any() or torch.isinf(r_w).any():
            no_nan_or_inf = False
            print(f"  [Validation Error] NaN or Inf found in layer {l_idx}")
            
    return {"shapes_ok": shapes_ok, "numerical_integrity": no_nan_or_inf}

def run_compatibility_test(model, tokenizer) -> bool:
    print("\n--- [Phase 2.1] Running JetMoE Compatibility Test ---", flush=True)
    try:
        sample_input = tokenizer("Hello, Mixture of Experts evaluation.", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**sample_input)
        assert not torch.isnan(out.logits).any(), "NaN found in baseline logits"
        print("  [Pass] Forward pass succeeded with clean numerical logits.", flush=True)
        return True
    except Exception as e:
        print(f"  [Fail] Compatibility test failed: {e}", flush=True)
        return False

def benchmark_jetmoe_method(method_name: str, apply_fn, model_pristine, tokenizer):
    print(f"\n{'='*70}\nSTARTING JETMOE BENCHMARK: {method_name}\n{'='*70}", flush=True)
    results = []
    runtime_records = []
    
    for target_k in JETMOE_TARGETS:
        print(f"\n>>> Compressing JetMoE-8B to {target_k} experts via {method_name}...", flush=True)
        t_start = time.time()
        torch.cuda.reset_peak_memory_stats()
        
        # Deepcopy pristine model to isolate checkpoints
        model = copy.deepcopy(model_pristine)
        
        # Apply layer-by-layer compression
        for l_idx, layer in enumerate(model.model.layers):
            apply_fn(layer.mlp, target_k)
            
        compression_time = time.time() - t_start
        peak_comp_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  [Compression Complete] Time: {compression_time:.2f}s | Peak VRAM: {peak_comp_vram:.2f} GB", flush=True)
        
        # Invariant validation gate
        val = validate_tensor_invariants(model, target_k)
        if not (val["shapes_ok"] and val["numerical_integrity"]):
            print(f"  [Gate Failed] Checkpoint at {target_k} experts failed invariant checks! Halting further targets.", flush=True)
            results.append({
                "model": "JetMoE-8B",
                "method": method_name,
                "experts": target_k,
                "perplexity": "FAILED_VALIDATION",
                "compression_time_sec": compression_time,
                "peak_vram_gb": peak_comp_vram,
                "tokens_per_sec": 0.0,
                "status": "FAIL"
            })
            break
            
        # PPL Evaluation
        print(f"  [Evaluating Perplexity] WikiText-2 (seq_len=512, stride=512)...", flush=True)
        eval_metrics = evaluate_wikitext_ppl(model, tokenizer, num_tokens=50000)
        ppl = eval_metrics["perplexity"]
        print(f"  [PPL @ {target_k} Experts]: {ppl:.4f} (Speed: {eval_metrics['tokens_per_sec']:.1f} tok/s, Peak VRAM: {eval_metrics['peak_vram_gb']:.2f} GB)", flush=True)
        
        res_row = {
            "model": "JetMoE-8B",
            "method": method_name,
            "experts": target_k,
            "perplexity": round(ppl, 4),
            "compression_time_sec": round(compression_time, 2),
            "peak_vram_gb": round(eval_metrics["peak_vram_gb"], 2),
            "tokens_per_sec": round(eval_metrics["tokens_per_sec"], 2),
            "status": "PASS"
        }
        results.append(res_row)
        
        runtime_records.append({
            "model": "JetMoE-8B",
            "method": method_name,
            "target_experts": target_k,
            "compression_time_sec": round(compression_time, 2),
            "eval_time_sec": round(eval_metrics["total_eval_time_sec"], 2),
            "peak_vram_gb": round(eval_metrics["peak_vram_gb"], 2),
            "tokens_per_sec": round(eval_metrics["tokens_per_sec"], 2)
        })
        
        # Save validated checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{method_name.lower()}_{target_k}exp.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  [Saved Checkpoint]: {ckpt_path}", flush=True)
        
        del model
        torch.cuda.empty_cache()
        
    return results, runtime_records

def main():
    print(f"Loading {MODEL_ID} in bfloat16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model_pristine = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    
    if not run_compatibility_test(model_pristine, tokenizer):
        sys.exit(1)
        
    all_results = []
    all_runtime = []
    
    # 1. Benchmark HC-SMoE
    hc_res, hc_rt = benchmark_jetmoe_method("HC-SMoE", apply_hcsmoe_jetmoe_layer, model_pristine, tokenizer)
    all_results.extend(hc_res)
    all_runtime.extend(hc_rt)
    
    # 2. Benchmark MC-SMoE
    mc_res, mc_rt = benchmark_jetmoe_method("MC-SMoE", apply_mcsmoe_jetmoe_layer, model_pristine, tokenizer)
    all_results.extend(mc_res)
    all_runtime.extend(mc_rt)
    
    # Write Results CSV
    res_df = pd.DataFrame(all_results)
    res_csv = os.path.join(RESULTS_DIR, "jetmoe_results.csv")
    res_df.to_csv(res_csv, index=False)
    print(f"\n[Saved] JetMoE results written to {res_csv}", flush=True)
    
    # Write Runtime & Memory CSV
    rt_df = pd.DataFrame(all_runtime)
    rt_csv = os.path.join(RESULTS_DIR, "runtime_memory.csv")
    if os.path.exists(rt_csv):
        existing_rt = pd.read_csv(rt_csv)
        rt_df = pd.concat([existing_rt, rt_df], ignore_index=True)
    rt_df.to_csv(rt_csv, index=False)
    print(f"[Saved] Runtime & Memory metrics written to {rt_csv}", flush=True)
    
    # Print Comparison Table
    print("\n" + "="*70)
    print("JETMOE-8B BENCHMARK COMPARISON TABLE (WikiText-2 PPL)")
    print("="*70)
    print(f"{'Method':<16} | {'7 Experts':<10} | {'6 Experts':<10} | {'4 Experts':<10}")
    print("-" * 52)
    print(f"{'CARE-Adaptive*':<16} | {'10.17':<10} | {'15.69':<10} | {'87.19':<10}")
    print(f"{'REAP*':<16} | {'8.10':<10} | {'9.81':<10} | {'17.48':<10}")
    
    for method in ["HC-SMoE", "MC-SMoE"]:
        sub = [r for r in all_results if r["method"] == method]
        p_dict = {r["experts"]: str(r["perplexity"]) for r in sub}
        print(f"{method:<16} | {p_dict.get(7, 'N/A'):<10} | {p_dict.get(6, 'N/A'):<10} | {p_dict.get(4, 'N/A'):<10}")
    print("* Reference baselines reused from existing artifacts.")

if __name__ == "__main__":
    main()
