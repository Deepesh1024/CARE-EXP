import os
import sys
import time
import copy
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from hc_smoe_engine import apply_hcsmoe_phi_layer
from mc_smoe_engine import apply_mcsmoe_phi_layer
from evaluate_ppl import evaluate_wikitext_ppl

MODEL_ID = "microsoft/Phi-3.5-MoE-instruct"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "phi")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

PHI_TARGETS = [14, 12, 10, 8]

REFERENCE_RESULTS = {
    "Original": {"16": 4.57},
    "CARE-Adaptive": {"14": 6.88, "12": 9.24, "10": 13.57, "8": 32.75},
    "REAP": {"14": 6.44, "12": 17.65, "10": 40.33, "8": 125.72},
    "Random": {"14": 8.50, "12": 23.34, "10": 44.74, "8": 71.90}
}

def validate_phi_invariants(model, target_k: int) -> Dict[str, bool]:
    shapes_ok = True
    no_nan_or_inf = True
    
    for l_idx, layer in enumerate(model.model.layers):
        moe = layer.block_sparse_moe
        gu_w = moe.gate_up_proj.data
        d_w = moe.down_proj.data
        r_w = moe.gate.weight.data
        
        if gu_w.shape[0] != target_k or d_w.shape[0] != target_k or r_w.shape[0] != target_k:
            shapes_ok = False
            print(f"  [Validation Error] Layer {l_idx} shapes: gu={gu_w.shape}, d={d_w.shape}, r={r_w.shape} != target {target_k}")
            
        if torch.isnan(gu_w).any() or torch.isinf(gu_w).any() or \
           torch.isnan(d_w).any() or torch.isinf(d_w).any() or \
           torch.isnan(r_w).any() or torch.isinf(r_w).any():
            no_nan_or_inf = False
            print(f"  [Validation Error] NaN or Inf found in layer {l_idx}")
            
    return {"shapes_ok": shapes_ok, "numerical_integrity": no_nan_or_inf}

def run_compatibility_test(model, tokenizer) -> bool:
    print("\n--- [Phase 3.1] Running Phi-3.5-MoE Compatibility Test ---", flush=True)
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

def benchmark_phi_method(method_name: str, apply_fn, model_pristine, tokenizer):
    print(f"\n{'='*70}\nSTARTING PHI-3.5-MOE BENCHMARK: {method_name}\n{'='*70}", flush=True)
    results = []
    runtime_records = []
    
    for target_k in PHI_TARGETS:
        print(f"\n>>> Compressing Phi-3.5-MoE to {target_k} experts via {method_name}...", flush=True)
        t_start = time.time()
        torch.cuda.reset_peak_memory_stats()
        
        model = copy.deepcopy(model_pristine)
        
        for l_idx, layer in enumerate(model.model.layers):
            apply_fn(layer.block_sparse_moe, target_k)
            
        compression_time = time.time() - t_start
        peak_comp_vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  [Compression Complete] Time: {compression_time:.2f}s | Peak VRAM: {peak_comp_vram:.2f} GB", flush=True)
        
        val = validate_phi_invariants(model, target_k)
        if not (val["shapes_ok"] and val["numerical_integrity"]):
            print(f"  [Gate Failed] Checkpoint at {target_k} experts failed invariant checks! Halting further targets.", flush=True)
            results.append({
                "model": "Phi-3.5-MoE",
                "method": method_name,
                "experts": target_k,
                "perplexity": "FAILED_VALIDATION",
                "compression_time_sec": compression_time,
                "peak_vram_gb": peak_comp_vram,
                "tokens_per_sec": 0.0,
                "status": "FAIL"
            })
            break
            
        print(f"  [Evaluating Perplexity] WikiText-2 (seq_len=512, stride=512)...", flush=True)
        eval_metrics = evaluate_wikitext_ppl(model, tokenizer, num_tokens=50000)
        ppl = eval_metrics["perplexity"]
        print(f"  [PPL @ {target_k} Experts]: {ppl:.4f} (Speed: {eval_metrics['tokens_per_sec']:.1f} tok/s, Peak VRAM: {eval_metrics['peak_vram_gb']:.2f} GB)", flush=True)
        
        res_row = {
            "model": "Phi-3.5-MoE",
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
            "model": "Phi-3.5-MoE",
            "method": method_name,
            "target_experts": target_k,
            "compression_time_sec": round(compression_time, 2),
            "eval_time_sec": round(eval_metrics["total_eval_time_sec"], 2),
            "peak_vram_gb": round(eval_metrics["peak_vram_gb"], 2),
            "tokens_per_sec": round(eval_metrics["tokens_per_sec"], 2)
        })
        
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
    hc_res, hc_rt = benchmark_phi_method("HC-SMoE", apply_hcsmoe_phi_layer, model_pristine, tokenizer)
    all_results.extend(hc_res)
    all_runtime.extend(hc_rt)
    
    # 2. Benchmark MC-SMoE
    mc_res, mc_rt = benchmark_phi_method("MC-SMoE", apply_mcsmoe_phi_layer, model_pristine, tokenizer)
    all_results.extend(mc_res)
    all_runtime.extend(mc_rt)
    
    res_df = pd.DataFrame(all_results)
    res_csv = os.path.join(RESULTS_DIR, "phi_results.csv")
    res_df.to_csv(res_csv, index=False)
    print(f"\n[Saved] Phi-3.5-MoE results written to {res_csv}", flush=True)
    
    rt_df = pd.DataFrame(all_runtime)
    rt_csv = os.path.join(RESULTS_DIR, "runtime_memory.csv")
    if os.path.exists(rt_csv):
        existing_rt = pd.read_csv(rt_csv)
        rt_df = pd.concat([existing_rt, rt_df], ignore_index=True)
    rt_df.to_csv(rt_csv, index=False)
    print(f"[Saved] Runtime & Memory metrics written to {rt_csv}", flush=True)
    
    print("\n" + "="*70)
    print("PHI-3.5-MOE BENCHMARK COMPARISON TABLE (WikiText-2 PPL)")
    print("="*70)
    print(f"{'Method':<16} | {'14 Experts':<11} | {'12 Experts':<11} | {'10 Experts':<11} | {'8 Experts':<11}")
    print("-" * 65)
    print(f"{'CARE-Adaptive*':<16} | {'6.88':<11} | {'9.24':<11} | {'13.57':<11} | {'32.75':<11}")
    print(f"{'REAP*':<16} | {'6.44':<11} | {'17.65':<11} | {'40.33':<11} | {'125.72':<11}")
    print(f"{'Random*':<16} | {'8.50':<11} | {'23.34':<11} | {'44.74':<11} | {'71.90':<11}")
    
    for method in ["HC-SMoE", "MC-SMoE"]:
        sub = [r for r in all_results if r["method"] == method]
        p_dict = {r["experts"]: str(r["perplexity"]) for r in sub}
        print(f"{method:<16} | {p_dict.get(14, 'N/A'):<11} | {p_dict.get(12, 'N/A'):<11} | {p_dict.get(10, 'N/A'):<11} | {p_dict.get(8, 'N/A'):<11}")
    print("* Reference baselines reused from existing artifacts.")

if __name__ == "__main__":
    main()
