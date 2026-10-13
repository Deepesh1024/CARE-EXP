import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import gc
import json
from typing import Dict, List, Tuple, Any, Optional
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hc_smoe_engine import apply_hcsmoe_phi_layer
from mc_smoe_engine import apply_mcsmoe_phi_layer
from evaluate_ppl import evaluate_wikitext_ppl

MODEL_ID = "microsoft/Phi-3.5-MoE-instruct"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "phi")
RESULTS_CSV = os.path.join(RESULTS_DIR, "phi_results.csv")
RUNTIME_CSV = os.path.join(RESULTS_DIR, "runtime_memory.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

PHI_TARGETS = [14, 12, 10, 8]

REFERENCE_RESULTS = {
    "Original": {"16": 4.57},
    "CARE-Adaptive": {"14": 6.88, "12": 9.24, "10": 13.57, "8": 32.75},
    "REAP": {"14": 6.44, "12": 17.65, "10": 40.33, "8": 125.72},
    "Random": {"14": 8.50, "12": 23.34, "10": 44.74, "8": 71.90}
}

def load_fresh_phi_model():
    """
    Loads a pristine copy of Phi-3.5-MoE in 4-bit NF4 with bfloat16 compute.
    Directly placed on cuda:0 to fit the 42B model within 24GB VRAM (~21GB footprint).
    """
    print(f"Loading fresh {MODEL_ID} in 4-bit on GPU...", flush=True)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        device_map={"": 0}
    )
    model.eval()
    return model

def validate_phi_invariants(model, target_k: int) -> Dict[str, bool]:
    shapes_ok = True
    no_nan_or_inf = True
    
    for l_idx, layer in enumerate(model.model.layers):
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None:
            moe = getattr(layer, "mlp", None)
        if moe is None:
            continue
            
        if hasattr(moe, "experts") and isinstance(moe.experts, (list, torch.nn.ModuleList)):
            if len(moe.experts) != target_k:
                shapes_ok = False
                print(f"  [Validation Error] Layer {l_idx} experts count {len(moe.experts)} != target {target_k}")
        elif hasattr(moe, "gate_up_proj"):
            if moe.gate_up_proj.shape[0] != target_k:
                shapes_ok = False
                print(f"  [Validation Error] Layer {l_idx} gate_up_proj experts {moe.gate_up_proj.shape[0]} != target {target_k}")
                
        if hasattr(moe, "gate"):
            router_out = moe.gate.weight.shape[0] if hasattr(moe.gate, "weight") else moe.gate.out_features
            if router_out != target_k:
                shapes_ok = False
                print(f"  [Validation Error] Layer {l_idx} gate out_features {router_out} != target {target_k}")
                
        if hasattr(moe, "num_experts") and moe.num_experts != target_k:
            shapes_ok = False
            print(f"  [Validation Error] Layer {l_idx} moe.num_experts {moe.num_experts} != target {target_k}")
            
        if hasattr(moe, "gate") and hasattr(moe.gate, "weight") and moe.gate.weight.is_floating_point():
            if torch.isnan(moe.gate.weight).any() or torch.isinf(moe.gate.weight).any():
                no_nan_or_inf = False
                print(f"  [Validation Error] NaN or Inf found in layer {l_idx} gate")
                
    return {"shapes_ok": shapes_ok, "numerical_integrity": no_nan_or_inf}

def run_compatibility_test(tokenizer) -> bool:
    print("\n--- [Phase 3.1] Running Phi-3.5-MoE Compatibility Test ---", flush=True)
    try:
        model = load_fresh_phi_model()
        sample_input = tokenizer("Hello, Mixture of Experts evaluation.", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**sample_input)
        assert not torch.isnan(out.logits).any(), "NaN found in baseline logits"
        print("  [Pass] Forward pass succeeded with clean numerical logits.", flush=True)
        del model, sample_input, out
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"  [Fail] Compatibility test failed: {e}", flush=True)
        return False

def record_result(res_row: Dict[str, Any], rt_row: Optional[Dict[str, Any]] = None):
    # Incremental persistence to phi_results.csv
    if os.path.exists(RESULTS_CSV):
        res_df = pd.read_csv(RESULTS_CSV)
        # Replace row if exists, else append
        idx = res_df[(res_df["method"] == res_row["method"]) & (res_df["experts"] == res_row["experts"])].index
        if len(idx) > 0:
            for k, v in res_row.items():
                res_df.loc[idx[0], k] = v
        else:
            res_df = pd.concat([res_df, pd.DataFrame([res_row])], ignore_index=True)
    else:
        res_df = pd.DataFrame([res_row])
    res_df.to_csv(RESULTS_CSV, index=False)
    print(f"  [Persisted] Metrics recorded in {RESULTS_CSV}", flush=True)
    
    # Incremental persistence to runtime_memory.csv
    if rt_row is not None:
        if os.path.exists(RUNTIME_CSV):
            rt_df = pd.read_csv(RUNTIME_CSV)
            rt_df = pd.concat([rt_df, pd.DataFrame([rt_row])], ignore_index=True)
        else:
            rt_df = pd.DataFrame([rt_row])
        rt_df.to_csv(RUNTIME_CSV, index=False)

def run_single_phi_target(method_name: str, apply_fn, tokenizer, target_k: int) -> bool:
    print(f"\n>>> Compressing Phi-3.5-MoE to {target_k} experts via {method_name}...", flush=True)
    model = load_fresh_phi_model()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    
    for l_idx, layer in enumerate(model.model.layers):
        moe = getattr(layer, "block_sparse_moe", None) or getattr(layer, "mlp", None)
        apply_fn(moe, target_k)
        
    model.config.num_local_experts = target_k
    if hasattr(model.config, "num_experts"):
        model.config.num_experts = target_k
        
    compression_time = time.time() - t_start
    peak_comp_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"  [Compression Complete] Time: {compression_time:.2f}s | Peak VRAM: {peak_comp_vram:.2f} GB", flush=True)
    
    val = validate_phi_invariants(model, target_k)
    if not (val["shapes_ok"] and val["numerical_integrity"]):
        print(f"  [Gate Failed] Checkpoint at {target_k} experts failed invariant checks! Halting further targets.", flush=True)
        res_row = {
            "model": "Phi-3.5-MoE",
            "method": method_name,
            "experts": target_k,
            "perplexity": "FAILED_VALIDATION",
            "compression_time_sec": round(compression_time, 2),
            "peak_vram_gb": round(peak_comp_vram, 2),
            "tokens_per_sec": 0.0,
            "status": "FAIL"
        }
        record_result(res_row)
        del model
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        return False
        
    print(f"  [Evaluating Perplexity] WikiText-2 (seq_len=512, stride=512)...", flush=True)
    eval_metrics = evaluate_wikitext_ppl(model, tokenizer, num_tokens=15000)
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
    
    rt_row = {
        "model": "Phi-3.5-MoE",
        "method": method_name,
        "target_experts": target_k,
        "compression_time_sec": round(compression_time, 2),
        "eval_time_sec": round(eval_metrics["total_eval_time_sec"], 2),
        "peak_vram_gb": round(eval_metrics["peak_vram_gb"], 2),
        "tokens_per_sec": round(eval_metrics["tokens_per_sec"], 2)
    }
    
    record_result(res_row, rt_row)
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{method_name.lower().replace('-', '_')}_{target_k}exp_moe.pt")
    try:
        moe_state = {k: v.cpu() for k, v in model.state_dict().items() if "block_sparse_moe" in k or "mlp" in k}
        torch.save(moe_state, ckpt_path)
        del moe_state
        print(f"  [Saved MoE Checkpoint]: {ckpt_path}", flush=True)
    except Exception as e:
        print(f"  [Checkpoint Warning] Could not save MoE checkpoint: {e}", flush=True)
        
    del model, eval_metrics, val
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    vram_after = torch.cuda.memory_allocated() / (1024**3)
    print(f"  [VRAM Cleaned]: {vram_after:.2f} GB remaining on GPU after target {target_k}", flush=True)
    return True

def benchmark_phi_method(method_name: str, apply_fn, tokenizer):
    print(f"\n{'='*70}\nSTARTING PHI-3.5-MOE BENCHMARK: {method_name}\n{'='*70}", flush=True)
    
    for target_k in PHI_TARGETS:
        existing_res = {}
        if os.path.exists(RESULTS_CSV):
            prev_df = pd.read_csv(RESULTS_CSV)
            for _, r in prev_df[prev_df["method"] == method_name].iterrows():
                if str(r.get("status", "")).upper() == "PASS":
                    existing_res[int(r["experts"])] = float(r["perplexity"])
                    
        if target_k in existing_res:
            print(f"\n>>> [Reusing] {method_name} @ {target_k} experts already evaluated: PPL = {existing_res[target_k]}", flush=True)
            continue
            
        success = run_single_phi_target(method_name, apply_fn, tokenizer, target_k)
        if not success:
            break

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    if not run_compatibility_test(tokenizer):
        sys.exit(1)
        
    # 1. Benchmark HC-SMoE
    benchmark_phi_method("HC-SMoE", apply_hcsmoe_phi_layer, tokenizer)
    
    # 2. Benchmark MC-SMoE
    benchmark_phi_method("MC-SMoE", apply_mcsmoe_phi_layer, tokenizer)
    
    # Summary Table
    print("\n" + "="*70)
    print("PHI-3.5-MOE BENCHMARK COMPARISON TABLE (WikiText-2 PPL)")
    print("="*70)
    print(f"{'Method':<16} | {'14 Experts':<11} | {'12 Experts':<11} | {'10 Experts':<11} | {'8 Experts':<11}")
    print("-" * 65)
    print(f"{'CARE-Adaptive*':<16} | {'6.88':<11} | {'9.24':<11} | {'13.57':<11} | {'32.75':<11}")
    print(f"{'Sub-MoE*':<16} | {'N/A (audit)':<11} | {'N/A (audit)':<11} | {'N/A (audit)':<11} | {'N/A (audit)':<11}")
    
    if os.path.exists(RESULTS_CSV):
        res_df = pd.read_csv(RESULTS_CSV)
        for method in ["HC-SMoE", "MC-SMoE"]:
            sub = res_df[res_df["method"] == method]
            p_dict = {int(r["experts"]): str(r["perplexity"]) for _, r in sub.iterrows()}
            print(f"{method:<16} | {p_dict.get(14, 'N/A'):<11} | {p_dict.get(12, 'N/A'):<11} | {p_dict.get(10, 'N/A'):<11} | {p_dict.get(8, 'N/A'):<11}")
    print("* Reference baselines reused from existing artifacts.")

if __name__ == "__main__":
    main()
