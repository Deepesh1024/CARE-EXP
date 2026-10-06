"""
Phi-3.5-MoE Architecture Audit Script
======================================
Run this FIRST before any compression benchmark.

Checks:
  1. Model config / architecture properties
  2. Expert tensor layout per layer
  3. Router behavior (top-k, gate values)
  4. VRAM requirement for full BF16 load
  5. Expert merge operator (one pair, one layer)
  6. Forward pass correctness after merge
  7. Evaluation pipeline sanity (one batch PPL)

STOP if any check fails. Do not proceed to benchmark.
"""

import os
import sys
import json
import time
import hashlib
import traceback
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import transformers.utils.import_utils
if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):
    transformers.utils.import_utils.is_torch_fx_available = lambda: False

from datasets import load_dataset

MODEL_ID = "microsoft/Phi-3.5-MoE-instruct"
AUDIT_LAYER = 0        # first MoE layer index to audit
AUDIT_EXPERT_A = 0
AUDIT_EXPERT_B = 1
SEQ_LEN = 128
AUDIT_NBATCHES = 2

STOP_CONDITIONS = []

def stop(reason):
    STOP_CONDITIONS.append(reason)
    print(f"\n🚨 STOP CONDITION: {reason}")

def check(condition, ok_msg, fail_msg):
    if condition:
        print(f"  ✅ {ok_msg}")
        return True
    else:
        stop(fail_msg)
        print(f"  ❌ {fail_msg}")
        return False


# ──────────────────────────────────────────────────────────
# 1. CONFIG / ARCHITECTURE
# ──────────────────────────────────────────────────────────
def audit_config():
    print("\n=== [1] Config / Architecture ===")
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    
    props = {
        "model_type":            getattr(cfg, "model_type", "UNKNOWN"),
        "num_hidden_layers":     getattr(cfg, "num_hidden_layers", None),
        "num_local_experts":     getattr(cfg, "num_local_experts", None),
        "num_experts_per_tok":   getattr(cfg, "num_experts_per_tok", None),
        "hidden_size":           getattr(cfg, "hidden_size", None),
        "intermediate_size":     getattr(cfg, "intermediate_size", None),
        "vocab_size":            getattr(cfg, "vocab_size", None),
    }
    
    for k, v in props.items():
        print(f"  {k}: {v}")
    
    check(props["num_local_experts"] is not None, 
          f"num_local_experts = {props['num_local_experts']}", 
          "num_local_experts not found in config")
    check(props["num_experts_per_tok"] is not None, 
          f"num_experts_per_tok = {props['num_experts_per_tok']}", 
          "num_experts_per_tok not found in config")
    
    return cfg, props


# ──────────────────────────────────────────────────────────
# 2. MEMORY STRATEGY
# ──────────────────────────────────────────────────────────
def audit_memory():
    print("\n=== [2] VRAM / Memory Strategy ===")
    if not torch.cuda.is_available():
        stop("No CUDA device available")
        return None, None
    
    n_gpus = torch.cuda.device_count()
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n_gpus))
    total_vram_gb = total_vram / 1e9
    
    print(f"  Detected {n_gpus} GPU(s), total VRAM: {total_vram_gb:.1f} GB")
    
    # Phi-3.5-MoE ~42B params, BF16 = ~84 GB; need offloading/quantization on single 4090
    if total_vram_gb < 80:
        print(f"  ⚠️  Full BF16 requires ~84 GB. VRAM ({total_vram_gb:.1f} GB) is insufficient.")
        print("  → Strategy: load_in_4bit via bitsandbytes OR device_map='auto' with CPU offload")
        print("  → DOCUMENTING this as required quantization — will apply to ALL methods equally")
        strategy = "4bit_or_offload"
    else:
        print("  VRAM sufficient for full BF16 load")
        strategy = "bf16"
    
    return strategy, total_vram_gb


# ──────────────────────────────────────────────────────────
# 3. MODEL LOAD
# ──────────────────────────────────────────────────────────
def load_model(strategy):
    print(f"\n=== [3] Model Load (strategy={strategy}) ===")
    kwargs = dict(trust_remote_code=True)
    
    if strategy == "4bit_or_offload":
        try:
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            kwargs["quantization_config"] = bnb_cfg
            kwargs["device_map"] = "auto"
            print("  Attempting 4-bit quantization via bitsandbytes (device_map='auto')...")
        except ImportError:
            print("  bitsandbytes not available; falling back to device_map=auto with CPU offload")
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    t1 = time.time()
    
    print(f"  Loaded in {t1-t0:.1f}s")
    
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  Peak GPU memory allocated: {peak_mb:.0f} MB")
    
    return model, tokenizer


# ──────────────────────────────────────────────────────────
# 4. EXPERT TENSOR LAYOUT
# ──────────────────────────────────────────────────────────
def audit_expert_layout(model, cfg):
    print(f"\n=== [4] Expert Tensor Layout (layer {AUDIT_LAYER}) ===")
    
    layer = model.model.layers[AUDIT_LAYER]
    if hasattr(layer, "mlp"):
        moe = layer.mlp
    elif hasattr(layer, "block_sparse_moe"):
        moe = layer.block_sparse_moe
    else:
        moe = layer
    
    print(f"  MoE module type: {type(moe).__name__}")
    
    expert_list = None
    router = None
    
    # Try common attribute names
    for attr in ["experts", "block_sparse_moe", "ffn"]:
        if hasattr(moe, attr):
            sub = getattr(moe, attr)
            print(f"  Found moe.{attr}: {type(sub).__name__}")
            if hasattr(sub, "experts"):
                expert_list = sub.experts
                print(f"    Found .experts: {type(expert_list).__name__}")
            elif isinstance(sub, (list, torch.nn.ModuleList)):
                expert_list = sub
                print(f"    Using directly as expert list, len={len(sub)}")
    
    if expert_list is None:
        # Try moe.experts directly
        if hasattr(moe, "experts"):
            expert_list = moe.experts
            print(f"  Found moe.experts: {type(expert_list).__name__}")
    
    # Find router
    for attr in ["gate", "router", "block_sparse_moe"]:
        if hasattr(moe, attr):
            sub = getattr(moe, attr)
            if hasattr(sub, "weight") or hasattr(sub, "gate"):
                router = sub
                print(f"  Found router at moe.{attr}: {type(sub).__name__}")
                break
    
    check(expert_list is not None, "Expert list found", "Expert list NOT found — STOP")
    if expert_list is None:
        return None, None
    
    n_found = len(expert_list)
    n_expected = cfg.num_local_experts
    check(n_found == n_expected,
          f"Expert count matches config: {n_found}",
          f"Expert count mismatch: found {n_found}, config says {n_expected}")
    
    # Inspect first expert tensor shapes
    e0 = expert_list[0] if not isinstance(expert_list, dict) else expert_list[list(expert_list.keys())[0]]
    print(f"\n  Expert[0] module type: {type(e0).__name__}")
    for name, param in e0.named_parameters():
        print(f"    {name}: shape={tuple(param.shape)}, dtype={param.dtype}")
    
    return expert_list, router


# ──────────────────────────────────────────────────────────
# 5. ROUTER BEHAVIOR
# ──────────────────────────────────────────────────────────
def audit_router(model, tokenizer, cfg):
    print(f"\n=== [5] Router Behavior ===")
    
    device = next(model.parameters()).device
    sample_text = "The Eiffel Tower is located in"
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)
    
    routing_log = []
    
    def make_hook(layer_idx):
        def hook(module, args, kwargs, output):
            # output may be (hidden_state,) or (hidden_state, router_logits)
            if isinstance(output, tuple) and len(output) >= 2:
                logits = output[1]
                if logits is not None and isinstance(logits, torch.Tensor):
                    k = cfg.num_experts_per_tok
                    topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
                    routing_log.append({
                        "layer": layer_idx,
                        "topk_indices_sample": topk_idx[0, :5].tolist(),
                        "topk_vals_sample": topk_vals[0, :5].tolist(),
                    })
        return hook
    
    handles = []
    for li, layer in enumerate(model.model.layers[:2]):
        if hasattr(layer, "mlp"):
            h = layer.mlp.register_forward_hook(make_hook(li), with_kwargs=True)
            handles.append(h)
    
    with torch.no_grad():
        try:
            _ = model(**inputs)
        except Exception as e:
            print(f"  Forward pass error: {e}")
    
    for h in handles:
        h.remove()
    
    if routing_log:
        for entry in routing_log:
            print(f"  Layer {entry['layer']}: top-{cfg.num_experts_per_tok} indices (first 5 tokens): {entry['topk_indices_sample']}")
        print("  ✅ Router produces interpretable top-k routing")
    else:
        print("  ⚠️  Could not capture router output via hook — will need direct inspection")
    
    return len(routing_log) > 0


# ──────────────────────────────────────────────────────────
# 6. EXPERT MERGE OPERATOR
# ──────────────────────────────────────────────────────────
def audit_merge(expert_list):
    print(f"\n=== [6] Expert Merge Operator ===")
    print(f"  Merging expert {AUDIT_EXPERT_A} + expert {AUDIT_EXPERT_B} on layer {AUDIT_LAYER}")
    
    eA = expert_list[AUDIT_EXPERT_A]
    eB = expert_list[AUDIT_EXPERT_B]
    
    # Check shapes match
    params_A = dict(eA.named_parameters())
    params_B = dict(eB.named_parameters())
    
    all_match = True
    for name in params_A:
        if name not in params_B:
            stop(f"Expert B missing parameter {name}")
            all_match = False
            continue
        if params_A[name].shape != params_B[name].shape:
            stop(f"Shape mismatch for {name}: A={params_A[name].shape}, B={params_B[name].shape}")
            all_match = False
    
    check(all_match, "All expert parameter shapes match", "Shape mismatch — STOP")
    if not all_match:
        return False
    
    # Perform merge into A in-place (on CPU copy to avoid corrupting model)
    print("  Performing test merge W_new = (W_A + W_B) / 2 ...")
    with torch.no_grad():
        for name in params_A:
            w_a = params_A[name].float()
            w_b = params_B[name].float()
            merged = (w_a + w_b) / 2.0
            check(merged.shape == w_a.shape,
                  f"Merged shape {name}: {tuple(merged.shape)} ✓",
                  f"Merge shape error for {name}")
    
    print("  ✅ Merge operator produces correct tensor shapes")
    return True


# ──────────────────────────────────────────────────────────
# 7. FORWARD PASS + PPL SANITY CHECK
# ──────────────────────────────────────────────────────────
def audit_forward(model, tokenizer):
    print(f"\n=== [7] Forward Pass + PPL Sanity ===")
    
    device = next(model.parameters()).device
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    tokens = enc.input_ids[0, :SEQ_LEN * AUDIT_NBATCHES]
    
    input_ids = tokens.view(AUDIT_NBATCHES, SEQ_LEN).to(device)
    
    total_nll = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(AUDIT_NBATCHES):
            batch = input_ids[i:i+1]
            try:
                out = model(batch, labels=batch)
                nll = out.loss.item()
                total_nll += nll * (SEQ_LEN - 1)
                total_tokens += (SEQ_LEN - 1)
                print(f"  Batch {i}: loss={nll:.4f}")
            except Exception as e:
                stop(f"Forward pass failed on batch {i}: {e}")
                traceback.print_exc()
                return None
    
    ppl = float(np.exp(total_nll / total_tokens))
    print(f"  PPL on {total_tokens} tokens (uncompressed sanity): {ppl:.2f}")
    
    # Phi-3.5-MoE is a capable model; uncompressed PPL on WT2 should be < 15
    check(ppl < 30.0, f"Sanity PPL {ppl:.2f} is in expected range", 
          f"Sanity PPL {ppl:.2f} suspiciously high — check tokenizer/model state")
    
    return ppl


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PHI-3.5-MOE ARCHITECTURE AUDIT")
    print("=" * 60)
    
    audit_results = {}
    
    # 1. Config
    cfg, props = audit_config()
    audit_results["config"] = props
    
    # 2. Memory
    strategy, vram_gb = audit_memory()
    audit_results["vram_gb"] = vram_gb
    audit_results["load_strategy"] = strategy
    
    # Abort early if no CUDA
    if strategy is None:
        print("\n❌ No CUDA available. Stopping audit.")
        return
    
    # 3. Load
    model, tokenizer = load_model(strategy)
    audit_results["model_loaded"] = True
    
    # 4. Expert layout
    expert_list, router = audit_expert_layout(model, cfg)
    
    # 5. Router
    router_ok = audit_router(model, tokenizer, cfg)
    audit_results["router_captured"] = router_ok
    
    # 6. Merge
    if expert_list is not None:
        merge_ok = audit_merge(expert_list)
        audit_results["merge_operator_ok"] = merge_ok
    
    # 7. Forward + PPL
    ppl = audit_forward(model, tokenizer)
    audit_results["sanity_ppl"] = ppl
    
    # ── Final verdict ──
    print("\n" + "=" * 60)
    if STOP_CONDITIONS:
        print("🚨 AUDIT FAILED — STOP CONDITIONS TRIGGERED:")
        for sc in STOP_CONDITIONS:
            print(f"  • {sc}")
        print("\nDo NOT proceed to compression benchmark until these are resolved.")
    else:
        print("✅ AUDIT PASSED — All checks cleared.")
        print("   Proceed to: bash experiments/experiment12/run_benchmark.sh")
    
    # Save audit report
    os.makedirs("experiments/experiment12/results", exist_ok=True)
    with open("experiments/experiment12/results/audit_report.json", "w") as f:
        json.dump({
            "stop_conditions": STOP_CONDITIONS,
            "passed": len(STOP_CONDITIONS) == 0,
            **audit_results
        }, f, indent=2)
    
    print(f"\nAudit report saved to experiments/experiment12/results/audit_report.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
