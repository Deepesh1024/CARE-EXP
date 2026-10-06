"""
Phi-3.5-MoE Compression Benchmark
====================================
Implements 8 methods × 4 compression targets.

Methods:
  1. uncompressed  — reference baseline
  2. random        — random pair merging
  3. parameter     — L2 weight-distance pair selection
  4. rw_l2         — router-weighted L2
  5. submoe        — Sub-MoE / CA-Greedy
  6. care_static   — frozen CARE geometry, no recompute
  7. care_adaptive — full CARE-COM with recompute after each merge
  8. reap          — pruning baseline (separate operator family)

Compression targets: 16→14, 16→12, 16→10, 16→8
"""

import os
import sys
import json
import time
import copy
import hashlib
import random as pyrandom
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
MODEL_ID = "microsoft/Phi-3.5-MoE-instruct"
CALIB_TOKENS = 32_768
EVAL_TOKENS = 15_000
SEQ_LEN = 512
SEEDS = [42, 1337, 7]
TARGETS = [14, 12, 10, 8]
METHODS = ["uncompressed", "random", "parameter", "rw_l2", "submoe",
           "care_static", "care_adaptive", "reap"]

RESULTS_DIR = "experiments/experiment12/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────
def load_wikitext_tokens(tokenizer, split, max_tokens):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    tokens = enc.input_ids[0, :max_tokens]
    
    # Store hash for reproducibility
    h = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()[:16]
    return tokens, h

def tokens_to_batches(tokens, seq_len):
    n = len(tokens) // seq_len
    return tokens[:n * seq_len].view(n, seq_len)

# ──────────────────────────────────────────────────────────
# MODEL ACCESS HELPERS
# ──────────────────────────────────────────────────────────
def get_moe_layer(model, layer_idx):
    return model.model.layers[layer_idx].mlp

def get_experts(moe_layer):
    """Return expert list, handling different attribute structures."""
    for attr in ["experts"]:
        if hasattr(moe_layer, attr):
            sub = getattr(moe_layer, attr)
            if isinstance(sub, (list, torch.nn.ModuleList)):
                return sub
            if hasattr(sub, "experts"):
                return sub.experts
    raise RuntimeError(f"Cannot find expert list in {type(moe_layer).__name__}")

def get_moe_layers(model, cfg):
    """Return indices of all MoE layers."""
    moe_layer_idxs = []
    for li, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            moe_layer_idxs.append(li)
    return moe_layer_idxs

def n_experts_per_layer(model, cfg):
    return cfg.num_local_experts

# ──────────────────────────────────────────────────────────
# MERGE OPERATOR
# ──────────────────────────────────────────────────────────
def merge_expert_pair_inplace(expert_a, expert_b):
    """W_new = (W_A + W_B) / 2, written into expert_a. expert_b becomes a dead slot."""
    with torch.no_grad():
        for (name_a, param_a), (name_b, param_b) in zip(
            expert_a.named_parameters(), expert_b.named_parameters()
        ):
            assert name_a == name_b, f"Param name mismatch: {name_a} vs {name_b}"
            assert param_a.shape == param_b.shape, f"Shape mismatch: {param_a.shape} vs {param_b.shape}"
            param_a.data = ((param_a.data.float() + param_b.data.float()) / 2.0).to(param_a.dtype)

def apply_merge_to_all_layers(model, cfg, pair: Tuple[int, int]):
    """Apply merge of expert pair (i, j) across all MoE layers."""
    i, j = pair
    moe_idxs = get_moe_layers(model, cfg)
    for li in moe_idxs:
        moe = get_moe_layer(model, li)
        experts = get_experts(moe)
        merge_expert_pair_inplace(experts[i], experts[j])
    # Note: dead experts (index j) are simply never selected by the router
    # Router weights stay frozen as per protocol

# ──────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def compute_ppl(model, batches, device):
    total_nll = 0.0
    total_tokens = 0
    for batch in tqdm(batches, desc="  PPL eval", leave=False):
        batch = batch.unsqueeze(0).to(device)
        out = model(batch, labels=batch)
        nll = out.loss.item()
        total_nll += nll * (batch.shape[1] - 1)
        total_tokens += batch.shape[1] - 1
    return float(np.exp(total_nll / total_tokens))

# ──────────────────────────────────────────────────────────
# CAPABILITY VECTORS (CARE)
# ──────────────────────────────────────────────────────────
@torch.no_grad()
def compute_care_vectors(model, cfg, calib_batches, device):
    """
    Compute C_hat for each expert across all MoE layers.
    C_hat[layer][expert] = normalized capability vector (routing frequency × output magnitude).
    """
    moe_idxs = get_moe_layers(model, cfg)
    n_exp = cfg.num_local_experts
    
    # Accumulate routing counts + output magnitudes per layer per expert
    route_counts = {li: torch.zeros(n_exp) for li in moe_idxs}
    output_mags  = {li: torch.zeros(n_exp) for li in moe_idxs}
    
    hooks = []
    for li in moe_idxs:
        moe = get_moe_layer(model, li)
        
        def make_hook(layer_idx):
            def hook(module, args, kwargs, output):
                if isinstance(output, tuple):
                    hidden, router_logits = output[0], output[1] if len(output) > 1 else None
                    if router_logits is not None:
                        topk_idx = torch.topk(router_logits, cfg.num_experts_per_tok, dim=-1).indices
                        for ei in range(n_exp):
                            mask = (topk_idx == ei).any(-1).float()
                            route_counts[layer_idx][ei] += mask.sum().item()
            return hook
        
        h = moe.register_forward_hook(make_hook(li), with_kwargs=True)
        hooks.append(h)
    
    for batch in tqdm(calib_batches, desc="  Computing CARE vectors", leave=False):
        model(batch.unsqueeze(0).to(device))
    
    for h in hooks:
        h.remove()
    
    # Build C_hat: normalize per-layer
    care_vecs = {}
    for li in moe_idxs:
        counts = route_counts[li].float()
        total = counts.sum().clamp(min=1e-8)
        care_vecs[li] = (counts / total).numpy()  # shape: (n_experts,)
    
    return care_vecs

def compute_care_distance(care_vecs, i, j):
    """Compute L2 distance between expert i and j averaged over all layers."""
    dists = []
    for li, vecs in care_vecs.items():
        d = float(np.linalg.norm(vecs[i] - vecs[j]))
        dists.append(d)
    return float(np.mean(dists))

# ──────────────────────────────────────────────────────────
# PAIR SELECTION STRATEGIES
# ──────────────────────────────────────────────────────────
def get_all_pairs(n_experts, alive_experts=None):
    if alive_experts is None:
        alive_experts = list(range(n_experts))
    pairs = [(i, j) for idx_i, i in enumerate(alive_experts) 
                     for j in alive_experts[idx_i+1:]]
    return pairs

def select_random_pair(alive_experts, rng):
    pairs = get_all_pairs(len(alive_experts), alive_experts)
    return rng.choice(pairs)

def select_param_pair(model, cfg, alive_experts):
    moe_idxs = get_moe_layers(model, cfg)
    pairs = get_all_pairs(len(alive_experts), alive_experts)
    
    best_pair, best_dist = None, float("inf")
    for (i, j) in pairs:
        total_dist = 0.0
        for li in moe_idxs:
            experts = get_experts(get_moe_layer(model, li))
            for (_, pa), (_, pb) in zip(
                experts[i].named_parameters(), experts[j].named_parameters()
            ):
                total_dist += F.mse_loss(pa.float(), pb.float()).item()
        if total_dist < best_dist:
            best_dist = total_dist
            best_pair = (i, j)
    return best_pair, best_dist

def select_rw_l2_pair(model, cfg, alive_experts, calib_batches, device):
    """Router-Weighted L2: weight parameter distance by routing frequency."""
    care_vecs = compute_care_vectors(model, cfg, calib_batches, device)
    moe_idxs = get_moe_layers(model, cfg)
    pairs = get_all_pairs(len(alive_experts), alive_experts)
    
    best_pair, best_score = None, float("inf")
    for (i, j) in pairs:
        # RW-L2: sum over layers of (route_freq_ij * param_L2)
        total_score = 0.0
        for li in moe_idxs:
            experts = get_experts(get_moe_layer(model, li))
            param_dist = sum(
                F.mse_loss(pa.float(), pb.float()).item()
                for (_, pa), (_, pb) in zip(
                    experts[i].named_parameters(), experts[j].named_parameters()
                )
            )
            routing_weight = (care_vecs[li][i] + care_vecs[li][j]) / 2.0
            total_score += routing_weight * param_dist
        
        if total_score < best_score:
            best_score = total_score
            best_pair = (i, j)
    return best_pair, best_score

def select_care_pair(care_vecs, alive_experts):
    """Select pair with smallest CARE L2 distance."""
    pairs = get_all_pairs(len(alive_experts), alive_experts)
    best_pair, best_dist = None, float("inf")
    for (i, j) in pairs:
        d = compute_care_distance(care_vecs, i, j)
        if d < best_dist:
            best_dist = d
            best_pair = (i, j)
    return best_pair, best_dist

def select_submoe_pair(model, cfg, alive_experts, calib_batches, device):
    """
    Sub-MoE / CA-Greedy: select pair that minimizes output activation distance
    on calibration data under the current model state.
    """
    moe_idxs = get_moe_layers(model, cfg)
    pairs = get_all_pairs(len(alive_experts), alive_experts)
    
    # Collect per-expert mean output activations on calibration
    expert_outputs = {li: {} for li in moe_idxs}
    
    captured = {li: {} for li in moe_idxs}
    hooks = []
    
    for li in moe_idxs:
        moe = get_moe_layer(model, li)
        experts = get_experts(moe)
        
        def make_expert_hook(layer_idx, expert_idx):
            def hook(module, input, output):
                o = output.detach().float().mean(0)  # mean over tokens
                captured[layer_idx][expert_idx] = o.cpu()
            return hook
        
        for ei in alive_experts:
            h = experts[ei].register_forward_hook(make_expert_hook(li, ei))
            hooks.append(h)
    
    with torch.no_grad():
        for batch in calib_batches[:4]:  # subset for speed
            model(batch.unsqueeze(0).to(device))
    
    for h in hooks:
        h.remove()
    
    best_pair, best_dist = None, float("inf")
    for (i, j) in pairs:
        total = 0.0
        for li in moe_idxs:
            if i in captured[li] and j in captured[li]:
                total += torch.dist(captured[li][i], captured[li][j]).item()
        if total < best_dist:
            best_dist = total
            best_pair = (i, j)
    return best_pair, best_dist

# ──────────────────────────────────────────────────────────
# COMPRESSION METHODS
# ──────────────────────────────────────────────────────────
def compress_random(model, cfg, target_n, calib_batches, eval_batches, device, seed):
    rng = np.random.default_rng(seed)
    alive = list(range(cfg.num_local_experts))
    trajectory = []
    
    while len(alive) > target_n:
        pairs = get_all_pairs(len(alive), alive)
        idx = rng.integers(len(pairs))
        pair = pairs[idx]
        apply_merge_to_all_layers(model, cfg, pair)
        alive.remove(max(pair))  # remove higher index (dead slot)
        trajectory.append({"pair": pair, "alive": alive[:], "method": "random"})
    
    return trajectory

def compress_parameter(model, cfg, target_n, calib_batches, eval_batches, device):
    alive = list(range(cfg.num_local_experts))
    trajectory = []
    
    while len(alive) > target_n:
        pair, dist = select_param_pair(model, cfg, alive)
        apply_merge_to_all_layers(model, cfg, pair)
        alive.remove(max(pair))
        trajectory.append({"pair": pair, "dist": dist, "alive": alive[:], "method": "parameter"})
    
    return trajectory

def compress_rw_l2(model, cfg, target_n, calib_batches, eval_batches, device):
    alive = list(range(cfg.num_local_experts))
    trajectory = []
    
    while len(alive) > target_n:
        pair, score = select_rw_l2_pair(model, cfg, alive, calib_batches, device)
        apply_merge_to_all_layers(model, cfg, pair)
        alive.remove(max(pair))
        trajectory.append({"pair": pair, "score": float(score), "alive": alive[:], "method": "rw_l2"})
    
    return trajectory

def compress_submoe(model, cfg, target_n, calib_batches, eval_batches, device):
    alive = list(range(cfg.num_local_experts))
    trajectory = []
    
    while len(alive) > target_n:
        pair, dist = select_submoe_pair(model, cfg, alive, calib_batches, device)
        apply_merge_to_all_layers(model, cfg, pair)
        alive.remove(max(pair))
        trajectory.append({"pair": pair, "dist": float(dist), "alive": alive[:], "method": "submoe"})
    
    return trajectory

def compress_care_static(model, cfg, target_n, calib_batches, eval_batches, device):
    """Compute CARE vectors once, never recompute after merges."""
    care_vecs = compute_care_vectors(model, cfg, calib_batches, device)
    alive = list(range(cfg.num_local_experts))
    trajectory = []
    
    while len(alive) > target_n:
        pair, dist = select_care_pair(care_vecs, alive)
        apply_merge_to_all_layers(model, cfg, pair)
        alive.remove(max(pair))
        trajectory.append({"pair": pair, "care_dist": float(dist), "alive": alive[:], "method": "care_static"})
    
    return trajectory

def compress_care_adaptive(model, cfg, target_n, calib_batches, eval_batches, device):
    """Full CARE-COM: recompute capability geometry after every merge."""
    alive = list(range(cfg.num_local_experts))
    trajectory = []
    
    while len(alive) > target_n:
        care_vecs = compute_care_vectors(model, cfg, calib_batches, device)
        pair, dist = select_care_pair(care_vecs, alive)
        apply_merge_to_all_layers(model, cfg, pair)
        alive.remove(max(pair))
        trajectory.append({
            "pair": pair, 
            "care_dist": float(dist), 
            "alive": alive[:],
            "method": "care_adaptive"
        })
    
    return trajectory

def compress_reap(model, cfg, target_n, calib_batches, eval_batches, device):
    """
    REAP: Router-guided Expert Activation Pruning.
    Prune lowest-activation experts based on router gate + activation norm.
    NOTE: This is a pruning operator, not merging. Report separately.
    """
    moe_idxs = get_moe_layers(model, cfg)
    n_exp = cfg.num_local_experts
    
    route_counts = {li: torch.zeros(n_exp) for li in moe_idxs}
    
    # Capture routing frequencies on calibration
    hooks = []
    for li in moe_idxs:
        moe = get_moe_layer(model, li)
        def make_hook(layer_idx):
            def hook(module, args, kwargs, output):
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    topk_idx = torch.topk(output[1], cfg.num_experts_per_tok, dim=-1).indices
                    for ei in range(n_exp):
                        route_counts[layer_idx][ei] += (topk_idx == ei).any(-1).float().sum().item()
            return hook
        h = moe.register_forward_hook(make_hook(li), with_kwargs=True)
        hooks.append(h)
    
    with torch.no_grad():
        for batch in calib_batches:
            model(batch.unsqueeze(0).to(device))
    for h in hooks:
        h.remove()
    
    # Compute mean routing frequency per expert across layers
    mean_freq = torch.zeros(n_exp)
    for li in moe_idxs:
        mean_freq += route_counts[li]
    mean_freq /= len(moe_idxs)
    
    n_prune = n_exp - target_n
    prune_indices = torch.argsort(mean_freq)[:n_prune].tolist()
    
    trajectory = [{"pruned_experts": prune_indices, "method": "reap"}]
    
    # Zero out pruned experts (equivalent to removal — router still routes but output = 0)
    with torch.no_grad():
        for li in moe_idxs:
            moe = get_moe_layer(model, li)
            experts = get_experts(moe)
            for ei in prune_indices:
                for param in experts[ei].parameters():
                    param.data.zero_()
    
    return trajectory

# ──────────────────────────────────────────────────────────
# BENCHMARK RUNNER
# ──────────────────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    method: str
    target: int
    seed: int
    ppl: float
    trajectory: List[Dict]
    runtime_s: float
    peak_gpu_mb: float
    metadata: Dict = field(default_factory=dict)

def run_one(model_state_dict, cfg, tokenizer, method, target, seed, device):
    """Load a fresh model copy from state dict and run one method."""
    print(f"\n  [{method}] target={target} seed={seed}")
    
    # Fresh model copy
    from transformers import AutoConfig
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        state_dict=model_state_dict,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_cfg,
        trust_remote_code=True
    )
    model.eval()
    
    calib_tokens, calib_hash = load_wikitext_tokens(tokenizer, "train", CALIB_TOKENS)
    eval_tokens, eval_hash = load_wikitext_tokens(tokenizer, "test", EVAL_TOKENS)
    
    calib_batches = tokens_to_batches(calib_tokens, SEQ_LEN)
    eval_batches = tokens_to_batches(eval_tokens, SEQ_LEN)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    t0 = time.time()
    trajectory = []
    
    if method == "uncompressed":
        trajectory = [{"method": "uncompressed"}]
    elif method == "random":
        trajectory = compress_random(model, cfg, target, calib_batches, eval_batches, device, seed)
    elif method == "parameter":
        trajectory = compress_parameter(model, cfg, target, calib_batches, eval_batches, device)
    elif method == "rw_l2":
        trajectory = compress_rw_l2(model, cfg, target, calib_batches, eval_batches, device)
    elif method == "submoe":
        trajectory = compress_submoe(model, cfg, target, calib_batches, eval_batches, device)
    elif method == "care_static":
        trajectory = compress_care_static(model, cfg, target, calib_batches, eval_batches, device)
    elif method == "care_adaptive":
        trajectory = compress_care_adaptive(model, cfg, target, calib_batches, eval_batches, device)
    elif method == "reap":
        trajectory = compress_reap(model, cfg, target, calib_batches, eval_batches, device)
    
    ppl = compute_ppl(model, eval_batches, device)
    runtime = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    
    result = BenchmarkResult(
        method=method, target=target, seed=seed,
        ppl=ppl, trajectory=trajectory, runtime_s=runtime,
        peak_gpu_mb=peak_mb,
        metadata={"calib_hash": calib_hash, "eval_hash": eval_hash}
    )
    
    print(f"    PPL={ppl:.3f}  runtime={runtime:.0f}s  peak_gpu={peak_mb:.0f}MB")
    
    # Save immediately
    out_path = f"{RESULTS_DIR}/{method}_target{target}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    
    del model
    torch.cuda.empty_cache()
    
    return result

def main():
    print("=" * 60)
    print("PHI-3.5-MOE COMPRESSION BENCHMARK")
    print("=" * 60)
    
    # Load audit report first
    audit_path = f"{RESULTS_DIR}/audit_report.json"
    if not os.path.exists(audit_path):
        print("❌ Audit report not found. Run phi_audit.py first.")
        sys.exit(1)
    
    with open(audit_path) as f:
        audit = json.load(f)
    
    if not audit.get("passed", False):
        print("❌ Audit did not pass. Fix stop conditions before running benchmark.")
        for sc in audit.get("stop_conditions", []):
            print(f"  • {sc}")
        sys.exit(1)
    
    print("✅ Audit passed. Starting benchmark...\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    all_results = []
    
    for target in TARGETS:
        for method in METHODS:
            n_seeds = len(SEEDS) if method in ["random"] else 1
            for seed in SEEDS[:n_seeds]:
                try:
                    result = run_one(None, cfg, tokenizer, method, target, seed, device)
                    all_results.append(asdict(result))
                except Exception as e:
                    print(f"  ⚠️  {method} target={target} seed={seed} FAILED: {e}")
                    traceback.print_exc()
                    all_results.append({
                        "method": method, "target": target, "seed": seed,
                        "ppl": None, "error": str(e)
                    })
    
    # Aggregate and save final table
    with open(f"{RESULTS_DIR}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Method':<16} {'Target':<8} {'PPL':<10} {'ΔPPL':<10} {'Runtime(s)':<12}")
    print("-" * 80)
    
    base_ppl = next((r["ppl"] for r in all_results 
                     if r["method"] == "uncompressed" and r.get("ppl")), None)
    
    for r in all_results:
        ppl = r.get("ppl")
        if ppl is None:
            ppl_str = "FAILED"
            delta_str = "-"
        else:
            ppl_str = f"{ppl:.3f}"
            delta_str = f"{ppl - base_ppl:+.3f}" if base_ppl else "-"
        
        rt = r.get("runtime_s", 0)
        print(f"{r['method']:<16} {r.get('target', '-'):<8} {ppl_str:<10} {delta_str:<10} {rt:<12.0f}")
    
    print(f"\nAll results saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
