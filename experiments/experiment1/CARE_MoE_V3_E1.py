"""
CARE-MoE V3 -- Oracle Benchmark Dataset v1.0 (Full Production Sweep)
=====================================================================
Target Model: allenai/OLMoE-1B-7B-0924
Objective: Immutable CARE-Oracle-v1.0 benchmark dataset with Hard Stability 
           Gates, Bootstrapped Paired Differences, and Complete Metadata.
Fix: Supports dynamic dispatch for Fused 3D Expert Parameters (transformers >= 4.40)
"""

import os
import json
import time
import copy
import math
import random
import signal
import itertools
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from tqdm.auto import tqdm
from datasets import load_dataset


# HOTFIX: PyTorch 2.1.1 compatibility for newer Transformers (OLMoE)
if not hasattr(torch, "library"):
    class DummyLibrary:
        pass
    torch.library = DummyLibrary()
if not hasattr(torch.library, "register_fake"):
    def dummy_register_fake(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    torch.library.register_fake = dummy_register_fake

from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
SEED = 42

CALIB_SIZES = [64, 128, 256, 512]
LAYERS_TO_EVALUATE = ["first", "middle", "last"]
MAX_PAIRS = None  

CALIB_BATCH_SIZE = 2  
EVAL_TOKENS_FOR_EXPERT_METRICS = 4096

GPU_ID = int(os.environ.get("CARE_MOE_GPU_ID", 0))
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results", "exp1")
SCATTER_DIR = os.path.join(OUTPUT_DIR, "scatterplots")
RESULTS_JSON_PATH = os.path.join(OUTPUT_DIR, "output.json")

METRIC_COLS = [
    "Random_Baseline", "Weight_Distance", "Weight_Cosine", 
    "Activation_Similarity", "Output_Similarity", "Routing_Similarity", 
    "Usage_Frequency", "Jaccard_Overlap"
]

_shutdown_requested = False

def _handle_shutdown_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n[SYSTEM] Signal {signum} intercepted. Halting cleanly.")

signal.signal(signal.SIGTERM, _handle_shutdown_signal)
signal.signal(signal.SIGINT, _handle_shutdown_signal)


# ============================================================
# METADATA & BOOTSTRAPPING
# ============================================================
def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return "Not inside a Git repository"

def get_benchmark_metadata():
    return {
        "benchmark_version": "CARE-Oracle-v1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "target_model": MODEL_ID,
        "merge_operator": "UniformAverage",
        "precision": str(DTYPE),
        "seed": SEED,
        "git_commit": get_git_hash(),
        "gpu_model": torch.cuda.get_device_name(GPU_ID) if torch.cuda.is_available() else "None",
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }

def load_checkpoint():
    if os.path.exists(RESULTS_JSON_PATH):
        with open(RESULTS_JSON_PATH, "r") as f:
            return json.load(f)
    return {
        "metadata": get_benchmark_metadata(),
        "architecture_summary": {},
        "expert_specialization": {},
        "results": [],
        "oracle_summaries": {},
        "correlations": None,
        "paired_differences": None,
        "split_sample_stability": {}
    }

def default_json_converter(o):
    if isinstance(o, (np.generic, np.ndarray)):
        return o.item() if (np.isscalar(o) or o.size == 1) else o.tolist()
    if isinstance(o, torch.Tensor):
        return o.item() if o.numel() == 1 else o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def save_checkpoint(data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_path = RESULTS_JSON_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=default_json_converter)
    os.replace(tmp_path, RESULTS_JSON_PATH)

def bootstrap_ci(x, y, metric_func, n_resamples=1000, ci=0.95):
    x, y = np.array(x), np.array(y)
    n = len(x)
    stats = []
    for _ in range(n_resamples):
        indices = np.random.randint(0, n, n)
        stat = metric_func(x[indices], y[indices])[0]
        if not np.isnan(stat):
            stats.append(stat)
    
    if not stats: return np.nan, np.nan
    alpha = (1.0 - ci) / 2.0
    return np.percentile(stats, alpha * 100), np.percentile(stats, (1.0 - alpha) * 100)

def bootstrap_paired_diff(m1, m2, oracle, n_resamples=1000, ci=0.95):
    m1, m2, oracle = np.array(m1), np.array(m2), np.array(oracle)
    n = len(oracle)
    diffs = []
    for _ in range(n_resamples):
        idx = np.random.randint(0, n, n)
        r1 = spearmanr(m1[idx], oracle[idx])[0]
        r2 = spearmanr(m2[idx], oracle[idx])[0]
        if not np.isnan(r1) and not np.isnan(r2):
            diffs.append(r1 - r2)
    if not diffs: return np.nan, np.nan, np.nan
    alpha = (1.0 - ci) / 2.0
    return np.mean(diffs), np.percentile(diffs, alpha * 100), np.percentile(diffs, (1.0 - alpha) * 100)

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean().item()


# ============================================================
# ARCHITECTURE & ZERO-ALLOCATION OPERATORS
# ============================================================
def find_moe_layers(model):
    found = []
    for name, module in model.named_modules():
        if hasattr(module, "experts"):
            # Matches both legacy ModuleList implementations and modern Fused 3D implementations (e.g. OLMoE)
            if isinstance(module.experts, torch.nn.ModuleList) or hasattr(module.experts, "gate_up_proj"):
                found.append((name, module))
    return found

class InPlaceParameterAveragingMerge:
    def __init__(self, template_expert):
        self.buffer = copy.deepcopy(template_expert).to(DEVICE, dtype=DTYPE)
        
    def merge(self, expert_a, expert_b):
        with torch.no_grad():
            for p_m, p_a, p_b in zip(self.buffer.parameters(), expert_a.parameters(), expert_b.parameters()):
                p_m.copy_(0.5 * p_a + 0.5 * p_b)
        return self.buffer

def flatten_expert_weights(expert):
    return torch.cat([p.detach().flatten().float() for p in expert.parameters()])


# ============================================================
# ACTIVATIONS & END-TO-END ORACLE
# ============================================================
@torch.no_grad()
def collect_calibration_activations(model, moe_block, input_ids, attention_mask, batch_size):
    hidden_list, router_list, spans = [], [], []

    h1 = moe_block.register_forward_pre_hook(lambda m, inputs: hidden_list.append(inputs[0].detach()))
    gate_module = getattr(moe_block, "gate", None) or getattr(moe_block, "router", None)
    
    # FIX: Safely unpack the tuple output from the OLMoE gate before calling .detach()
    h2 = gate_module.register_forward_hook(
        lambda m, inputs, output: router_list.append((output[0] if isinstance(output, tuple) else output).detach())
    )

    n = input_ids.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        model(input_ids=input_ids[start:end].to(DEVICE), attention_mask=attention_mask[start:end].to(DEVICE))
        spans.append((start, end))

    h1.remove()
    h2.remove()

    hidden_all, router_all = [], []
    for hs, rl, (start, end) in zip(hidden_list, router_list, spans):
        mask = attention_mask[start:end]
        b, seq, hid = hs.shape
        hs_flat, mask_flat = hs.reshape(b * seq, hid), mask.reshape(b * seq).bool()
        hidden_all.append(hs_flat[mask_flat].cpu())
        router_all.append(rl.reshape(b * seq, -1)[mask_flat].cpu())

    return torch.cat(hidden_all, dim=0), torch.cat(router_all, dim=0)

def install_experts(moe_block, idx_list, modules):
    for idx, mod in zip(idx_list, modules):
        moe_block.experts[idx] = mod

@torch.no_grad()
def run_oracle_pair(model, moe_block, i, j, input_ids, attention_mask, batch_size, top_k):
    start_time = time.time()
    
    gate_module = getattr(moe_block, "gate", None) or getattr(moe_block, "router", None)
    is_fused = hasattr(moe_block.experts, "gate_up_proj")
    
    if is_fused:
        gate_i = moe_block.experts.gate_up_proj.data[i].clone()
        gate_j = moe_block.experts.gate_up_proj.data[j].clone()
        down_i = moe_block.experts.down_proj.data[i].clone()
        down_j = moe_block.experts.down_proj.data[j].clone()
        
        merged_gate = 0.5 * (gate_i + gate_j)
        merged_down = 0.5 * (down_i + down_j)
    else:
        orig_i, orig_j = moe_block.experts[i], moe_block.experts[j]
        merge_operator = InPlaceParameterAveragingMerge(orig_i)
        merged_expert = merge_operator.merge(orig_i, orig_j)

    total_tokens, kl_sum, ce_orig_sum, ce_merged_sum, l2_hidden_sum = 0, 0.0, 0.0, 0.0, 0.0
    top1_agree_sum, topk_agree_sum, entropy_orig_sum, entropy_merged_sum = 0, 0, 0.0, 0.0
    
    n = input_ids.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ids = input_ids[start:end].to(DEVICE)
        mask = attention_mask[start:end].to(DEVICE)
        shift_mask, flat_mask = mask[:, 1:].bool(), mask.view(-1).bool()
        
        # --- ORIGINAL FORWARD ---
        if not is_fused:
            install_experts(moe_block, [i, j], [orig_i, orig_j])
            
        orig_moe_out, orig_router_out = [], []
        h_moe = moe_block.register_forward_hook(lambda m, inp, out: orig_moe_out.append(out[0] if isinstance(out, tuple) else out))
        h_gate = gate_module.register_forward_hook(lambda m, inp, out: orig_router_out.append(out[0] if isinstance(out, tuple) else out))
        
        logits_orig = model(input_ids=ids, attention_mask=mask).logits
        h_moe.remove(); h_gate.remove()
        
        # --- MERGED FORWARD ---
        if is_fused:
            moe_block.experts.gate_up_proj.data[i].copy_(merged_gate)
            moe_block.experts.gate_up_proj.data[j].copy_(merged_gate)
            moe_block.experts.down_proj.data[i].copy_(merged_down)
            moe_block.experts.down_proj.data[j].copy_(merged_down)
        else:
            install_experts(moe_block, [i, j], [merged_expert, merged_expert])
            
        merged_moe_out, merged_router_out = [], []
        h_moe = moe_block.register_forward_hook(lambda m, inp, out: merged_moe_out.append(out[0] if isinstance(out, tuple) else out))
        h_gate = gate_module.register_forward_hook(lambda m, inp, out: merged_router_out.append(out[0] if isinstance(out, tuple) else out))
        
        logits_merged = model(input_ids=ids, attention_mask=mask).logits
        h_moe.remove(); h_gate.remove()
        
        # --- RESTORE EXPERTS ---
        if is_fused:
            moe_block.experts.gate_up_proj.data[i].copy_(gate_i)
            moe_block.experts.gate_up_proj.data[j].copy_(gate_j)
            moe_block.experts.down_proj.data[i].copy_(down_i)
            moe_block.experts.down_proj.data[j].copy_(down_j)
        else:
            install_experts(moe_block, [i, j], [orig_i, orig_j])

        # --- LOGITS & STATS ---
        shift_logits_orig, shift_logits_merged = logits_orig[:, :-1, :], logits_merged[:, :-1, :]
        shift_labels = ids[:, 1:]

        logp_orig = F.log_softmax(shift_logits_orig.float(), dim=-1)
        logp_merged = F.log_softmax(shift_logits_merged.float(), dim=-1)
        
        kl_tok = (logp_orig.exp() * (logp_orig - logp_merged)).sum(dim=-1)
        kl_sum += kl_tok[shift_mask].sum().item()

        ce_orig_sum += F.nll_loss(logp_orig.reshape(-1, logp_orig.size(-1)), shift_labels.reshape(-1), reduction="none").reshape(shift_labels.shape)[shift_mask].sum().item()
        ce_merged_sum += F.nll_loss(logp_merged.reshape(-1, logp_merged.size(-1)), shift_labels.reshape(-1), reduction="none").reshape(shift_labels.shape)[shift_mask].sum().item()

        h_orig_flat = orig_moe_out[0].view(-1, orig_moe_out[0].shape[-1])[flat_mask]
        h_merged_flat = merged_moe_out[0].view(-1, merged_moe_out[0].shape[-1])[flat_mask]
        l2_hidden_sum += torch.norm(h_orig_flat.float() - h_merged_flat.float(), p=2, dim=-1).sum().item()

        r_orig_flat = orig_router_out[0].view(-1, orig_router_out[0].shape[-1])[flat_mask].float()
        r_merged_flat = merged_router_out[0].view(-1, merged_router_out[0].shape[-1])[flat_mask].float()
        
        entropy_orig_sum += compute_entropy(r_orig_flat) * r_orig_flat.shape[0]
        entropy_merged_sum += compute_entropy(r_merged_flat) * r_merged_flat.shape[0]

        top1_agree_sum += (r_orig_flat.argmax(dim=-1) == r_merged_flat.argmax(dim=-1)).sum().item()
        
        topk_orig = torch.topk(r_orig_flat, k=top_k, dim=-1).indices
        topk_merged = torch.topk(r_merged_flat, k=top_k, dim=-1).indices
        
        for tok_idx in range(topk_orig.shape[0]):
            intersect = len(set(topk_orig[tok_idx].tolist()) & set(topk_merged[tok_idx].tolist()))
            topk_agree_sum += (intersect / top_k)

        total_tokens += shift_mask.sum().item()

    return {
        "_Total_Tokens": total_tokens,
        "_KL_Sum": kl_sum,
        "_CE_Orig_Sum": ce_orig_sum,
        "_CE_Merged_Sum": ce_merged_sum,
        "_L2_Hidden_Sum": l2_hidden_sum,
        "_Top1_Agree_Sum": top1_agree_sum,
        "_TopK_Agree_Sum": topk_agree_sum,
        "_Entropy_Orig_Sum": entropy_orig_sum,
        "_Entropy_Merged_Sum": entropy_merged_sum,
        "Oracle_KL": kl_sum / max(total_tokens, 1),
        "Runtime_Sec": time.time() - start_time,
        "Max_VRAM_MB": torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2) if torch.cuda.is_available() else 0.0
    }

def aggregate_oracle_stats(stats_A, stats_B):
    total = max(stats_A["_Total_Tokens"] + stats_B["_Total_Tokens"], 1)
    return {
        "Oracle_KL": (stats_A["_KL_Sum"] + stats_B["_KL_Sum"]) / total,
        "CrossEntropy_Delta": ((stats_A["_CE_Merged_Sum"] + stats_B["_CE_Merged_Sum"]) - (stats_A["_CE_Orig_Sum"] + stats_B["_CE_Orig_Sum"])) / total,
        "Hidden_L2_Drift": (stats_A["_L2_Hidden_Sum"] + stats_B["_L2_Hidden_Sum"]) / total,
        "Router_Entropy_Orig": (stats_A["_Entropy_Orig_Sum"] + stats_B["_Entropy_Orig_Sum"]) / total,
        "Router_Entropy_Merged": (stats_A["_Entropy_Merged_Sum"] + stats_B["_Entropy_Merged_Sum"]) / total,
        "Top1_Routing_Agreement": (stats_A["_Top1_Agree_Sum"] + stats_B["_Top1_Agree_Sum"]) / total,
        "TopK_Routing_Agreement": (stats_A["_TopK_Agree_Sum"] + stats_B["_TopK_Agree_Sum"]) / total,
        "Runtime_Sec": stats_A["Runtime_Sec"] + stats_B["Runtime_Sec"],
        "Max_VRAM_MB": max(stats_A["Max_VRAM_MB"], stats_B["Max_VRAM_MB"])
    }

def format_single_oracle_stats(stats):
    t = max(stats["_Total_Tokens"], 1)
    return {
        "Oracle_KL": stats["_KL_Sum"] / t,
        "CrossEntropy_Delta": (stats["_CE_Merged_Sum"] - stats["_CE_Orig_Sum"]) / t,
        "Hidden_L2_Drift": stats["_L2_Hidden_Sum"] / t,
        "Router_Entropy_Orig": stats["_Entropy_Orig_Sum"] / t,
        "Router_Entropy_Merged": stats["_Entropy_Merged_Sum"] / t,
        "Top1_Routing_Agreement": stats["_Top1_Agree_Sum"] / t,
        "TopK_Routing_Agreement": stats["_TopK_Agree_Sum"] / t,
        "Runtime_Sec": stats["Runtime_Sec"],
        "Max_VRAM_MB": stats["Max_VRAM_MB"]
    }

# ============================================================
# FIGURE GENERATION PIPELINE
# ============================================================
def generate_publication_figures(results_df, corr_df, num_experts):
    print("\n[VISUALIZATION] Rendering publication-ready figures...")
    
    df_512 = results_df[results_df["Seq_Len"] == 512]
    for layer in df_512["Layer"].unique():
        layer_group = df_512[df_512["Layer"] == layer]
        matrix = np.full((num_experts, num_experts), np.nan)
        for _, r in layer_group.iterrows():
            a, b = int(r["Expert_A"]), int(r["Expert_B"])
            matrix[a, b] = matrix[b, a] = r["Oracle_KL"]
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(matrix), cmap="viridis", square=True)
        plt.title(f"Oracle KL Divergence - Layer '{layer}' (S=512)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"oracle_heatmap_{layer}.png"), dpi=200)
        plt.close()

    if not corr_df.empty:
        summary_corr = corr_df.groupby("Metric")[["Pearson_r", "Spearman_r"]].mean()
        plt.figure(figsize=(8, 6))
        sns.heatmap(summary_corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Mean Proxy Correlation vs Oracle KL (Across All Settings)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "correlation_summary_matrix.png"), dpi=200)
        plt.close()

    for col in METRIC_COLS:
        plt.figure(figsize=(6, 5))
        plt.scatter(results_df[col], results_df["Oracle_KL"], alpha=0.3, s=15, c="crimson")
        plt.xlabel(col)
        plt.ylabel("Oracle KL Divergence")
        plt.title(f"{col} vs Oracle KL (N={len(results_df)})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(SCATTER_DIR, f"{col}_vs_oracle_kl.png"), dpi=200)
        plt.close()

    ranked = results_df[results_df["Seq_Len"] == 512].sort_values("Oracle_KL", ascending=True)
    if not ranked.empty:
        top_20 = ranked.iloc[:20]
        labels = [f"L:{r['Layer']} E{int(r['Expert_A'])}-E{int(r['Expert_B'])}" for _, r in top_20.iterrows()]
        plt.figure(figsize=(8, 6))
        plt.barh(labels[::-1], top_20["Oracle_KL"].values[::-1], color="navy")
        plt.xlabel("Oracle KL Divergence (Lower = Safer Merge)")
        plt.title("Top 20 Lowest-Loss Merge Candidates (S=512)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "top_20_safest_merges.png"), dpi=200)
        plt.close()

    print(f"[VISUALIZATION] Complete. Figures written to {os.path.abspath(OUTPUT_DIR)}")

# ============================================================
# MAIN ORCHESTRATION
# ============================================================
def main():
    os.makedirs(SCATTER_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    checkpoint = load_checkpoint()
    
    print(f"[SYSTEM] Initializing {MODEL_ID} (SDPA Optimized)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map={"": GPU_ID}, attn_implementation="sdpa"
    ).eval()

    moe_layers = find_moe_layers(model)
    top_k = getattr(model.config, "num_experts_per_tok", 8) 
    
    layer_indices = {}
    if "first" in LAYERS_TO_EVALUATE: layer_indices["first"] = 0
    if "middle" in LAYERS_TO_EVALUATE: layer_indices["middle"] = len(moe_layers) // 2
    if "last" in LAYERS_TO_EVALUATE: layer_indices["last"] = len(moe_layers) - 1

    # Use the explicit path if the generic 'wikitext' fails due to hub changes
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in raw["text"] if len(t.strip()) > 0]
    random.Random(SEED).shuffle(texts)

    all_tokens = []
    for t in texts:
        if len(all_tokens) >= max(CALIB_SIZES): break
        enc = tokenizer(t, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        if enc["attention_mask"].sum().item() >= 8: all_tokens.append(enc)

    completed_pairs = {(r["Seq_Len"], r["Layer"], r["Expert_A"], r["Expert_B"]) for r in checkpoint.get("results", [])}
    
    # Establish total number of experts gracefully
    test_experts = moe_layers[0][1].experts
    num_experts = test_experts.num_experts if hasattr(test_experts, "num_experts") else len(test_experts)
    
    all_pairs = list(itertools.combinations(range(num_experts), 2))
    if MAX_PAIRS: all_pairs = random.Random(SEED).sample(all_pairs, min(MAX_PAIRS, len(all_pairs)))

    # Generate static random baseline for absolute stability
    static_random_scores = {pair: np.random.rand() for pair in all_pairs}

    try:
        for seq_len in CALIB_SIZES:
            print(f"\n--- Engaging Calibration Sequence Size: {seq_len} ---")
            calib_subset = all_tokens[:seq_len]
            calib_input_ids = torch.cat([s["input_ids"] for s in calib_subset], dim=0)
            calib_attn_mask = torch.cat([s["attention_mask"] for s in calib_subset], dim=0)

            for layer_label, layer_idx in layer_indices.items():
                print(f"[PROCESS] Target Layer: {layer_label}")
                layer_name, moe_block = moe_layers[layer_idx]
                experts = moe_block.experts
                is_fused = hasattr(experts, "gate_up_proj")
                
                # Fetch flattened weights dynamically
                flat_weights = []
                for e in range(num_experts):
                    if is_fused:
                        w1 = experts.gate_up_proj.data[e].flatten()
                        w2 = experts.down_proj.data[e].flatten()
                        flat_weights.append(torch.cat([w1, w2]).float().cpu())
                    else:
                        flat_weights.append(flatten_expert_weights(experts[e]))
                
                calib_hidden, calib_router_logits = collect_calibration_activations(
                    model, moe_block, calib_input_ids, calib_attn_mask, batch_size=CALIB_BATCH_SIZE
                )
                
                router_probs = F.softmax(calib_router_logits.float(), dim=-1)
                topk_idx = torch.topk(router_probs, k=top_k, dim=-1).indices
                routed_to = {e: (topk_idx == e).any(dim=-1) for e in range(num_experts)}
                
                # Specialization & Dead Expert Analysis
                layer_spec = {}
                for e in range(num_experts):
                    tokens_routed = routed_to[e].sum().item()
                    mean_weight = router_probs[:, e][routed_to[e]].mean().item() if tokens_routed > 0 else 0.0
                    layer_spec[f"E{e}"] = {
                        "routed_tokens": tokens_routed,
                        "mean_routing_weight": float(mean_weight),
                        "routing_frequency": float(tokens_routed / max(1, router_probs.shape[0]))
                    }
                checkpoint["expert_specialization"][f"{layer_label}_S{seq_len}"] = layer_spec
                
                dead_experts = [e for e, data in layer_spec.items() if data["routing_frequency"] < 0.01]
                if dead_experts: print(f"[WARNING] Layer {layer_label} contains {len(dead_experts)} dead experts.")

                # Phase 0: The Hard Stability Gate
                if seq_len == 256:
                    print(f"[GATE] Executing Split-Sample Stability Test for Layer {layer_label}...")
                    split_idx = 128
                    test_pairs = random.Random(SEED).sample(all_pairs, min(30, len(all_pairs)))
                    oracle_A, oracle_B = [], []
                    
                    for (i, j) in tqdm(test_pairs, desc="Stability Gate Verification"):
                        sA = run_oracle_pair(model, moe_block, i, j, calib_input_ids[:split_idx], calib_attn_mask[:split_idx], CALIB_BATCH_SIZE, top_k)["Oracle_KL"]
                        sB = run_oracle_pair(model, moe_block, i, j, calib_input_ids[split_idx:], calib_attn_mask[split_idx:], CALIB_BATCH_SIZE, top_k)["Oracle_KL"]
                        oracle_A.append(sA)
                        oracle_B.append(sB)
                    
                    gate_rs, _ = spearmanr(oracle_A, oracle_B)
                    checkpoint["split_sample_stability"][f"{layer_label}_S{seq_len}"] = gate_rs
                    
                    if gate_rs < 0.90:
                        print(f"[CRITICAL FAILURE] Oracle Stability r_s = {gate_rs:.4f}. Ground truth is noise. Halting layer analysis.")
                        continue
                    else:
                        print(f"[PASS] Oracle Stability Verified (r_s = {gate_rs:.4f}). Proceeding to proxy mapping.")

                n_eval = min(EVAL_TOKENS_FOR_EXPERT_METRICS, calib_hidden.shape[0])
                eval_hidden = calib_hidden[torch.randperm(calib_hidden.shape[0])[:n_eval]].to(DEVICE, dtype=DTYPE)

                # Dynamically extract expert activations (Handles both Fused Tensors and Modules)
                expert_outputs, expert_activations = {}, {}
                with torch.no_grad():
                    if is_fused:
                        for idx in range(num_experts):
                            gate_up = experts.gate_up_proj.data[idx]
                            down = experts.down_proj.data[idx]
                            
                            gate_up_out = F.linear(eval_hidden, gate_up)
                            gate, up = gate_up_out.chunk(2, dim=-1)
                            act = F.silu(gate) * up
                            out = F.linear(act, down)
                            
                            expert_activations[idx] = act.float().cpu()
                            expert_outputs[idx] = out.float().cpu()
                    else:
                        for idx, expert in enumerate(experts):
                            captured = {}
                            h = expert.register_forward_pre_hook(lambda m, inp, cap=captured: cap.update({"act": inp[0].detach()}))
                            out = expert(eval_hidden)
                            h.remove()
                            expert_outputs[idx] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
                            expert_activations[idx] = captured["act"].detach().float().cpu()

                pairs_to_run = [(i, j) for (i, j) in all_pairs if (seq_len, layer_label, i, j) not in completed_pairs]
                
                for pair_count, (i, j) in enumerate(tqdm(pairs_to_run, desc=f"Proxy/Oracle Mapping")):
                    union = (routed_to[i] | routed_to[j]).sum().item()
                    intersect = (routed_to[i] & routed_to[j]).sum().item()
                    
                    row = {
                        "Seq_Len": seq_len, "Layer": layer_label,
                        "Expert_A": i, "Expert_B": j,
                        "Random_Baseline": static_random_scores[(i, j)],
                        "Weight_Distance": torch.norm(flat_weights[i] - flat_weights[j]).item(),
                        "Weight_Cosine": F.cosine_similarity(flat_weights[i].unsqueeze(0), flat_weights[j].unsqueeze(0)).item(),
                        "Activation_Similarity": F.cosine_similarity(expert_activations[i], expert_activations[j], dim=-1).mean().item(),
                        "Output_Similarity": F.cosine_similarity(expert_outputs[i], expert_outputs[j], dim=-1).mean().item(),
                        "Routing_Similarity": float(pearsonr(router_probs[:, i].numpy(), router_probs[:, j].numpy())[0]),
                        "Usage_Frequency": float(union / max(1, router_probs.shape[0])),
                        "Jaccard_Overlap": float(intersect / max(union, 1))
                    }
                    
                    if seq_len == 256:
                        split_idx = 128
                        stats_A = run_oracle_pair(model, moe_block, i, j, calib_input_ids[:split_idx], calib_attn_mask[:split_idx], CALIB_BATCH_SIZE, top_k)
                        stats_B = run_oracle_pair(model, moe_block, i, j, calib_input_ids[split_idx:], calib_attn_mask[split_idx:], CALIB_BATCH_SIZE, top_k)
                        
                        row["Oracle_KL_SplitA"] = stats_A["Oracle_KL"]
                        row["Oracle_KL_SplitB"] = stats_B["Oracle_KL"]
                        row.update(aggregate_oracle_stats(stats_A, stats_B))
                    else:
                        stats = run_oracle_pair(model, moe_block, i, j, calib_input_ids, calib_attn_mask, CALIB_BATCH_SIZE, top_k)
                        row.update(format_single_oracle_stats(stats))
                        
                    checkpoint["results"].append(row)

                    if (pair_count + 1) % CHECKPOINT_EVERY_N_PAIRS == 0 or _shutdown_requested: save_checkpoint(checkpoint)
                    if _shutdown_requested: break
                
                if _shutdown_requested: break
            if _shutdown_requested: break
    finally:
        save_checkpoint(checkpoint)

    if _shutdown_requested: return

    # ---------------- Statistical Engine & Summaries ----------------
    results_df = pd.DataFrame(checkpoint["results"])
    if results_df.empty: return

    print("\n[ANALYSIS] Computing Oracle Normalization Summaries...")
    for (seq_len, layer), group in results_df.groupby(["Seq_Len", "Layer"]):
        kl = group["Oracle_KL"]
        checkpoint["oracle_summaries"][f"{layer}_S{seq_len}"] = {
            "Total_KL": float(kl.sum()), "Mean_KL": float(kl.mean()), 
            "Median_KL": float(kl.median()), "P95_KL": float(np.percentile(kl, 95))
        }

    print("[ANALYSIS] Computing Bootstrapped Proxy Paired Differences...")
    corr_rows, paired_diffs = [], []
    for (seq_len, layer), group in results_df.groupby(["Seq_Len", "Layer"]):
        for col in METRIC_COLS:
            if group[col].nunique() <= 1: continue
            pear_r, pear_p = pearsonr(group[col], group["Oracle_KL"])
            spear_r, spear_p = spearmanr(group[col], group["Oracle_KL"])
            p_low, p_high = bootstrap_ci(group[col], group["Oracle_KL"], pearsonr)
            s_low, s_high = bootstrap_ci(group[col], group["Oracle_KL"], spearmanr)

            corr_rows.append({
                "Seq_Len": seq_len, "Layer": layer, "Metric": col,
                "Pearson_r": pear_r, "Pearson_95CI_Lower": p_low, "Pearson_95CI_Upper": p_high,
                "Spearman_r": spear_r, "Spearman_95CI_Lower": s_low, "Spearman_95CI_Upper": s_high
            })
            
        m_pairs = [("Output_Similarity", "Weight_Distance"), ("Output_Similarity", "Activation_Similarity"), ("Routing_Similarity", "Random_Baseline")]
        for m1, m2 in m_pairs:
            if m1 in group.columns and m2 in group.columns:
                mean_diff, ci_low, ci_high = bootstrap_paired_diff(group[m1], group[m2], group["Oracle_KL"])
                paired_diffs.append({
                    "Seq_Len": seq_len, "Layer": layer, "Metric_1": m1, "Metric_2": m2,
                    "Delta_Spearman_Mean": mean_diff, "Delta_95CI_Lower": ci_low, "Delta_95CI_Upper": ci_high
                })

    checkpoint["correlations"] = pd.DataFrame(corr_rows).to_dict(orient="records")
    checkpoint["paired_differences"] = pd.DataFrame(paired_diffs).to_dict(orient="records")
    save_checkpoint(checkpoint)
    
    generate_publication_figures(results_df, pd.DataFrame(corr_rows), num_experts)
    
    print(f"\n[EXECUTION COMPLETE] CARE-Oracle-v1.0 Benchmark Finalized -> {os.path.abspath(RESULTS_JSON_PATH)}")

if __name__ == "__main__":
    main()