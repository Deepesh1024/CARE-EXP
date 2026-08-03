"""
CARE-MoE V3 -- Oracle Benchmark Dataset v1.0
============================================
Execution Phase: Split-Sample Stability Verification & Bootstrapped Proxy Correlation.
Target model: allenai/OLMoE-1B-7B-0924.
"""

import os
import json
import time
import copy
import math
import random
import signal
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION & HARDWARE GATES
# ============================================================
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
SEED = 42

CALIB_SIZES = [64, 128, 256, 512]
LAYERS_TO_EVALUATE = ["first", "middle", "last"]

MAX_PAIRS = 50  # Enforced Smoke Test Limit
CALIB_BATCH_SIZE = 2  # Hard-capped to prevent RTX 4090 OOM
EVAL_TOKENS_FOR_EXPERT_METRICS = 4096

GPU_ID = int(os.environ.get("CARE_MOE_GPU_ID", 0))
DEVICE = f"cuda:{GPU_ID}"
DTYPE = torch.bfloat16

CHECKPOINT_EVERY_N_PAIRS = 25
OUTPUT_DIR = "./output"
SCATTER_DIR = os.path.join(OUTPUT_DIR, "scatterplots")
RESULTS_JSON_PATH = os.path.join(OUTPUT_DIR, "output.json")

METRIC_COLS = [
    "Random_Baseline", "Weight_Distance", "Weight_Cosine", 
    "Activation_Similarity", "Output_Similarity", "Routing_Similarity", 
    "Usage_Frequency"
]

_shutdown_requested = False

def _handle_shutdown_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n[SYSTEM] Signal {signum} intercepted. Halting and checkpointing.")

signal.signal(signal.SIGTERM, _handle_shutdown_signal)
signal.signal(signal.SIGINT, _handle_shutdown_signal)


# ============================================================
# UTILITIES, METADATA & BOOTSTRAPPING
# ============================================================
def get_benchmark_metadata():
    return {
        "benchmark_version": "CARE-Oracle-v1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "target_model": MODEL_ID,
        "merge_operator": "Uniform Parameter Averaging (Baseline)",
        "precision": str(DTYPE),
        "seed": SEED,
        "infrastructure_target": "ADAMOPS (Open-Source Framework Integration)",
    }

def load_checkpoint():
    if os.path.exists(RESULTS_JSON_PATH):
        with open(RESULTS_JSON_PATH, "r") as f:
            return json.load(f)
    return {
        "metadata": get_benchmark_metadata(),
        "architecture_summary": {},
        "expert_usage_profiles": {},
        "results": [],
        "correlations": None, 
        "split_sample_stability": {}
    }

def save_checkpoint(data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_path = RESULTS_JSON_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
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
    
    if not stats:
        return np.nan, np.nan
    
    alpha = (1.0 - ci) / 2.0
    return np.percentile(stats, alpha * 100), np.percentile(stats, (1.0 - alpha) * 100)

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
        if hasattr(module, "experts") and isinstance(module.experts, torch.nn.ModuleList):
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
    h2 = gate_module.register_forward_hook(lambda m, inputs, output: router_list.append(output.detach()))

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
def run_oracle_pair(model, moe_block, i, j, merge_operator, input_ids, attention_mask, batch_size, top_k):
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    
    orig_i, orig_j = moe_block.experts[i], moe_block.experts[j]
    merged_expert = merge_operator.merge(orig_i, orig_j)

    gate_module = getattr(moe_block, "gate", None) or getattr(moe_block, "router", None)
    
    total_tokens, kl_sum, ce_orig_sum, ce_merged_sum, l2_hidden_sum = 0, 0.0, 0.0, 0.0, 0.0
    top1_agree_sum, topk_agree_sum, entropy_orig_sum, entropy_merged_sum = 0, 0, 0.0, 0.0

    n = input_ids.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ids = input_ids[start:end].to(DEVICE)
        mask = attention_mask[start:end].to(DEVICE)
        shift_mask, flat_mask = mask[:, 1:].bool(), mask.view(-1).bool()
        
        install_experts(moe_block, [i, j], [orig_i, orig_j])
        orig_moe_out, orig_router_out = [], []
        h_moe = moe_block.register_forward_hook(lambda m, inp, out: orig_moe_out.append(out[0] if isinstance(out, tuple) else out))
        h_gate = gate_module.register_forward_hook(lambda m, inp, out: orig_router_out.append(out))
        
        logits_orig = model(input_ids=ids, attention_mask=mask).logits
        h_moe.remove(); h_gate.remove()
        
        install_experts(moe_block, [i, j], [merged_expert, merged_expert])
        merged_moe_out, merged_router_out = [], []
        h_moe = moe_block.register_forward_hook(lambda m, inp, out: merged_moe_out.append(out[0] if isinstance(out, tuple) else out))
        h_gate = gate_module.register_forward_hook(lambda m, inp, out: merged_router_out.append(out))
        
        logits_merged = model(input_ids=ids, attention_mask=mask).logits
        h_moe.remove(); h_gate.remove()

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

    install_experts(moe_block, [i, j], [orig_i, orig_j])

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
        "Max_VRAM_MB": torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
    }

def aggregate_oracle_stats(stats_A, stats_B):
    total = stats_A["_Total_Tokens"] + stats_B["_Total_Tokens"]
    if total == 0: total = 1
    
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
# MAIN ORCHESTRATION
# ============================================================
def main():
    os.makedirs(SCATTER_DIR, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    checkpoint = load_checkpoint()
    
    print(f"[SYSTEM] Initializing {MODEL_ID} with SDPA Memory Optimization...")
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

    print("[SYSTEM] Fetching Calibration Vectors...")
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in raw["text"] if len(t.strip()) > 0]
    random.Random(SEED).shuffle(texts)

    all_tokens = []
    for t in texts:
        if len(all_tokens) >= max(CALIB_SIZES): break
        enc = tokenizer(t, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        if enc["attention_mask"].sum().item() >= 8: all_tokens.append(enc)

    completed_pairs = {(r["Seq_Len"], r["Layer"], r["Expert_A"], r["Expert_B"]) for r in checkpoint.get("results", [])}

    try:
        for seq_len in CALIB_SIZES:
            print(f"\n--- Engaging Calibration Sequence Size: {seq_len} ---")
            calib_subset = all_tokens[:seq_len]
            calib_input_ids = torch.cat([s["input_ids"] for s in calib_subset], dim=0)
            calib_attn_mask = torch.cat([s["attention_mask"] for s in calib_subset], dim=0)

            for layer_label, layer_idx in layer_indices.items():
                print(f"[PROCESS] Target Layer: {layer_label} (Idx {layer_idx})")
                layer_name, moe_block = moe_layers[layer_idx]
                experts, num_experts = moe_block.experts, len(moe_block.experts)
                
                merge_operator = InPlaceParameterAveragingMerge(experts[0])
                flat_weights = [flatten_expert_weights(e) for e in experts]
                
                calib_hidden, calib_router_logits = collect_calibration_activations(
                    model, moe_block, calib_input_ids, calib_attn_mask, batch_size=CALIB_BATCH_SIZE
                )
                
                router_probs = F.softmax(calib_router_logits.float(), dim=-1)
                topk_idx = torch.topk(router_probs, k=top_k, dim=-1).indices
                routed_to = {e: (topk_idx == e).any(dim=-1) for e in range(num_experts)}
                
                # Phase 7: Expert Usage Profiling
                usage_freq_dist = {e: routed_to[e].float().mean().item() for e in range(num_experts)}
                checkpoint["expert_usage_profiles"][f"{layer_label}_S{seq_len}"] = usage_freq_dist
                dead_experts = [e for e, freq in usage_freq_dist.items() if freq < 0.01]
                if dead_experts:
                    print(f"[WARNING] Layer {layer_label} contains {len(dead_experts)} dead experts (< 1% routing frequency).")
                
                n_eval = min(EVAL_TOKENS_FOR_EXPERT_METRICS, calib_hidden.shape[0])
                eval_hidden = calib_hidden[torch.randperm(calib_hidden.shape[0])[:n_eval]].to(DEVICE, dtype=DTYPE)

                expert_outputs, expert_activations = {}, {}
                with torch.no_grad():
                    for idx, expert in enumerate(experts):
                        captured = {}
                        h = expert.register_forward_pre_hook(lambda m, inp, cap=captured: cap.update({"act": inp[0].detach()}))
                        out = expert(eval_hidden)
                        h.remove()
                        expert_outputs[idx] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
                        expert_activations[idx] = captured["act"].detach().float().cpu()

                all_pairs = list(itertools.combinations(range(num_experts), 2))
                if MAX_PAIRS:
                    all_pairs = random.Random(SEED).sample(all_pairs, min(MAX_PAIRS, len(all_pairs)))

                pairs_to_run = [(i, j) for (i, j) in all_pairs if (seq_len, layer_label, i, j) not in completed_pairs]
                
                for pair_count, (i, j) in enumerate(tqdm(pairs_to_run, desc=f"L={layer_label}, S={seq_len}")):
                    row = {
                        "Seq_Len": seq_len, "Layer": layer_label,
                        "Expert_A": i, "Expert_B": j,
                        "Random_Baseline": np.random.rand(),
                        "Weight_Distance": torch.norm(flat_weights[i] - flat_weights[j]).item(),
                        "Weight_Cosine": F.cosine_similarity(flat_weights[i].unsqueeze(0), flat_weights[j].unsqueeze(0)).item(),
                        "Activation_Similarity": F.cosine_similarity(expert_activations[i], expert_activations[j], dim=-1).mean().item(),
                        "Output_Similarity": F.cosine_similarity(expert_outputs[i], expert_outputs[j], dim=-1).mean().item(),
                        "Routing_Similarity": pearsonr(router_probs[:, i].numpy(), router_probs[:, j].numpy())[0],
                        "Usage_Frequency": (routed_to[i] | routed_to[j]).float().mean().item(),
                    }
                    
                    # Phase 3: Split-Sample Stability Verification at N=256
                    if seq_len == 256:
                        split_idx = 128
                        stats_A = run_oracle_pair(
                            model, moe_block, i, j, merge_operator,
                            calib_input_ids[:split_idx], calib_attn_mask[:split_idx], CALIB_BATCH_SIZE, top_k
                        )
                        stats_B = run_oracle_pair(
                            model, moe_block, i, j, merge_operator,
                            calib_input_ids[split_idx:], calib_attn_mask[split_idx:], CALIB_BATCH_SIZE, top_k
                        )
                        row["Oracle_KL_SplitA"] = stats_A["Oracle_KL"]
                        row["Oracle_KL_SplitB"] = stats_B["Oracle_KL"]
                        
                        # Mathematically exact combination without third forward pass
                        row.update(aggregate_oracle_stats(stats_A, stats_B))
                    else:
                        stats = run_oracle_pair(
                            model, moe_block, i, j, merge_operator,
                            calib_input_ids, calib_attn_mask, CALIB_BATCH_SIZE, top_k
                        )
                        row.update(format_single_oracle_stats(stats))
                        
                    checkpoint["results"].append(row)

                    if (pair_count + 1) % CHECKPOINT_EVERY_N_PAIRS == 0 or _shutdown_requested:
                        save_checkpoint(checkpoint)
                    if _shutdown_requested: break
                
                del merge_operator
                if _shutdown_requested: break
            if _shutdown_requested: break
    finally:
        save_checkpoint(checkpoint)

    if _shutdown_requested: return

    # ---------------- Phase 3 Validation & Bootstrapping ----------------
    results_df = pd.DataFrame(checkpoint["results"])
    if results_df.empty: return
    
    print("\n[ANALYSIS] Validating Phase 3 Oracle Stability Gates...")
    split_stability_report = {}
    stability_failed = False
    
    if 256 in results_df["Seq_Len"].values:
        for layer, group in results_df[results_df["Seq_Len"] == 256].groupby("Layer"):
            if "Oracle_KL_SplitA" in group.columns and "Oracle_KL_SplitB" in group.columns:
                spearman_r, _ = spearmanr(group["Oracle_KL_SplitA"], group["Oracle_KL_SplitB"])
                split_stability_report[layer] = spearman_r
                
                if spearman_r < 0.90:
                    print(f"  [SEVERE] Layer {layer} Stability Failed! Spearman r = {spearman_r:.4f}")
                    stability_failed = True
                else:
                    print(f"  [PASS] Layer {layer} Stability Verified (r = {spearman_r:.4f})")
    
    checkpoint["split_sample_stability"] = split_stability_report
    if stability_failed:
        print("\n[CRITICAL WARNING] Dataset variance is corrupting the Oracle ground truth. Do not proceed to proxy optimization.")

    print("\n[ANALYSIS] Executing Non-Parametric Bootstrapped Correlations...")
    corr_rows = []
    for (seq_len, layer), group in results_df.groupby(["Seq_Len", "Layer"]):
        for col in METRIC_COLS:
            if group[col].nunique() <= 1: continue
            
            pear_r, pear_p = pearsonr(group[col], group["Oracle_KL"])
            spear_r, spear_p = spearmanr(group[col], group["Oracle_KL"])
            p_low, p_high = bootstrap_ci(group[col], group["Oracle_KL"], pearsonr)
            s_low, s_high = bootstrap_ci(group[col], group["Oracle_KL"], spearmanr)

            corr_rows.append({
                "Seq_Len": seq_len, "Layer": layer, "Metric": col,
                "Pearson_r": pear_r, "Pearson_p": pear_p,
                "Pearson_95CI_Lower": p_low, "Pearson_95CI_Upper": p_high,
                "Spearman_r": spear_r, "Spearman_p": spear_p,
                "Spearman_95CI_Lower": s_low, "Spearman_95CI_Upper": s_high
            })
            
    if corr_rows:
        checkpoint["correlations"] = pd.DataFrame(corr_rows).to_dict(orient="records")
        save_checkpoint(checkpoint)
        print("[SYSTEM] Data Checkpointed. Execution Terminated.")

if __name__ == "__main__":
    main()