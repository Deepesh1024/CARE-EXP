"""
CARE-MoE V3 -- Oracle Benchmark Dataset v1.0 (Full Production Sweep)
=====================================================================
Target Model: allenai/OLMoE-1B-7B-0924
Objective: Immutable CARE-Oracle-v1.0 benchmark dataset with Hard Stability 
           Gates, Bootstrapped Paired Differences, and Complete Metadata.
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
DEVICE = f"cuda:{GPU_ID}"
DTYPE = torch.bfloat16

CHECKPOINT_EVERY_N_PAIRS = 50
OUTPUT_DIR = "./output"
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
    orig_i, orig_j = moe_block.experts[i], moe_block.experts[j]
    merged_expert = merge_operator.merge(orig_i, orig_j)
    gate_module = getattr(moe_block, "gate", None) or getattr(moe_block, "router", None)
    
    total_tokens, kl_sum, ce_orig_sum, ce_merged_sum, l2_hidden_sum = 0, 0.0, 0.0, 0.0, 0.0
    
    n = input_ids.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ids = input_ids[start:end].to(DEVICE)
        mask = attention_mask[start:end].to(DEVICE)
        shift_mask, flat_mask = mask[:, 1:].bool(), mask.view(-1).bool()
        
        install_experts(moe_block, [i, j], [orig_i, orig_j])
        orig_moe_out = []
        h_moe = moe_block.register_forward_hook(lambda m, inp, out: orig_moe_out.append(out[0] if isinstance(out, tuple) else out))
        logits_orig = model(input_ids=ids, attention_mask=mask).logits
        h_moe.remove()
        
        install_experts(moe_block, [i, j], [merged_expert, merged_expert])
        merged_moe_out = []
        h_moe = moe_block.register_forward_hook(lambda m, inp, out: merged_moe_out.append(out[0] if isinstance(out, tuple) else out))
        logits_merged = model(input_ids=ids, attention_mask=mask).logits
        h_moe.remove()

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

        total_tokens += shift_mask.sum().item()

    install_experts(moe_block, [i, j], [orig_i, orig_j])

    return {
        "Oracle_KL": kl_sum / max(total_tokens, 1),
        "CrossEntropy_Delta": (ce_merged_sum - ce_orig_sum) / max(total_tokens, 1),
        "Hidden_L2_Drift": l2_hidden_sum / max(total_tokens, 1),
        "Runtime_Sec": time.time() - start_time
    }


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
    num_experts = len(moe_layers[0][1].experts)
    
    layer_indices = {}
    if "first" in LAYERS_TO_EVALUATE: layer_indices["first"] = 0
    if "middle" in LAYERS_TO_EVALUATE: layer_indices["middle"] = len(moe_layers) // 2
    if "last" in LAYERS_TO_EVALUATE: layer_indices["last"] = len(moe_layers) - 1

    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in raw["text"] if len(t.strip()) > 0]
    random.Random(SEED).shuffle(texts)

    all_tokens = []
    for t in texts:
        if len(all_tokens) >= max(CALIB_SIZES): break
        enc = tokenizer(t, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        if enc["attention_mask"].sum().item() >= 8: all_tokens.append(enc)

    completed_pairs = {(r["Seq_Len"], r["Layer"], r["Expert_A"], r["Expert_B"]) for r in checkpoint.get("results", [])}
    
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
                merge_operator = InPlaceParameterAveragingMerge(experts[0])
                flat_weights = [flatten_expert_weights(e) for e in experts]
                
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

                # Phase 0: The Hard Stability Gate (Evaluated once per layer prior to proxy generation)
                if seq_len == 256:
                    print(f"[GATE] Executing Split-Sample Stability Test for Layer {layer_label}...")
                    split_idx = 128
                    test_pairs = random.Random(SEED).sample(all_pairs, min(30, len(all_pairs)))
                    oracle_A, oracle_B = [], []
                    
                    for (i, j) in tqdm(test_pairs, desc="Stability Gate Verification"):
                        sA = run_oracle_pair(model, moe_block, i, j, merge_operator, calib_input_ids[:split_idx], calib_attn_mask[:split_idx], CALIB_BATCH_SIZE, top_k)["Oracle_KL"]
                        sB = run_oracle_pair(model, moe_block, i, j, merge_operator, calib_input_ids[split_idx:], calib_attn_mask[split_idx:], CALIB_BATCH_SIZE, top_k)["Oracle_KL"]
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

                expert_outputs, expert_activations = {}, {}
                with torch.no_grad():
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
                        "Routing_Similarity": pearsonr(router_probs[:, i].numpy(), router_probs[:, j].numpy())[0],
                        "Usage_Frequency": float(union / max(1, router_probs.shape[0])),
                        "Jaccard_Overlap": float(intersect / max(union, 1))
                    }
                    
                    row.update(run_oracle_pair(model, moe_block, i, j, merge_operator, calib_input_ids, calib_attn_mask, CALIB_BATCH_SIZE, top_k))
                    checkpoint["results"].append(row)

                    if (pair_count + 1) % CHECKPOINT_EVERY_N_PAIRS == 0 or _shutdown_requested: save_checkpoint(checkpoint)
                    if _shutdown_requested: break
                
                del merge_operator
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
            
        # Execute paired difference mapping for prominent proxy combinations
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
    print(f"\n[EXECUTION COMPLETE] CARE-Oracle-v1.0 Benchmark Finalized -> {os.path.abspath(RESULTS_JSON_PATH)}")

if __name__ == "__main__":
    main()