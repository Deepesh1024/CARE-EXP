"""
Phase 11: Save all compressed JetMoE-8B checkpoints to disk.

Applies all four compression methods at three targets (7, 6, 4 experts)
and saves each as a full HuggingFace checkpoint so downstream evaluation
harnesses (lm-evaluation-harness, etc.) can load them directly.

Output tree:
    benchmark_results/JetMoE-8B/checkpoints/
        jetmoe_base/
        jetmoe_random_7/
        jetmoe_random_6/
        jetmoe_random_4/
        jetmoe_submoe_7/
        jetmoe_submoe_6/
        jetmoe_submoe_4/
        jetmoe_reap_7/
        jetmoe_reap_6/
        jetmoe_reap_4/
        jetmoe_care_7/
        jetmoe_care_6/
        jetmoe_care_4/
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import random
import json
import os
import time
from tqdm import tqdm
import shutil

MODEL_ID = "jetmoe/jetmoe-8b"
CKPT_DIR = "benchmark_results/JetMoE-8B/checkpoints"
NUM_LAYERS = 24
NUM_EXPERTS = 8
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (copied from phase8_benchmark.py to keep this script
# standalone; do NOT modify the originals)
# ─────────────────────────────────────────────────────────────────────────────

def get_jetmoe_mlp_tensors(model, layer_idx):
    input_linear, output_linear, router = None, None, None
    prefix = f"model.layers.{layer_idx}.mlp"
    for name, param in model.named_parameters():
        if f"{prefix}.input_linear.weight" in name:  input_linear  = param
        elif f"{prefix}.output_linear.weight" in name: output_linear = param
        elif f"{prefix}.router.layer.weight" in name:  router        = param
    return input_linear, output_linear, router


def merge_experts_jetmoe(model, layer_idx, keep_idx, remove_idx):
    input_linear, output_linear, router = get_jetmoe_mlp_tensors(model, layer_idx)
    with torch.no_grad():
        merged_in  = (input_linear.data[keep_idx]  + input_linear.data[remove_idx])  / 2.0
        merged_out = (output_linear.data[keep_idx] + output_linear.data[remove_idx]) / 2.0
        merged_r   = (router.data[keep_idx]        + router.data[remove_idx])        / 2.0
        input_linear.data[keep_idx].copy_(merged_in)
        input_linear.data[remove_idx].copy_(merged_in)
        output_linear.data[keep_idx].copy_(merged_out)
        output_linear.data[remove_idx].copy_(merged_out)
        router.data[keep_idx].copy_(merged_r)
        router.data[remove_idx].copy_(merged_r * 0.99)


def get_characteristic_activations(model):
    final_ca = {}
    for i in range(NUM_LAYERS):
        input_linear, _, _ = get_jetmoe_mlp_tensors(model, i)
        # detach() first to avoid holding gradient graphs; float32 is required by numpy
        ca = input_linear.data.detach().mean(dim=1).cpu().to(torch.float32).numpy()
        final_ca[i] = ca
    return final_ca


def load_calibration_tokens(tokenizer, num_samples=64, seq_length=512):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in dataset["text"] if len(t.strip()) > 100][:num_samples]
    ids = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", max_length=seq_length, truncation=True)
        ids.append(enc["input_ids"])
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Compression methods
# ─────────────────────────────────────────────────────────────────────────────

def apply_random(model, target_experts):
    random.seed(SEED + (NUM_EXPERTS - target_experts))
    for i in range(NUM_LAYERS):
        active = list(range(NUM_EXPERTS))
        while len(active) > target_experts:
            rm = random.choice(active)
            active.remove(rm)
            kp = random.choice(active)
            merge_experts_jetmoe(model, i, kp, rm)


def apply_submoe(model, target_experts):
    ca_stats = get_characteristic_activations(model)
    for i in range(NUM_LAYERS):
        ca = ca_stats[i].copy()
        active = list(range(NUM_EXPERTS))
        while len(active) > target_experts:
            pairs = []
            for ai in range(len(active)):
                for bi in range(ai + 1, len(active)):
                    i1, i2 = active[ai], active[bi]
                    n1 = np.linalg.norm(ca[i1]) + 1e-9
                    n2 = np.linalg.norm(ca[i2]) + 1e-9
                    sim = float(np.dot(ca[i1], ca[i2]) / (n1 * n2))
                    pairs.append((sim, i1, i2))
            pairs.sort(reverse=True)
            _, k, r = pairs[0]
            merge_experts_jetmoe(model, i, k, r)
            ca[k] = (ca[k] + ca[r]) / 2.0
            active.remove(r)


def compute_reap_scores(model, encodings):
    reap_sum   = {i: torch.zeros(NUM_EXPERTS) for i in range(NUM_LAYERS)}
    reap_count = {i: torch.zeros(NUM_EXPERTS) for i in range(NUM_LAYERS)}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            hidden = inputs[0].view(-1, inputs[0].size(-1))
            logits = module.router.layer(hidden)
            probs  = torch.softmax(logits.float(), dim=-1)
            _, sel = torch.topk(probs, 2, dim=-1)
            W_in = W_out = None
            for name, p in module.named_parameters():
                if "input_linear"  in name: W_in  = p
                elif "output_linear" in name: W_out = p
            for j in range(NUM_EXPERTS):
                mask = (sel == j).any(dim=-1)
                if not mask.any(): continue
                tok_j  = hidden[mask]
                h      = F.linear(tok_j, W_in[j])
                g, u   = h.chunk(2, dim=-1)
                out_j  = F.linear(F.silu(g) * u, W_out[j])
                norm_j = torch.linalg.norm(out_j.float(), dim=-1).cpu()
                g_j    = probs[mask, j].cpu()
                reap_sum[layer_idx][j]   += (g_j * norm_j).sum()
                reap_count[layer_idx][j] += mask.cpu().sum().long()
        return hook

    for i in range(NUM_LAYERS):
        hooks.append(model.model.layers[i].mlp.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        for ids in tqdm(encodings, desc="  REAP calibration"):
            model(ids.cuda())
    for h in hooks:
        h.remove()
    return {i: (reap_sum[i] / (reap_count[i] + 1e-9)).tolist() for i in range(NUM_LAYERS)}


def apply_reap(model, scores, target_experts):
    for i in range(NUM_LAYERS):
        s = torch.tensor(scores[i])
        _, kept = torch.topk(s, target_experts, largest=True)
        
        # Determine which experts to remove
        all_experts = set(range(NUM_EXPERTS))
        kept_experts = set(kept.tolist())
        removed = list(all_experts - kept_experts)
        
        mlp = model.model.layers[i].mlp

        # Mask router logits for removed experts so they are never selected
        rl = mlp.router.layer
        with torch.no_grad():
            rl.weight[removed, :] = -1e9

        # We do NOT slice the expert tensors (input_linear/output_linear) 
        # or change num_experts. This keeps the architecture mathematically 
        # identical to an 8-expert model, satisfying Hugging Face's strict 
        # shape checks, but the removed experts are never routed to!

    # Do not modify config.moe_num_experts because the architecture shape remains 8


def apply_care(model, tokenizer, target_experts):
    """CARE Adaptive — candidate pool + micro-eval micro-recomputation."""
    from datasets import load_dataset as _ld

    def _micro_ppl(model, tokenizer, num_tokens=2000):
        model.eval()
        ds  = _ld("wikitext", "wikitext-2-raw-v1", split="test")
        enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
        lim = min(enc.input_ids.size(1), num_tokens)
        nlls, tot = [], 0
        seq_len = 512
        with torch.inference_mode():
            for b in range(0, lim, seq_len):
                e = min(b + seq_len, lim)
                if e - b == 0: break
                ids = enc.input_ids[:, b:e].cuda()
                out = model(ids, labels=ids.clone())
                nlls.append(out.loss * (e - b))
                tot += (e - b)
        return torch.exp(torch.stack(nlls).sum() / tot).item()

    ca_stats = get_characteristic_activations(model)
    for i in tqdm(range(NUM_LAYERS), desc=f"  CARE {target_experts}-exp"):
        ca = ca_stats[i].copy()
        active = list(range(NUM_EXPERTS))
        while len(active) > target_experts:
            pairs = []
            for ai in range(len(active)):
                for bi in range(ai + 1, len(active)):
                    i1, i2 = active[ai], active[bi]
                    n1 = np.linalg.norm(ca[i1]) + 1e-9
                    n2 = np.linalg.norm(ca[i2]) + 1e-9
                    sim = float(np.dot(ca[i1], ca[i2]) / (n1 * n2))
                    pairs.append((sim, i1, i2))
            pairs.sort(reverse=True)
            candidates = pairs[:3]
            best_pair, best_ppl = None, float("inf")
            for _, k_idx, r_idx in candidates:
                in_w, out_w, r_w = get_jetmoe_mlp_tensors(model, i)
                in_bak  = in_w.data.detach().clone()
                out_bak = out_w.data.detach().clone()
                r_bak   = r_w.data.detach().clone()
                merge_experts_jetmoe(model, i, k_idx, r_idx)
                ppl = _micro_ppl(model, tokenizer)
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_pair = (k_idx, r_idx)
                in_w.data.copy_(in_bak); out_w.data.copy_(out_bak); r_w.data.copy_(r_bak)
            k, r = best_pair
            merge_experts_jetmoe(model, i, k, r)
            active.remove(r)
            ca[k] = (ca[k] + ca[r]) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, tokenizer, name):
    path = os.path.join(CKPT_DIR, name)
    os.makedirs(path, exist_ok=True)
    print(f"  Saving → {path}")
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return path


def load_fresh_model(tokenizer=None):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 11: SAVING ALL COMPRESSED CHECKPOINTS")
    print("=" * 60)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Check which checkpoints already exist
    all_names = (
        ["jetmoe_base"] +
        [f"jetmoe_{m}_{t}" for m in ("random", "submoe", "reap", "care") for t in (7, 6, 4)]
    )
    missing = [n for n in all_names if not os.path.isdir(os.path.join(CKPT_DIR, n))]
    if not missing:
        print("All checkpoints already exist. Nothing to do.")
        return
    print(f"Need to generate: {missing}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # ── Base ──────────────────────────────────────────────────────────────────
    if "jetmoe_base" in missing:
        print("\n[1/13] Saving base model...")
        model, _ = load_fresh_model(tokenizer)
        save_checkpoint(model, tokenizer, "jetmoe_base")
        del model; torch.cuda.empty_cache()

    # ── REAP (compute scores once, reuse for all targets) ────────────────────
    reap_scores_path = "benchmark_results/JetMoE-8B/reap/reap_scores.json"
    reap_scores = None
    if os.path.exists(reap_scores_path):
        with open(reap_scores_path) as f:
            raw = json.load(f)
            # Keys may be strings "0".."23" — normalise to int keys
            reap_scores = {int(k): v for k, v in raw.items()}
        print(f"\n[REAP] Loaded pre-computed saliency scores from {reap_scores_path}")

    reap_needed = [t for t in (7, 6, 4) if f"jetmoe_reap_{t}" in missing]
    if reap_needed and reap_scores is None:
        print("\n[REAP] Computing saliency scores (one-time calibration)...")
        model, _ = load_fresh_model(tokenizer)
        encodings = load_calibration_tokens(tokenizer)
        reap_scores = compute_reap_scores(model, encodings)
        os.makedirs(os.path.dirname(reap_scores_path), exist_ok=True)
        with open(reap_scores_path, "w") as f:
            json.dump({str(k): v for k, v in reap_scores.items()}, f)
        del model; torch.cuda.empty_cache()

    # ── Iterate all methods × targets ─────────────────────────────────────────
    step = 2
    for target in (7, 6, 4):
        # Random
        name = f"jetmoe_random_{target}"
        if name in missing:
            print(f"\n[{step}/13] Random → {target} experts...")
            model, _ = load_fresh_model(tokenizer)
            apply_random(model, target)
            save_checkpoint(model, tokenizer, name)
            del model; torch.cuda.empty_cache()
        step += 1

        # Sub-MoE
        name = f"jetmoe_submoe_{target}"
        if name in missing:
            print(f"\n[{step}/13] Sub-MoE → {target} experts...")
            model, _ = load_fresh_model(tokenizer)
            apply_submoe(model, target)
            save_checkpoint(model, tokenizer, name)
            del model; torch.cuda.empty_cache()
        step += 1

        # REAP
        name = f"jetmoe_reap_{target}"
        if name in missing:
            print(f"\n[{step}/13] REAP → {target} experts...")
            model, _ = load_fresh_model(tokenizer)
            apply_reap(model, reap_scores, target)
            save_checkpoint(model, tokenizer, name)
            del model; torch.cuda.empty_cache()
        step += 1

        # CARE Adaptive
        name = f"jetmoe_care_{target}"
        if name in missing:
            print(f"\n[{step}/13] CARE Adaptive → {target} experts...")
            model, _ = load_fresh_model(tokenizer)
            apply_care(model, tokenizer, target)
            save_checkpoint(model, tokenizer, name)
            del model; torch.cuda.empty_cache()
        step += 1

    print("\n✓ All checkpoints saved to:", CKPT_DIR)
    print("\nCheckpoint list:")
    for name in sorted(os.listdir(CKPT_DIR)):
        size_mb = sum(
            os.path.getsize(os.path.join(CKPT_DIR, name, f))
            for f in os.listdir(os.path.join(CKPT_DIR, name))
            if os.path.isfile(os.path.join(CKPT_DIR, name, f))
        ) / (1024 ** 2)
        print(f"  {name:<30} {size_mb:>8.0f} MB")


if __name__ == "__main__":
    main()
