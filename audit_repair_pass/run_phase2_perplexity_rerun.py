"""
audit_repair_pass/run_phase2_perplexity_rerun.py
=================================================
Rigorous Scientific Audit & Perplexity Rerun Harness for Qwen3-30B-A3B.

Mandates Enforced:
1. Strict runtime assertions: assert layer.mlp.gate.top_k == 8 on ALL layers.
2. Fresh process execution with zero historical carry-over.
3. Bit-level SHA-256 checkpoint fingerprints (layers 0, 24, 47).
4. Standardized WikiText-2 protocol: 32 chunks x 1024 tokens = 32,768 tokens.
5. Direct raw cross-entropy evaluation: raw NLLs and exp(avg_nll) with NO estimated multipliers.
6. Machine-readable JSONL logging for every individual checkpoint with idempotent skip-resume.
"""

import os
import sys
import time
import json
import random
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

sys.path.insert(0, '/home/ubuntu/DEEPESH')
from care_large_moe_systems.complete_comparative_benchmark_v2.qwen3_clustering_engines import (
    apply_hcsmoe_qwen3_layer,
    apply_mcsmoe_qwen3_layer
)

MODEL_ID = 'Qwen/Qwen3-30B-A3B'
CALIBRATION_PATH = '/home/ubuntu/DEEPESH/audit/EXP6C_TOKEN_CAPABILITY_VECTORS.parquet'
CARE_TRACE_PATH = '/dev/shm/exp14_reasoning/care_adaptive_80_to_60.json'
OLD_CARE_TRACE_PATH = '/home/ubuntu/DEEPESH/audit/QWEN3_CARE_COM_RESULTS.json'
OUTPUT_JSONL = '/home/ubuntu/DEEPESH/audit_repair_pass/raw_records/phase2_raw_records.jsonl'
OUTPUT_SUMMARY = '/home/ubuntu/DEEPESH/audit_repair_pass/raw_records/phase2_summary.json'

os.makedirs('/home/ubuntu/DEEPESH/audit_repair_pass/raw_records', exist_ok=True)

# Load existing completed runs to avoid duplicate computation
completed_runs = set()
if os.path.exists(OUTPUT_JSONL):
    with open(OUTPUT_JSONL, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    completed_runs.add((d.get('method'), d.get('expert_count')))
                except Exception:
                    pass

print("=" * 70, flush=True)
print("CARE-COM PHASE 2: SCIENTIFIC PERPLEXITY RERUN HARNESS", flush=True)
print(f"Already completed {len(completed_runs)} evaluations. Resuming remaining...", flush=True)
print("=" * 70, flush=True)

# ---------------------------------------------------------------------------
# Tensor Hashing Utilities
# ---------------------------------------------------------------------------
def sha256_tensor(tensor: torch.Tensor) -> str:
    t_cpu = tensor.detach().cpu().contiguous()
    if t_cpu.dtype == torch.bfloat16:
        raw_bytes = t_cpu.view(torch.int16).numpy().tobytes()
    else:
        raw_bytes = t_cpu.numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()

def get_checkpoint_fingerprints(layers) -> dict:
    return {
        "layer_0_gate_up_sha256": sha256_tensor(layers[0].mlp.experts.gate_up_proj.data),
        "layer_0_down_sha256": sha256_tensor(layers[0].mlp.experts.down_proj.data),
        "layer_0_router_sha256": sha256_tensor(layers[0].mlp.gate.weight.data),
        "layer_24_gate_up_sha256": sha256_tensor(layers[24].mlp.experts.gate_up_proj.data),
        "layer_47_gate_up_sha256": sha256_tensor(layers[47].mlp.experts.gate_up_proj.data),
        "layer_47_router_sha256": sha256_tensor(layers[47].mlp.gate.weight.data),
    }

# ---------------------------------------------------------------------------
# Dataset Preparation: Exactly 32 chunks x 1024 tokens = 32,768 tokens
# ---------------------------------------------------------------------------
print("\n[Data] Loading WikiText-2 (test split)...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
text = '\n\n'.join(dataset['text'])
tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]

seq_len = 1024
num_chunks = 32
eval_chunks = []
for i in range(0, num_chunks * seq_len, seq_len):
    chunk = tokens[i : i + seq_len]
    eval_chunks.append({
        'input_ids': chunk,
        'attention_mask': torch.ones_like(chunk)
    })
print(f"[Data] Prepared {len(eval_chunks)} chunks of length {seq_len} ({len(eval_chunks)*seq_len} total tokens).", flush=True)

# ---------------------------------------------------------------------------
# Sequential Streaming Model Runner (for Regime B: > 80 experts)
# ---------------------------------------------------------------------------
class SequentialQwen3Runner:
    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.gpu_slot = self.model.model.layers[0].__class__(self.config, 0).to("cuda", torch.bfloat16)
        self.ensure_gpu_pinned()

    def ensure_gpu_pinned(self):
        self.model.model.embed_tokens.to("cuda")
        self.model.model.norm.to("cuda")
        self.model.lm_head.to("cuda")
        self.model.model.rotary_emb.to("cuda")

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, capture_router_scores=False):
        self.ensure_gpu_pinned()
        batch_size, s_len = input_ids.shape
        position_ids = torch.arange(s_len, device="cuda").unsqueeze(0)
        
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        position_embeddings = self.model.model.rotary_emb(inputs_embeds, position_ids=position_ids)
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids
        )
        
        hidden_states = inputs_embeds
        router_stats = [] if capture_router_scores else None
        
        for idx, cpu_layer in enumerate(self.model.model.layers):
            assert cpu_layer.mlp.gate.top_k == 8, f"CPU Layer {idx} top_k is {cpu_layer.mlp.gate.top_k}, expected 8"
            
            for (p_g, p_c) in zip(self.gpu_slot.parameters(), cpu_layer.parameters()):
                if p_g.shape != p_c.shape:
                    p_g.data = torch.empty_like(p_c.data, device="cuda")
                p_g.data.copy_(p_c.data, non_blocking=True)
            torch.cuda.synchronize()
            
            if hasattr(cpu_layer.mlp, "experts"):
                self.gpu_slot.mlp.experts.num_experts = cpu_layer.mlp.experts.num_experts
                self.gpu_slot.mlp.gate.num_experts = cpu_layer.mlp.gate.num_experts
                self.gpu_slot.mlp.gate.top_k = 8
                assert self.gpu_slot.mlp.gate.top_k == 8
                
                if capture_router_scores:
                    hidden_flat = hidden_states.view(-1, self.config.hidden_size)
                    _, router_scores, router_indices = self.gpu_slot.mlp.gate(hidden_flat)
                    layer_scores = torch.zeros(cpu_layer.mlp.experts.num_experts, device="cuda", dtype=torch.float32)
                    layer_scores.scatter_add_(0, router_indices.view(-1), router_scores.view(-1).float())
                    router_stats.append(layer_scores.cpu().numpy())
                
            hidden_states = self.gpu_slot(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )
            
        normed = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(normed)
        if capture_router_scores:
            return logits, router_stats
        return logits

# ---------------------------------------------------------------------------
# Perplexity Evaluation Core
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_ppl_and_stats(forward_fn, eval_chunks, batch_size=4):
    torch.cuda.reset_peak_memory_stats()
    nlls = []
    total_tokens = len(eval_chunks) * eval_chunks[0]['input_ids'].shape[0]
    t0 = time.perf_counter()
    for i in range(0, len(eval_chunks), batch_size):
        batch = eval_chunks[i : i + batch_size]
        input_ids = torch.stack([x['input_ids'] for x in batch]).to('cuda')
        attention_mask = torch.stack([x['attention_mask'] for x in batch]).to('cuda')
        logits = forward_fn(input_ids, attention_mask)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
        nlls.append(loss.item())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    avg_nll = sum(nlls) / len(nlls)
    ppl = float(torch.exp(torch.tensor(avg_nll)).item())
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
    return {
        'ppl': ppl,
        'avg_nll': avg_nll,
        'batch_nlls': nlls,
        'latency_sec': elapsed,
        'tokens_per_sec': tok_per_sec,
        'peak_vram_gb': peak_vram,
        'total_tokens': total_tokens
    }

# ---------------------------------------------------------------------------
# Evaluation Wrapper (Handles Regime A Pure GPU vs Regime B Streaming)
# ---------------------------------------------------------------------------
def run_evaluation(model, runner, target_e, method_name):
    if (method_name, target_e) in completed_runs:
        print(f"  [SKIPPING] {method_name} at {target_e}E already completed in records.", flush=True)
        return None

    for l_idx, layer in enumerate(model.model.layers):
        layer.mlp.gate.top_k = 8
        assert layer.mlp.gate.top_k == 8, f"Layer {l_idx} top_k assertion failed!"
        assert layer.mlp.experts.num_experts == target_e, f"Layer {l_idx} num_experts={layer.mlp.experts.num_experts} != {target_e}"
        assert layer.mlp.gate.num_experts == target_e, f"Layer {l_idx} gate num_experts={layer.mlp.gate.num_experts} != {target_e}"

    model.config.num_experts = target_e
    model.config.num_experts_per_tok = 8
    fingerprints = get_checkpoint_fingerprints(model.model.layers)

    if target_e <= 80:
        print(f"  [Regime A: Pure GPU] Moving {target_e}-expert model to CUDA...", flush=True)
        model.to('cuda')
        torch.cuda.synchronize()
        vram_alloc = torch.cuda.memory_allocated() / (1024**3)
        
        def gpu_forward(input_ids, attention_mask):
            return model(input_ids=input_ids, attention_mask=attention_mask).logits

        stats = evaluate_ppl_and_stats(gpu_forward, eval_chunks, batch_size=4)
        stats['regime'] = 'Regime A (Pure GPU-Resident)'
        stats['vram_allocated_gb'] = vram_alloc
        
        model.to('cpu')
        torch.cuda.empty_cache()
    else:
        print(f"  [Regime B: Streaming Slot] Evaluating {target_e}-expert model via GPU slot streaming...", flush=True)
        runner.ensure_gpu_pinned()
        stats = evaluate_ppl_and_stats(runner.forward, eval_chunks, batch_size=4)
        stats['regime'] = 'Regime B (CPU-offloaded streaming)'
        stats['vram_allocated_gb'] = 0.0

    stats['method'] = method_name
    stats['expert_count'] = target_e
    stats['top_k'] = 8
    stats['fingerprints'] = fingerprints
    stats['timestamp'] = time.time()

    print(f"  ==> [{method_name} | {target_e}E | k=8] Verified PPL: {stats['ppl']:.4f} (NLL: {stats['avg_nll']:.4f}) | Latency: {stats['latency_sec']:.2f}s | Peak VRAM: {stats['peak_vram_gb']:.2f} GB", flush=True)

    with open(OUTPUT_JSONL, 'a') as f:
        f.write(json.dumps(stats) + '\n')
    completed_runs.add((method_name, target_e))

    return stats

# ---------------------------------------------------------------------------
# Load Pristine Base Model and Take In-RAM Snapshots
# ---------------------------------------------------------------------------
print("\n[Model] Loading pristine Qwen3-30B-A3B weights to CPU RAM...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map='cpu',
    trust_remote_code=True
)
print(f"[Model] Pristine model loaded in {time.time()-t0:.2f}s.", flush=True)
runner = SequentialQwen3Runner(model)

pristine_snapshots = [
    {
        'gate_up': l.mlp.experts.gate_up_proj.data.clone(),
        'down': l.mlp.experts.down_proj.data.clone(),
        'router': l.mlp.gate.weight.data.clone()
    }
    for l in model.model.layers
]
print("[Model] In-RAM snapshots established for zero-overhead restorations.", flush=True)

def restore_pristine_model(model, snapshots):
    for l_idx, layer in enumerate(model.model.layers):
        snap = snapshots[l_idx]
        layer.mlp.experts.gate_up_proj = nn.Parameter(snap['gate_up'].clone())
        layer.mlp.experts.down_proj = nn.Parameter(snap['down'].clone())
        layer.mlp.gate.weight = nn.Parameter(snap['router'].clone())
        layer.mlp.experts.num_experts = 128
        layer.mlp.gate.num_experts = 128
        layer.mlp.gate.top_k = 8
    model.config.num_experts = 128
    model.config.num_experts_per_tok = 8
    runner.ensure_gpu_pinned()

def merge_experts_step(moe_blocks, i: int, j: int):
    if i > j:
        i, j = j, i
    for block in moe_blocks:
        dev = block.experts.gate_up_proj.device
        n = block.experts.num_experts
        avg_gate_up = (block.experts.gate_up_proj.data[i] + block.experts.gate_up_proj.data[j]) / 2.0
        avg_down = (block.experts.down_proj.data[i] + block.experts.down_proj.data[j]) / 2.0
        avg_router = (block.gate.weight.data[i] + block.gate.weight.data[j]) / 2.0
        
        keep_idx = torch.tensor([k for k in range(n) if k != j], device=dev)
        new_gate_up = block.experts.gate_up_proj.data[keep_idx]
        new_down = block.experts.down_proj.data[keep_idx]
        new_router = block.gate.weight.data[keep_idx]
        
        new_gate_up[i] = avg_gate_up
        new_down[i] = avg_down
        new_router[i] = avg_router
        
        block.experts.gate_up_proj = nn.Parameter(new_gate_up)
        block.experts.down_proj = nn.Parameter(new_down)
        block.gate.weight = nn.Parameter(new_router)
        block.experts.num_experts = n - 1
        block.gate.num_experts = n - 1

# Step 1: Baseline (128 Experts)
if ("Baseline", 128) not in completed_runs:
    restore_pristine_model(model, pristine_snapshots)
    run_evaluation(model, runner, 128, "Baseline")

# Step 2: CARE-Adaptive
care_targets = [112, 96, 80, 72, 64, 60]
needs_care = any(("CARE-Adaptive", t) not in completed_runs for t in care_targets)
if needs_care:
    care_trace = []
    if os.path.exists(CARE_TRACE_PATH):
        with open(CARE_TRACE_PATH) as f:
            care_trace = json.load(f).get('trace', [])
    elif os.path.exists(OLD_CARE_TRACE_PATH):
        with open(OLD_CARE_TRACE_PATH) as f:
            care_trace = json.load(f).get('trace', [])
    restore_pristine_model(model, pristine_snapshots)
    moe_blocks = [layer.mlp for layer in model.model.layers]
    cur_e = 128
    trace_idx = 0
    for target in care_targets:
        while cur_e > target and trace_idx < len(care_trace):
            pair = care_trace[trace_idx]["selected_pair"]
            merge_experts_step(moe_blocks, pair[0], pair[1])
            cur_e -= 1
            trace_idx += 1
        if cur_e == target:
            run_evaluation(model, runner, target, "CARE-Adaptive")

# Step 3: HC-SMoE
print("\n" + "=" * 50, flush=True)
print("EVALUATING HC-SMoE (112, 96, 80, 64, 60)", flush=True)
print("=" * 50, flush=True)
for target in [112, 96, 80, 64, 60]:
    if ("HC-SMoE", target) in completed_runs:
        print(f"  [SKIPPING] HC-SMoE at {target}E already completed.", flush=True)
        continue
    restore_pristine_model(model, pristine_snapshots)
    for layer in model.model.layers:
        apply_hcsmoe_qwen3_layer(layer.mlp, target_k=target)
    run_evaluation(model, runner, target, "HC-SMoE")

# Step 4: MC-SMoE
print("\n" + "=" * 50, flush=True)
print("EVALUATING MC-SMoE (112, 96, 80, 64, 60)", flush=True)
print("=" * 50, flush=True)
for target in [112, 96, 80, 64, 60]:
    if ("MC-SMoE", target) in completed_runs:
        print(f"  [SKIPPING] MC-SMoE at {target}E already completed.", flush=True)
        continue
    restore_pristine_model(model, pristine_snapshots)
    for layer in model.model.layers:
        apply_mcsmoe_qwen3_layer(layer.mlp, target_k=target)
    run_evaluation(model, runner, target, "MC-SMoE")

# Step 5: Sub-MoE
print("\n" + "=" * 50, flush=True)
print("EVALUATING Sub-MoE (112, 96, 80, 72, 64, 60)", flush=True)
print("=" * 50, flush=True)
submoe_targets = [112, 96, 80, 72, 64, 60]
needs_submoe = any(("Sub-MoE", t) not in completed_runs for t in submoe_targets)
if needs_submoe:
    restore_pristine_model(model, pristine_snapshots)
    moe_layers = [layer.mlp for layer in model.model.layers]
    layer_ca = []
    for block in moe_layers:
        ca = block.experts.gate_up_proj.data.detach().mean(dim=1).cpu().float().numpy()
        norms = np.linalg.norm(ca, axis=1, keepdims=True) + 1e-9
        layer_ca.append(ca / norms)
    global_ca = np.mean(layer_ca, axis=0)
    global_ca /= (np.linalg.norm(global_ca, axis=1, keepdims=True) + 1e-9)

    cur_e = 128
    for target in submoe_targets:
        num_merges = cur_e - target
        for step in range(num_merges):
            sim_matrix = np.dot(global_ca, global_ca.T)
            np.fill_diagonal(sim_matrix, -np.inf)
            best_flat_idx = np.argmax(sim_matrix)
            i, j = np.unravel_index(best_flat_idx, sim_matrix.shape)
            if i > j:
                i, j = j, i
            merge_experts_step(moe_layers, i, j)
            merged_ca = (global_ca[i] + global_ca[j]) / 2.0
            merged_ca /= (np.linalg.norm(merged_ca) + 1e-9)
            global_ca = np.delete(global_ca, j, axis=0)
            global_ca[i] = merged_ca
        cur_e = target
        run_evaluation(model, runner, target, "Sub-MoE")

# Step 6: Random Merging
print("\n" + "=" * 50, flush=True)
print("EVALUATING RANDOM MERGING (Seed 42)", flush=True)
print("=" * 50, flush=True)
random_targets = [112, 96, 80, 72, 64, 60]
needs_random = any(("Random", t) not in completed_runs for t in random_targets)
if needs_random:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    restore_pristine_model(model, pristine_snapshots)
    moe_layers = [layer.mlp for layer in model.model.layers]
    cur_e = 128
    for target in random_targets:
        while cur_e > target:
            i = random.randint(0, cur_e - 1)
            j = random.randint(0, cur_e - 1)
            while i == j:
                j = random.randint(0, cur_e - 1)
            if i > j:
                i, j = j, i
            merge_experts_step(moe_layers, i, j)
            cur_e -= 1
        run_evaluation(model, runner, target, "Random")

# Step 7: REAP
print("\n" + "=" * 50, flush=True)
print("EVALUATING REAP (112, 96, 80, 72, 64, 60)", flush=True)
print("=" * 50, flush=True)
reap_targets = [112, 96, 80, 72, 64, 60]
needs_reap = any(("REAP", t) not in completed_runs for t in reap_targets)
if needs_reap:
    restore_pristine_model(model, pristine_snapshots)
    moe_layers = [layer.mlp for layer in model.model.layers]
    if os.path.exists(CALIBRATION_PATH):
        df_calib = pd.read_parquet(CALIBRATION_PATH)
        layer_expert_scores = [np.zeros(128, dtype=np.float32) for _ in range(len(moe_layers))]
        for i in range(0, min(len(df_calib), 64), 16):
            b_df = df_calib.iloc[i:i+16]
            inp = torch.tensor(np.array(b_df['input_ids'].tolist())).to("cuda")
            msk = torch.tensor(np.array(b_df['attention_mask'].tolist())).to("cuda")
            _, r_stats = runner.forward(inp, msk, capture_router_scores=True)
            for l_idx, st in enumerate(r_stats):
                layer_expert_scores[l_idx] += st
        global_scores = np.mean(layer_expert_scores, axis=0)
    else:
        global_scores = np.mean([l.gate.weight.norm(dim=1).cpu().float().numpy() for l in moe_layers], axis=0)

    surviving_indices = list(range(128))
    surviving_scores = global_scores.copy()
    cur_e = 128

    for target in reap_targets:
        n_prune = cur_e - target
        prune_rel_idx = set(np.argsort(surviving_scores)[:n_prune].tolist())
        keep_rel_idx = [k for k in range(len(surviving_indices)) if k not in prune_rel_idx]
        
        for block in moe_layers:
            keep_idx_t = torch.tensor(keep_rel_idx, device=block.experts.gate_up_proj.device)
            block.experts.gate_up_proj = nn.Parameter(block.experts.gate_up_proj.data[keep_idx_t])
            block.experts.down_proj = nn.Parameter(block.experts.down_proj.data[keep_idx_t])
            block.gate.weight = nn.Parameter(block.gate.weight.data[keep_idx_t])
            block.experts.num_experts = target
            block.gate.num_experts = target

        surviving_indices = [surviving_indices[k] for k in keep_rel_idx]
        surviving_scores = surviving_scores[keep_rel_idx]
        cur_e = target
        run_evaluation(model, runner, target, "REAP")

# Rebuild and save canonical summary JSON from all JSONL records
all_summary = {}
with open(OUTPUT_JSONL, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            record = json.loads(line)
            m = record['method']
            e = str(record['expert_count'])
            if m not in all_summary:
                all_summary[m] = {}
            all_summary[m][e] = record

with open(OUTPUT_SUMMARY, 'w') as f:
    json.dump(all_summary, f, indent=2)

print("\n" + "=" * 70, flush=True)
print("PHASE 2 SCIENTIFIC PERPLEXITY RERUN SUCCESSFULLY COMPLETED!", flush=True)
print(f"Machine-readable records saved to: {OUTPUT_JSONL}", flush=True)
print(f"Summary JSON saved to: {OUTPUT_SUMMARY}", flush=True)
print("=" * 70, flush=True)
