"""
care_large_moe_systems/complete_comparative_benchmark_v2/audit_fix_hcsmoe_mcsmoe_k8.py
Scientific Audit & Re-Evaluation Script:
Strictly verifies and enforces top_k = 8 on Qwen3-30B-A3B for:
- HC-SMoE at 64E and 60E
- MC-SMoE at 80E, 64E, and 60E
Evaluates across exactly 32 chunks x 1024 tokens = 32,768 tokens on WikiText-2.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/home/ubuntu/DEEPESH')
from care_large_moe_systems.complete_comparative_benchmark_v2.qwen3_clustering_engines import (
    apply_hcsmoe_qwen3_layer,
    apply_mcsmoe_qwen3_layer
)

MODEL_ID = 'Qwen/Qwen3-30B-A3B'
JSON_PATH = '/home/ubuntu/DEEPESH/care_large_moe_systems/complete_comparative_benchmark_v2/hc_mc_smoe_results.json'

print("="*65, flush=True)
print("AUDIT & VERIFICATION OF HC-SMoE AND MC-SMoE AT STRICT k=8", flush=True)
print("="*65, flush=True)

# 1. Dataset Preparation: Exactly 32 chunks of 1024 tokens = 32,768 tokens
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
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
print(f"Prepared {len(eval_chunks)} chunks of length {seq_len} ({len(eval_chunks)*seq_len} total tokens).", flush=True)

@torch.no_grad()
def evaluate_ppl_and_latency(forward_fn, eval_chunks, batch_size=4):
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
        'latency_sec': elapsed,
        'tokens_per_sec': tok_per_sec,
        'peak_vram_gb': peak_vram,
        'total_tokens': total_tokens
    }

# 2. Load pristine model into CPU RAM
print("\nLoading pristine Qwen3-30B-A3B into CPU RAM...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map='cpu',
    trust_remote_code=True
)
print(f"Pristine model loaded in {time.time()-t0:.2f}s.", flush=True)

pristine_snapshots = [
    {
        'gate_up': l.mlp.experts.gate_up_proj.data.clone(),
        'down': l.mlp.experts.down_proj.data.clone(),
        'router': l.mlp.gate.weight.data.clone()
    }
    for l in model.model.layers
]
print("Pristine snapshots created.", flush=True)

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

# Read existing results
with open(JSON_PATH, 'r') as f:
    results = json.load(f)

# Evaluate on GPU at strict k=8
def evaluate_gpu_k8(model, target_e):
    # Enforce top-k = 8 on ALL layers
    for l in model.model.layers:
        l.mlp.gate.top_k = 8
    model.config.num_experts = target_e
    model.config.num_experts_per_tok = 8
    
    print(f"  Transferring {target_e}-expert model to GPU...", flush=True)
    model.to('cuda')
    torch.cuda.synchronize()
    allocated_gb = torch.cuda.memory_allocated() / (1024**3)
    
    def gpu_forward(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
        
    metrics = evaluate_ppl_and_latency(gpu_forward, eval_chunks, batch_size=4)
    metrics['regime'] = 'Regime A (Pure GPU-Resident)'
    metrics['vram_allocated_gb'] = allocated_gb
    metrics['expert_count'] = target_e
    metrics['top_k'] = 8
    
    # Measure decode throughput
    decode_timings = []
    sample_ids = eval_chunks[0]['input_ids'][:16].unsqueeze(0).to('cuda')
    for _ in range(3):
        torch.cuda.synchronize()
        t_d0 = time.perf_counter()
        with torch.no_grad():
            gen_out = model.generate(
                sample_ids,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True
            )
        torch.cuda.synchronize()
        decode_timings.append((gen_out.shape[1] - sample_ids.shape[1]) / (time.perf_counter() - t_d0))
    metrics['decode_tokens_per_sec_mean'] = sum(decode_timings) / len(decode_timings)
    metrics['decode_tokens_per_sec_std'] = (sum((x - metrics['decode_tokens_per_sec_mean'])**2 for x in decode_timings)/len(decode_timings))**0.5
    
    model.to('cpu')
    torch.cuda.empty_cache()
    return metrics

# 1. Audit MC-SMoE at 80E, 64E, 60E
for target_e in [80, 64, 60]:
    print(f"\n--- [AUDIT RE-EVAL] MC-SMoE at {target_e} experts (STRICT k=8) ---", flush=True)
    restore_pristine_model(model, pristine_snapshots)
    for layer in model.model.layers:
        apply_mcsmoe_qwen3_layer(layer.mlp, target_k=target_e)
    metrics = evaluate_gpu_k8(model, target_e)
    results['MC-SMoE']['expert_sweep'][str(target_e)] = metrics
    print(f"  [RESULT MC-SMoE | {target_e}E | k=8] Verified PPL: {metrics['ppl']:.4f} | Decode: {metrics['decode_tokens_per_sec_mean']:.2f} tok/s | VRAM: {metrics['vram_allocated_gb']:.2f} GB", flush=True)

# 2. Audit HC-SMoE at 64E, 60E
for target_e in [64, 60]:
    print(f"\n--- [AUDIT RE-EVAL] HC-SMoE at {target_e} experts (STRICT k=8) ---", flush=True)
    restore_pristine_model(model, pristine_snapshots)
    for layer in model.model.layers:
        apply_hcsmoe_qwen3_layer(layer.mlp, target_k=target_e)
    metrics = evaluate_gpu_k8(model, target_e)
    results['HC-SMoE']['expert_sweep'][str(target_e)] = metrics
    print(f"  [RESULT HC-SMoE | {target_e}E | k=8] Verified PPL: {metrics['ppl']:.4f} | Decode: {metrics['decode_tokens_per_sec_mean']:.2f} tok/s | VRAM: {metrics['vram_allocated_gb']:.2f} GB", flush=True)

# Save verified JSON
with open(JSON_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*65, flush=True)
print("[AUDIT COMPLETE] Verified hc_mc_smoe_results.json with strict top_k=8!", flush=True)
print("="*65, flush=True)
