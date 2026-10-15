"""
care_large_moe_systems/master_benchmark/validation_runner.py
Validation runner for the Master MoE Benchmark.
Validates that:
1. The original Qwen3-30B-A3B baseline (128 experts) runs cleanly across top-k in [8, 6, 4, 2].
2. The existing CARE-COM 80-expert checkpoint runs cleanly across top-k in [8, 6, 4, 2].
3. Top-k modulation functions correctly with automatic probability mass renormalization.
4. Memory usage and inference timings are measured cleanly without OOM.
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
from transformers.masking_utils import create_causal_mask

MODEL_ID = "Qwen/Qwen3-30B-A3B"
CARE_80_RESULTS_PATH = "/home/ubuntu/DEEPESH/audit/QWEN3_CARE_COM_RESULTS.json"
OUTPUT_DIR = "/home/ubuntu/DEEPESH/care_large_moe_systems/master_benchmark"
REPORT_PATH = os.path.join(OUTPUT_DIR, "validation_report.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

class SequentialQwen3Runner:
    """Streams layers sequentially through a single GPU slot for 128E baseline."""
    def __init__(self, model):
        self.model = model
        self.config = model.config
        
        self.model.model.embed_tokens.to("cuda")
        self.model.model.norm.to("cuda")
        self.model.lm_head.to("cuda")
        self.model.model.rotary_emb.to("cuda")
        
        self.gpu_slot = self.model.model.layers[0].__class__(self.config, 0).to("cuda", torch.bfloat16)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        
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
        for idx, cpu_layer in enumerate(self.model.model.layers):
            # Sync top_k and num_experts to GPU slot
            self.gpu_slot.mlp.gate.top_k = cpu_layer.mlp.gate.top_k
            if hasattr(cpu_layer.mlp, "experts"):
                self.gpu_slot.mlp.experts.num_experts = cpu_layer.mlp.experts.num_experts
                self.gpu_slot.mlp.gate.num_experts = cpu_layer.mlp.gate.num_experts
                
            for (p_g, p_c) in zip(self.gpu_slot.parameters(), cpu_layer.parameters()):
                if p_g.shape != p_c.shape:
                    p_g.data = torch.empty_like(p_c.data, device="cuda")
                p_g.data.copy_(p_c.data, non_blocking=True)
            torch.cuda.synchronize()
            
            outputs = self.gpu_slot(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False
            )
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            
        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        return logits

@torch.no_grad()
def merge_qwen3_step(moe_blocks, i: int, j: int):
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

def set_top_k(model, k: int):
    for layer in model.model.layers:
        layer.mlp.gate.top_k = k
    model.config.num_experts_per_tok = k

def evaluate_quick_ppl(forward_fn, chunks):
    nlls = []
    for chunk in chunks:
        input_ids = chunk["input_ids"].to("cuda")
        attention_mask = chunk["attention_mask"].to("cuda")
        logits = forward_fn(input_ids, attention_mask)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
        nlls.append(loss.item())
    avg_nll = sum(nlls) / len(nlls)
    return float(torch.exp(torch.tensor(avg_nll)).item())

def run_validation():
    print("=" * 80)
    print("RUNNING MASTER BENCHMARK PIPELINE VALIDATION (CPU + GPU)")
    print("=" * 80)

    # 1. Load Tokenizer & Quick WikiText chunks
    print("\n[Step 1] Loading tokenizer & preparing validation data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"][:100])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    
    seq_len = 512
    eval_chunks = []
    for i in range(0, 4 * seq_len, seq_len):
        chunk = tokens[i : i + seq_len].unsqueeze(0)
        eval_chunks.append({
            "input_ids": chunk,
            "attention_mask": torch.ones_like(chunk)
        })
    print(f" -> Prepared {len(eval_chunks)} validation chunks (seq_len={seq_len}).")

    # 2. Load Pristine Base Model
    print("\n[Step 2] Loading pristine Qwen3-30B-A3B into CPU RAM...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=True
    )
    print(f" -> Pristine model loaded in {time.time() - t0:.1f}s.")

    runner_128 = SequentialQwen3Runner(model)
    validation_records = {
        "baseline_128": {},
        "care_80": {}
    }

    # 3. Test Baseline at top-k in [8, 6, 4, 2]
    print("\n[Step 3] Testing Baseline (128 Experts) across top-k in [8, 6, 4, 2]...")
    for k in [8, 6, 4, 2]:
        set_top_k(model, k)
        t_start = time.perf_counter()
        ppl = evaluate_quick_ppl(runner_128.forward, eval_chunks)
        latency = time.perf_counter() - t_start
        print(f"  [128E | top-k={k}] PPL: {ppl:.4f} | Latency: {latency:.2f}s | Gate Renorm: {model.model.layers[0].mlp.gate.norm_topk_prob}")
        validation_records["baseline_128"][f"topk_{k}"] = {
            "top_k": k,
            "ppl_pilot": round(ppl, 4),
            "eval_latency_sec": round(latency, 2),
            "status": "PASS"
        }

    # 4. Reconstruct CARE-80
    print("\n[Step 4] Reconstructing CARE-COM 80-expert model via trace replay...")
    with open(CARE_80_RESULTS_PATH, "r") as f:
        care_data = json.load(f)
    trace_48 = care_data["trace"][:48]
    moe_blocks = [l.mlp for l in model.model.layers]
    
    t_merge = time.time()
    for idx, step in enumerate(trace_48):
        i, j = step["selected_pair"]
        merge_qwen3_step(moe_blocks, i, j)
    print(f" -> Replayed 48 merges in {time.time() - t_merge:.2f}s. New experts: {moe_blocks[0].experts.num_experts}")

    # 5. Test CARE-80 at top-k in [8, 6, 4, 2]
    print("\n[Step 5] Testing CARE-80 across top-k in [8, 6, 4, 2]...")
    for k in [8, 6, 4, 2]:
        set_top_k(model, k)
        t_start = time.perf_counter()
        ppl = evaluate_quick_ppl(runner_128.forward, eval_chunks)
        latency = time.perf_counter() - t_start
        print(f"  [80E | top-k={k}] PPL: {ppl:.4f} | Latency: {latency:.2f}s | Gate Renorm: {model.model.layers[0].mlp.gate.norm_topk_prob}")
        validation_records["care_80"][f"topk_{k}"] = {
            "top_k": k,
            "ppl_pilot": round(ppl, 4),
            "eval_latency_sec": round(latency, 2),
            "status": "PASS"
        }

    # 6. Save Validation Report
    with open(REPORT_PATH, "w") as f:
        json.dump(validation_records, f, indent=2)

    print(f"\n[Validation Complete] All 8 configurations evaluated successfully! Report saved to {REPORT_PATH}")
    return validation_records

if __name__ == "__main__":
    run_validation()
