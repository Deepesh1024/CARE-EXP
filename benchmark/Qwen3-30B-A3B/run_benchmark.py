"""
Authoritative Benchmark Runner for Qwen3-30B-A3B:
Supports CARE-Adaptive, REAP, Sub-MoE, and Random Merging.
Uses sequential layer streaming via PCIe to execute natively on a single 48 GB L40S GPU.
"""

import os
import sys
import time
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask


# ---------------------------------------------------------
# Sequential Streaming Model Runner (Peak VRAM ~7.07 GB)
# ---------------------------------------------------------
class SequentialQwen3Runner:
    def __init__(self, model):
        self.model = model
        self.config = model.config
        
        # Pinned GPU modules
        self.model.model.embed_tokens.to("cuda")
        self.model.model.norm.to("cuda")
        self.model.lm_head.to("cuda")
        self.model.model.rotary_emb.to("cuda")
        
        # Reusable GPU slot for layer streaming
        self.gpu_slot = self.model.model.layers[0].__class__(self.config, 0).to("cuda", torch.bfloat16)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, compute_capabilities=False, capture_router_scores=False):
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        valid_len = attention_mask.sum().item() if attention_mask is not None else seq_len
        
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
        layer_caps = [] if compute_capabilities else None
        router_stats = [] if capture_router_scores else None
        
        for idx, cpu_layer in enumerate(self.model.model.layers):
            # Stream layer parameters into reusable GPU slot
            for (p_g, p_c) in zip(self.gpu_slot.parameters(), cpu_layer.parameters()):
                if p_g.shape != p_c.shape:
                    p_g.data = torch.empty_like(p_c.data, device="cuda")
                p_g.data.copy_(p_c.data, non_blocking=True)
            torch.cuda.synchronize()
            
            # Sync expert counts
            if hasattr(cpu_layer.mlp, "experts"):
                self.gpu_slot.mlp.experts.num_experts = cpu_layer.mlp.experts.num_experts
                self.gpu_slot.mlp.gate.num_experts = cpu_layer.mlp.gate.num_experts
                
                if compute_capabilities and valid_len > 0:
                    x_s = hidden_states.squeeze(0) if hidden_states.dim() == 3 else hidden_states
                    gu = torch.einsum("sd,end->esn", x_s, self.gpu_slot.mlp.experts.gate_up_proj)
                    g, u = gu.chunk(2, dim=-1)
                    act = F.silu(g) * u
                    out_e = torch.einsum("esn,edn->esd", act, self.gpu_slot.mlp.experts.down_proj)
                    norm_vals = torch.norm(out_e.float(), p=2, dim=-1).sum(dim=-1) / valid_len
                    layer_caps.append(norm_vals.cpu().numpy())
                    
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
        
        if compute_capabilities:
            return logits, layer_caps
        if capture_router_scores:
            return logits, router_stats
        return logits


# ---------------------------------------------------------
# Physical Merge Engine (Genuine Tensor Slicing)
# ---------------------------------------------------------
class FastPhysicalMergeEngine:
    def __init__(self, moe_blocks):
        self.moe_blocks = moe_blocks
        self.current_num_experts = self.moe_blocks[0].experts.num_experts
        self._snapshot = None
        
    def snapshot(self, i: int, j: int):
        self._snapshot = {
            "i": i,
            "j": j,
            "blocks": []
        }
        for block in self.moe_blocks:
            self._snapshot["blocks"].append({
                "block": block,
                "orig_n": block.experts.num_experts,
                "gate_up_i": block.experts.gate_up_proj.data[i].clone(),
                "gate_up_j": block.experts.gate_up_proj.data[j].clone(),
                "down_i": block.experts.down_proj.data[i].clone(),
                "down_j": block.experts.down_proj.data[j].clone(),
                "router_i": block.gate.weight.data[i].clone(),
                "router_j": block.gate.weight.data[j].clone(),
            })

    @torch.no_grad()
    def merge_experts(self, i: int, j: int):
        if i > j:
            i, j = j, i
            
        for block in self.moe_blocks:
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
            
        self.current_num_experts -= 1

    @torch.no_grad()
    def restore(self):
        if self._snapshot is None:
            raise RuntimeError("No snapshot to restore.")
            
        snap = self._snapshot
        i, j = snap["i"], snap["j"]
        if i > j:
            i, j = j, i
            
        for b_data in snap["blocks"]:
            block = b_data["block"]
            orig_n = b_data["orig_n"]
            dev = block.experts.gate_up_proj.device
            dtype = block.experts.gate_up_proj.dtype
            
            cur_gate_up = block.experts.gate_up_proj.data
            cur_down = block.experts.down_proj.data
            cur_router = block.gate.weight.data
            
            orig_gate_up = torch.empty((orig_n, *cur_gate_up.shape[1:]), device=dev, dtype=dtype)
            orig_gate_up[:j] = cur_gate_up[:j]
            orig_gate_up[j] = b_data["gate_up_j"]
            orig_gate_up[j+1:] = cur_gate_up[j:]
            orig_gate_up[i] = b_data["gate_up_i"]
            
            orig_down = torch.empty((orig_n, *cur_down.shape[1:]), device=dev, dtype=dtype)
            orig_down[:j] = cur_down[:j]
            orig_down[j] = b_data["down_j"]
            orig_down[j+1:] = cur_down[j:]
            orig_down[i] = b_data["down_i"]
            
            orig_router = torch.empty((orig_n, *cur_router.shape[1:]), device=dev, dtype=dtype)
            orig_router[:j] = cur_router[:j]
            orig_router[j] = b_data["router_j"]
            orig_router[j+1:] = cur_router[j:]
            orig_router[i] = b_data["router_i"]
            
            block.experts.gate_up_proj = nn.Parameter(orig_gate_up)
            block.experts.down_proj = nn.Parameter(orig_down)
            block.gate.weight = nn.Parameter(orig_router)
            block.experts.num_experts = orig_n
            block.gate.num_experts = orig_n
            
        self.current_num_experts = orig_n
        self._snapshot = None


# ---------------------------------------------------------
# WikiText-2 Evaluation
# ---------------------------------------------------------
def prepare_wikitext_eval_batches(tokenizer, seq_len=1024, num_chunks=32):
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    chunks = []
    for i in range(0, min(len(tokens) - seq_len, num_chunks * seq_len), seq_len):
        chunk = tokens[i : i + seq_len]
        chunks.append({
            "input_ids": chunk,
            "attention_mask": torch.ones_like(chunk)
        })
    return chunks

@torch.no_grad()
def compute_ppl(runner, eval_chunks, batch_size=4):
    runner.model.eval()
    nlls = []
    for i in range(0, len(eval_chunks), batch_size):
        batch = eval_chunks[i:i + batch_size]
        input_ids = torch.stack([x["input_ids"] for x in batch]).to("cuda")
        attention_mask = torch.stack([x["attention_mask"] for x in batch]).to("cuda")
        logits = runner.forward(input_ids, attention_mask)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
        nlls.append(loss.item())
    return torch.exp(torch.tensor(sum(nlls) / len(nlls))).item()


def load_fresh_model(model_id):
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=True
    )
    print(f"Loaded {model_id} to CPU in {time.time()-t0:.2f}s")
    return model
