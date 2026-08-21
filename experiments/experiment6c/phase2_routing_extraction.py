"""
EXPERIMENT 6C - PHASE 2: ROUTER-BASED ENVIRONMENT EXTRACTION
============================================================
For every token/chunk, records the actual MoE router behavior.
Constructs the exposure-weighted formulation:
  tau_i = sum_x p(i|x) tau(x) / sum_x p(i|x)
and a Top-K version. Keeps them separate.
"""

import os
import sys
import gc
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIRS, CHECKPOINTS, MODEL_ID, DEVICE, DTYPE, NUM_LAYERS, NUM_EXPERTS, NUM_EXPERTS_PER_TOK, ensure_dirs

class RouterHook:
    def __init__(self):
        self.records = []
        self.layer_idx = 0
        
    def hook_fn(self, module, input, output):
        router_logits = output
        if isinstance(output, tuple):
            router_logits = output[0]
        
        # In OLMoE, it's typically flattened to (batch_size * seq_len, num_experts)
        # Reshape to (batch_size, seq_len, num_experts) if needed, but since batch_size=1, 
        # both (seq_len, num_experts) and (1, seq_len, num_experts) are fine.
        # We will force it to (seq_len, num_experts) for consistency.
        if router_logits.dim() == 3:
            router_logits = router_logits.squeeze(0)
        
        with torch.no_grad():
            probs = torch.softmax(router_logits.float(), dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=NUM_EXPERTS_PER_TOK, dim=-1)
            
            self.records.append({
                "layer_idx": self.layer_idx,
                "router_probs": probs.cpu().numpy(),
                "topk_indices": topk_indices.cpu().numpy(),
            })
            self.layer_idx += 1

    def register(self, router_module):
        return router_module.register_forward_hook(self.hook_fn)

def extract_routing_for_checkpoint(checkpoint_name, hf_revision, df_tokens):
    print(f"\n[{checkpoint_name}] Loading model revision: {hf_revision}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=hf_revision,
        torch_dtype=DTYPE,
        device_map=DEVICE if DEVICE == "cuda:0" else None
    )
    if DEVICE == "mps":
        model = model.to(DEVICE)
    model.eval()

    # Find routers
    moe_blocks = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "OlmoeSparseMoeBlock":
            moe_blocks.append(module)
            
    routers = []
    for moe_block in moe_blocks:
        router = None
        for n, m in moe_block.named_modules():
            if m.__class__.__name__ == "OlmoeTopKRouter" or "gate" in n:
                if hasattr(m, 'weight'):
                    router = m
                    break
        if router is None and hasattr(moe_block, "gate"):
            router = moe_block.gate
        if router is not None:
            routers.append(router)
            
    if len(routers) != NUM_LAYERS:
        print(f"WARNING: Expected {NUM_LAYERS} routers, found {len(routers)}")

    # Initialize environment storage
    # Dict[layer_idx, Dict[expert_idx, {"weighted": np.zeros(10), "topk": np.zeros(10), "weight_sum": 0, "topk_sum": 0}]]
    env = {l: {e: {"weighted_tau": np.zeros(10, dtype=np.float32), 
                   "topk_tau": np.zeros(10, dtype=np.float32),
                   "weight_sum": 0.0,
                   "topk_sum": 0} for e in range(NUM_EXPERTS)} for l in range(NUM_LAYERS)}

    print(f"[{checkpoint_name}] Processing {len(df_tokens)} sequences...")
    for idx, row in tqdm(df_tokens.iterrows(), total=len(df_tokens)):
        input_ids = torch.tensor(row['input_ids']).unsqueeze(0).to(DEVICE)
        attention_mask = torch.tensor(row['attention_mask']).unsqueeze(0).to(DEVICE)
        tau_x = row['tau_x_norm'] # 10D vector
        
        hook = RouterHook()
        handles = [hook.register(r) for r in routers]
        
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
        for h in handles:
            h.remove()
            
        # Process hooks
        # hook.records contains 1 entry per layer
        for record in hook.records:
            l = record["layer_idx"]
            probs = record["router_probs"] # (seq_len, num_experts)
            topk_idx = record["topk_indices"] # (seq_len, top_k)
            mask = row['attention_mask'] # (seq_len,)
            
            # Aggregate over sequence
            for pos in range(len(mask)):
                if mask[pos] == 0:
                    continue
                
                # Weighted Exposure
                for e in range(NUM_EXPERTS):
                    p_val = probs[pos, e]
                    env[l][e]["weighted_tau"] += tau_x * p_val
                    env[l][e]["weight_sum"] += p_val
                    
                # TopK Exposure
                for e in topk_idx[pos]:
                    env[l][e]["topk_tau"] += tau_x
                    env[l][e]["topk_sum"] += 1
                    
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Finalize tau_i for this checkpoint
    # Convert to dataframe
    records = []
    for l in range(NUM_LAYERS):
        for e in range(NUM_EXPERTS):
            d = env[l][e]
            w_tau = d["weighted_tau"] / d["weight_sum"] if d["weight_sum"] > 0 else np.zeros(10)
            t_tau = d["topk_tau"] / d["topk_sum"] if d["topk_sum"] > 0 else np.zeros(10)
            
            records.append({
                "checkpoint": checkpoint_name,
                "layer_idx": l,
                "expert_idx": e,
                "tau_weighted": w_tau.tolist(),
                "tau_topk": t_tau.tolist(),
                "weight_sum": float(d["weight_sum"]),
                "topk_sum": int(d["topk_sum"])
            })
            
    return records

def main():
    ensure_dirs()
    token_path = os.path.join(DIRS["token_vectors"], "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet")
    if not os.path.exists(token_path):
        print(f"Error: Run phase1 first to generate {token_path}")
        sys.exit(1)
        
    df_tokens = pd.read_parquet(token_path)
    
    all_env_records = []
    
    for ckpt_name, hf_rev in CHECKPOINTS.items():
        records = extract_routing_for_checkpoint(ckpt_name, hf_rev, df_tokens)
        all_env_records.extend(records)
        
    df_env = pd.DataFrame(all_env_records)
    out_path = os.path.join(DIRS["routing"], "EXP6C_ROUTING_ENVIRONMENT.parquet")
    df_env.to_parquet(out_path, engine="pyarrow")
    print(f"[SUCCESS] Routing environments saved to {out_path}")

if __name__ == "__main__":
    main()
