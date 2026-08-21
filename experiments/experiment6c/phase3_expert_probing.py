"""
EXPERIMENT 6C - PHASE 3: EXPERT FUNCTIONAL PROBING
============================================================
Extracts the expert functional capability vector C_ik.
For each axis k, we feed tokens belonging to k into the expert
and measure its output norm (activation strength), completely
independent of the router.

C_i = [C_i1, ..., C_i10]
C_hat_i = C_i / ||C_i||
"""

import os
import sys
import gc
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIRS, CHECKPOINTS, MODEL_ID, DEVICE, DTYPE, NUM_LAYERS, NUM_EXPERTS, ensure_dirs

class MoEInputHook:
    def __init__(self):
        self.inputs = {}
        self.layer_idx = 0
        
    def hook_fn(self, module, input, output):
        # input[0] is hidden_states before the MoE block
        self.inputs[self.layer_idx] = input[0].detach().cpu()
        self.layer_idx += 1

    def register(self, moe_module):
        return moe_module.register_forward_hook(self.hook_fn)

def probe_experts_for_checkpoint(checkpoint_name, hf_revision, df_tokens):
    print(f"\n[{checkpoint_name}] Probing expert functional vectors...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=hf_revision,
        torch_dtype=DTYPE,
        device_map=DEVICE if DEVICE == "cuda:0" else None
    )
    if DEVICE == "mps":
        model = model.to(DEVICE)
    model.eval()

    # Find MoE blocks
    moe_blocks = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "OlmoeSparseMoeBlock":
            moe_blocks.append(module)
            
    if len(moe_blocks) != NUM_LAYERS:
        print(f"WARNING: Expected {NUM_LAYERS} MoE blocks, found {len(moe_blocks)}")

    # We will accumulate the mean output norm for each expert, per axis
    # C_sum: Dict[layer_idx, Dict[expert_idx, np.zeros(10)]]
    # C_count: np.zeros(10)
    C_sum = {l: {e: np.zeros(10, dtype=np.float32) for e in range(NUM_EXPERTS)} for l in range(NUM_LAYERS)}
    C_count = np.zeros(10, dtype=np.float32)


    print(f"[{checkpoint_name}] Passing {len(df_tokens)} sequences...")
    for idx, row in tqdm(df_tokens.iterrows(), total=len(df_tokens)):
        input_ids = torch.tensor(row['input_ids']).unsqueeze(0).to(DEVICE)
        attention_mask = torch.tensor(row['attention_mask']).unsqueeze(0).to(DEVICE)
        axis_idx = int(row['axis_idx'])
        
        hook = MoEInputHook()
        handles = [hook.register(m) for m in moe_blocks]
        
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
        for h in handles:
            h.remove()
            
        # Process the captured hidden states manually through all experts
        mask = row['attention_mask'] # (seq_len,)
        valid_len = mask.sum()
        
        with torch.no_grad():
            for l in range(NUM_LAYERS):
                moe = moe_blocks[l]
                x = hook.inputs[l][0].to(DEVICE).to(DTYPE) # (seq_len, hidden_size)
                
                # Compute sequentially to avoid OOM from duplicating model weights
                out_all = []
                for e in range(NUM_EXPERTS):
                    exp = moe.experts[e]
                    gate = F.silu(F.linear(x, exp.gate_proj.weight))
                    up = F.linear(x, exp.up_proj.weight)
                    out_e = F.linear(gate * up, exp.down_proj.weight)
                    out_all.append(out_e)
                    
                out = torch.stack(out_all, dim=0) # (num_experts, seq_len, hidden_size)
                
                # Mask out padding tokens
                mask_t = torch.tensor(mask).to(DEVICE).unsqueeze(0).unsqueeze(-1)
                out = out * mask_t
                
                # Compute L2 norm of the output for each token, then mean over valid tokens
                # out_norm: (num_experts, seq_len)
                out_norm = torch.norm(out.float(), p=2, dim=-1)
                mean_norm = out_norm.sum(dim=1) / valid_len # (num_experts,)
                
                mean_norm = mean_norm.detach().cpu().numpy()
            
                for e in range(NUM_EXPERTS):
                    C_sum[l][e][axis_idx] += mean_norm[e]
                
        C_count[axis_idx] += 1
                    
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Finalize C_i
    records = []
    C_matrix = np.zeros((NUM_LAYERS, NUM_EXPERTS, 10))
    layer_zero_counts = {}
    
    for l in range(NUM_LAYERS):
        zero_count = 0
        for e in range(NUM_EXPERTS):
            # C_i raw vector
            c_raw = C_sum[l][e] / C_count # (10,)
            C_matrix[l, e] = c_raw
            
            mag = np.linalg.norm(c_raw)
            if mag == 0:
                zero_count += 1
                
            c_hat = c_raw / mag if mag > 0 else np.zeros(10)
            
            records.append({
                "checkpoint": checkpoint_name,
                "layer_idx": l,
                "expert_idx": e,
                "C_raw": c_raw.tolist(),
                "C_hat": c_hat.tolist(),
                "C_mag": float(mag),
                "probe_count": C_count.tolist()
            })
            
        layer_zero_counts[l] = zero_count
        assert zero_count < NUM_EXPERTS, f"Validation Error: Entire Layer {l} contains zero vectors!"
        
    assert C_matrix.shape == (NUM_LAYERS, NUM_EXPERTS, 10), f"Validation Error: C_matrix shape is {C_matrix.shape}"
    print(f"[{checkpoint_name}] Validation Passed: Matrix shape {C_matrix.shape}, no completely dead layers.")
            
    return records

def generate_validation_report(df_c):
    import hashlib
    import numpy as np
    
    out_path = os.path.join(DIRS["root"], "phase3_validation.md")
    
    with open(out_path, "w") as f:
        f.write("# Phase 3: Expert Probing Validation Report\n\n")
        
        for ckpt in df_c["checkpoint"].unique():
            df_ckpt = df_c[df_c["checkpoint"] == ckpt]
            f.write(f"## Checkpoint: {ckpt}\n\n")
            
            # General Integrity Checks
            f.write("### Data Integrity\n")
            c_raw_all = np.stack(df_ckpt["C_raw"].values)
            has_nans = np.isnan(c_raw_all).any()
            has_infs = np.isinf(c_raw_all).any()
            f.write(f"- Contains NaNs: **{has_nans}**\n")
            f.write(f"- Contains Infs: **{has_infs}**\n\n")
            
            if has_nans or has_infs:
                print(f"[ERROR] NaNs or Infs detected in {ckpt}!")
            
            f.write("### Layer-by-Layer Audit\n\n")
            f.write("| Layer | Coverage | Zero Vectors | Min $\\|C\\|$ | Median $\\|C\\|$ | Mean $\\|C\\|$ | Max $\\|C\\|$ | Mean CV ($C_k$) | Probe Count |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            
            for l in range(NUM_LAYERS):
                df_l = df_ckpt[df_ckpt["layer_idx"] == l]
                num_exp = len(df_l)
                mags = df_l["C_mag"].values
                zero_count = np.sum(mags == 0)
                
                c_mat = np.stack(df_l["C_raw"].values) # (64, 10)
                
                # Discriminability (Coefficient of Variation)
                # Compute variance / mean across experts for each capability axis k
                # Adding a tiny epsilon to avoid division by zero
                eps = 1e-9
                axis_means = np.mean(c_mat, axis=0) # (10,)
                axis_stds = np.std(c_mat, axis=0) # (10,)
                cv_per_axis = axis_stds / (np.abs(axis_means) + eps)
                mean_cv = np.mean(cv_per_axis)
                
                probe_c = df_l.iloc[0]["probe_count"]
                probe_str = f"[{int(probe_c[0])}, ..., {int(probe_c[-1])}]"
                
                f.write(f"| {l} | {num_exp}/{NUM_EXPERTS} | {zero_count} | {np.min(mags):.4f} | {np.median(mags):.4f} | {np.mean(mags):.4f} | {np.max(mags):.4f} | {mean_cv:.4f} | {probe_str} |\n")
            f.write("\n")
            
        h = hashlib.md5(pd.util.hash_pandas_object(df_c.astype(str)).values).hexdigest()
        f.write(f"**Data Checksum (MD5):** `{h}`\n")
        
    print(f"[SUCCESS] Validation report saved to {out_path}")

def main():
    ensure_dirs()
    token_path = os.path.join(DIRS["token_vectors"], "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet")
    if not os.path.exists(token_path):
        print(f"Error: Run phase1 first to generate {token_path}")
        sys.exit(1)
        
    df_tokens = pd.read_parquet(token_path)
    
    all_c_records = []
    
    for ckpt_name, hf_rev in CHECKPOINTS.items():
        records = probe_experts_for_checkpoint(ckpt_name, hf_rev, df_tokens)
        all_c_records.extend(records)
        
    df_c = pd.DataFrame(all_c_records)
    out_path = os.path.join(DIRS["expert_vectors"], "EXP6C_EXPERT_CAPABILITY_VECTORS.parquet")
    df_c.to_parquet(out_path, engine="pyarrow")
    print(f"[SUCCESS] Expert capability vectors saved to {out_path}")
    
    generate_validation_report(df_c)

if __name__ == "__main__":
    main()
