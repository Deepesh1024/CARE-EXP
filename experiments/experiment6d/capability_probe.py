"""
EXPERIMENT 6D - CAPABILITY PROBE
============================================================
The standardized measurement methodology for extracting
the expert functional capability state (C ∈ R^10).
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, DTYPE

def probe_expert_capability(model, moe_blocks, df_tokens, target_layer, target_expert):
    """
    Measures the 10D functional state of a specific expert using the fixed token pool.
    This exactly replicates the measurement logic from Experiment 6C.
    """
    model.eval()
    c_raw = np.zeros(10, dtype=np.float32)
    
    # We sample a small fixed number of tokens per axis for the intervention measurement.
    # Note: In a production run without limits, we would use the entire df_tokens.
    # We use a deterministic subset here for stability.
    SAMPLES_PER_AXIS = 20 
    
    for axis_idx in range(10):
        axis_tokens = df_tokens[df_tokens["axis_idx"] == axis_idx].head(SAMPLES_PER_AXIS)
        axis_norm_sum = 0.0
        
        for _, row in axis_tokens.iterrows():
            input_ids = torch.tensor(row['input_ids']).unsqueeze(0).to(DEVICE)
            mask = torch.tensor(row['attention_mask']).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
                x = out.hidden_states[target_layer]
                
                exp = moe_blocks[target_layer].experts[target_expert]
                gate = torch.nn.functional.silu(torch.nn.functional.linear(x, exp.gate_proj.weight))
                up = torch.nn.functional.linear(x, exp.up_proj.weight)
                out_e = torch.nn.functional.linear(gate * up, exp.down_proj.weight)
                
                mask_t = mask.unsqueeze(-1)
                out_e = out_e * mask_t
                
                valid_len = mask.sum().item()
                if valid_len > 0:
                    # L2 norm over the hidden dimension, mean over the sequence
                    axis_norm_sum += (torch.norm(out_e.float(), p=2, dim=-1).sum(dim=1) / valid_len).item()
                    
        c_raw[axis_idx] = axis_norm_sum / len(axis_tokens)
        
    return c_raw
