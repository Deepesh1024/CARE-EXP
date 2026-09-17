import torch
import numpy as np

def compute_capability_vectors(model, moe_blocks, df_tokens, target_layer, active_experts, device="cuda:0"):
    """
    Computes capability vectors for the specified active experts using the existing CARE probe infrastructure.
    Re-implements the logic from experiment6d/capability_probe.py but handles a subset of experts.
    """
    model.eval()
    
    # C is [num_experts, 10]
    num_experts_total = len(moe_blocks[target_layer].experts)
    C = np.zeros((num_experts_total, 10), dtype=np.float32)
    
    SAMPLES_PER_AXIS = 20
    
    from tqdm import tqdm
    
    print(f"Computing Capability Vectors for {len(active_experts)} experts...")
    for axis_idx in tqdm(range(10), desc="Capability Axes"):
        axis_tokens = df_tokens[df_tokens["axis_idx"] == axis_idx].head(SAMPLES_PER_AXIS)
        
        # Cache hidden states for the 20 tokens
        xs = []
        masks = []
        for _, row in axis_tokens.iterrows():
            input_ids = torch.tensor(row['input_ids']).unsqueeze(0).to(device)
            mask = torch.tensor(row['attention_mask']).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
                xs.append(out.hidden_states[target_layer])
                masks.append(mask)
        
        for exp_idx in active_experts:
            axis_norm_sum = 0.0
            for x, mask in zip(xs, masks):
                exp = moe_blocks[target_layer].experts[exp_idx]
                gate = torch.nn.functional.silu(torch.nn.functional.linear(x, exp.gate_proj.weight))
                up = torch.nn.functional.linear(x, exp.up_proj.weight)
                out_e = torch.nn.functional.linear(gate * up, exp.down_proj.weight)
                
                mask_t = mask.unsqueeze(-1)
                out_e = out_e * mask_t
                
                valid_len = mask.sum().item()
                if valid_len > 0:
                    axis_norm_sum += (torch.norm(out_e.float(), p=2, dim=-1).sum(dim=1) / valid_len).item()
                    
            C[exp_idx, axis_idx] = axis_norm_sum / len(axis_tokens)
            
    # L2 normalize the capability vectors as done in typical CARE implementations
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    norms[norms == 0] = 1.0 # Prevent division by zero
    C_normalized = C / norms
    
    return C_normalized

def pairwise_capability_distances(C, active_experts):
    """
    Computes pairwise L2 distances between active experts in capability space.
    """
    distances = {}
    for i in active_experts:
        distances[i] = {}
        for j in active_experts:
            if i != j:
                distances[i][j] = np.linalg.norm(C[i] - C[j])
    return distances
