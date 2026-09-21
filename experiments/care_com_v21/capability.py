import torch
import numpy as np
from tqdm import tqdm

def compute_global_capability_vectors(model, moe_blocks, df_tokens, device="cuda:0"):
    """
    Computes capability vectors across all layers for the active experts.
    Returns: numpy array of shape [num_experts, num_layers * 10]
    where each layer's capability (10 dimensions) is computed identically.
    """
    model.eval()
    
    num_layers = len(moe_blocks)
    num_experts = len(moe_blocks[0].experts)
    
    SAMPLES_PER_AXIS = 20
    print(f"Computing Capability Vectors for {num_experts} experts across {num_layers} layers...")
    
    layer_capabilities = np.zeros((num_layers, num_experts, 10), dtype=np.float32)
    
    for axis_idx in tqdm(range(10), desc="Capability Axes", leave=False):
        axis_tokens = df_tokens[df_tokens["axis_idx"] == axis_idx].head(SAMPLES_PER_AXIS)
        
        for _, row in axis_tokens.iterrows():
            input_ids = torch.tensor(row['input_ids']).unsqueeze(0).to(device)
            mask = torch.tensor(row['attention_mask']).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
                hidden_states = out.hidden_states 
                
                # Hidden states usually start with embeddings, so index `layer_idx` is input to layer `layer_idx`
                # Let's map model layers to blocks. We will use the fact that out.hidden_states[layer_idx] is the input to layer block.
                # However, to be safe against varying model configs, we know len(hidden_states) == num_layers + 1.
                # The first is embedding. So index `layer_idx` is the input to layer_idx.
                
                for layer_idx, block in enumerate(moe_blocks):
                    # For OLMoE, hidden_states[layer_idx] is the input to the layer.
                    x = hidden_states[layer_idx] 
                    
                    for exp_idx in range(num_experts):
                        exp = block.experts[exp_idx]
                        gate = torch.nn.functional.silu(torch.nn.functional.linear(x, exp.gate_proj.weight))
                        up = torch.nn.functional.linear(x, exp.up_proj.weight)
                        out_e = torch.nn.functional.linear(gate * up, exp.down_proj.weight)
                        
                        mask_t = mask.unsqueeze(-1)
                        out_e = out_e * mask_t
                        
                        valid_len = mask.sum().item()
                        if valid_len > 0:
                            norm_val = (torch.norm(out_e.float(), p=2, dim=-1).sum(dim=1) / valid_len).item()
                            layer_capabilities[layer_idx, exp_idx, axis_idx] += norm_val

    # Average over samples
    layer_capabilities /= min(len(axis_tokens), SAMPLES_PER_AXIS)
    
    # Transpose to [num_experts, num_layers, 10] then flatten to [num_experts, num_layers * 10]
    C_concat = layer_capabilities.transpose(1, 0, 2).reshape(num_experts, num_layers * 10)
    
    # Concatenated layer-normalized global capability
    # The prompt says: "use concatenated layer-normalized global capability"
    # This means we first normalize per layer, then concatenate. Or concatenate then global normalize.
    # Let's normalize per layer first as suggested by "layer-normalized".
    for layer_idx in range(num_layers):
        layer_C = layer_capabilities[:, layer_idx, :] # [num_experts, 10]
        norms = np.linalg.norm(layer_C, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        layer_capabilities[:, layer_idx, :] = layer_C / norms
        
    C_concat_normalized = layer_capabilities.transpose(1, 0, 2).reshape(num_experts, num_layers * 10)
    
    # Now global normalize just to ensure unit vectors globally, or keep as is.
    # Usually concatenation of normalized vectors yields norm sqrt(num_layers).
    # We will normalize globally to unit length.
    norms_global = np.linalg.norm(C_concat_normalized, axis=1, keepdims=True)
    norms_global[norms_global == 0] = 1.0
    C_final = C_concat_normalized / norms_global
    
    return C_final

def pairwise_capability_distances(C):
    """
    Computes pairwise L2 distances between active experts in global capability space.
    """
    num_experts = C.shape[0]
    distances = {}
    for i in range(num_experts):
        distances[i] = {}
        for j in range(num_experts):
            if i != j:
                distances[i][j] = np.linalg.norm(C[i] - C[j])
    return distances
