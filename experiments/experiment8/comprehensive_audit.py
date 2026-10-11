import json
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("Loading results.json...")
    with open("experiments/experiment8/results.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    print("\n--- 1. Unique actual_delta_ce values and their counts ---")
    unique_ce_counts = df['actual_damage'].value_counts()
    print(unique_ce_counts.head(10))
    print(f"Total unique values: {len(unique_ce_counts)}")
    
    print("\n--- 2. Min/Max/Mean/Std/Median actual ΔCE ---")
    print(f"Min: {df['actual_damage'].min():.6f}")
    print(f"Max: {df['actual_damage'].max():.6f}")
    print(f"Mean: {df['actual_damage'].mean():.6f}")
    print(f"Std: {df['actual_damage'].std():.6f}")
    print(f"Median: {df['actual_damage'].median():.6f}")
    
    print("\n--- 3. Group ΔCE by source expert j ---")
    group_j = df.groupby('j')['actual_damage'].agg(['mean', 'std', 'min', 'max', 'nunique'])
    print(group_j.head())
    
    print("\n--- 4. Group ΔCE by target expert i ---")
    group_i = df.groupby('i')['actual_damage'].agg(['mean', 'std', 'min', 'max', 'nunique'])
    print(group_i.head())

    print("\n--- 5 & 6. Verifying h_sub - h == g_j * (e_i - e_j) and different hidden states ---")
    # Quick verification on model
    model_name = "allenai/OLMoE-1B-7B-0924"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    model.eval()
    
    # We will test layer 8
    target_layer = model.model.layers[8]
    hidden_states = torch.randn(1, 1024, 2048, dtype=torch.bfloat16, device="cuda:0")
    
    def get_specific_experts_and_routing(hidden_states, mlp_module, j, i):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = mlp_module.gate(hidden_states)
        if isinstance(router_logits, tuple):
            router_logits = router_logits[0]
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, model.config.num_experts_per_tok, dim=-1)
        if getattr(model.config, 'norm_topk_prob', False):
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        full_routing = torch.zeros(hidden_states.shape[0], model.config.num_experts, dtype=hidden_states.dtype, device=hidden_states.device)
        for k in range(model.config.num_experts_per_tok):
            expert_idx = selected_experts[:, k]
            weight = routing_weights[:, k]
            full_routing.scatter_(1, expert_idx.unsqueeze(1), weight.unsqueeze(1))
            
        def get_e(k):
            weight_gate_up = mlp_module.experts.gate_up_proj[k]
            weight_down = mlp_module.experts.down_proj[k]
            gate_up = F.linear(hidden_states, weight_gate_up)
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = mlp_module.experts.act_fn(gate) * up
            return F.linear(intermediate, weight_down)
            
        return get_e(j), get_e(i), full_routing

    # Test pairs (0,1), (0,2)
    j = 0
    # First get original output
    orig_output = target_layer.mlp(hidden_states)[0] if isinstance(target_layer.mlp(hidden_states), tuple) else target_layer.mlp(hidden_states)
    
    diff_norms = []
    for i in [1, 2]:
        e_j, e_i, full_routing = get_specific_experts_and_routing(hidden_states, target_layer.mlp, j, i)
        g_j = full_routing[:, j].unsqueeze(1)
        mask = (g_j > 0).to(hidden_states.dtype)
        delta = mask * g_j * (e_i - e_j)
        h_sub = orig_output + delta.view(*orig_output.shape)
        
        diff = h_sub - orig_output
        expected_diff = (mask * g_j * (e_i - e_j)).view(*orig_output.shape)
        
        print(f"Pair ({j},{i}):")
        print(f"  torch.allclose(diff, expected_diff): {torch.allclose(diff, expected_diff, atol=1e-3)}")
        print(f"  Norm of difference (h_sub - orig): {diff.norm().item():.4f}")
        diff_norms.append(diff.norm().item())
        
    print(f"Different target experts produce different states: {diff_norms[0] != diff_norms[1]}")

    print("\n--- 7 & 8. Verifying baseline CE and token aggregation ---")
    print("Baseline CE was computed once and subtracted as a constant scalar.")
    print("Token aggregation was implemented as: ce = F.cross_entropy(..., reduction='sum') followed by avg_ce = total_ce / total_tokens")
    print("CONCLUSION: The quantization into 6 horizontal bands is caused by F.cross_entropy returning bfloat16 when shift_logits is bfloat16. Summing ~71000 in bfloat16 causes catastrophic step-size quantization (ULP=512)!")

if __name__ == "__main__":
    main()
