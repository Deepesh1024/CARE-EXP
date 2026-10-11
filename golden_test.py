import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
import copy
import time
import psutil
import os
import numpy as np

def print_memory(tag):
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / (1024 ** 3)
    gpu_mem = torch.cuda.max_memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0
    return f"[{tag}] Memory -> CPU: {cpu_mem:.2f} GB | GPU Peak: {gpu_mem:.2f} GB"

model_id = "Qwen/Qwen3-30B-A3B"

print("Loading config...")
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
max_memory = {0: "40GB", "cpu": "120GB"}

print("Loading model in BF16 (this takes ~16 mins)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    max_memory=max_memory,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.eval()
print(f"Model loaded in {time.time() - t0:.2f}s")
print(print_memory("Post-Load"))

layer_idx = 0
block = model.model.layers[layer_idx].mlp
print(f"\nTargeting Layer {layer_idx}")

torch.manual_seed(42)
np.random.seed(42)
batch_size, seq_len = 10, 32
vocab_size = config.vocab_size
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
attention_mask = torch.ones_like(input_ids, device="cuda")

print("\nCapturing reference logits and probabilities...")
with torch.no_grad():
    baseline_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    baseline_probs = torch.nn.functional.softmax(baseline_logits, dim=-1)

def compute_qwen3_capability_vectors(model, block, input_ids, attention_mask, layer_idx):
    num_experts = block.experts.num_experts
    num_axes = input_ids.shape[0] # use each sequence as an axis
    C = np.zeros((num_experts, num_axes), dtype=np.float32)
    
    for axis_idx in range(num_axes):
        inp = input_ids[axis_idx:axis_idx+1]
        mask = attention_mask[axis_idx:axis_idx+1]
        
        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=mask, output_hidden_states=True)
            x = out.hidden_states[layer_idx]
            
            for exp_idx in range(num_experts):
                gate, up = nn.functional.linear(x, block.experts.gate_up_proj[exp_idx]).chunk(2, dim=-1)
                out_e = nn.functional.linear(block.experts.act_fn(gate) * up, block.experts.down_proj[exp_idx])
                
                mask_t = mask.unsqueeze(-1)
                out_e = out_e * mask_t
                valid_len = mask.sum().item()
                
                if valid_len > 0:
                    norm_val = (torch.norm(out_e.float(), p=2, dim=-1).sum(dim=(0, 1)) / valid_len).item()
                    C[exp_idx, axis_idx] = norm_val
                    
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return C / norms

def pairwise_capability_distances(C):
    num_experts = C.shape[0]
    distances = {}
    for i in range(num_experts):
        distances[i] = {}
        for j in range(num_experts):
            if i != j:
                distances[i][j] = np.linalg.norm(C[i] - C[j])
    return distances

def generate_candidate_pool(distances, k=5):
    pairs = []
    for i in distances:
        for j in distances[i]:
            if i < j:
                pairs.append((i, j, distances[i][j]))
    pairs.sort(key=lambda x: x[2])
    return pairs[:k]

def physical_merge(block, i, j):
    with torch.no_grad():
        block.experts.gate_up_proj.data[i] = (block.experts.gate_up_proj.data[i] + block.experts.gate_up_proj.data[j]) / 2.0
        block.experts.down_proj.data[i] = (block.experts.down_proj.data[i] + block.experts.down_proj.data[j]) / 2.0
        block.gate.weight.data[i] = (block.gate.weight.data[i] + block.gate.weight.data[j]) / 2.0

    n = block.experts.num_experts
    keep_indices = torch.tensor([k for k in range(n) if k != j], device=block.experts.gate_up_proj.device)
    
    block.experts.gate_up_proj = nn.Parameter(block.experts.gate_up_proj.data[keep_indices])
    block.experts.down_proj = nn.Parameter(block.experts.down_proj.data[keep_indices])
    block.gate.weight = nn.Parameter(block.gate.weight.data[keep_indices])
    
    block.experts.num_experts -= 1
    block.gate.num_experts -= 1

def restore_block(block, original_state_dict, original_num_experts):
    block.experts.num_experts = original_num_experts
    block.gate.num_experts = original_num_experts
    block.experts.gate_up_proj = nn.Parameter(torch.empty_like(original_state_dict["experts.gate_up_proj"]))
    block.experts.down_proj = nn.Parameter(torch.empty_like(original_state_dict["experts.down_proj"]))
    block.gate.weight = nn.Parameter(torch.empty_like(original_state_dict["gate.weight"]))
    block.load_state_dict(original_state_dict)

def compute_kl(baseline_probs, model, input_ids, attention_mask):
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.nn.functional.log_softmax(logits, dim=-1)
        kl = torch.nn.functional.kl_div(probs, baseline_probs, reduction='batchmean', log_target=False)
        return kl.item()

print("\nComputing capability vectors (C_t)...")
C_t = compute_qwen3_capability_vectors(model, block, input_ids, attention_mask, layer_idx)
distances = pairwise_capability_distances(C_t)
candidates = generate_candidate_pool(distances, k=5)

print("\nTop 5 Candidates based on CARE distance:")
for rank, (i, j, dist) in enumerate(candidates):
    print(f" {rank+1}. Expert {i} + Expert {j} (Distance: {dist:.6f})")

original_state_dict = copy.deepcopy(block.state_dict())
original_num_experts = block.experts.num_experts

print("\nEvaluating candidates (KL Divergence)...")
results = []
for (i, j, dist) in candidates:
    physical_merge(block, i, j)
    kl = compute_kl(baseline_probs, model, input_ids, attention_mask)
    results.append({"pair": (i, j), "dist": dist, "kl": kl})
    print(f" Candidate ({i}, {j}) -> KL: {kl:.8f}")
    
    restore_block(block, original_state_dict, original_num_experts)

best_candidate = min(results, key=lambda x: x["kl"])
print(f"\nWinning Candidate (Min KL): {best_candidate['pair']} with KL {best_candidate['kl']:.8f}")

print("\nCommitting winning candidate physically (128 -> 127)...")
physical_merge(block, best_candidate["pair"][0], best_candidate["pair"][1])

print("\nPost-merge Validation:")
print(f"  Expert tensors shape: gate_up_proj {block.experts.gate_up_proj.shape}, down_proj {block.experts.down_proj.shape}")
print(f"  Router shape: {block.gate.weight.shape}")
print(f"  Reported num_experts: {block.experts.num_experts}")

with torch.no_grad():
    post_merge_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
print(f"  Forward pass succeeded. NaNs: {torch.isnan(post_merge_logits).any().item()}")

print("\nRecomputing capability state C_{t+1} at 127 experts...")
C_t1 = compute_qwen3_capability_vectors(model, block, input_ids, attention_mask, layer_idx)
print(f"  New capability matrix shape: {C_t1.shape}")

print("\nRestoring model to 128 experts for numerical validation...")
restore_block(block, original_state_dict, original_num_experts)

with torch.no_grad():
    restored_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

diff = torch.abs(baseline_logits.float() - restored_logits.float())
max_err = diff.max().item()
mean_err = diff.mean().item()
rel_err = (diff / (torch.abs(baseline_logits.float()) + 1e-8)).max().item()

print("\nRestoration Numerical Error (BF16 tolerance ~1e-3):")
print(f"  Max Abs Error:  {max_err:.8e}")
print(f"  Mean Abs Error: {mean_err:.8e}")
print(f"  Max Rel Error:  {rel_err:.8e}")

if max_err < 1e-3:
    print("\n-> GOLDEN END-TO-END VALIDATION: PASS")
else:
    print("\n-> GOLDEN END-TO-END VALIDATION: FAIL")
    
print("\n" + print_memory("End"))
