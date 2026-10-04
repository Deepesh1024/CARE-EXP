import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import itertools

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
SEQ_LEN = 512
BATCH_SIZE = 16
LAYER_IDX = 8

def main():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    print("Loading WikiText-2...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    # SPLIT A: First 25% of the data
    total_tokens = encodings.input_ids.size(1)
    split_a_tokens = encodings.input_ids[:, :total_tokens // 4]
    
    num_batches = split_a_tokens.size(1) // (BATCH_SIZE * SEQ_LEN)
    
    u_i = torch.zeros(64, dtype=torch.float64, device="cuda")
    rw_l2_matrix = torch.zeros((64, 64), dtype=torch.float64, device="cuda")
    
    layer_module = model.model.layers[LAYER_IDX].mlp
    
    total_valid_tokens = 0
    
    print("Computing Usage and RW-L2 on Split A...")
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            start_idx = i * BATCH_SIZE * SEQ_LEN
            end_idx = start_idx + BATCH_SIZE * SEQ_LEN
            batch_ids = split_a_tokens[:, start_idx:end_idx].reshape(BATCH_SIZE, SEQ_LEN).cuda()
            
            # We just need the inputs to layer 8 MLP.
            # We can use a forward hook to capture the input to layer 8.
            captured_inputs = []
            def hook(module, args, kwargs):
                captured_inputs.append(args[0].detach())
            
            handle = layer_module.register_forward_pre_hook(hook, with_kwargs=True)
            _ = model(batch_ids)
            handle.remove()
            
            mlp_input = captured_inputs[0] # (B, S, H)
            
            # Compute routing probabilities
            routing_logits = layer_module.gate(mlp_input) # (B, S, 64)
            if isinstance(routing_logits, tuple):
                routing_logits = routing_logits[0]
            routing_probs = F.softmax(routing_logits, dim=-1, dtype=torch.float32) # (B, S, 64)
            
            # Valid tokens mask (all tokens are valid in this packed formulation)
            # Accumulate raw usage (marginal routing mass)
            u_i += routing_probs.sum(dim=(0, 1)).to(torch.float64)
            total_valid_tokens += BATCH_SIZE * SEQ_LEN
            
            # Compute expert outputs. Since this is just a clean pass, we can compute all 64 expert outputs
            # efficiently by flattening the batch.
            flat_input = mlp_input.view(-1, mlp_input.size(-1)) # (B*S, H)
            
            # (B*S, 64, H) - this would be 16 * 512 * 64 * 2048 * 2 bytes = 2.1 GB, completely safe to store
            expert_outputs = torch.zeros((flat_input.size(0), 64, flat_input.size(1)), dtype=torch.bfloat16, device="cuda")
            
            for exp_idx in range(64):
                expert_outputs[:, exp_idx, :] = layer_module.experts[exp_idx](flat_input)
                
            # Now compute distance. 
            # We want \mathbb{E}_x [ P(j|x) ||e_i(x) - e_j(x)||_2^2 ]
            # expert_outputs is (N, 64, H)
            flat_probs = routing_probs.view(-1, 64) # (N, 64)
            
            # We can do this efficiently using broadcasting:
            # ||e_i - e_j||^2 = ||e_i||^2 + ||e_j||^2 - 2 e_i^T e_j
            e_norms = torch.sum(expert_outputs.float()**2, dim=-1) # (N, 64)
            
            # We need the full 64x64 distance matrix for each token
            # Doing it via loop over pairs to avoid OOM for a N x 64 x 64 x H tensor
            expert_outputs_f = expert_outputs.float()
            
            for j in range(64):
                # e_j: (N, H)
                e_j = expert_outputs_f[:, j, :]
                dist_j = e_norms + e_norms[:, j:j+1] - 2 * torch.matmul(expert_outputs_f, e_j.unsqueeze(-1)).squeeze(-1) # (N, 64)
                # Ensure non-negative due to float precision
                dist_j = torch.clamp(dist_j, min=0.0)
                
                # Weight by P(j|x)
                weighted_dist = flat_probs[:, j:j+1] * dist_j # (N, 64)
                rw_l2_matrix[:, j] += weighted_dist.sum(dim=0).to(torch.float64)
                
            del expert_outputs, expert_outputs_f, mlp_input, routing_logits, routing_probs
            torch.cuda.empty_cache()

    # Normalize by total tokens to get expectation
    u_i = u_i / total_valid_tokens
    rw_l2_matrix = rw_l2_matrix / total_valid_tokens
    
    # Save the 2016 symmetric unordered pairs
    results = []
    
    for i, j in itertools.combinations(range(64), 2):
        sym_usage = u_i[i].item() + u_i[j].item()
        # Symmetric RW-L2 is D(i,j) + D(j,i)
        sym_rw_l2 = rw_l2_matrix[i, j].item() + rw_l2_matrix[j, i].item()
        
        results.append({
            "pair": [i, j],
            "usage_sum": sym_usage,
            "usage_i": u_i[i].item(),
            "usage_j": u_i[j].item(),
            "rw_l2_symmetric": sym_rw_l2,
            "rw_l2_i_to_j": rw_l2_matrix[i, j].item(),
            "rw_l2_j_to_i": rw_l2_matrix[j, i].item()
        })
        
    os.makedirs("experiments/experiment9", exist_ok=True)
    with open("experiments/experiment9/controls.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Done! Saved controls to experiments/experiment9/controls.json")

if __name__ == "__main__":
    main()
