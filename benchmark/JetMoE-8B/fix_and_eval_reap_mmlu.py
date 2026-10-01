import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import os
import time
from tqdm import tqdm

import lm_eval
from lm_eval.models.huggingface import HFLM

def get_calibration_dataset(tokenizer, num_samples=64, seq_length=512):
    print("\n[REAP] Loading calibration dataset (WikiText-2 Train)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [text for text in dataset["text"] if len(text.strip()) > 100]
    calibration_texts = texts[:num_samples]
    
    encodings = []
    for text in calibration_texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=seq_length, truncation=True)
        encodings.append(inputs["input_ids"])
    return encodings

def compute_reap_scores(model, encodings):
    print("\n[REAP] Computing expert saliency scores...")
    start_time = time.time()
    
    num_layers = 24
    num_experts = 8
    reap_sum = {i: torch.zeros(num_experts, device="cpu") for i in range(num_layers)}
    reap_count = {i: torch.zeros(num_experts, device="cpu") for i in range(num_layers)}
    
    hooks = []
    
    def get_mlp_hook(layer_idx):
        def hook(module, inputs, output):
            hidden_states = inputs[0]
            N_tokens = hidden_states.view(-1, hidden_states.size(-1))
            
            router_logits = module.router.layer(N_tokens)
            probs = torch.softmax(router_logits.float(), dim=-1)
            
            topk_weights, selected_experts = torch.topk(probs, 2, dim=-1)
            
            W_in, W_out = None, None
            for name, param in module.named_parameters():
                if "input_linear" in name: W_in = param
                elif "output_linear" in name: W_out = param
            
            for j in range(num_experts):
                mask = (selected_experts == j).any(dim=-1)
                if not mask.any(): continue
                
                tokens_for_j = N_tokens[mask]
                
                h = F.linear(tokens_for_j, W_in[j])
                gate, up = h.chunk(2, dim=-1)
                h_act = F.silu(gate) * up
                out_j = F.linear(h_act, W_out[j])
                
                norm_j = torch.linalg.norm(out_j.float(), dim=-1).cpu()
                g_j = probs[mask, j].cpu()
                mask_cpu = mask.cpu()
                
                reap_sum[layer_idx][j] += (g_j * norm_j).sum()
                reap_count[layer_idx][j] += mask_cpu.sum().long()
                
        return hook

    print("[REAP] Attaching hooks to MLP layers...")
    for i in range(num_layers):
        mlp_module = model.model.layers[i].mlp
        hooks.append(mlp_module.register_forward_hook(get_mlp_hook(i)))
        
    print(f"[REAP] Running {len(encodings)} calibration sequences...")
    with torch.no_grad():
        for input_ids in tqdm(encodings):
            model(input_ids.cuda())
            
    for h in hooks:
        h.remove()
        
    final_scores = {}
    for i in range(num_layers):
        scores = reap_sum[i] / (reap_count[i] + 1e-9)
        final_scores[i] = scores.tolist()
        
    print(f"[REAP] Saliency computation finished in {time.time() - start_time:.2f}s.")
    return final_scores

def get_masking_hook(removed_indices):
    def hook(module, inputs, output):
        output[..., removed_indices] = -float('inf')
        return output
    return hook

def apply_reap_hooks(model, reap_scores, target_experts):
    hooks = []
    for i in range(24):
        scores = torch.tensor(reap_scores[str(i)] if str(i) in reap_scores else reap_scores[i])
        _, retained = torch.topk(scores, target_experts, largest=True)
        retained = retained.tolist()
        removed = [j for j in range(8) if j not in retained]
        
        layer = model.model.layers[i].mlp.router.layer
        h = layer.register_forward_hook(get_masking_hook(removed))
        hooks.append(h)
    return hooks

def main():
    print("Loading base model for calibration...")
    model = AutoModelForCausalLM.from_pretrained(
        "jetmoe/jetmoe-8b", 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("jetmoe/jetmoe-8b", trust_remote_code=True)
    
    encodings = get_calibration_dataset(tokenizer)
    reap_scores = compute_reap_scores(model, encodings)
        
    for target in [7, 6, 4]:
        print(f"\n=========================================")
        print(f" Evaluating REAP {target} Experts on MMLU ")
        print(f"=========================================\n")
        
        print(f"Applying REAP hooks (target={target})...")
        hooks = apply_reap_hooks(model, reap_scores, target)
        
        print("Running lm-eval...")
        lm_eval_model = HFLM(pretrained=model, backend="causal", batch_size="auto")
        
        results = lm_eval.simple_evaluate(
            model=lm_eval_model,
            tasks=["mmlu"],
            num_fewshot=5,
            batch_size="auto"
        )
        
        acc = results["results"]["mmlu"]["acc,none"]
        print(f"\n>>> REAP {target} MMLU Accuracy: {acc:.4f} <<<\n")
        
        out_dir = f"benchmark_results/JetMoE-8B/multicapability_results/reap_mmlu_fixed"
        os.makedirs(out_dir, exist_ok=True)
        # with open(f"{out_dir}/reap_{target}.json", "w") as f:
        #     json.dump(results, f, indent=4)
            
        for h in hooks:
            h.remove()

if __name__ == "__main__":
    main()
