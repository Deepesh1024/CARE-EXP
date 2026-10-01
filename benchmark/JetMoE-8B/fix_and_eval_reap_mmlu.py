import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

# Create LM-Eval task dynamically
import lm_eval
from lm_eval.models.huggingface import HFLM

def get_masking_hook(removed_indices):
    def hook(module, inputs, output):
        # output is the router logits: [batch, seq_len, num_experts]
        # We set the logits of removed experts to -infinity
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
        
        # Attach forward hook to the router layer to mask logits
        layer = model.model.layers[i].mlp.router.layer
        h = layer.register_forward_hook(get_masking_hook(removed))
        hooks.append(h)
    return hooks

def main():
    print("Loading REAP scores...")
    with open("benchmark_results/JetMoE-8B/reap/reap_scores.json", "r") as f:
        reap_scores = json.load(f)
        
    for target in [7, 6, 4]:
        print(f"\n=========================================")
        print(f" Evaluating REAP {target} Experts on MMLU ")
        print(f"=========================================\n")
        
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            "jetmoe/jetmoe-8b", 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
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
        
        # Save result
        out_dir = f"benchmark_results/JetMoE-8B/multicapability_results/reap_mmlu_fixed"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/reap_{target}.json", "w") as f:
            json.dump(results, f, indent=4)
            
        for h in hooks:
            h.remove()
        
        del model
        del lm_eval_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
