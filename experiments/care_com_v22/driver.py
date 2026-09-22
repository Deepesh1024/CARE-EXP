import os
import json
import torch
import gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.care_com_v21.core import PhysicalMergeEngine
from experiments.care_com_v22.config import CareComV22Config
from experiments.care_com_v22.baselines import run_random_baseline, run_static_baseline
from experiments.care_com_v22.adaptive import run_adaptive_baseline
from experiments.care_com_v22.analysis import generate_analysis

def prepare_data(tokenizer, max_length=1024, num_eval_chunks=32):
    """Loads wikitext for evaluation."""
    print("Loading WikiText...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    
    chunks = []
    for i in range(0, min(len(tokens) - max_length, num_eval_chunks * max_length), max_length):
        chunk = tokens[i : i + max_length]
        chunks.append({
            "input_ids": chunk,
            "attention_mask": torch.ones_like(chunk)
        })
        
    # Also need CARE tokens
    df_tokens = tokens[:4096].unsqueeze(0)
    
    return chunks, df_tokens

def load_fresh_model(config):
    """Loads a fresh uncompressed model."""
    print(f"Loading {config.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=config.device
    )
    model.eval()
    return model

def run_care_com_v22_pilot():
    config = CareComV22Config()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    eval_chunks, df_tokens = prepare_data(tokenizer, max_length=1024, num_eval_chunks=config.max_ppl_batches * config.ppl_batch_size)
    
    df_tokens = df_tokens.to(config.device)
    
    # 1. Random Baselines
    for seed in config.random_seeds:
        model = load_fresh_model(config)
        engine = PhysicalMergeEngine(model)
        
        trace, ppl_log = run_random_baseline(model, engine, eval_chunks, config, seed=seed)
        
        with open(os.path.join(config.trajectories_dir, f"random_seed_{seed}.json"), "w") as f:
            json.dump({"trace": trace, "ppl": ppl_log}, f, indent=2)
            
        del model
        del engine
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        
    # 2. Static v1 CARE-COM Baseline
    model = load_fresh_model(config)
    engine = PhysicalMergeEngine(model)
    
    trace, ppl_log = run_static_baseline(model, engine, df_tokens, eval_chunks, config)
    
    with open(os.path.join(config.trajectories_dir, "static_v1.json"), "w") as f:
        json.dump({"trace": trace, "ppl": ppl_log}, f, indent=2)
        
    del model
    del engine
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. Adaptive v2.1 CARE-COM Baseline
    model = load_fresh_model(config)
    engine = PhysicalMergeEngine(model)
    
    trace, ppl_log = run_adaptive_baseline(model, engine, df_tokens, eval_chunks, config)
    
    with open(os.path.join(config.trajectories_dir, "adaptive_v21.json"), "w") as f:
        json.dump({"trace": trace, "ppl": ppl_log}, f, indent=2)
        
    del model
    del engine
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n[v2.2] All baselines completed. Generating analysis tables and figures...")
    generate_analysis(config)
    
if __name__ == "__main__":
    run_care_com_v22_pilot()
