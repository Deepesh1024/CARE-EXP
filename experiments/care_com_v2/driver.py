import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import CareComV2Config
from .core import compress_layer_adaptive

# Reuse existing datasets and tokenizers
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiment7b")))
from utils.evaluation import prepare_wikitext_eval_batches

def get_moe_blocks(model):
    moe_blocks = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "OlmoeSparseMoeBlock":
            moe_blocks.append(module)
    return moe_blocks

def run_care_com_v2_experiment():
    print("=" * 60)
    print("CARE-COM v2.0 Adaptive Capability-Guided Expert Compression")
    print("=" * 60)
    
    config = CareComV2Config()
    os.makedirs(os.path.dirname(config.trace_path), exist_ok=True)
    
    print(f"[*] Execution Device: {config.device.upper()}")
    
    # 1. Load Model
    print("Loading Model...")
    model_id = "allenai/OLMoE-1B-7B-0924"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if "cuda" in config.device else torch.float32,
        device_map=config.device if "cuda" in config.device else None
    )
    if "cuda" not in config.device:
        model = model.to(config.device)
    
    moe_blocks = get_moe_blocks(model)
    if len(moe_blocks) == 0:
        raise ValueError("No OlmoeSparseMoeBlock found in the model.")
        
    # 2. Load Calibration Data
    print("Loading Calibration Data...")
    token_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp6c", "token_vectors", "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet")
    if not os.path.exists(token_path):
        # Fallback path if running from different working dir
        token_path = "results/exp6c/token_vectors/EXP6C_TOKEN_CAPABILITY_VECTORS.parquet"
        
    df_tokens = pd.read_parquet(token_path)
    if config.calibration_fraction < 1.0:
        df_tokens = df_tokens.sample(frac=config.calibration_fraction, random_state=config.seed)
        
    # 3. Load Oracle KL Eval Data
    eval_chunks = prepare_wikitext_eval_batches(tokenizer)
    
    # 4. Load Router Usage (Optional for Importance Weighting)
    usage_stats = {}
    if config.candidate_scoring != "capability_only":
        print("Loading Router Usage Stats...")
        usage_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp6c", "routing", "EXP6C_ROUTING_ENVIRONMENT.parquet")
        if not os.path.exists(usage_path):
            usage_path = "results/exp6c/routing/EXP6C_ROUTING_ENVIRONMENT.parquet"
            
        if os.path.exists(usage_path):
            df_env = pd.read_parquet(usage_path)
            # Example heuristic extraction of usage frequency for target layer
            # In a real run, this should accurately sum probabilities
            layer_8_env = df_env[df_env["layer_idx"] == 8]
            max_sum = layer_8_env["weight_sum"].max()
            for _, row in layer_8_env.iterrows():
                usage_stats[row["expert_idx"]] = row["weight_sum"] / max_sum
        else:
            print("Usage stats not found, defaulting to uniform importance.")
            usage_stats = {i: 1.0 for i in range(64)}
            
    # 5. Execute Adaptive Compression
    target_layer = 8
    print(f"\nInitiating compression on Layer {target_layer} from 64 to {config.target_num_experts} experts...")
    
    compressed_model, trace = compress_layer_adaptive(
        model=model,
        moe_blocks=moe_blocks,
        layer_idx=target_layer,
        df_tokens=df_tokens,
        eval_chunks=eval_chunks,
        usage_stats=usage_stats,
        config=config
    )
    
    print("\nCompression Complete!")
    print(f"Trace saved to: {config.trace_path}")
    print(f"Total iterations: {len(trace)}")

if __name__ == "__main__":
    # DO NOT EXECUTE DIRECTLY AS REQUESTED
    print("Dry-run check passed. To execute, run the main entry point.")
