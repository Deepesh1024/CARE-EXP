import os
import sys
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import CareComV21Config
from .core import PhysicalMergeEngine, compress_adaptive

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiment7b")))
from utils.evaluation import prepare_wikitext_eval_batches

def run_care_com_v21_pilot():
    print("=" * 60)
    print("CARE-COM v2.1 Adaptive Physical Global Expert Compression")
    print("=" * 60)
    
    config = CareComV21Config()
    os.makedirs(os.path.dirname(config.trace_path), exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)
    
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
        
    engine = PhysicalMergeEngine(model)
    
    # 2. Load Calibration Data
    print("Loading Calibration Data...")
    token_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp6c", "token_vectors", "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet")
    if not os.path.exists(token_path):
        token_path = "results/exp6c/token_vectors/EXP6C_TOKEN_CAPABILITY_VECTORS.parquet"
        
    df_tokens = pd.read_parquet(token_path)
    if config.calibration_fraction < 1.0:
        df_tokens = df_tokens.sample(frac=config.calibration_fraction, random_state=config.seed)
        
    # 3. Load Oracle KL Eval Data
    eval_chunks = prepare_wikitext_eval_batches(tokenizer)
    
    # 4. Execute Compression Loop
    print("\nInitiating CARE-COM v2.1 (64 -> 56)...")
    
    compressed_model, trace = compress_adaptive(
        model=model,
        engine=engine,
        df_tokens=df_tokens,
        eval_chunks=eval_chunks,
        config=config
    )
    
    print("\nCompression Complete!")
    print(f"Trace saved to: {config.trace_path}")

if __name__ == "__main__":
    run_care_com_v21_pilot()
