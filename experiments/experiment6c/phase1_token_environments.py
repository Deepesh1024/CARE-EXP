"""
EXPERIMENT 6C - PHASE 1: TOKEN / CHUNK CAPABILITY VECTORS
==========================================================
Constructs the common 10-dimensional empirical capability/task basis
for tokens (inputs). For each chunk x, constructs tau(x) in R^10.
"""

import os
import gc
import sys
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIRS, CAPABILITY_AXES, MODEL_ID, ensure_dirs

def download_and_vectorize_dataset():
    """
    Downloads the defined subset of MMLU and ARC.
    Constructs the tau(x) 10D capability vector for each text sample.
    """
    ensure_dirs()
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    all_chunks = []
    
    axes_keys = list(CAPABILITY_AXES.keys())
    
    for axis_idx, axis_id in enumerate(axes_keys):
        axis_info = CAPABILITY_AXES[axis_id]
        print(f"Processing {axis_id} from {axis_info['source']}...")
        
        samples_collected = 0
        max_samples = axis_info["max_samples"]
        
        for category in axis_info["categories"]:
            if samples_collected >= max_samples:
                break
                
            try:
                # Load the dataset
                if axis_info["source"] == "cais/mmlu":
                    # MMLU uses category as the config name
                    ds = load_dataset(axis_info["source"], category, split="test")
                elif axis_info["source"] == "ai2_arc":
                    ds = load_dataset(axis_info["source"], category, split="test")
                else:
                    raise ValueError(f"Unknown source {axis_info['source']}")
                
                for item in ds:
                    if samples_collected >= max_samples:
                        break
                        
                    # Construct text depending on dataset format
                    if axis_info["source"] == "cais/mmlu":
                        text = item["question"] + "\nChoices:\n" + "\n".join(item["choices"]) + f"\nAnswer: {item['answer']}"
                    else: # ARC
                        text = item["question"] + "\nChoices:\n" + str(item["choices"]) + f"\nAnswer: {item['answerKey']}"
                    
                    # Tokenize to check length, pad/truncate to 512 for consistent MoE router evaluation
                    tokens = tokenizer(
                        text, 
                        max_length=512, 
                        padding="max_length", 
                        truncation=True, 
                        return_tensors="pt"
                    )
                    
                    if tokens.attention_mask.sum().item() < 16:
                        continue # Skip too short
                        
                    # Create tau(x) 10D vector
                    # We use a hard one-hot assignment for the dataset source category
                    tau_x = np.zeros(10, dtype=np.float32)
                    tau_x[axis_idx] = 1.0
                    
                    all_chunks.append({
                        "chunk_id": f"{axis_id}_{category}_{samples_collected}",
                        "axis_id": axis_id,
                        "axis_idx": axis_idx,
                        "category": category,
                        "source": axis_info["source"],
                        "input_ids": tokens.input_ids[0].numpy(),
                        "attention_mask": tokens.attention_mask[0].numpy(),
                        "tau_x_raw": tau_x,
                        "tau_x_norm": tau_x # Already normalized since it's one-hot
                    })
                    
                    samples_collected += 1
                    
            except Exception as e:
                print(f"  [Warning] Could not load {category} from {axis_info['source']}: {e}")
                
    if len(all_chunks) == 0:
        raise RuntimeError("No data collected! Check dataset connectivity or configuration.")
        
    print(f"Collected total of {len(all_chunks)} evaluation chunks across 10 axes.")
    
    # Save as Parquet via pandas
    df = pd.DataFrame(all_chunks)
    out_path = os.path.join(DIRS["token_vectors"], "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet")
    df.to_parquet(out_path, engine="pyarrow")
    print(f"[SUCCESS] Token capability vectors saved to {out_path}")

if __name__ == "__main__":
    download_and_vectorize_dataset()
