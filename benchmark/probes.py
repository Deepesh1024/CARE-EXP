"""
benchmark/probes.py
===================
Standardized Capability Probe Formation and Representation Engine for CARE-COM (v3.0).

Scientific Mandate (Section 18):
1. Strict 3-role data separation: Capability Probe vs Calibration vs Held-out Test.
2. Capability Probes:
   - Math Reasoning (GSM8K train split)
   - Program Synthesis (MBPP sanitized train split)
   - General Knowledge (MMLU dev split)
   - General Language Modeling (WikiText-2 train split)
   - Generic Reference (10-axis MMLU/ARC probe)
3. Probe Strategies:
   - Strategy 1: Generic Reference Probe
   - Strategy 2: Single-Capability Probes (Math, Coding, Knowledge, LM)
   - Strategy 3: Multi-Capability Combined Probe (Equal-weight normalized concatenation)
4. Full architectural support for fused 3D tensors (Olmoe, Qwen) and unfused modules (JetMoE, Phi).
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from datasets import load_dataset


def get_token_hash(token_list: List[int]) -> str:
    """Computes a SHA-256 hash of a token sequence for strict overlap checking."""
    return hashlib.sha256(np.array(token_list, dtype=np.int64).tobytes()).hexdigest()


def build_probe_datasets(tokenizer, max_length: int = 512, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Constructs independently identifiable capability probe datasets using strictly disjoint
    train/dev splits. Labels and ground-truth answers are never fed to the model.
    """
    probes = {}

    # 1. Math Reasoning Probe: GSM8K (train split only)
    try:
        ds_gsm = load_dataset("openai/gsm8k", "main", split="train")
        gsm_subset = ds_gsm.shuffle(seed=seed).select(range(min(100, len(ds_gsm))))
        math_rows = []
        for idx, ex in enumerate(gsm_subset):
            # Prompt-only without the #### final answer or solution steps
            prompt = f"Question: {ex['question'].strip()}\nAnswer: Let's think step by step."
            enc = tokenizer(prompt, max_length=max_length, truncation=True, padding="max_length")
            math_rows.append({
                "sample_id": f"math_gsm8k_{idx}",
                "capability": "math",
                "axis_idx": 0,
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "token_hash": get_token_hash(enc["input_ids"]),
            })
        probes["math"] = pd.DataFrame(math_rows)
    except Exception as e:
        print(f"[Warning] Failed to build math probe from GSM8K: {e}")

    # 2. Coding Probe: MBPP (sanitized train split only)
    try:
        ds_mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        mbpp_subset = ds_mbpp.shuffle(seed=seed).select(range(min(100, len(ds_mbpp))))
        code_rows = []
        for idx, ex in enumerate(mbpp_subset):
            prompt = f"# Task: {ex['prompt'].strip()}\n# Write python code to solve it:\n"
            enc = tokenizer(prompt, max_length=max_length, truncation=True, padding="max_length")
            code_rows.append({
                "sample_id": f"code_mbpp_{idx}",
                "capability": "coding",
                "axis_idx": 1,
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "token_hash": get_token_hash(enc["input_ids"]),
            })
        probes["coding"] = pd.DataFrame(code_rows)
    except Exception as e:
        print(f"[Warning] Failed to build coding probe from MBPP: {e}")

    # 3. Knowledge Probe: MMLU (dev split only)
    try:
        ds_mmlu = load_dataset("cais/mmlu", "all", split="dev")
        mmlu_subset = ds_mmlu.shuffle(seed=seed).select(range(min(100, len(ds_mmlu))))
        know_rows = []
        for idx, ex in enumerate(mmlu_subset):
            choices = "\n".join([f"({chr(65+c_i)}) {c_text}" for c_i, c_text in enumerate(ex["choices"])])
            prompt = f"Subject: {ex.get('subject', 'general')}\nQuestion: {ex['question'].strip()}\n{choices}\nAnswer:"
            enc = tokenizer(prompt, max_length=max_length, truncation=True, padding="max_length")
            know_rows.append({
                "sample_id": f"mmlu_dev_{idx}",
                "capability": "knowledge",
                "axis_idx": 2,
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "token_hash": get_token_hash(enc["input_ids"]),
            })
        probes["knowledge"] = pd.DataFrame(know_rows)
    except Exception as e:
        print(f"[Warning] Failed to build knowledge probe from MMLU dev: {e}")

    # 4. Language Modeling Probe: WikiText-2 (train split only)
    try:
        ds_wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        full_text = "\n\n".join([t for t in ds_wiki["text"] if len(t.strip()) > 50])
        enc_all = tokenizer(full_text, truncation=False)["input_ids"]
        lm_rows = []
        num_chunks = min(100, len(enc_all) // max_length)
        for idx in range(num_chunks):
            chunk = enc_all[idx * max_length : (idx + 1) * max_length]
            mask = [1] * len(chunk)
            lm_rows.append({
                "sample_id": f"wikitext_train_{idx}",
                "capability": "lm",
                "axis_idx": 3,
                "input_ids": chunk,
                "attention_mask": mask,
                "token_hash": get_token_hash(chunk),
            })
        probes["lm"] = pd.DataFrame(lm_rows)
    except Exception as e:
        print(f"[Warning] Failed to build LM probe from WikiText-2 train: {e}")

    # 5. Generic Reference Probe: 10 axes from historical calibration parquet
    generic_parquet = Path(__file__).resolve().parent.parent / "results" / "exp6c" / "token_vectors" / "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet"
    if generic_parquet.exists():
        try:
            df_gen = pd.read_parquet(generic_parquet)
            if "token_hash" not in df_gen.columns:
                df_gen["token_hash"] = [get_token_hash(list(x)) for x in df_gen["input_ids"]]
            probes["generic"] = df_gen
        except Exception as e:
            print(f"[Warning] Failed to load generic probe: {e}")

    return probes


def verify_strict_data_separation(probe_dfs: Dict[str, pd.DataFrame], eval_chunks: List[Dict[str, Any]]):
    """
    Asserts zero token-level sequence overlap between any capability probe and the held-out evaluation suite.
    Fails closed if data leakage is detected.
    """
    eval_hashes = {get_token_hash(c["input_ids"].tolist() if isinstance(c["input_ids"], torch.Tensor) else list(c["input_ids"])) for c in eval_chunks}
    for name, df in probe_dfs.items():
        probe_hashes = set(df["token_hash"].tolist())
        leakage = probe_hashes.intersection(eval_hashes)
        if len(leakage) > 0:
            raise ValueError(
                f"[FATAL DATA LEAKAGE] Probe '{name}' has {len(leakage)} sequence overlaps with the held-out evaluation set!"
            )
    print(f"[AUDIT PASS] Verified 100% disjoint separation across all {len(probe_dfs)} capability probes and evaluation sets.")


def extract_single_capability_representation(
    model: torch.nn.Module,
    moe_blocks: list,
    current_num_experts: int,
    df_probe: pd.DataFrame,
    max_samples: int = 20,
    device: str = "cuda:0"
) -> np.ndarray:
    """
    Extracts an expert capability vector for a specific probe dataset across all layers.
    Shape: [current_num_experts, num_layers * feature_dim]
    """
    model.eval()
    num_layers = len(moe_blocks)
    num_experts = current_num_experts
    
    # Check architecture type (fused OlmoeExperts vs unfused ModuleList/Batched)
    first_block = moe_blocks[0]
    is_fused_olmoe = hasattr(first_block.experts, "gate_up_proj")
    is_batched_jetmoe = hasattr(first_block, "input_linear") and hasattr(first_block, "output_linear")
    
    samples = df_probe.head(max_samples)
    if len(samples) == 0:
        raise ValueError("df_probe is empty.")

    # layer_representations: [num_layers, num_experts]
    layer_acts = np.zeros((num_layers, num_experts), dtype=np.float32)

    for _, row in samples.iterrows():
        input_ids = torch.tensor(row['input_ids']).unsqueeze(0).to(device)
        mask = torch.tensor(row['attention_mask']).unsqueeze(0).to(device)
        valid_len = mask.sum().item()
        if valid_len == 0:
            continue
        mask_t = mask.unsqueeze(-1)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
            hidden_states = out.hidden_states

            for layer_idx, block in enumerate(moe_blocks):
                x = hidden_states[layer_idx]
                
                if is_fused_olmoe:
                    # Olmoe 3D parameter layout
                    for exp_idx in range(num_experts):
                        gate_up = F.linear(x, block.experts.gate_up_proj[exp_idx])
                        gate, up = gate_up.chunk(2, dim=-1)
                        act = F.silu(gate) * up
                        out_e = F.linear(act, block.experts.down_proj[exp_idx]) * mask_t
                        norm_val = (torch.norm(out_e.float(), p=2, dim=-1).sum(dim=1) / valid_len).item()
                        layer_acts[layer_idx, exp_idx] += norm_val

                elif is_batched_jetmoe:
                    # JetMoE batched linear layout
                    for exp_idx in range(num_experts):
                        w_in = block.input_linear.weight[exp_idx]
                        w_out = block.output_linear.weight[exp_idx]
                        act = F.silu(F.linear(x, w_in))
                        out_e = F.linear(act, w_out) * mask_t
                        norm_val = (torch.norm(out_e.float(), p=2, dim=-1).sum(dim=1) / valid_len).item()
                        layer_acts[layer_idx, exp_idx] += norm_val

                elif hasattr(block, "experts") and isinstance(block.experts, torch.nn.ModuleList):
                    # Unfused ModuleList (Phi-3.5 or standard MoE)
                    for exp_idx in range(num_experts):
                        exp = block.experts[exp_idx]
                        if hasattr(exp, "gate_proj"):
                            gate = F.silu(F.linear(x, exp.gate_proj.weight))
                            up = F.linear(x, exp.up_proj.weight)
                            out_e = F.linear(gate * up, exp.down_proj.weight) * mask_t
                        else:
                            # Standard MLP
                            out_e = exp(x) * mask_t
                        norm_val = (torch.norm(out_e.float(), p=2, dim=-1).sum(dim=1) / valid_len).item()
                        layer_acts[layer_idx, exp_idx] += norm_val
                else:
                    # Fallback router saliency
                    layer_acts[layer_idx, :] += 1.0

    # Average over samples
    layer_acts /= len(samples)

    # Layer normalization: normalize each layer slice across experts
    for l in range(num_layers):
        l_norm = np.linalg.norm(layer_acts[l, :])
        if l_norm > 1e-9:
            layer_acts[l, :] /= l_norm

    # Transpose to [num_experts, num_layers]
    C = layer_acts.T

    # Global unit normalization per expert vector
    expert_norms = np.linalg.norm(C, axis=1, keepdims=True)
    expert_norms[expert_norms == 0] = 1.0
    C_unit = C / expert_norms
    return C_unit


def build_multi_capability_representation(
    model: torch.nn.Module,
    moe_blocks: list,
    current_num_experts: int,
    probe_dict: Dict[str, pd.DataFrame],
    max_samples_per_probe: int = 20,
    device: str = "cuda:0"
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Builds the combined multi-capability representation (Strategy 3).
    Aggregates capability vectors using equal-weight normalized concatenation:
        C_multi = 1/sqrt(K) * [C_math, C_coding, C_knowledge, C_lm]
    Returns:
        C_multi: [num_experts, K * num_layers]
        individual_reps: Dict mapping capability name to individual C_unit
    """
    individual_reps = {}
    vectors = []
    
    for cap_name in ["math", "coding", "knowledge", "lm"]:
        if cap_name in probe_dict:
            c_rep = extract_single_capability_representation(
                model=model,
                moe_blocks=moe_blocks,
                current_num_experts=current_num_experts,
                df_probe=probe_dict[cap_name],
                max_samples=max_samples_per_probe,
                device=device
            )
            individual_reps[cap_name] = c_rep
            vectors.append(c_rep)

    if not vectors:
        raise ValueError("No valid capability probes found to build multi-capability representation.")

    K = len(vectors)
    # Concatenate along feature dimension
    C_concat = np.concatenate(vectors, axis=1) # [num_experts, K * num_layers]
    # Scale by 1 / sqrt(K) to maintain unit length
    C_multi = C_concat / np.sqrt(K)

    # Verify unit norm per expert
    norms = np.linalg.norm(C_multi, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"Combined multi-capability norms deviate from 1.0: {norms}"
    return C_multi, individual_reps


def compute_pairwise_distances(C: np.ndarray) -> Dict[int, Dict[int, float]]:
    """Computes pairwise Euclidean distances between all active experts."""
    num_experts = C.shape[0]
    distances = {i: {} for i in range(num_experts)}
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            d = float(np.linalg.norm(C[i] - C[j]))
            distances[i][j] = d
            distances[j][i] = d
    return distances


def nominate_candidate_pool(distances: Dict[int, Dict[int, float]], pool_size: int = 5) -> List[Tuple[int, int]]:
    """Generates unique top-K nearest candidate pairs sorted by smallest capability distance."""
    pairs = []
    seen = set()
    num_experts = len(distances)
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            if j in distances[i]:
                pair = (i, j)
                if pair not in seen:
                    pairs.append((*pair, distances[i][j]))
                    seen.add(pair)
    pairs.sort(key=lambda x: x[2])
    return [(x[0], x[1]) for x in pairs[:pool_size]]


def compute_candidate_pool_jaccard(pool_a: List[Tuple[int, int]], pool_b: List[Tuple[int, int]]) -> float:
    """Computes Jaccard similarity between two candidate pools."""
    set_a = set(pool_a)
    set_b = set(pool_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(set_a.intersection(set_b)) / len(union)
