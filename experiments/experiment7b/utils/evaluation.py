"""
EXPERIMENT 7B - EVALUATION PIPELINE
===================================
Provides evaluation functions for:
1. Wikitext-2 Oracle KL Divergence (In-Domain, directly comparable to CARE-COM).
2. AI2 ARC-Challenge Downstream Capability Evaluation (Loss, Margin, Accuracy).
"""

import os
import torch
import torch.nn.functional as F
from datasets import load_dataset

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    WIKITEXT_DATASET, WIKITEXT_CONFIG, WIKITEXT_SPLIT,
    EVAL_TOKENS_LIMIT, MAX_SEQ_LEN, BATCH_SIZE,
    ARC_DATASET_NAME, ARC_DATASET_SUBSET, EXP7A_DATA_DIR,
    DEVICE
)

def prepare_wikitext_eval_batches(tokenizer, max_tokens=EVAL_TOKENS_LIMIT, seq_len=MAX_SEQ_LEN):
    """Tokenizes Wikitext-2 text into independent padded sequences, matching Exp 1 exactly."""
    print(f"[Evaluation] Loading {WIKITEXT_DATASET} ({WIKITEXT_SPLIT})...")
    raw = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split=WIKITEXT_SPLIT)
    
    # Filter non-empty texts
    texts = [t for t in raw["text"] if len(t.strip()) > 0]
    
    import random
    # Match Exp 1 exact shuffle logic
    random.Random(42).shuffle(texts)
    
    chunks = []
    max_seqs = max_tokens // seq_len
    
    for t in texts:
        if len(chunks) >= max_seqs: 
            break
        enc = tokenizer(t, truncation=True, max_length=seq_len, padding="max_length", return_tensors="pt")
        if enc["attention_mask"].sum().item() >= 8:
            chunks.append({
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0]
            })
            
    print(f"[Evaluation] Prepared {len(chunks)} sequences ({len(chunks)*seq_len} tokens).")
    return chunks

@torch.no_grad()
def collect_baseline_wikitext_logits(model, eval_chunks, batch_size=BATCH_SIZE):
    """Precomputes and caches baseline model logits for fast KL computation."""
    print("[Evaluation] Precomputing baseline Wikitext logits...")
    baseline_logprobs = []
    
    for i in range(0, len(eval_chunks), batch_size):
        batch = eval_chunks[i:i + batch_size]
        input_ids = torch.stack([x["input_ids"] for x in batch]).to(DEVICE)
        attention_mask = torch.stack([x["attention_mask"] for x in batch]).to(DEVICE)
        
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        logp = F.log_softmax(shift_logits, dim=-1).cpu()
        baseline_logprobs.append(logp)
        
    return baseline_logprobs

@torch.no_grad()
def evaluate_wikitext_oracle_kl(model, eval_chunks, baseline_logprobs, batch_size=BATCH_SIZE):
    """
    Evaluates token-level Oracle KL divergence against cached baseline:
    D_KL(P_base || P_curr) = sum(P_base * (log P_base - log P_curr))
    """
    total_kl = 0.0
    total_tokens = 0
    batch_idx = 0
    
    for i in range(0, len(eval_chunks), batch_size):
        batch = eval_chunks[i:i + batch_size]
        input_ids = torch.stack([x["input_ids"] for x in batch]).to(DEVICE)
        attention_mask = torch.stack([x["attention_mask"] for x in batch]).to(DEVICE)
        
        logits_curr = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        shift_logits_curr = logits_curr[:, :-1, :]
        logp_curr = F.log_softmax(shift_logits_curr, dim=-1).cpu()
        
        logp_base = baseline_logprobs[batch_idx]
        batch_idx += 1
        
        # KL(P_base || P_curr)
        p_base = logp_base.exp()
        kl_per_token = (p_base * (logp_base - logp_curr)).sum(dim=-1)
        
        # Mask out padding tokens (shift mask to match shifted logits)
        shift_mask = attention_mask[:, 1:].bool().cpu()
        
        total_kl += kl_per_token[shift_mask].sum().item()
        total_tokens += shift_mask.sum().item()
        
    mean_kl = total_kl / max(total_tokens, 1)
    return mean_kl

# ══════════════════════════════════════════════════════════
# ARC-Challenge Capability Pipeline (Matching Exp 7A)
# ══════════════════════════════════════════════════════════

def format_arc_question(item):
    """Formats ARC multiple-choice item into prompt and choices."""
    question = item["question"]
    choices = item["choices"]
    prompt = f"Question: {question}\nAnswer:"
    
    candidates = []
    correct_idx = -1
    for i, (label, text) in enumerate(zip(choices["label"], choices["text"])):
        candidates.append(f" {text}")
        if label == item["answerKey"]:
            correct_idx = i
            
    return prompt, candidates, correct_idx

@torch.no_grad()
def compute_choice_logprob(model, tokenizer, prompt, choice):
    """Computes total log probability of choice continuation given prompt."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + choice, add_special_tokens=False)
    
    choice_len = len(full_ids) - len(prompt_ids)
    if choice_len <= 0:
        return torch.tensor(-1e9, device=DEVICE)
        
    input_ids = torch.tensor([full_ids], device=DEVICE)
    logits = model(input_ids=input_ids).logits[0] # [seq_len, vocab]
    
    # Continuation tokens
    choice_logits = logits[len(prompt_ids)-1:-1]
    choice_targets = input_ids[0, len(prompt_ids):]
    
    log_probs = F.log_softmax(choice_logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=choice_targets.unsqueeze(-1)).squeeze(-1)
    return target_log_probs.sum()

@torch.no_grad()
def evaluate_arc_dataset(model, tokenizer, dataset):
    """
    Evaluates ARC multiple-choice dataset.
    Returns:
    - accuracy: fraction of correctly chosen answers
    - avg_p_correct: mean softmax probability of the correct answer
    - avg_margin: mean (P(correct) - max(P(incorrect)))
    - avg_loss: mean (-log P(correct))
    - avg_logit_margin: mean (logit(correct) - max(logit(incorrect)))
    """
    correct_count = 0
    total = len(dataset)
    sum_p_correct = 0.0
    sum_margin = 0.0
    sum_loss = 0.0
    sum_logit_margin = 0.0
    
    for item in dataset:
        prompt, candidates, correct_idx = format_arc_question(item)
        if correct_idx == -1:
            continue
            
        log_probs = []
        for choice in candidates:
            lp = compute_choice_logprob(model, tokenizer, prompt, choice)
            log_probs.append(lp)
            
        log_probs = torch.stack(log_probs)
        probs = F.softmax(log_probs, dim=0)
        
        pred_idx = torch.argmax(probs).item()
        if pred_idx == correct_idx:
            correct_count += 1
            
        p_corr = probs[correct_idx].item()
        sum_p_correct += p_corr
        
        # Prob margin
        inc_probs = torch.cat([probs[:correct_idx], probs[correct_idx+1:]])
        max_inc_prob = torch.max(inc_probs).item() if len(inc_probs) > 0 else 0.0
        sum_margin += (p_corr - max_inc_prob)
        
        # Loss & Logit margin
        sum_loss += (-log_probs[correct_idx].item())
        inc_log_probs = torch.cat([log_probs[:correct_idx], log_probs[correct_idx+1:]])
        max_inc_lp = torch.max(inc_log_probs).item() if len(inc_log_probs) > 0 else 0.0
        sum_logit_margin += (log_probs[correct_idx].item() - max_inc_lp)
        
    return {
        "accuracy": correct_count / total,
        "avg_p_correct": sum_p_correct / total,
        "avg_margin": sum_margin / total,
        "avg_loss": sum_loss / total,
        "avg_logit_margin": sum_logit_margin / total
    }

def load_arc_partition(partition_name="D_proxy.pt"):
    """Loads ARC partition from results/exp7a/data/ if available."""
    local_path = os.path.join(EXP7A_DATA_DIR, partition_name)
    if os.path.exists(local_path):
        print(f"[Evaluation] Loading ARC partition from {local_path}...")
        return torch.load(local_path)
        
    print(f"[Evaluation] Downloading fresh ARC-Challenge ({ARC_DATASET_SUBSET})...")
    ds = load_dataset(ARC_DATASET_NAME, ARC_DATASET_SUBSET, split="validation")
    return [item for item in ds]
