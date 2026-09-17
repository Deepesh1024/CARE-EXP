import torch
import torch.nn.functional as F

@torch.no_grad()
def collect_baseline_wikitext_logits(model, eval_chunks, batch_size, max_eval_batches=None, device="cuda:0"):
    """
    Precomputes baseline model logits for fast KL computation.
    """
    model.eval()
    baseline_logprobs = []
    
    num_batches = len(eval_chunks) // batch_size + (1 if len(eval_chunks) % batch_size != 0 else 0)
    if max_eval_batches is not None:
        num_batches = min(num_batches, max_eval_batches)
        
    from tqdm import tqdm
    for i in tqdm(range(num_batches), desc="Precomputing Baseline Logits"):
        batch = eval_chunks[i * batch_size : (i + 1) * batch_size]
        input_ids = torch.stack([x["input_ids"] for x in batch]).to(device)
        attention_mask = torch.stack([x["attention_mask"] for x in batch]).to(device)
        
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        shift_logits = logits[:, :-1, :]
        logp = F.log_softmax(shift_logits, dim=-1).cpu()
        baseline_logprobs.append(logp)
        
    return baseline_logprobs

@torch.no_grad()
def evaluate_temporary_pruning(model, eval_chunks, baseline_logprobs, batch_size, max_eval_batches=None, device="cuda:0"):
    """
    Evaluates token-level Oracle KL divergence against cached baseline.
    Assumes the model's router has ALREADY been hooked/patched to route away from the pruned expert.
    """
    model.eval()
    total_kl = 0.0
    total_tokens = 0
    
    num_batches = len(eval_chunks) // batch_size + (1 if len(eval_chunks) % batch_size != 0 else 0)
    if max_eval_batches is not None:
        num_batches = min(num_batches, max_eval_batches)
        
    from tqdm import tqdm
    for i in tqdm(range(num_batches), desc="Evaluating Candidate", leave=False):
        batch = eval_chunks[i * batch_size : (i + 1) * batch_size]
        input_ids = torch.stack([x["input_ids"] for x in batch]).to(device)
        attention_mask = torch.stack([x["attention_mask"] for x in batch]).to(device)
        
        logits_curr = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        shift_logits_curr = logits_curr[:, :-1, :]
        logp_curr = F.log_softmax(shift_logits_curr, dim=-1).cpu()
        
        logp_base = baseline_logprobs[i]
        
        p_base = logp_base.exp()
        kl_per_token = (p_base * (logp_base - logp_curr)).sum(dim=-1)
        
        shift_mask = attention_mask[:, 1:].bool().cpu()
        
        total_kl += kl_per_token[shift_mask].sum().item()
        total_tokens += shift_mask.sum().item()
        
    return total_kl / max(total_tokens, 1)
