import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

def prepare_wikitext_eval_batches(tokenizer, config):
    """
    Loads WikiText-2 and chunks it into fixed-length sequences.
    Re-uses the CARE-COM v2.2 data loading protocol exactly.
    """
    dataset_name = config['evaluation']['dataset']
    if dataset_name == "wikitext":
        dataset_name = "Salesforce/wikitext"
    try:
        dataset = load_dataset(
            dataset_name,
            config['evaluation']['subset'],
            split=config['evaluation']['split']
        )
    except Exception:
        dataset = load_dataset(
            config['evaluation']['dataset'],
            config['evaluation']['subset'],
            split=config['evaluation']['split']
        )
    
    # Simple chunking approach to match v2.2 exactly
    text = "\n\n".join(dataset["text"])
    
    seq_len = config['evaluation']['max_length']
    num_chunks = config['evaluation']['num_eval_chunks']
    
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    
    chunks = []
    # Take contiguous non-overlapping sequences of max_length
    for i in range(0, min(len(tokens) - seq_len, num_chunks * seq_len), seq_len):
        chunk = tokens[i : i + seq_len]
        chunks.append({
            "input_ids": chunk,
            "attention_mask": torch.ones_like(chunk)
        })
        
    print(f"[Benchmark] Prepared {len(chunks)} evaluation sequences of length {seq_len}.")
    return chunks

@torch.no_grad()
def compute_ppl(model, eval_chunks, device="cuda", batch_size=4):
    """
    Computes exact language model perplexity on the prepared chunks:
    PPL = exp( sum(valid_token_nll) / total_valid_tokens )
    Computes cross-entropy directly from shifted logits, strictly avoiding any
    auxiliary router losses in outputs.loss and correctly weighting every valid token.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    
    for i in tqdm(range(0, len(eval_chunks), batch_size), desc="Computing PPL", leave=False):
        batch = eval_chunks[i:i + batch_size]
        input_ids = torch.stack([x["input_ids"] for x in batch]).to(device)
        attention_mask = torch.stack([x["attention_mask"] for x in batch]).to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.shape)
        
        valid_loss = (token_losses * shift_mask).sum().item()
        num_valid = shift_mask.sum().item()
        
        total_nll += valid_loss
        total_tokens += num_valid
        
    if total_tokens == 0:
        raise ValueError("Cannot compute PPL: zero valid prediction tokens found in evaluation chunks.")
    avg_nll = total_nll / total_tokens
    ppl = torch.exp(torch.tensor(avg_nll)).item()
    return ppl

@torch.no_grad()
def compute_calibration_distributions(model, df_tokens, batch_size=32, device="cuda"):
    """
    Computes the baseline probability distributions for marginal KL measurement.
    """
    model.eval()
    all_probs = []
    
    for i in range(0, len(df_tokens), batch_size):
        batch_df = df_tokens.iloc[i:i+batch_size]
        
        import numpy as np
        input_ids = torch.tensor(np.array(batch_df['input_ids'].tolist())).to(device)
        attention_mask = torch.tensor(np.array(batch_df['attention_mask'].tolist())).to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits[:, -1, :], dim=-1)
        all_probs.append(probs.cpu())
        
    return torch.cat(all_probs, dim=0)

@torch.no_grad()
def evaluate_marginal_kl(model, df_tokens, baseline_probs, batch_size=32, device="cuda"):
    """
    Computes the KL divergence of the current model against the baseline distributions.
    """
    model.eval()
    total_kl = 0.0
    num_samples = len(df_tokens)
    
    for i in range(0, num_samples, batch_size):
        batch_df = df_tokens.iloc[i:i+batch_size]
        
        import numpy as np
        input_ids = torch.tensor(np.array(batch_df['input_ids'].tolist())).to(device)
        attention_mask = torch.tensor(np.array(batch_df['attention_mask'].tolist())).to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        current_probs = F.softmax(outputs.logits[:, -1, :], dim=-1)
        
        # Target = baseline_probs, Input = log(current_probs)
        target_probs = baseline_probs[i:i+batch_size].to(device)
        log_current_probs = F.log_softmax(outputs.logits[:, -1, :], dim=-1)
        
        # KL(P || Q) = sum P * log(P/Q) = sum P * (log P - log Q)
        kl = F.kl_div(log_current_probs, target_probs, reduction='batchmean')
        
        batch_actual_size = len(batch_df)
        total_kl += kl.item() * batch_actual_size
        
    return total_kl / num_samples


def compute_causal_kl_from_logits(base_logits: torch.Tensor, comp_logits: torch.Tensor, attention_mask: torch.Tensor):
    """
    Computes token-wise KL divergence for causally aligned next-token prediction:
    Target token at index t+1 is predicted by logits at index t.
    Compares base_logits[..., :-1, :] vs comp_logits[..., :-1, :]
    Masks with attention_mask[..., 1:].
    Returns (sum_kl, valid_tokens).
    """
    shift_base = base_logits[..., :-1, :].contiguous()
    shift_comp = comp_logits[..., :-1, :].contiguous()
    shift_mask = attention_mask[..., 1:].bool().contiguous()

    p_log = F.log_softmax(shift_base, dim=-1)
    p = torch.exp(p_log)
    q_log = F.log_softmax(shift_comp, dim=-1)
    tok_kl = (p * (p_log - q_log)).sum(dim=-1)

    valid_kl = (tok_kl * shift_mask).sum().item()
    valid_tokens = shift_mask.sum().item()
    return valid_kl, valid_tokens


@torch.no_grad()
def compute_heldout_kl(base_model, comp_model, eval_chunks, device="cuda", batch_size=4, max_chunks=32):
    """
    Computes true held-out KL divergence KL(P_orig || P_comp) over held-out test chunks:
    KL(P || Q) = sum_v P(v) * (log P(v) - log Q(v))
    Causally aligned: compares logits[..., :-1, :] and masks with attention_mask[..., 1:].
    Accumulates KL over valid target tokens only.
    """
    base_model.eval()
    comp_model.eval()
    total_kl = 0.0
    total_tokens = 0
    chunks_to_eval = eval_chunks[:max_chunks]
    
    for i in range(0, len(chunks_to_eval), batch_size):
        batch = chunks_to_eval[i:i + batch_size]
        input_ids = torch.stack([x["input_ids"] for x in batch]).to(device)
        attention_mask = torch.stack([x["attention_mask"] for x in batch]).to(device)
        
        base_logits = base_model(input_ids=input_ids, attention_mask=attention_mask).logits
        comp_logits = comp_model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        batch_kl, batch_tokens = compute_causal_kl_from_logits(base_logits, comp_logits, attention_mask)
        total_kl += batch_kl
        total_tokens += batch_tokens

    if total_tokens == 0:
        raise ValueError("Cannot compute held-out KL: zero valid target tokens found across all evaluation chunks.")
    return total_kl / total_tokens


