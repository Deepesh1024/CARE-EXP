import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

def prepare_wikitext_eval_batches(tokenizer, config):
    """
    Loads WikiText-2 and chunks it into fixed-length sequences.
    Re-uses the CARE-COM v2.2 data loading protocol exactly.
    """
    print("[Benchmark] Loading WikiText-2 for PPL evaluation...")
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
    Computes perplexity on the prepared chunks.
    Matches the fixed v2.2 methodology where pad tokens are masked out.
    """
    model.eval()
    nlls = []
    
    for i in tqdm(range(0, len(eval_chunks), batch_size), desc="Computing PPL", leave=False):
        batch = eval_chunks[i:i + batch_size]
        
        input_ids = torch.stack([x["input_ids"] for x in batch]).to(device)
        attention_mask = torch.stack([x["attention_mask"] for x in batch]).to(device)
        
        # Mask padding tokens with -100 so CrossEntropyLoss ignores them (though in our chunking there is no padding)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        nlls.append(outputs.loss.item())
        
    avg_nll = sum(nlls) / len(nlls) if nlls else 0
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
        
        input_ids = torch.tensor(batch_df['input_ids'].tolist()).to(device)
        attention_mask = torch.tensor(batch_df['attention_mask'].tolist()).to(device)
        
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
