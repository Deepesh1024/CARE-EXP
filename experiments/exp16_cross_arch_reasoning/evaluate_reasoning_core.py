import os
import sys
import time
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import torch
from datasets import load_dataset

# -----------------------------------------------------------------------------
# Robust Answer Extraction and Normalization
# -----------------------------------------------------------------------------

def extract_gsm8k_answer(text: str) -> Optional[Any]:
    if not text:
        return None
    
    # 1. Standard GSM8K delimiter: #### <num>
    match = re.search(r"####\s*([0-9\.,\-]+)", text)
    if match:
        raw = match.group(1).replace(",", "").strip()
        try:
            return float(raw) if "." in raw else int(raw)
        except ValueError:
            pass

    # 2. LaTeX boxed: \boxed{<num>}
    box_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if box_matches:
        raw = box_matches[-1].replace(",", "").replace("$", "").strip()
        try:
            return float(raw) if "." in raw else int(raw)
        except ValueError:
            pass

    # 3. Explicit phrasing: The answer is <num>
    match = re.search(r"[Tt]he (?:final )?answer is\s*:?\s*([0-9\.,\-]+)", text)
    if match:
        raw = match.group(1).replace(",", "").strip()
        try:
            return float(raw) if "." in raw else int(raw)
        except ValueError:
            pass

    # 4. Fallback: Last number appearing in text
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)
    if nums:
        raw = nums[-1].replace(",", "").strip()
        try:
            return float(raw) if "." in raw else int(raw)
        except ValueError:
            pass

    return None


def normalize_gsm8k_gold(gold_text: str) -> Optional[Any]:
    if not gold_text:
        return None
    match = re.search(r"####\s*([0-9\.,\-]+)", gold_text)
    if match:
        raw = match.group(1).replace(",", "").strip()
        try:
            return float(raw) if "." in raw else int(raw)
        except ValueError:
            return raw
    return None


def _clean_latex_str(s: str) -> str:
    s = s.strip()
    s = s.replace("$", "").replace("\\%", "%")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\left", "", s)
    s = re.sub(r"\\right", "", s)
    s = s.strip().rstrip(".")
    return s


def extract_math500_answer(text: str) -> Optional[str]:
    if not text:
        return None
        
    # Match innermost and balanced \boxed{...}
    # First search for full \boxed{...}
    box_matches = []
    start = 0
    while True:
        pos = text.find(r"\boxed{", start)
        if pos == -1:
            break
        idx = pos + 7
        depth = 1
        while idx < len(text) and depth > 0:
            if text[idx] == '{':
                depth += 1
            elif text[idx] == '}':
                depth -= 1
            idx += 1
        if depth == 0:
            box_matches.append(text[pos+7:idx-1])
            start = idx
        else:
            break
            
    if box_matches:
        return _clean_latex_str(box_matches[-1])

    # Phrasing fallback: The answer is ...
    match = re.search(r"[Tt]he (?:final )?answer is\s*:?\s*([^\n\.]+)", text)
    if match:
        return _clean_latex_str(match.group(1))

    return None


def normalize_math_gold(gold_text: str) -> str:
    if not gold_text:
        return ""
    # Gold may already be boxed or plain
    box_matches = re.findall(r"\\boxed\{([^{}]+)\}", gold_text)
    if box_matches:
        return _clean_latex_str(box_matches[-1])
    return _clean_latex_str(gold_text)


def is_math_equivalent(pred: Optional[str], gold: str) -> bool:
    if pred is None or gold == "":
        return False
    pred_c = _clean_latex_str(pred)
    gold_c = _clean_latex_str(gold)
    if pred_c == gold_c:
        return True
        
    # Float numerical equivalence
    try:
        f_pred = float(pred_c)
        f_gold = float(gold_c)
        if abs(f_pred - f_gold) < 1e-4:
            return True
    except ValueError:
        pass

    # Simple fraction equivalence e.g. 1/2 vs \frac{1}{2}
    frac_match = re.match(r"\\frac\{([0-9\.\-]+)\}\{([0-9\.\-]+)\}", gold_c)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            val = num / den
            if abs(float(pred_c) - val) < 1e-4:
                return True
        except Exception:
            pass

    return False


# -----------------------------------------------------------------------------
# Core Evaluation Function (Resumable + Streaming JSONL)
# -----------------------------------------------------------------------------

def evaluate_benchmark(
    model,
    tokenizer,
    benchmark_key: str,
    model_label: str,
    architecture_type: str,  # 'base' or 'instruct'
    config: Dict[str, Any],
    raw_gen_dir: str
) -> Dict[str, Any]:
    os.makedirs(raw_gen_dir, exist_ok=True)
    bench_cfg = config["benchmarks"][benchmark_key]
    num_samples = bench_cfg["num_samples"]
    sample_indices = bench_cfg.get("sample_indices", list(range(num_samples)))
    
    dataset_name = bench_cfg["dataset_name"]
    dataset_config = bench_cfg.get("dataset_config", None)
    split = bench_cfg["split"]
    
    print(f"\n{'='*70}", flush=True)
    print(f"EVALUATING {benchmark_key.upper()} ({len(sample_indices)} Qs) on {model_label}", flush=True)
    print(f"{'='*70}", flush=True)
    
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
        
    samples = [ds[i] for i in sample_indices]
    
    jsonl_filename = f"{model_label}_{benchmark_key}_{len(samples)}q.jsonl"
    results_path = os.path.join(raw_gen_dir, jsonl_filename)
    
    # Resumption check
    completed_indices = set()
    existing_records = []
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    completed_indices.add(rec["idx"])
                    existing_records.append(rec)
        print(f"  [Resumption] Found {len(completed_indices)} already completed questions in {jsonl_filename}", flush=True)
        
    correct_count = sum(1 for r in existing_records if r.get("is_correct", False))
    parse_fail_count = sum(1 for r in existing_records if r.get("parse_failed", False))
    total_tokens = sum(r.get("generated_tokens", 0) for r in existing_records)
    
    t_start = time.time()
    
    with open(results_path, "a") as f_out:
        for local_idx, sample in enumerate(samples):
            global_idx = sample_indices[local_idx]
            if global_idx in completed_indices:
                continue
                
            if benchmark_key == "gsm8k":
                question = sample["question"]
                gold_raw = sample["answer"]
                gold_answer = normalize_gsm8k_gold(gold_raw)
                
                if architecture_type == "instruct":
                    prompt_text = bench_cfg["prompt_format_instruct"].format(question=question)
                    messages = [{"role": "user", "content": prompt_text}]
                    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    else:
                        prompt = f"<|user|>\n{prompt_text}<|end|>\n<|assistant|>\n"
                else:
                    prompt = bench_cfg["prompt_format_base"].format(question=question)
                    
            elif benchmark_key == "math500":
                problem = sample["problem"]
                gold_raw = sample.get("answer", "")
                gold_answer = normalize_math_gold(gold_raw)
                
                if architecture_type == "instruct":
                    prompt_text = bench_cfg["prompt_format_instruct"].format(problem=problem)
                    messages = [{"role": "user", "content": prompt_text}]
                    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    else:
                        prompt = f"<|user|>\n{prompt_text}<|end|>\n<|assistant|>\n"
                else:
                    prompt = bench_cfg["prompt_format_base"].format(problem=problem)
            else:
                raise ValueError(f"Unknown benchmark: {benchmark_key}")

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            t_gen_0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=bench_cfg.get("max_new_tokens", 512),
                    do_sample=bench_cfg.get("do_sample", False),
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    use_cache=True
                )
            gen_time = time.time() - t_gen_0
            gen_ids = outputs[0][inputs.input_ids.shape[1]:]
            num_tokens = len(gen_ids)
            total_tokens += num_tokens
            tok_per_sec = num_tokens / gen_time if gen_time > 0 else 0.0
            
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            if benchmark_key == "gsm8k":
                pred_answer = extract_gsm8k_answer(gen_text)
                parse_failed = (pred_answer is None)
                is_correct = (pred_answer == gold_answer) if (pred_answer is not None and gold_answer is not None) else False
            else:
                pred_answer = extract_math500_answer(gen_text)
                parse_failed = (pred_answer is None)
                is_correct = is_math_equivalent(pred_answer, str(gold_answer))
                
            if is_correct:
                correct_count += 1
            if parse_failed:
                parse_fail_count += 1
                
            record = {
                "idx": global_idx,
                "benchmark": benchmark_key,
                "prompt": prompt,
                "gold_raw": str(gold_raw),
                "gold_answer": str(gold_answer),
                "generation": gen_text,
                "predicted_answer": str(pred_answer) if pred_answer is not None else None,
                "parse_failed": parse_failed,
                "is_correct": is_correct,
                "gen_time_sec": round(gen_time, 3),
                "generated_tokens": num_tokens,
                "tokens_per_sec": round(tok_per_sec, 2)
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()
            
            cur_processed = len(completed_indices) + (local_idx + 1)
            if cur_processed % 10 == 0 or cur_processed == len(samples):
                cur_acc = (correct_count / cur_processed) * 100.0
                print(f"  [{benchmark_key.upper()}] {cur_processed}/{len(samples)} | Acc: {cur_acc:.1f}% ({correct_count}/{cur_processed}) | Parse Fails: {parse_fail_count} | Speed: {tok_per_sec:.1f} tok/s | Elapsed: {time.time()-t_start:.1f}s", flush=True)

    total_q = len(samples)
    final_acc = (correct_count / total_q) * 100.0 if total_q > 0 else 0.0
    parse_fail_rate = (parse_fail_count / total_q) * 100.0 if total_q > 0 else 0.0
    elapsed_total = time.time() - t_start
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    
    print(f"\n>>> [{benchmark_key.upper()} COMPLETE] Accuracy: {final_acc:.2f}% ({correct_count}/{total_q}) | Parse Failures: {parse_fail_rate:.1f}% | Peak VRAM: {peak_vram_gb:.2f} GB", flush=True)
    
    return {
        "benchmark": benchmark_key,
        "total_questions": total_q,
        "correct": correct_count,
        "accuracy": round(final_acc, 2),
        "parse_failures": parse_fail_count,
        "parse_failure_rate": round(parse_fail_rate, 2),
        "total_tokens": total_tokens,
        "avg_throughput": round(total_tokens / elapsed_total, 2) if elapsed_total > 0 else 0.0,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "elapsed_sec": round(elapsed_total, 2),
        "results_file": results_path
    }
