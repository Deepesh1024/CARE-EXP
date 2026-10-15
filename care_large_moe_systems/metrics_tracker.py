"""
care_large_moe_systems/metrics_tracker.py
Precise metrics tracking for TTFT, ITL, throughput, and system memory.
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional

class MetricsTracker:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.is_cuda = (device == "cuda" or "cuda" in str(device)) and torch.cuda.is_available()

    def get_memory_stats(self) -> Dict[str, float]:
        stats = {
            "cpu_ram_rss_mb": psutil.Process().memory_info().rss / (1024 * 1024),
            "cpu_ram_vms_mb": psutil.Process().memory_info().vms / (1024 * 1024),
        }
        if self.is_cuda:
            stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            stats["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            stats["gpu_max_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 * 1024)
        else:
            stats["gpu_allocated_mb"] = 0.0
            stats["gpu_reserved_mb"] = 0.0
            stats["gpu_max_allocated_mb"] = 0.0
            stats["gpu_max_reserved_mb"] = 0.0
        return stats

    def reset_peak_memory(self):
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()

    def synchronize(self):
        if self.is_cuda:
            torch.cuda.synchronize()

    def benchmark_generation(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Executes generation and measures:
        - TTFT (Time to first token / prefill latency)
        - Per-token inter-token latency (ITL)
        - Prefill & decode throughput
        - Query end-to-end latency
        - Peak VRAM during generation
        """
        # Encode inputs
        encoded = tokenizer([prompt] * batch_size, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        prompt_len = input_ids.shape[1]

        # Reset peak memory
        self.reset_peak_memory()
        self.synchronize()

        # Step 1: Prefill (First Token)
        t_prefill_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
        self.synchronize()
        t_prefill_end = time.perf_counter()

        ttft_sec = t_prefill_end - t_prefill_start
        ttft_ms = ttft_sec * 1000.0

        current_input_ids = next_token
        if attention_mask is not None:
            current_attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=self.device, dtype=attention_mask.dtype)], dim=1)
        else:
            current_attention_mask = None

        # Step 2: Autoregressive Decode Loop
        inter_token_latencies_ms: List[float] = []
        tokens_generated = 1

        t_decode_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                t_tok_start = time.perf_counter()
                outputs = model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                past_key_values = outputs.past_key_values
                self.synchronize()
                t_tok_end = time.perf_counter()

                inter_token_latencies_ms.append((t_tok_end - t_tok_start) * 1000.0)
                current_input_ids = next_token
                if current_attention_mask is not None:
                    current_attention_mask = torch.cat([current_attention_mask, torch.ones((batch_size, 1), device=self.device, dtype=current_attention_mask.dtype)], dim=1)
                tokens_generated += 1

        self.synchronize()
        t_decode_end = time.perf_counter()
        decode_time_sec = t_decode_end - t_decode_start
        total_time_sec = t_decode_end - t_prefill_start

        mem_stats = self.get_memory_stats()

        # Compute summary statistics
        itls = np.array(inter_token_latencies_ms) if inter_token_latencies_ms else np.array([0.0])
        prefill_throughput_tok_sec = (prompt_len * batch_size) / max(ttft_sec, 1e-6)
        decode_throughput_tok_sec = ((tokens_generated - 1) * batch_size) / max(decode_time_sec, 1e-6)
        e2e_throughput_tok_sec = (tokens_generated * batch_size) / max(total_time_sec, 1e-6)

        return {
            "batch_size": batch_size,
            "prompt_tokens": prompt_len,
            "generated_tokens": tokens_generated,
            "ttft_ms": round(float(ttft_ms), 3),
            "decode_time_sec": round(float(decode_time_sec), 4),
            "e2e_time_sec": round(float(total_time_sec), 4),
            "mean_itl_ms": round(float(np.mean(itls)), 3),
            "std_itl_ms": round(float(np.std(itls)), 3),
            "p50_itl_ms": round(float(np.percentile(itls, 50)), 3),
            "p90_itl_ms": round(float(np.percentile(itls, 90)), 3),
            "p95_itl_ms": round(float(np.percentile(itls, 95)), 3),
            "prefill_throughput_tok_sec": round(float(prefill_throughput_tok_sec), 2),
            "decode_throughput_tok_sec": round(float(decode_throughput_tok_sec), 2),
            "e2e_throughput_tok_sec": round(float(e2e_throughput_tok_sec), 2),
            "peak_vram_mb": round(float(mem_stats["gpu_max_allocated_mb"]), 2),
            "allocated_vram_mb": round(float(mem_stats["gpu_allocated_mb"]), 2),
            "cpu_ram_rss_mb": round(float(mem_stats["cpu_ram_rss_mb"]), 2),
        }
