"""
care_large_moe_systems/workloads.py
Defines standardized, fixed workload prompts and generation parameters for systems benchmarking.
"""

from typing import List, Dict, Any

# Fixed prompt texts calibrated for target token lengths
SHORT_PROMPT_32 = (
    "Question: Explain why sparse Mixture-of-Experts models achieve higher training efficiency "
    "than dense models while maintaining constant active compute per token."
)

MEDIUM_PROMPT_256 = (
    "Mixture-of-Experts (MoE) is an architectural paradigm that replaces dense feed-forward "
    "networks with a set of sparse expert networks coordinated by a trainable routing mechanism. "
    "During forward inference, each token is dynamically assigned to a top-k subset of experts. "
    "While this sparsity decouples total model capacity from per-token compute, it introduces "
    "unique systems challenges including memory bandwidth pressure, load imbalance across experts, "
    "and parameter communication overhead in distributed deployments. Analyze how expert consolidation "
    "and physical merging algorithms like CARE-COM, Sub-MoE, and REAP affect memory access patterns, "
    "routing entropy, and cache utilization during autoregressive generation on modern GPU architectures. "
    "Specifically discuss whether reducing the expert count improves inference latency when top-k is held constant."
)

LONG_PROMPT_2048 = (
    ("In modern deep learning systems, the computational cost of autoregressive language generation "
     "is dominated by memory bandwidth rather than pure arithmetic throughput. For dense transformers, "
     "the arithmetic intensity during token generation is roughly one FLOP per byte of memory loaded, "
     "making decode throughput strictly memory-bound. In sparse Mixture-of-Experts models, this bandwidth "
     "bottleneck is exacerbated because different tokens in a sequence may activate disjoint sets of experts, "
     "forcing the GPU to fetch non-contiguous weight matrices from high-bandwidth memory (HBM) into on-chip SRAM. "
     "When batch size is small (e.g., batch size = 1), only a small fraction of expert weights are reused across tokens, "
     "causing low operational intensity. As batch size increases, tokens across the batch are routed to different experts, "
     "leading to expert load imbalance where popular 'hub' experts process many tokens while specialized experts remain idle. "
     "System-level optimizations such as grouped GEMM, kernel fusion, and expert parallelism attempt to alleviate these "
     "overheads, but physical consolidation remains an attractive technique to shrink the total parameter footprint. "
     ) * 16  # Repeated to generate ~2048 tokens
)

WORKLOADS: Dict[str, Dict[str, Any]] = {
    "W1_short": {
        "name": "Interactive Query (Short)",
        "prompt_text": SHORT_PROMPT_32,
        "target_input_tokens": 32,
        "max_new_tokens": 64,
        "batch_sizes": [1, 4, 8],
        "description": "Measures Time to First Token (TTFT) and p50/p95 latency under interactive chat conditions."
    },
    "W2_medium": {
        "name": "Instruction / Code Generation (Medium)",
        "prompt_text": MEDIUM_PROMPT_256,
        "target_input_tokens": 256,
        "max_new_tokens": 128,
        "batch_sizes": [1, 4],
        "description": "Measures inter-token latency (ITL) and sustained decode throughput."
    },
    "W3_long": {
        "name": "Document Context (Long)",
        "prompt_text": LONG_PROMPT_2048,
        "target_input_tokens": 2048,
        "max_new_tokens": 512,
        "batch_sizes": [1],
        "description": "Measures prefill throughput and peak memory under long-context load."
    }
}
