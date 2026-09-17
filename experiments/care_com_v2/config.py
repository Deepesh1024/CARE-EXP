import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class CareComV2Config:
    target_num_experts: int = 56
    candidate_pool_size: int = 8
    
    # Ablation Hooks
    candidate_scoring: str = "capability_plus_usage" # Options: "capability_only", "usage_only", "capability_plus_usage"
    adaptive_recompute: bool = True
    capability_redistribution: bool = True
    
    calibration_fraction: float = 1.0
    max_eval_batches: Optional[int] = None
    eval_batch_size: int = 4
    device: str = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    seed: int = 42
    save_trace: bool = True
    trace_path: str = "results/care_com_v2/trace.json"
