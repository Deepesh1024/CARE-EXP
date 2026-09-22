import os
import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class CareComV22Config:
    # Model details
    model_name: str = "allenai/OLMoE-1B-7B-0924"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Trajectory
    trajectory: List[int] = field(default_factory=lambda: list(range(64, 55, -1))) # 64 -> 56
    checkpoints: List[int] = field(default_factory=lambda: [64, 60, 56])
    
    # Baselines
    random_seeds: List[int] = field(default_factory=lambda: [42, 1024, 2027])
    
    # Adaptive properties (from v2.1)
    candidate_pool_size: int = 5
    
    # Evaluation
    eval_batch_size: int = 4
    max_eval_batches: int = 16
    ppl_batch_size: int = 4
    max_ppl_batches: int = 64 # Use a larger number of batches for checkpoint PPL calculation
    
    # Output paths
    output_dir: str = "results/care_com_v22"
    trajectories_dir: str = field(init=False)
    tables_dir: str = field(init=False)
    figures_dir: str = field(init=False)
    
    def __post_init__(self):
        self.trajectories_dir = os.path.join(self.output_dir, "trajectories")
        self.tables_dir = os.path.join(self.output_dir, "tables")
        self.figures_dir = os.path.join(self.output_dir, "figures")
        
        os.makedirs(self.trajectories_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
