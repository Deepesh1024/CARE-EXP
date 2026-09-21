import os
import torch

class CareComV21Config:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.seed = 42
        
        # We start with 64 experts and decrease one by one down to 56
        self.trajectory = list(range(64, 55, -1)) # [64, 63, ..., 56]
        self.checkpoints = [64, 60, 56]
        
        self.candidate_pool_size = 5
        self.eval_batch_size = 4
        self.max_eval_batches = 100 # Limit to speed up evaluation
        
        self.calibration_fraction = 1.0
        
        self.results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "care_com_v21")
        self.trace_path = os.path.join(self.results_dir, "trajectories", "v21_trace.json")
        self.model_save_dir = os.path.join(self.results_dir, "checkpoints")
