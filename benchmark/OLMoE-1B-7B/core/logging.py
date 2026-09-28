import os
import json

class BenchmarkLogger:
    def __init__(self, log_path, config):
        self.log_path = log_path
        self.config = config
        
        # Initialize JSON file
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                json.dump({
                    "config": config,
                    "ppl": {},
                    "trace": []
                }, f, indent=2)
                
    def log_ppl(self, experts, ppl_val):
        with open(self.log_path, "r") as f:
            data = json.load(f)
            
        data["ppl"][str(experts)] = ppl_val
        
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2)
            
    def log_step(self, step_data):
        """
        step_data must contain:
        { "method": "...", "seed": 42, "step": 1, "experts_before": 64, 
          "experts_after": 63, "candidate_pairs": [[i,j], ...], 
          "selected_pair": [i,j], "selection_score": ..., 
          "capability_distance": ..., "marginal_kl": ..., 
          "cumulative_kl": ..., "wall_time_sec": ..., 
          "peak_memory_mb": ... }
        """
        with open(self.log_path, "r") as f:
            data = json.load(f)
            
        data["trace"].append(step_data)
        
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2)
