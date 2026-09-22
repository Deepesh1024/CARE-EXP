import os
import json
import torch
import gc

from experiments.care_com_v22.config import CareComV22Config
from experiments.care_com_v22.baselines import run_random_baseline, run_static_baseline
from experiments.care_com_v22.adaptive import run_adaptive_baseline
from experiments.care_com_v22.tests.test_v22 import MockModel
from experiments.care_com_v21.core import PhysicalMergeEngine
from unittest.mock import patch

def run_smoke_test():
    config = CareComV22Config()
    config.trajectory = [64, 63, 62] # Smoke test to 62
    config.checkpoints = [64, 62]
    config.device = "cpu"
    config.eval_batch_size = 1
    config.max_eval_batches = 1
    config.ppl_batch_size = 1
    config.max_ppl_batches = 1
    config.random_seeds = [42]
    
    eval_chunks = [{"input_ids": torch.ones((1, 16), dtype=torch.long), "attention_mask": torch.ones((1, 16), dtype=torch.long)}]
    df_tokens = torch.ones((1, 16), dtype=torch.long)
    
    # 1. Random
    print("--- Running Random Smoke Test ---")
    with patch('experiments.care_com_v22.baselines.evaluate_marginal_damage', return_value=0.1):
        with patch('experiments.care_com_v22.baselines.collect_current_logits', return_value=None):
            with patch('experiments.care_com_v22.baselines.compute_ppl', return_value=12.0):
                model = MockModel(num_experts=64)
                engine = PhysicalMergeEngine(model)
                trace, ppl = run_random_baseline(model, engine, eval_chunks, config, seed=42)
                assert engine.current_num_experts == 62
                assert len(trace) == 2
                
    # 2. Static
    print("\n--- Running Static Smoke Test ---")
    with patch('experiments.care_com_v22.baselines.evaluate_marginal_damage', return_value=0.2):
        with patch('experiments.care_com_v22.baselines.collect_current_logits', return_value=None):
            with patch('experiments.care_com_v22.baselines.compute_ppl', return_value=12.5):
                with patch('experiments.care_com_v22.baselines.compute_global_capability_vectors', return_value=torch.randn(64, 16)):
                    model = MockModel(num_experts=64)
                    engine = PhysicalMergeEngine(model)
                    trace, ppl = run_static_baseline(model, engine, df_tokens, eval_chunks, config)
                    assert engine.current_num_experts == 62
                    assert len(trace) == 2
                    
    # 3. Adaptive
    print("\n--- Running Adaptive Smoke Test ---")
    with patch('experiments.care_com_v22.adaptive.evaluate_marginal_damage', return_value=0.3):
        with patch('experiments.care_com_v22.adaptive.collect_current_logits', return_value=None):
            with patch('experiments.care_com_v22.adaptive.compute_ppl', return_value=11.5):
                with patch('experiments.care_com_v22.adaptive.compute_global_capability_vectors') as mock_cap:
                    # Return correctly sized tensor for current N
                    def side_effect(model, blocks, tokens, device):
                        return torch.randn(engine.current_num_experts, 16)
                    mock_cap.side_effect = side_effect
                    model = MockModel(num_experts=64)
                    engine = PhysicalMergeEngine(model)
                    trace, ppl = run_adaptive_baseline(model, engine, df_tokens, eval_chunks, config)
                    assert engine.current_num_experts == 62
                    assert len(trace) == 2
                    
    print("\n[SUCCESS] Smoke test passed! Logic correctly brings model from 64 to 62 experts without error.")

if __name__ == "__main__":
    run_smoke_test()
