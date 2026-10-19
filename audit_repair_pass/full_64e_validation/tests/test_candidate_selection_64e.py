"""
audit_repair_pass/full_64e_validation/tests/test_candidate_selection_64e.py
==========================================================================
Comprehensive Unit & Invariant Test Suite for Full 64-Expert Candidate Selection (v2.3).
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from audit_repair_pass.full_64e_validation.care_capability_v23 import (
    compute_global_capability_vectors_v23,
    pairwise_capability_distances_v23,
    generate_candidate_pool_v23
)
from experiments.care_com_v21.core import PhysicalMergeEngine

class MockFusedExperts(nn.Module):
    def __init__(self, num_experts, hidden_dim, intermediate_dim):
        super().__init__()
        self.num_experts = num_experts
        # Sliced shapes matching HF OlmoeExperts layout
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate_dim, hidden_dim) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_dim, intermediate_dim) * 0.02)

class MockRouter(nn.Module):
    def __init__(self, num_experts, hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_dim) * 0.02)

class MockFusedMoEBlock(nn.Module):
    def __init__(self, num_experts, hidden_dim, intermediate_dim):
        super().__init__()
        self.experts = MockFusedExperts(num_experts, hidden_dim, intermediate_dim)
        self.gate = MockRouter(num_experts, hidden_dim)
        self.num_experts = num_experts

class MockModelConfig:
    def __init__(self, num_experts, num_layers):
        self.num_experts = num_experts
        self.num_hidden_layers = num_layers

class MockFusedModel(nn.Module):
    def __init__(self, num_experts=64, num_layers=4, hidden_dim=32, intermediate_dim=64):
        super().__init__()
        self.config = MockModelConfig(num_experts, num_layers)
        self.layers = nn.ModuleList([
            MockFusedMoEBlock(num_experts, hidden_dim, intermediate_dim) for _ in range(num_layers)
        ])
        
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        batch, seq_len = input_ids.shape
        hidden_states = [torch.randn(batch, seq_len, 32) for _ in range(len(self.layers) + 1)]
        class Output:
            pass
        out = Output()
        out.hidden_states = hidden_states
        return out

class TestCandidateSelection64E(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.num_experts = 64
        self.num_layers = 4
        self.model = MockFusedModel(num_experts=self.num_experts, num_layers=self.num_layers)
        self.engine = PhysicalMergeEngine(self.model)
        
        # Calibration tokens mock
        self.df_tokens = pd.DataFrame({
            'axis_idx': [i % 10 for i in range(40)],
            'input_ids': [[1, 2, 3, 4] for _ in range(40)],
            'attention_mask': [[1, 1, 1, 1] for _ in range(40)]
        })

    def test_invariant_1_full_64e_shape(self):
        """Invariant 1: Capability vector shape is exactly [num_experts, num_layers * 10]."""
        C = compute_global_capability_vectors_v23(
            self.model, self.engine.moe_blocks, self.engine.current_num_experts, self.df_tokens, device="cpu"
        )
        self.assertEqual(C.shape, (64, self.num_layers * 10))
        # Check unit norm globally
        norms = np.linalg.norm(C, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_invariant_2_layer_normalization_axis_correctness(self):
        """Invariant 2: Layer normalization slices layer dimension, ensuring experts 16..63 are normalized."""
        C = compute_global_capability_vectors_v23(
            self.model, self.engine.moe_blocks, self.engine.current_num_experts, self.df_tokens, device="cpu"
        )
        # Verify non-zero and finite values across all experts including 16..63
        for e in range(64):
            expert_vec = C[e]
            self.assertFalse(np.isnan(expert_vec).any())
            self.assertFalse(np.isinf(expert_vec).any())
            self.assertGreater(np.linalg.norm(expert_vec), 0.0)

    def test_invariant_3_candidate_pool_spans_full_64e(self):
        """Invariant 3: Candidate pool can nominate expert pairs with indices >= 16."""
        # Force expert 62 and 63 to have identical weights to guarantee minimal distance
        for block in self.engine.moe_blocks:
            block.experts.gate_up_proj.data[63] = block.experts.gate_up_proj.data[62].clone()
            block.experts.down_proj.data[63] = block.experts.down_proj.data[62].clone()
        
        C = compute_global_capability_vectors_v23(
            self.model, self.engine.moe_blocks, self.engine.current_num_experts, self.df_tokens, device="cpu"
        )
        distances = pairwise_capability_distances_v23(C)
        candidates = generate_candidate_pool_v23(distances, pool_size=5)
        
        self.assertEqual(len(candidates), 5)
        # Verify that pair (62, 63) is nominated due to identical representation
        self.assertIn((62, 63), candidates)
        # Verify all nominated indices satisfy 0 <= i < j < 64
        for i, j in candidates:
            self.assertTrue(0 <= i < j < 64)

    def test_invariant_4_dynamic_tracking_across_sequential_merges(self):
        """Invariant 4: As experts are merged from 64 down to 48, candidate generator adjusts N_t with zero index errors."""
        target_experts = 48
        step = 1
        
        while self.engine.current_num_experts > target_experts:
            current_n = self.engine.current_num_experts
            C = compute_global_capability_vectors_v23(
                self.model, self.engine.moe_blocks, current_n, self.df_tokens, device="cpu"
            )
            self.assertEqual(C.shape[0], current_n)
            
            distances = pairwise_capability_distances_v23(C)
            candidates = generate_candidate_pool_v23(distances, pool_size=5)
            
            # Assert candidate validity
            for i, j in candidates:
                self.assertTrue(0 <= i < j < current_n, f"Step {step}: Invalid pair ({i}, {j}) for N={current_n}")
                
            # Perform merge on candidate 0
            best_i, best_j = candidates[0]
            self.engine.merge_experts(best_i, best_j)
            self.assertEqual(self.engine.current_num_experts, current_n - 1)
            step += 1
            
        self.assertEqual(self.engine.current_num_experts, 48)

    def test_invariant_5_zero_stale_indices(self):
        """Invariant 5: Capability distances contain exactly N_t keys and no stale indices."""
        self.engine.merge_experts(10, 20)
        current_n = self.engine.current_num_experts
        self.assertEqual(current_n, 63)
        
        C = compute_global_capability_vectors_v23(
            self.model, self.engine.moe_blocks, current_n, self.df_tokens, device="cpu"
        )
        distances = pairwise_capability_distances_v23(C)
        self.assertEqual(len(distances), 63)
        self.assertNotIn(63, distances)

    def test_invariant_6_unfused_modulelist_compatibility(self):
        """Invariant 6: Dual-architecture dispatch functions cleanly on unfused nn.ModuleList models."""
        class MockUnfusedExpert(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.gate_proj = nn.Linear(dim, dim, bias=False)
                self.up_proj = nn.Linear(dim, dim, bias=False)
                self.down_proj = nn.Linear(dim, dim, bias=False)
                
        class MockUnfusedBlock(nn.Module):
            def __init__(self, num_experts, dim):
                super().__init__()
                self.experts = nn.ModuleList([MockUnfusedExpert(dim) for _ in range(num_experts)])
                self.gate = nn.Linear(dim, num_experts, bias=False)
                self.num_experts = num_experts

        class MockUnfusedModel(nn.Module):
            def __init__(self, num_experts=16, num_layers=2, dim=32):
                super().__init__()
                self.layers = nn.ModuleList([MockUnfusedBlock(num_experts, dim) for _ in range(num_layers)])
            def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
                batch, seq_len = input_ids.shape
                hidden_states = [torch.randn(batch, seq_len, 32) for _ in range(3)]
                class Out: pass
                o = Out()
                o.hidden_states = hidden_states
                return o

        unfused_model = MockUnfusedModel(num_experts=16, num_layers=2, dim=32)
        moe_blocks = [layer for layer in unfused_model.layers]
        C = compute_global_capability_vectors_v23(
            unfused_model, moe_blocks, 16, self.df_tokens, device="cpu"
        )
        self.assertEqual(C.shape, (16, 20))
        distances = pairwise_capability_distances_v23(C)
        candidates = generate_candidate_pool_v23(distances, pool_size=5)
        self.assertEqual(len(candidates), 5)
        for i, j in candidates:
            self.assertTrue(0 <= i < j < 16)

if __name__ == "__main__":
    unittest.main()
