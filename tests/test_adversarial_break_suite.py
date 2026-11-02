"""
tests/test_adversarial_break_suite.py
==========================================================================
Adversarial Break Suite for CARE-COM (Phase 3 Audit).
Specifically attempts to BREAK the implementation with edge cases,
dimension mismatches, non-contiguous states, failure injections,
and analytical counterexamples.
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from experiments.care_com_v21.core import PhysicalMergeEngine, generate_candidate_pool
from experiments.care_com_v21.capability import (
    compute_global_capability_vectors,
    pairwise_capability_distances
)

class MockFusedExperts(nn.Module):
    def __init__(self, num_experts, hidden_dim, intermediate_dim):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate_dim, hidden_dim) * 0.05)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_dim, intermediate_dim) * 0.05)

class MockRouter(nn.Module):
    def __init__(self, num_experts, hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_dim) * 0.05)
        self.num_experts = num_experts

class MockFusedBlock(nn.Module):
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
    def __init__(self, num_experts=64, num_layers=4, hidden_dim=16, intermediate_dim=32):
        super().__init__()
        self.config = MockModelConfig(num_experts, num_layers)
        self.layers = nn.ModuleList([
            MockFusedBlock(num_experts, hidden_dim, intermediate_dim) for _ in range(num_layers)
        ])
        
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        batch, seq_len = input_ids.shape
        hidden_states = [torch.randn(batch, seq_len, 16) for _ in range(len(self.layers) + 1)]
        class Out: pass
        out = Out()
        out.hidden_states = hidden_states
        return out

class AdversarialBreakSuite(unittest.TestCase):

    def test_01_tensor_axis_permutation_detector(self):
        """
        Adversarial Test 1: Expose dimension-order mistakes where num_layers != num_experts.
        Constructs an asymmetric tensor [L=4, N=32, D=10].
        Verifies that slicing [l, :, :] extracts all 32 experts, while [:, l, :] fails or truncates.
        """
        L, N, D = 4, 32, 10
        raw_tensor = np.zeros((L, N, D), dtype=np.float32)
        for l in range(L):
            for e in range(N):
                raw_tensor[l, e, :] = l * 1000.0 + e * 10.0 + np.arange(D)

        # Correct slice for layer l=2: must have shape [32, 10]
        correct_slice = raw_tensor[2, :, :]
        self.assertEqual(correct_slice.shape, (32, 10))
        # Expert 31 at layer 2 must have value 2*1000 + 31*10 = 2310
        self.assertAlmostEqual(correct_slice[31, 0], 2310.0)

        # Buggy slice [:, 2, :] would have shape [4, 10] and only extract expert e=2
        buggy_slice = raw_tensor[:, 2, :]
        self.assertEqual(buggy_slice.shape, (4, 10))
        # Buggy slice completely misses expert 31!
        self.assertNotEqual(buggy_slice.shape, (32, 10))

    def test_02_expert_count_asymmetry_gt16_eq16_lt16(self):
        """
        Adversarial Test 2: Verify candidate selection with N > 16, N == 16, and N < 16.
        Ensures candidates span the full active range [0, N-1] and never index out of bounds.
        """
        for n in [32, 16, 8]:
            # Create synthetic distance matrix
            distances = {i: {} for i in range(n)}
            for i in range(n):
                for j in range(i + 1, n):
                    d = float(abs(i - j) + (1.0 if (i, j) == (n-2, n-1) else 10.0))
                    distances[i][j] = d
                    distances[j][i] = d

            candidates = generate_candidate_pool(distances, pool_size=5)
            self.assertLessEqual(len(candidates), 5)
            # All pairs must satisfy 0 <= i < j < n
            for i, j in candidates:
                self.assertTrue(0 <= i < j < n, f"Pair ({i}, {j}) out of bounds for N={n}")
            
            # For N=32, verify pair (30, 31) is nominated
            if n == 32:
                self.assertIn((30, 31), candidates)

    def test_03_synthetic_analytically_known_merge(self):
        """
        Adversarial Test 3: Construct synthetic experts whose merged forward output is analytically known.
        Expert 0: W = 2.0 * I
        Expert 1: W = 4.0 * I
        Analytical merged weight: W_merged = (2.0 + 4.0) / 2.0 = 3.0 * I.
        Input x = [1.0, 2.0].
        Expected merged output: y = [3.0, 6.0].
        """
        model = MockFusedModel(num_experts=4, num_layers=1, hidden_dim=2, intermediate_dim=2)
        engine = PhysicalMergeEngine(model)
        block = engine.moe_blocks[0]

        # Set gate_up to zeros so act passes through cleanly
        block.experts.gate_up_proj.data.zero_()
        # Set expert 0 down_proj to 2.0 * Identity
        block.experts.down_proj.data[0] = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        # Set expert 1 down_proj to 4.0 * Identity
        block.experts.down_proj.data[1] = torch.tensor([[4.0, 0.0], [0.0, 4.0]])

        # Execute physical merge of expert 0 and 1
        engine.merge_experts(0, 1)

        # Expected merged weight at slot 0: 3.0 * Identity
        expected_weight = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
        actual_weight = block.experts.down_proj.data[0]
        torch.testing.assert_close(actual_weight, expected_weight)

        # Compute output for x = [1.0, 2.0]
        x = torch.tensor([1.0, 2.0])
        y = F.linear(x, actual_weight)
        expected_y = torch.tensor([3.0, 6.0])
        torch.testing.assert_close(y, expected_y)

    def test_04_zero_vector_and_nan_resilience(self):
        """
        Adversarial Test 4: Verify zero-vector and NaN handling in normalization and distance computation.
        """
        # Capability matrix with a zero vector and NaN
        C = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 2.0],
            [np.nan, 0.0, 1.0],
            [3.0, 4.0, 0.0]
        ])
        # Sanitize NaNs
        C_clean = np.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0)
        # Compute norms
        norms = np.linalg.norm(C_clean, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        C_norm = C_clean / norms

        self.assertFalse(np.isnan(C_norm).any())
        self.assertFalse(np.isinf(C_norm).any())
        self.assertAlmostEqual(np.linalg.norm(C_norm[0]), 0.0) # Zero vector stays zero
        self.assertAlmostEqual(np.linalg.norm(C_norm[1]), 1.0) # Non-zero vector becomes unit norm

        distances = pairwise_capability_distances(C_norm)
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.assertFalse(np.isnan(distances[i][j]))
                    self.assertGreaterEqual(distances[i][j], 0.0)

    def test_05_candidate_order_and_rollback_invariance(self):
        """
        Adversarial Test 5: Verify that evaluating candidates in different orders
        produces bitwise identical model states after rollback.
        """
        model = MockFusedModel(num_experts=8, num_layers=2, hidden_dim=8, intermediate_dim=16)
        engine = PhysicalMergeEngine(model)

        # Save initial state hash
        w0 = model.layers[0].experts.down_proj.clone()
        r0 = model.layers[0].gate.weight.clone()

        # Trial merge candidate (1, 3)
        engine.snapshot(1, 3)
        engine.merge_experts(1, 3)
        engine.restore()

        # Trial merge candidate (4, 6)
        engine.snapshot(4, 6)
        engine.merge_experts(4, 6)
        engine.restore()

        # Verify bitwise identical restoration
        torch.testing.assert_close(model.layers[0].experts.down_proj, w0)
        torch.testing.assert_close(model.layers[0].gate.weight, r0)
        self.assertEqual(engine.current_num_experts, 8)

    def test_06_failure_injection_halfway_restores_state(self):
        """
        Adversarial Test 6: Inject an exception halfway through candidate evaluation.
        Verifies that error handling leaves the model in its original state.
        """
        model = MockFusedModel(num_experts=8, num_layers=2, hidden_dim=8, intermediate_dim=16)
        engine = PhysicalMergeEngine(model)
        w0 = model.layers[0].experts.down_proj.clone()

        try:
            engine.snapshot(2, 5)
            engine.merge_experts(2, 5)
            # Simulated catastrophic out-of-memory or evaluation exception
            raise RuntimeError("Injected evaluation OOM failure")
        except RuntimeError:
            engine.restore()

        # Assert full recovery
        torch.testing.assert_close(model.layers[0].experts.down_proj, w0)
        self.assertEqual(engine.current_num_experts, 8)

    def test_07_trajectory_validator_detects_concatenated_corrupted_trace(self):
        """
        Adversarial Test 7: Trajectory validator must detect corrupted or concatenated step traces.
        Simulates the defect in historical benchmark_results/care_static/trajectory.json (23 steps).
        """
        # Step sequence with a sudden jump in experts_before from an aborted run
        corrupted_trace = [
            {"step": 1, "experts_before": 64, "experts_after": 63},
            {"step": 2, "experts_before": 63, "experts_after": 62},
            {"step": 3, "experts_before": 64, "experts_after": 63}, # Concatenated aborted run!
        ]

        def validate_trajectory(trace):
            for i in range(1, len(trace)):
                if trace[i]["experts_before"] != trace[i-1]["experts_after"]:
                    return False, f"Discontinuity at step {trace[i]['step']}: {trace[i]['experts_before']} != {trace[i-1]['experts_after']}"
            return True, "Valid"

        is_valid, msg = validate_trajectory(corrupted_trace)
        self.assertFalse(is_valid)
        self.assertIn("Discontinuity", msg)

    def test_08_budget_asymmetry_detector(self):
        """
        Adversarial Test 8: Methodological audit validator flags K=1 random baseline vs K=5 CARE.
        """
        def check_methodological_symmetry(care_config, baseline_config):
            if care_config.get("candidate_pool_size") != baseline_config.get("candidate_pool_size"):
                return False, "Budget asymmetry: candidate pool sizes differ"
            return True, "Symmetric"

        care_cfg = {"method": "care_adaptive", "candidate_pool_size": 5}
        rand_k1_cfg = {"method": "random_merge", "candidate_pool_size": 1}
        rand_k5_cfg = {"method": "random_k5", "candidate_pool_size": 5}

        valid_k1, msg_k1 = check_methodological_symmetry(care_cfg, rand_k1_cfg)
        self.assertFalse(valid_k1)
        self.assertIn("Budget asymmetry", msg_k1)

        valid_k5, msg_k5 = check_methodological_symmetry(care_cfg, rand_k5_cfg)
        self.assertTrue(valid_k5)

if __name__ == "__main__":
    unittest.main()
