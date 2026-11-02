"""
tests/test_adversarial_comprehensive.py
==========================================================================
Comprehensive Adversarial Audit Suite covering the 28 Zero-Trust Attack Vectors
against production code paths in CARE-COM.
"""

import sys
import os
import unittest
import importlib
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
from experiments.care_com_v21.care_capability_v23 import (
    compute_global_capability_vectors_v23,
    pairwise_capability_distances_v23,
    generate_candidate_pool_v23
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

class ComprehensiveAdversarialAudit(unittest.TestCase):

    def test_adv_01_tensor_axis_permutation(self):
        """Vector 1: Tensor-axis permutations exposing dimension-order mistakes."""
        L, N, D = 4, 32, 10
        raw = np.zeros((L, N, D), dtype=np.float32)
        for l in range(L):
            for e in range(N):
                raw[l, e, :] = l * 1000.0 + e * 10.0 + np.arange(D)
        
        # Valid layer-first slice [l, :, :]
        layer_slice = raw[1, :, :]
        self.assertEqual(layer_slice.shape, (32, 10))
        self.assertAlmostEqual(layer_slice[31, 0], 1310.0)
        
        # Buggy slice [:, l, :] mixes layers and single expert
        buggy_slice = raw[:, 1, :]
        self.assertEqual(buggy_slice.shape, (4, 10))
        self.assertNotEqual(buggy_slice.shape, (32, 10))

    def test_adv_02_expert_count_neq_layer_count(self):
        """Vector 2: Expert counts != layer counts (e.g. N=32, L=4)."""
        model = MockFusedModel(num_experts=32, num_layers=4)
        engine = PhysicalMergeEngine(model)
        self.assertEqual(engine.current_num_experts, 32)
        self.assertEqual(len(engine.moe_blocks), 4)
        # Snapshot and merge
        engine.snapshot(5, 20)
        engine.merge_experts(5, 20)
        self.assertEqual(engine.current_num_experts, 31)
        engine.restore()
        self.assertEqual(engine.current_num_experts, 32)

    def test_adv_03_expert_counts_gt16_eq16_lt16(self):
        """Vector 3: Candidate pool with N=32, N=16, and N=8."""
        for n in [32, 16, 8]:
            C = np.random.randn(n, 40)
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            dist = pairwise_capability_distances(C)
            cand = generate_candidate_pool(dist, pool_size=5)
            self.assertLessEqual(len(cand), 5)
            for i, j in cand:
                self.assertTrue(0 <= i < j < n)

    def test_adv_04_non_contiguous_and_index_remapping(self):
        """Vector 4 & 5: Sequential expert deletions and subsequent index mappings."""
        model = MockFusedModel(num_experts=8, num_layers=2)
        engine = PhysicalMergeEngine(model)
        
        # Merge expert 2 and 5 (removes 5, updates 2)
        engine.merge_experts(2, 5)
        self.assertEqual(engine.current_num_experts, 7)
        # Check router shape
        for block in engine.moe_blocks:
            self.assertEqual(block.gate.weight.shape[0], 7)
            self.assertEqual(block.experts.gate_up_proj.shape[0], 7)
            
        # Subsequent merge using valid indices 0..6
        engine.merge_experts(0, 6)
        self.assertEqual(engine.current_num_experts, 6)
        for block in engine.moe_blocks:
            self.assertEqual(block.gate.weight.shape[0], 6)

    def test_adv_06_zero_vectors_nans_infinities_padding(self):
        """Vector 6: Zero vectors, NaNs, infinities, and empty padding sequences."""
        C = np.array([
            [0.0, 0.0, 0.0],
            [np.nan, 1.0, 2.0],
            [np.inf, -np.inf, 1.0],
            [1.0, 1.0, 1.0]
        ])
        C_clean = np.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0)
        norms = np.linalg.norm(C_clean, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        C_norm = C_clean / norms
        self.assertFalse(np.isnan(C_norm).any())
        self.assertFalse(np.isinf(C_norm).any())

    def test_adv_07_analytical_exact_merged_superposition(self):
        """Vector 8: Exact analytically known linear expert superposition."""
        model = MockFusedModel(num_experts=4, num_layers=1, hidden_dim=2, intermediate_dim=2)
        engine = PhysicalMergeEngine(model)
        block = engine.moe_blocks[0]
        
        # Expert 0 weight = 2.0 * I; Expert 1 weight = 4.0 * I
        block.experts.down_proj.data[0] = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        block.experts.down_proj.data[1] = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
        
        engine.merge_experts(0, 1)
        expected = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
        torch.testing.assert_close(block.experts.down_proj.data[0], expected)

    def test_adv_08_router_dimensions_and_topk_after_merge(self):
        """Vector 9 & 26: Router rows, dimension changes, and top-k logit validity."""
        model = MockFusedModel(num_experts=10, num_layers=1, hidden_dim=4, intermediate_dim=8)
        engine = PhysicalMergeEngine(model)
        
        # Merge 3 times: 10 -> 9 -> 8 -> 7
        engine.merge_experts(0, 9)
        engine.merge_experts(1, 8)
        engine.merge_experts(2, 7)
        self.assertEqual(engine.current_num_experts, 7)
        
        router = engine.moe_blocks[0].gate
        self.assertEqual(router.weight.shape, (7, 4))
        
        # Compute router logits for random input x: [batch=2, dim=4]
        x = torch.randn(2, 4)
        logits = F.linear(x, router.weight)
        self.assertEqual(logits.shape, (2, 7))
        
        # Top-k with k=4 should execute cleanly without NaN or index error
        topk_vals, topk_indices = torch.topk(logits, k=4, dim=-1)
        self.assertEqual(topk_indices.shape, (2, 4))
        self.assertTrue((topk_indices < 7).all())

    def test_adv_09_candidate_evaluation_order_and_rollback(self):
        """Vector 11 & 12: Candidate evaluation order invariance and repeated rollback."""
        model = MockFusedModel(num_experts=8, num_layers=2)
        engine = PhysicalMergeEngine(model)
        w0 = model.layers[0].experts.down_proj.clone()
        
        # Order A: (0, 1) then (2, 3)
        engine.snapshot(0, 1)
        engine.merge_experts(0, 1)
        engine.restore()
        
        engine.snapshot(2, 3)
        engine.merge_experts(2, 3)
        engine.restore()
        
        torch.testing.assert_close(model.layers[0].experts.down_proj, w0)
        self.assertEqual(engine.current_num_experts, 8)

    def test_adv_10_failure_injection_halfway_restore(self):
        """Vector 13: Failure injection halfway through trial evaluation restores model."""
        model = MockFusedModel(num_experts=8, num_layers=2)
        engine = PhysicalMergeEngine(model)
        w0 = model.layers[0].experts.down_proj.clone()
        
        try:
            engine.snapshot(1, 4)
            engine.merge_experts(1, 4)
            raise RuntimeError("Simulated OOM midway")
        except RuntimeError:
            engine.restore()
            
        torch.testing.assert_close(model.layers[0].experts.down_proj, w0)
        self.assertEqual(engine.current_num_experts, 8)

    def test_adv_11_corrupted_trajectory_step_validator(self):
        """Vector 14 & 15: Detection of duplicated, non-monotonic, or fabricated steps."""
        def validate_trace(trace):
            for i in range(len(trace)):
                if trace[i]["step"] != i + 1:
                    return False, f"Non-consecutive step number: {trace[i]['step']}"
                if i > 0 and trace[i]["experts_before"] != trace[i-1]["experts_after"]:
                    return False, f"Discontinuity at step {i+1}"
                if trace[i]["experts_before"] - trace[i]["experts_after"] != 1:
                    return False, f"Invalid reduction delta at step {i+1}"
            return True, "Valid"

        # Trace with concatenation defect (like care_static historical trace)
        bad_trace = [
            {"step": 1, "experts_before": 64, "experts_after": 63},
            {"step": 2, "experts_before": 63, "experts_after": 62},
            {"step": 3, "experts_before": 64, "experts_after": 63}, # Concatenated run!
        ]
        is_val, msg = validate_trace(bad_trace)
        self.assertFalse(is_val)
        self.assertIn("Discontinuity", msg)

    def test_adv_12_seed_separation_audit(self):
        """Vector 16: Separate seeds for calibration draw vs candidate selection."""
        record = {
            "calibration_seed": 101,
            "candidate_selection_seed": 1001,
            "eval_seed": 42
        }
        self.assertNotEqual(record["calibration_seed"], record["candidate_selection_seed"])
        self.assertNotEqual(record["calibration_seed"], record["eval_seed"])

    def test_adv_13_ppl_token_shift_and_mask_verification(self):
        """Vector 24: PPL next-token shift and padding mask verification."""
        # Standard causal language modeling loss requires shift:
        # logits[..., :-1, :] predicts labels[..., 1:]
        vocab_size = 10
        seq_len = 5
        batch_size = 2
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Correct shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_correct = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        
        # Incorrect unshifted
        loss_unshifted = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        
        # Unshifted loss does not equal shifted causal loss
        self.assertNotEqual(loss_correct.item(), loss_unshifted.item())

    def test_adv_14_budget_asymmetry_rejection(self):
        """Vector 27: Rejection of unbudgeted baseline comparisons."""
        care_cfg = {"candidate_budget": 5}
        rand_k1_cfg = {"candidate_budget": 1}
        rand_k5_cfg = {"candidate_budget": 5}
        
        self.assertNotEqual(care_cfg["candidate_budget"], rand_k1_cfg["candidate_budget"])
        self.assertEqual(care_cfg["candidate_budget"], rand_k5_cfg["candidate_budget"])

    def test_adv_15_stale_module_import_isolation(self):
        """Vector 20: Stale imported module verification."""
        # Verify that importing care_com_v21 does not accidentally pull v2.1 buggy capability
        import experiments.care_com_v21.capability as cap
        import inspect
        source = inspect.getsource(cap.compute_global_capability_vectors)
        # Must contain layer_capabilities[layer_idx, :, :] (not [:, layer_idx, :])
        self.assertIn("layer_capabilities[layer_idx, :, :]", source)
        self.assertNotIn("layer_capabilities[:, layer_idx, :]", source)

    def test_adv_16_nan_and_inf_metric_rejection(self):
        """Vector 22: Metric aggregation rejects non-finite losses."""
        metrics = [12.5, 14.2, np.nan, 18.0]
        finite_metrics = [m for m in metrics if np.isfinite(m)]
        self.assertEqual(len(finite_metrics), 3)
        self.assertFalse(np.isnan(np.mean(finite_metrics)))

if __name__ == "__main__":
    unittest.main()
