"""
tests/test_router_semantics_and_identity.py
===========================================
Unit tests for:
1. Fix 2: CARE Router Semantics
   - Historical uniform arithmetic mean: (W_i + W_j) / 2.0 (default, used across all historical experiments)
   - Manuscript Eq. 6 frequency-weighted merging: (f_i W_i + f_j W_j) / (f_i + f_j)
   - Bitwise transactional snapshot/restore parity
2. Fix 3: CARE-Adaptive Identity Tracking
   - active_to_original tracking through sequential multi-step merges
   - original_to_cluster tracking of merged expert clusters
   - Local vs. original ID logging and candidate pool translation
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
from typing import Dict, List

# Add repository root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.care_com_v21.core import PhysicalMergeEngine, generate_candidate_pool


class MockFusedMoEBlock(nn.Module):
    """Mock OLMoE-style fused MoE block for testing."""
    def __init__(self, num_experts=8, hidden_dim=16, intermediate_dim=32):
        super().__init__()
        self.num_experts_per_tok = 2
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        
        # Experts with fused gate_up_proj and down_proj
        self.experts = nn.Module()
        self.experts.num_experts = num_experts
        self.experts.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_dim, hidden_dim) * 0.1
        )
        self.experts.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_dim, intermediate_dim) * 0.1
        )
        
        # Router
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        # Initialize gate weights with deterministic identifiable values
        with torch.no_grad():
            for e in range(num_experts):
                self.gate.weight[e] = (e + 1) * 1.0


class MockMoEModel(nn.Module):
    """Mock Causal MoE model with 2 MoE blocks."""
    def __init__(self, num_experts=8, num_layers=2):
        super().__init__()
        self.config = type("Config", (), {
            "num_experts": num_experts,
            "num_local_experts": num_experts,
            "hidden_size": 16,
            "num_layers": num_layers,
        })()
        self.blocks = nn.ModuleList([MockFusedMoEBlock(num_experts=num_experts) for _ in range(num_layers)])


class TestRouterSemantics(unittest.TestCase):
    """
    Verifies CARE router merging semantics.
    Historical Note:
      All historical experiments (care_com_v21, care_com_v23, and benchmark runs)
      empirically executed 'uniform' router merging: (W_i + W_j) / 2.0.
      'frequency_weighted' implements the theoretical formulation in manuscript Eq. 6:
      W_merged = (f_i * W_i + f_j * W_j) / (f_i + f_j).
    """

    def setUp(self):
        torch.manual_seed(42)
        self.model = MockMoEModel(num_experts=8, num_layers=2)

    def test_uniform_router_merging(self):
        """Tests that router_merge_mode='uniform' performs exact (W_i + W_j) / 2.0 averaging."""
        engine = PhysicalMergeEngine(self.model, router_merge_mode="uniform")
        
        block = engine.moe_blocks[0]
        w0 = block.gate.weight[0].clone()
        w1 = block.gate.weight[1].clone()
        expected_uniform = (w0 + w1) / 2.0
        
        engine.merge_experts(0, 1)
        
        # Merged router row at index 0 should equal expected_uniform
        self.assertTrue(torch.allclose(block.gate.weight[0], expected_uniform, atol=1e-6))
        self.assertEqual(engine.current_num_experts, 7)
        self.assertEqual(block.gate.weight.shape[0], 7)

    def test_frequency_weighted_router_merging(self):
        """Tests that router_merge_mode='frequency_weighted' matches manuscript Eq. 6."""
        # Frequencies: expert 0 has freq 0.8, expert 1 has freq 0.2
        frequencies = {0: torch.tensor([0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                       1: torch.tensor([0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])}
        engine = PhysicalMergeEngine(self.model, router_merge_mode="frequency_weighted", expert_frequencies=frequencies)
        
        block = engine.moe_blocks[0]
        w0 = block.gate.weight[0].clone()
        w1 = block.gate.weight[1].clone()
        expected_freq_weighted = (0.8 * w0 + 0.2 * w1) / (0.8 + 0.2)
        
        engine.merge_experts(0, 1)
        
        self.assertTrue(torch.allclose(block.gate.weight[0], expected_freq_weighted, atol=1e-6))
        self.assertEqual(engine.current_num_experts, 7)

    def test_frequency_weighted_consecutive_merges(self):
        """Tests that frequency_weighted accumulates cluster frequencies correctly across consecutive merges."""
        # Layer 0: expert 0 has freq 0.8, expert 1 has freq 0.2, expert 2 has freq 0.5
        frequencies = {
            0: [0.8, 0.2, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
            1: [0.8, 0.2, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
        }
        engine = PhysicalMergeEngine(self.model, router_merge_mode="frequency_weighted", expert_frequencies=frequencies)
        block = engine.moe_blocks[0]
        w0 = block.gate.weight[0].clone()
        w1 = block.gate.weight[1].clone()
        w2 = block.gate.weight[2].clone()

        # Merge 1: local (0, 1) [orig 0, 1]
        engine.merge_experts(0, 1)
        expected_w01 = (0.8 * w0 + 0.2 * w1) / 1.0
        self.assertTrue(torch.allclose(block.gate.weight[0], expected_w01, atol=1e-6))
        # Frequency of active expert 0 should now be 0.8 + 0.2 = 1.0
        self.assertAlmostEqual(engine.expert_frequencies[0][0], 1.0)
        self.assertEqual(len(engine.expert_frequencies[0]), 7)

        # Merge 2: local (0, 1) which corresponds to cluster {0, 1} and original expert 2
        engine.merge_experts(0, 1)
        expected_w012 = (1.0 * expected_w01 + 0.5 * w2) / 1.5
        self.assertTrue(torch.allclose(block.gate.weight[0], expected_w012, atol=1e-6))
        # Total frequency of cluster {0, 1, 2} should be 1.5
        self.assertAlmostEqual(engine.expert_frequencies[0][0], 1.5)
        self.assertEqual(len(engine.expert_frequencies[0]), 6)

    def test_frequency_weighted_fallback_when_zero(self):
        """Tests that frequency_weighted gracefully falls back to uniform when total frequency is 0."""
        frequencies = torch.zeros(8)
        engine = PhysicalMergeEngine(self.model, router_merge_mode="frequency_weighted", expert_frequencies=frequencies)
        
        block = engine.moe_blocks[0]
        w0 = block.gate.weight[0].clone()
        w1 = block.gate.weight[1].clone()
        expected_uniform = (w0 + w1) / 2.0
        
        engine.merge_experts(0, 1)
        self.assertTrue(torch.allclose(block.gate.weight[0], expected_uniform, atol=1e-6))

    def test_transactional_snapshot_restore_parity(self):
        """Tests that snapshot and restore restore exact weights and identity states bitwise."""
        engine = PhysicalMergeEngine(self.model, router_merge_mode="uniform")
        
        block = engine.moe_blocks[0]
        orig_router = block.gate.weight.clone()
        orig_gate_up = block.experts.gate_up_proj.clone()
        orig_active = list(engine.active_to_original)
        orig_clusters = {k: list(v) for k, v in engine.original_to_cluster.items()}
        
        engine.snapshot(0, 1)
        engine.merge_experts(0, 1)
        
        # State modified
        self.assertEqual(engine.current_num_experts, 7)
        self.assertEqual(len(engine.active_to_original), 7)
        
        # Restore
        engine.restore()
        
        # Assert bitwise parity
        self.assertEqual(engine.current_num_experts, 8)
        self.assertEqual(engine.active_to_original, orig_active)
        self.assertEqual(engine.original_to_cluster, orig_clusters)
        self.assertTrue(torch.equal(block.gate.weight, orig_router))
        self.assertTrue(torch.equal(block.experts.gate_up_proj, orig_gate_up))
        self.assertEqual(self.model.config.num_local_experts, 8)


class TestIdentityTracking(unittest.TestCase):
    """
    Verifies CARE-Adaptive active-to-original identity tracking.
    Guarantees:
      1. active_to_original length == current_num_experts at every step.
      2. Merged pairs update surviving cluster lists.
      3. Deleted expert indices never reappear in active pool.
      4. Multi-step merges accurately record local vs original pairs.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.model = MockMoEModel(num_experts=8, num_layers=2)

    def test_initial_identity_mapping(self):
        """Initial state: local index i maps to original ID i."""
        engine = PhysicalMergeEngine(self.model)
        self.assertEqual(engine.active_to_original, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(engine.get_active_mapping(), {i: i for i in range(8)})
        for i in range(8):
            self.assertEqual(engine.get_cluster(i), [i])

    def test_single_merge_identity(self):
        """Merging local (1, 3): original 3 merged into original 1, 3 removed from active."""
        engine = PhysicalMergeEngine(self.model)
        engine.merge_experts(1, 3)
        
        # Active length 7
        self.assertEqual(engine.current_num_experts, 7)
        # Local indices:
        # local 0 -> orig 0
        # local 1 -> orig 1 (cluster [1, 3])
        # local 2 -> orig 2
        # local 3 -> orig 4 (formerly local 4 shifted down)
        # local 4 -> orig 5
        # local 5 -> orig 6
        # local 6 -> orig 7
        expected_active = [0, 1, 2, 4, 5, 6, 7]
        self.assertEqual(engine.active_to_original, expected_active)
        self.assertEqual(engine.get_cluster(1), [1, 3])
        self.assertEqual(engine.get_original_id(1), 1)
        self.assertEqual(engine.get_original_id(3), 4)

    def test_four_step_sequential_merges(self):
        """Simulates 4 sequential merges (8 -> 7 -> 6 -> 5 -> 4 experts) and verifies tracking."""
        engine = PhysicalMergeEngine(self.model)
        
        # Step 1: Merge local (0, 1) [orig 0, 1] -> 7 experts
        orig_i = engine.get_original_id(0)
        orig_j = engine.get_original_id(1)
        self.assertEqual((orig_i, orig_j), (0, 1))
        engine.merge_experts(0, 1)
        self.assertEqual(engine.current_num_experts, 7)
        self.assertEqual(engine.active_to_original, [0, 2, 3, 4, 5, 6, 7])
        self.assertEqual(engine.get_cluster(0), [0, 1])
        
        # Step 2: Merge local (0, 1) [orig 0, 2] -> 6 experts
        orig_i = engine.get_original_id(0)
        orig_j = engine.get_original_id(1)
        self.assertEqual((orig_i, orig_j), (0, 2))
        engine.merge_experts(0, 1)
        self.assertEqual(engine.current_num_experts, 6)
        self.assertEqual(engine.active_to_original, [0, 3, 4, 5, 6, 7])
        self.assertEqual(engine.get_cluster(0), [0, 1, 2])
        
        # Step 3: Merge local (2, 3) [orig 4, 5] -> 5 experts
        # active is [0, 3, 4, 5, 6, 7], so local 2 is orig 4, local 3 is orig 5
        orig_i = engine.get_original_id(2)
        orig_j = engine.get_original_id(3)
        self.assertEqual((orig_i, orig_j), (4, 5))
        engine.merge_experts(2, 3)
        self.assertEqual(engine.current_num_experts, 5)
        self.assertEqual(engine.active_to_original, [0, 3, 4, 6, 7])
        self.assertEqual(engine.get_cluster(2), [4, 5])
        
        # Step 4: Merge local (0, 2) [orig 0, 4] -> 4 experts
        # active is [0, 3, 4, 6, 7], so local 0 is orig 0, local 2 is orig 4
        orig_i = engine.get_original_id(0)
        orig_j = engine.get_original_id(2)
        self.assertEqual((orig_i, orig_j), (0, 4))
        engine.merge_experts(0, 2)
        self.assertEqual(engine.current_num_experts, 4)
        self.assertEqual(engine.active_to_original, [0, 3, 6, 7])
        # Cluster of local 0 now contains original {0, 1, 2, 4, 5}
        self.assertEqual(engine.get_cluster(0), [0, 1, 2, 4, 5])
        
        # Config check
        self.assertEqual(self.model.config.num_local_experts, 4)
        self.assertEqual(self.model.config.num_experts, 4)


if __name__ == "__main__":
    unittest.main()
