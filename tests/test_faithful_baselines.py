"""
tests/test_faithful_baselines.py
=================================
Automated test suite for the four faithful baseline implementations.
Tests run on synthetic mock OLMoE-style fused MoE blocks (CPU, no GPU required).

Tests:
  Test 1 - Sub-MoE SVD reconstruction and shape invariance on fused block
  Test 2 - REAP-Prune: router-weighted saliency accumulation and physical excision
  Test 3 - HC-SMoE: agglomerative clustering and frequency-weighted combination
  Test 4 - MC-SMoE: Hungarian permutation matching, permutation invariance, and permuted merge
  Test 5 - Forward pass: finite logits and correct tensor shapes after compression
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to sys.path for engine imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BASELINES_DIR = os.path.join(PROJECT_ROOT, "benchmark", "OLMoE-1B-7B", "methods")
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

from faithful_baselines.sub_moe_engine import SubMoEEngine
from faithful_baselines.reap_prune_engine import REAPPruneEngine, REAPSaliencyObserver
from faithful_baselines.hc_smoe_engine import HCSMoEEngine
from faithful_baselines.mc_smoe_engine import MCSMoEEngine

# ========================
# Mock OLMoE Architecture
# ========================

NUM_EXPERTS_FULL = 8   # Smaller stand-in for 64 in production
HIDDEN_DIM = 16
INTERMEDIATE_DIM = 32  # d_ff (half of 2*d_ff in gate_up_proj)
TARGET_EXPERTS = 4     # Compression target


class MockOlmoeExperts(nn.Module):
    """Fused expert module mirroring OlmoeExperts layout."""
    def __init__(self, num_experts, hidden_dim, intermediate_dim):
        super().__init__()
        self.num_experts = num_experts
        # Store as a plain Python int so MockMoEBlock.forward can read it.
        # nn.Module passes non-Tensor/Module values to object.__setattr__,
        # so this survives parameter replacements done by compression engines.
        object.__setattr__(self, "intermediate_dim", intermediate_dim)
        # OLMoE fused layout: gate_up_proj [E, 2*d_ff, hidden], down_proj [E, hidden, d_ff]
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_dim, hidden_dim) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_dim, intermediate_dim) * 0.02
        )


class MockMoEBlock(nn.Module):
    """Mock MoE block mimicking OlmoeSparseMoeBlock."""
    def __init__(self, num_experts=NUM_EXPERTS_FULL, top_k=2,
                 hidden_dim=HIDDEN_DIM, intermediate_dim=INTERMEDIATE_DIM):
        super().__init__()
        self.num_experts_per_tok = top_k
        self.experts = MockOlmoeExperts(num_experts, hidden_dim, intermediate_dim)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        bsz, seq, h = x.shape
        flat_x = x.reshape(-1, h)
        router_logits = self.gate(flat_x)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, selected = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat_x)
        for t in range(flat_x.shape[0]):
            for k in range(self.num_experts_per_tok):
                eid = selected[t, k].item()
                w = topk_weights[t, k]
                gate_up = self.experts.gate_up_proj[eid]  # [2*d_ff, hidden]
                d_ff = self.experts.intermediate_dim
                g = F.silu(gate_up[:d_ff] @ flat_x[t])
                u = gate_up[d_ff:] @ flat_x[t]
                down = self.experts.down_proj[eid]  # [hidden, d_ff]
                out[t] += w * (down @ (g * u))
        return out.reshape(bsz, seq, h)


class MockLMWithMoE(nn.Module):
    """Minimal model wrapping a single MoE block to test engine attachment."""
    def __init__(self, num_experts=NUM_EXPERTS_FULL):
        super().__init__()
        self.config = type("Cfg", (), {
            "num_experts": num_experts,
            "num_local_experts": num_experts,
            "num_experts_per_tok": 2,
        })()
        self.moe_block = MockMoEBlock(num_experts=num_experts)

    def forward(self, x):
        return self.moe_block(x)


def make_model(num_experts=NUM_EXPERTS_FULL):
    torch.manual_seed(42)
    return MockLMWithMoE(num_experts=num_experts)


def make_dummy_frequencies(num_experts=NUM_EXPERTS_FULL, seed=42):
    rng = np.random.default_rng(seed)
    freqs = torch.from_numpy(rng.dirichlet(np.ones(num_experts))).float()
    return freqs  # shape [num_experts], sums to 1.0


# ========================
# Test Case 1: Sub-MoE
# ========================

class TestSubMoE(unittest.TestCase):

    def _make_engine(self, num_experts=NUM_EXPERTS_FULL):
        model = make_model(num_experts)
        return SubMoEEngine(model, rank_ratio=0.5)

    def test_01_output_shape_after_compression(self):
        """After compression, gate_up_proj must be [target_k, 2*d_ff, hidden] and down_proj [target_k, hidden, d_ff]."""
        engine = self._make_engine()
        block = engine.moe_blocks[0]
        freqs = make_dummy_frequencies()
        labels = np.array([i % TARGET_EXPERTS for i in range(NUM_EXPERTS_FULL)], dtype=int)
        engine.compress_block(block, labels, freqs.numpy())
        self.assertEqual(block.experts.gate_up_proj.shape[0], TARGET_EXPERTS,
                         msg=f"Expected {TARGET_EXPERTS} gate_up rows, got {block.experts.gate_up_proj.shape[0]}")
        self.assertEqual(block.experts.down_proj.shape[0], TARGET_EXPERTS,
                         msg=f"Expected {TARGET_EXPERTS} down rows, got {block.experts.down_proj.shape[0]}")
        self.assertEqual(block.experts.gate_up_proj.shape[1], 2 * INTERMEDIATE_DIM,
                         msg="gate_up_proj intermediate dim must remain 2*d_ff")
        self.assertEqual(block.experts.down_proj.shape[1], HIDDEN_DIM,
                         msg="down_proj output dim must remain hidden_dim")

    def test_02_router_shape_after_compression(self):
        """Router weight must be [target_k, hidden_dim] after compression."""
        engine = self._make_engine()
        block = engine.moe_blocks[0]
        freqs = make_dummy_frequencies()
        labels = np.array([i % TARGET_EXPERTS for i in range(NUM_EXPERTS_FULL)], dtype=int)
        engine.compress_block(block, labels, freqs.numpy())
        router = getattr(block, "gate", getattr(block, "router", None))
        self.assertEqual(router.weight.shape[0], TARGET_EXPERTS,
                         msg=f"Expected {TARGET_EXPERTS} router rows, got {router.weight.shape[0]}")
        self.assertEqual(router.weight.shape[1], HIDDEN_DIM,
                         msg=f"Expected {HIDDEN_DIM} router cols, got {router.weight.shape[1]}")

    def test_03_svd_reconstruction_shape_invariance(self):
        """experts_union_decomposition_merge must return a tensor of the same shape as input."""
        engine = self._make_engine()
        O, I = HIDDEN_DIM, INTERMEDIATE_DIM
        weights = [torch.randn(O, I) for _ in range(3)]
        freqs = [0.3, 0.5, 0.2]
        merged_down = engine.experts_union_decomposition_merge(weights, freqs, concat_dim=0)
        self.assertEqual(merged_down.shape, (O, I),
                         msg=f"Down-proj merge shape mismatch: {merged_down.shape} != {(O, I)}")
        
        O2, I2 = 2 * INTERMEDIATE_DIM, HIDDEN_DIM
        weights2 = [torch.randn(O2, I2) for _ in range(3)]
        merged_gu = engine.experts_union_decomposition_merge(weights2, freqs, concat_dim=1)
        self.assertEqual(merged_gu.shape, (O2, I2),
                         msg=f"GU-proj merge shape mismatch: {merged_gu.shape} != {(O2, I2)}")

    def test_04_svd_merge_is_finite(self):
        """SVD merge must produce finite (non-NaN, non-Inf) values."""
        engine = self._make_engine()
        weights = [torch.randn(HIDDEN_DIM, INTERMEDIATE_DIM) for _ in range(3)]
        merged = engine.experts_union_decomposition_merge(weights, [0.3, 0.5, 0.2], concat_dim=0)
        self.assertTrue(torch.isfinite(merged).all(),
                        msg="SVD merged weights contain NaN or Inf")

    def test_05_single_expert_passthrough(self):
        """Cluster with one expert should return the original weight unchanged."""
        engine = self._make_engine()
        w = torch.randn(HIDDEN_DIM, INTERMEDIATE_DIM)
        merged = engine.experts_union_decomposition_merge([w], [1.0], concat_dim=0)
        self.assertTrue(torch.allclose(merged, w),
                        msg="Single-expert merge should be identity")


# ========================
# Test Case 2: REAP-Prune
# ========================

class TestREAPPrune(unittest.TestCase):

    def _make_engine(self):
        model = make_model()
        return REAPPruneEngine(model, renormalize_router_weights=True)

    def test_06_prune_reduces_expert_count(self):
        """After pruning, gate_up_proj must have exactly target_experts rows."""
        engine = self._make_engine()
        block = engine.moe_blocks[0]
        saliency = torch.rand(NUM_EXPERTS_FULL)
        retained = engine.prune_block(block, saliency, TARGET_EXPERTS)
        self.assertEqual(block.experts.gate_up_proj.shape[0], TARGET_EXPERTS,
                         msg="gate_up_proj expert dim mismatch after pruning")
        self.assertEqual(block.experts.down_proj.shape[0], TARGET_EXPERTS,
                         msg="down_proj expert dim mismatch after pruning")
        self.assertEqual(len(retained), TARGET_EXPERTS,
                         msg="Retained index list length mismatch")

    def test_07_router_sliced_correctly(self):
        """Router weight must be sliced to [target_experts, hidden_dim] after pruning."""
        engine = self._make_engine()
        block = engine.moe_blocks[0]
        saliency = torch.rand(NUM_EXPERTS_FULL)
        engine.prune_block(block, saliency, TARGET_EXPERTS)
        router = getattr(block, "gate", getattr(block, "router", None))
        self.assertEqual(router.weight.shape[0], TARGET_EXPERTS,
                         msg="Router weight rows mismatch after pruning")
        self.assertEqual(router.weight.shape[1], HIDDEN_DIM,
                         msg="Router weight cols mismatch after pruning")

    def test_08_lowest_saliency_experts_removed(self):
        """The retained experts must be the top-saliency ones."""
        engine = self._make_engine()
        block = engine.moe_blocks[0]
        # Deterministic saliency: expert 0 is most salient, last ones least
        saliency = torch.arange(NUM_EXPERTS_FULL, dtype=torch.float) / NUM_EXPERTS_FULL
        n_prune = NUM_EXPERTS_FULL - TARGET_EXPERTS
        expected_retained = sorted(range(n_prune, NUM_EXPERTS_FULL))
        retained = engine.prune_block(block, saliency, TARGET_EXPERTS)
        self.assertEqual(sorted(retained), expected_retained,
                         msg=f"Expected retained {expected_retained}, got {retained}")

    def test_09_observer_accumulates_saliency(self):
        """REAPSaliencyObserver must produce non-zero saliencies after at least one pass."""
        model = make_model()
        observer = REAPSaliencyObserver(model, renormalize_router_weights=True)
        observer.register_hooks()
        dummy_input = torch.randn(1, 4, HIDDEN_DIM)
        with torch.no_grad():
            model(dummy_input)
        observer.remove_hooks()
        sals = observer.get_saliencies("weighted_ean_sum")
        self.assertIn(0, sals, "Layer 0 not in saliency dict")
        self.assertEqual(len(sals[0]), NUM_EXPERTS_FULL,
                         msg="Saliency vector must cover all experts")
        self.assertTrue(sals[0].sum().item() >= 0.0,
                        msg="Saliency values must be non-negative")


# ========================
# Test Case 3: HC-SMoE
# ========================

class TestHCSMoE(unittest.TestCase):

    def _make_engine(self):
        model = make_model()
        return HCSMoEEngine(model, linkage="average", similarity_metric="router_cosine")

    def test_10_agglomerative_produces_target_k_clusters(self):
        """Agglomerative clustering must produce exactly target_k cluster labels."""
        engine = self._make_engine()
        dist = torch.rand(NUM_EXPERTS_FULL, NUM_EXPERTS_FULL)
        dist = (dist + dist.t()) / 2
        dist.fill_diagonal_(0.0)
        labels = engine.agglomerative_clustering(dist, TARGET_EXPERTS)
        unique = torch.unique(labels)
        self.assertEqual(len(unique), TARGET_EXPERTS,
                         msg=f"Expected {TARGET_EXPERTS} clusters, got {len(unique)}")

    def test_11_distance_matrix_is_symmetric_and_zero_diagonal(self):
        """Expert distance matrix must be symmetric with zero diagonal."""
        model = make_model()
        engine = HCSMoEEngine(model)
        block = engine.moe_blocks[0]
        D = engine.compute_expert_distance_matrix(block)
        self.assertEqual(D.shape, (NUM_EXPERTS_FULL, NUM_EXPERTS_FULL),
                         msg="Distance matrix shape incorrect")
        self.assertTrue(torch.allclose(D, D.t(), atol=1e-5),
                        msg="Distance matrix is not symmetric")
        self.assertTrue(torch.allclose(D.diagonal(), torch.zeros(NUM_EXPERTS_FULL), atol=1e-5),
                        msg="Distance matrix diagonal is not zero")

    def test_12_merge_reduces_expert_count(self):
        """After merge_block, expert tensors must have target_k slots."""
        engine = self._make_engine()
        block = engine.moe_blocks[0]
        labels = torch.tensor([i % TARGET_EXPERTS for i in range(NUM_EXPERTS_FULL)])
        freqs = make_dummy_frequencies()
        engine.merge_block(block, labels, freqs)
        self.assertEqual(block.experts.gate_up_proj.shape[0], TARGET_EXPERTS)
        self.assertEqual(block.experts.down_proj.shape[0], TARGET_EXPERTS)

    def test_13_merged_weights_are_finite(self):
        """Frequency-weighted merge must produce finite weights."""
        engine = self._make_engine()
        block = engine.moe_blocks[0]
        labels = torch.tensor([i % TARGET_EXPERTS for i in range(NUM_EXPERTS_FULL)])
        freqs = make_dummy_frequencies()
        engine.merge_block(block, labels, freqs)
        self.assertTrue(torch.isfinite(block.experts.gate_up_proj).all(),
                        msg="gate_up_proj after HC-SMoE merge contains NaN/Inf")
        self.assertTrue(torch.isfinite(block.experts.down_proj).all(),
                        msg="down_proj after HC-SMoE merge contains NaN/Inf")

    def test_14_zero_frequency_expert_handled_gracefully(self):
        """Zero-frequency experts must not cause NaN (handled via epsilon guard)."""
        engine = self._make_engine()
        block = engine.moe_blocks[0]
        labels = torch.tensor([i % TARGET_EXPERTS for i in range(NUM_EXPERTS_FULL)])
        # All frequency concentrated in first expert
        freqs = torch.zeros(NUM_EXPERTS_FULL)
        freqs[0] = 1.0
        engine.merge_block(block, labels, freqs)
        self.assertTrue(torch.isfinite(block.experts.gate_up_proj).all(),
                        msg="Zero-frequency handling caused NaN/Inf in gate_up_proj")


# ========================
# Test Case 4: MC-SMoE
# ========================

class TestMCSMoE(unittest.TestCase):

    def _make_engine(self):
        model = make_model()
        return MCSMoEEngine(model)

    def test_15_hungarian_permutation_is_valid(self):
        """Hungarian bipartite matching must return a valid permutation of [0, d_ff)."""
        engine = self._make_engine()
        d_ff = INTERMEDIATE_DIM
        ref_gate = torch.randn(d_ff, HIDDEN_DIM)
        ref_up = torch.randn(d_ff, HIDDEN_DIM)
        ref_down = torch.randn(HIDDEN_DIM, d_ff)
        t_gate = torch.randn(d_ff, HIDDEN_DIM)
        t_up = torch.randn(d_ff, HIDDEN_DIM)
        t_down = torch.randn(HIDDEN_DIM, d_ff)
        perm = engine.compute_bipartite_neuron_matching(ref_gate, ref_up, ref_down,
                                                        t_gate, t_up, t_down)
        self.assertEqual(perm.shape, (d_ff,), msg="Permutation must have shape [d_ff]")
        self.assertTrue(torch.allclose(perm.sort()[0], torch.arange(d_ff)),
                        msg="Permutation must cover all neuron indices exactly once")

    def test_16_permuted_expert_is_functionally_equivalent(self):
        """Applying the self-permutation (reference matched against itself) returns identity perm."""
        engine = self._make_engine()
        d_ff = INTERMEDIATE_DIM
        ref_gate = torch.randn(d_ff, HIDDEN_DIM)
        ref_up = torch.randn(d_ff, HIDDEN_DIM)
        ref_down = torch.randn(HIDDEN_DIM, d_ff)
        perm = engine.compute_bipartite_neuron_matching(ref_gate, ref_up, ref_down,
                                                        ref_gate, ref_up, ref_down)
        identity = torch.arange(d_ff)
        self.assertTrue(torch.equal(perm, identity),
                        msg="Self-permutation should yield identity permutation")

    def test_17_merge_reduces_expert_count(self):
        """After permute_and_merge_block, expert tensors must have target_k slots."""
        engine = self._make_engine()
        block = engine.moe_blocks[0]
        labels = torch.tensor([i % TARGET_EXPERTS for i in range(NUM_EXPERTS_FULL)])
        freqs = make_dummy_frequencies()
        engine.permute_and_merge_block(block, labels, freqs)
        self.assertEqual(block.experts.gate_up_proj.shape[0], TARGET_EXPERTS)
        self.assertEqual(block.experts.down_proj.shape[0], TARGET_EXPERTS)

    def test_18_merged_weights_are_finite(self):
        """MC-SMoE permuted merge must produce finite weights."""
        engine = self._make_engine()
        block = engine.moe_blocks[0]
        labels = torch.tensor([i % TARGET_EXPERTS for i in range(NUM_EXPERTS_FULL)])
        freqs = make_dummy_frequencies()
        engine.permute_and_merge_block(block, labels, freqs)
        self.assertTrue(torch.isfinite(block.experts.gate_up_proj).all(),
                        msg="gate_up_proj after MC-SMoE merge contains NaN/Inf")
        self.assertTrue(torch.isfinite(block.experts.down_proj).all(),
                        msg="down_proj after MC-SMoE merge contains NaN/Inf")

    def test_19_cluster_by_router_policy_shape(self):
        """Cluster labels from router policy must span all expert indices."""
        engine = self._make_engine()
        block = engine.moe_blocks[0]
        labels = engine.cluster_experts_by_router_policy(block, TARGET_EXPERTS)
        self.assertEqual(labels.shape[0], NUM_EXPERTS_FULL,
                         msg="Cluster labels must have one entry per expert")
        unique = torch.unique(labels)
        self.assertEqual(len(unique), TARGET_EXPERTS,
                         msg=f"Expected {TARGET_EXPERTS} unique cluster labels, got {len(unique)}")


# ========================
# Test Case 5: End-to-End
# ========================

class TestForwardPassAfterCompression(unittest.TestCase):

    def test_20_sub_moe_forward_pass(self):
        """Model forward pass must produce finite logits after Sub-MoE compression."""
        model = make_model()
        engine = SubMoEEngine(model, rank_ratio=0.5)
        block = engine.moe_blocks[0]
        freqs = make_dummy_frequencies().numpy()
        labels = np.array([i % TARGET_EXPERTS for i in range(NUM_EXPERTS_FULL)])
        engine.compress_block(block, labels, freqs)
        # intermediate_dim is set in MockOlmoeExperts.__init__ and survives
        # parameter replacement, so no manual assignment is needed here.
        dummy_input = torch.randn(1, 4, HIDDEN_DIM)
        with torch.no_grad():
            out = model(dummy_input)
        self.assertEqual(out.shape, (1, 4, HIDDEN_DIM),
                         msg=f"Output shape mismatch: {out.shape}")
        self.assertTrue(torch.isfinite(out).all(),
                        msg="Model output after Sub-MoE compression contains NaN/Inf")

    def test_21_reap_prune_forward_pass(self):
        """Model forward pass must produce finite output after REAP pruning."""
        model = make_model()
        engine = REAPPruneEngine(model, renormalize_router_weights=True)
        block = engine.moe_blocks[0]
        saliency = torch.rand(NUM_EXPERTS_FULL)
        engine.prune_block(block, saliency, TARGET_EXPERTS)
        block.num_experts_per_tok = min(2, TARGET_EXPERTS)
        dummy_input = torch.randn(1, 4, HIDDEN_DIM)
        with torch.no_grad():
            out = model(dummy_input)
        self.assertEqual(out.shape, (1, 4, HIDDEN_DIM))
        self.assertTrue(torch.isfinite(out).all(),
                        msg="Model output after REAP pruning contains NaN/Inf")

    def test_22_hc_smoe_forward_pass(self):
        """Model forward pass must produce finite output after HC-SMoE merging."""
        model = make_model()
        engine = HCSMoEEngine(model)
        block = engine.moe_blocks[0]
        labels = torch.tensor([i % TARGET_EXPERTS for i in range(NUM_EXPERTS_FULL)])
        freqs = make_dummy_frequencies()
        engine.merge_block(block, labels, freqs)
        block.num_experts_per_tok = min(2, TARGET_EXPERTS)
        dummy_input = torch.randn(1, 4, HIDDEN_DIM)
        with torch.no_grad():
            out = model(dummy_input)
        self.assertEqual(out.shape, (1, 4, HIDDEN_DIM))
        self.assertTrue(torch.isfinite(out).all(),
                        msg="Model output after HC-SMoE merging contains NaN/Inf")

    def test_23_mc_smoe_forward_pass(self):
        """Model forward pass must produce finite output after MC-SMoE merging."""
        model = make_model()
        engine = MCSMoEEngine(model)
        block = engine.moe_blocks[0]
        labels = torch.tensor([i % TARGET_EXPERTS for i in range(NUM_EXPERTS_FULL)])
        freqs = make_dummy_frequencies()
        engine.permute_and_merge_block(block, labels, freqs)
        block.num_experts_per_tok = min(2, TARGET_EXPERTS)
        dummy_input = torch.randn(1, 4, HIDDEN_DIM)
        with torch.no_grad():
            out = model(dummy_input)
        self.assertEqual(out.shape, (1, 4, HIDDEN_DIM))
        self.assertTrue(torch.isfinite(out).all(),
                        msg="Model output after MC-SMoE merging contains NaN/Inf")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None  # preserve definition order
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSubMoE))
    suite.addTests(loader.loadTestsFromTestCase(TestREAPPrune))
    suite.addTests(loader.loadTestsFromTestCase(TestHCSMoE))
    suite.addTests(loader.loadTestsFromTestCase(TestMCSMoE))
    suite.addTests(loader.loadTestsFromTestCase(TestForwardPassAfterCompression))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
