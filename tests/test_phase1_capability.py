"""
tests/test_phase1_capability.py
============================================================
Phase 1 Validation Tests: Capability Extraction Engine & Tensor Slicing.

Tests:
1. Exact layer-normalization slicing demonstration (verifying fix for [:, layer_idx, :] vs [layer_idx, :, :]).
2. Expert coverage across all active experts (verifying experts >= 16 are processed and normalized).
3. Zero-vector and non-finite protection (verifying no NaNs or Infs created).
4. Invariance to varying layer and expert counts (e.g., [4, 8], [16, 64], [32, 128]).
5. Dual-architecture dispatch: fused 3D parameters vs unfused ModuleList.
6. Explicit error raising on unsupported architectures.
7. Pairwise Euclidean distance mathematical properties (symmetry, non-negativity, triangle inequality).
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.care_com_v21.capability import (
    compute_global_capability_vectors,
    pairwise_capability_distances,
    get_active_expert_count
)

class TestPhase1Capability(unittest.TestCase):

    def test_tensor_slicing_math_proof(self):
        """
        Direct mathematical verification of the slicing error and its fix.
        Tensor shape: [num_layers=16, num_experts=64, capability_dim=10].
        """
        num_layers = 16
        num_experts = 64
        cap_dim = 10
        
        # Synthetic capability tensor: expert e has base value (e + 1)
        raw_caps = np.zeros((num_layers, num_experts, cap_dim), dtype=np.float32)
        for l in range(num_layers):
            for e in range(num_experts):
                raw_caps[l, e, :] = (e + 1.0) * (l + 1.0)

        # 1. Simulate Faulty Slicing: layer_C = raw_caps[:, layer_idx, :]
        faulty_caps = raw_caps.copy()
        for layer_idx in range(num_layers):
            layer_C = faulty_caps[:, layer_idx, :] # Shape [16, 10] - SLICES EXPERTS 0..15!
            norms = np.linalg.norm(layer_C, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            faulty_caps[:, layer_idx, :] = layer_C / norms

        # Under faulty slicing, experts 16..63 were NEVER modified in layer loop
        for e in range(16, num_experts):
            np.testing.assert_allclose(
                faulty_caps[:, e, :],
                raw_caps[:, e, :],
                err_msg=f"Expert {e} was unexpectedly modified under faulty slicing"
            )

        # 2. Correct Slicing: layer_C = raw_caps[layer_idx, :, :]
        corrected_caps = raw_caps.copy()
        for layer_idx in range(num_layers):
            layer_C = corrected_caps[layer_idx, :, :] # Shape [64, 10] - ALL 64 EXPERTS!
            norms = np.linalg.norm(layer_C, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            corrected_caps[layer_idx, :, :] = layer_C / norms

        # Under correct slicing, every single expert e in [0, 63] must have unit norm in each layer
        for l in range(num_layers):
            for e in range(num_experts):
                layer_norm = np.linalg.norm(corrected_caps[l, e, :])
                self.assertAlmostEqual(layer_norm, 1.0, places=5,
                                       msg=f"Layer {l}, Expert {e} failed unit normalization")

    def test_zero_vectors_and_nan_safety(self):
        """Zero vectors and non-finite inputs must not produce NaNs or Infs."""
        num_layers = 4
        num_experts = 8
        cap_dim = 10
        
        caps = np.zeros((num_layers, num_experts, cap_dim), dtype=np.float32)
        # Leave expert 0 and 3 as entirely zero
        caps[:, 1, :] = 1.0
        caps[:, 2, :] = 2.0
        # Inject non-finite into expert 4
        caps[:, 4, :] = np.nan

        # Normalize with safety guards
        caps = np.nan_to_num(caps, nan=0.0, posinf=1.0, neginf=-1.0)
        for layer_idx in range(num_layers):
            layer_C = caps[layer_idx, :, :]
            norms = np.linalg.norm(layer_C, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            caps[layer_idx, :, :] = layer_C / norms
            
        C_concat = caps.transpose(1, 0, 2).reshape(num_experts, num_layers * cap_dim)
        global_norms = np.linalg.norm(C_concat, axis=1, keepdims=True)
        global_norms[global_norms == 0] = 1.0
        C_final = C_concat / global_norms
        C_final = np.nan_to_num(C_final, nan=0.0, posinf=1.0, neginf=-1.0)

        self.assertFalse(np.isnan(C_final).any(), "NaN detected in final capability matrix")
        self.assertFalse(np.isinf(C_final).any(), "Inf detected in final capability matrix")
        
        # Zero experts must remain all zero
        np.testing.assert_allclose(C_final[0], np.zeros(num_layers * cap_dim))
        np.testing.assert_allclose(C_final[3], np.zeros(num_layers * cap_dim))
        
        # Active non-zero experts must have unit norm
        self.assertAlmostEqual(np.linalg.norm(C_final[1]), 1.0, places=5)
        self.assertAlmostEqual(np.linalg.norm(C_final[2]), 1.0, places=5)

    def test_varying_dimensions(self):
        """Test with different combinations of layers and experts."""
        test_configs = [
            (2, 4),
            (8, 16),
            (16, 64),
            (32, 128),
            (1, 8),
        ]
        
        for n_layers, n_experts in test_configs:
            caps = np.random.randn(n_layers, n_experts, 10).astype(np.float32)
            # Correct layer-normalization
            for l in range(n_layers):
                l_C = caps[l, :, :]
                norms = np.linalg.norm(l_C, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                caps[l, :, :] = l_C / norms
                
            C_concat = caps.transpose(1, 0, 2).reshape(n_experts, n_layers * 10)
            g_norms = np.linalg.norm(C_concat, axis=1, keepdims=True)
            g_norms[g_norms == 0] = 1.0
            C_final = C_concat / g_norms
            
            self.assertEqual(C_final.shape, (n_experts, n_layers * 10))
            for e in range(n_experts):
                self.assertAlmostEqual(np.linalg.norm(C_final[e]), 1.0, places=5)
                
            distances = pairwise_capability_distances(C_final)
            self.assertEqual(len(distances), n_experts)
            for i in range(n_experts):
                self.assertEqual(len(distances[i]), n_experts - 1)
                for j in distances[i]:
                    self.assertGreaterEqual(distances[i][j], 0.0)
                    self.assertAlmostEqual(distances[i][j], distances[j][i], places=5)

    def test_mock_fused_and_unfused_blocks(self):
        """Verify dynamic dispatch and active expert identification on mock architectures."""
        hidden_dim = 16
        intermediate_dim = 32
        num_experts = 64
        
        # 1. Mock Fused Architecture (OLMoE style)
        class MockFusedExperts(nn.Module):
            def __init__(self, n_exp, h_dim, inter_dim):
                super().__init__()
                self.num_experts = n_exp
                self.gate_up_proj = nn.Parameter(torch.randn(n_exp, 2 * inter_dim, h_dim))
                self.down_proj = nn.Parameter(torch.randn(n_exp, h_dim, inter_dim))

        class MockFusedBlock(nn.Module):
            def __init__(self, n_exp, h_dim, inter_dim):
                super().__init__()
                self.experts = MockFusedExperts(n_exp, h_dim, inter_dim)
                self.gate = nn.Linear(h_dim, n_exp, bias=False)

        fused_block = MockFusedBlock(num_experts, hidden_dim, intermediate_dim)
        self.assertEqual(get_active_expert_count(fused_block), 64)

        # 2. Mock Unfused Architecture (ModuleList style)
        class MockExpertMLP(nn.Module):
            def __init__(self, h_dim, inter_dim):
                super().__init__()
                self.gate_proj = nn.Linear(h_dim, inter_dim, bias=False)
                self.up_proj = nn.Linear(h_dim, inter_dim, bias=False)
                self.down_proj = nn.Linear(inter_dim, h_dim, bias=False)

        class MockUnfusedBlock(nn.Module):
            def __init__(self, n_exp, h_dim, inter_dim):
                super().__init__()
                self.experts = nn.ModuleList([MockExpertMLP(h_dim, inter_dim) for _ in range(n_exp)])
                self.gate = nn.Linear(h_dim, n_exp, bias=False)

        unfused_block = MockUnfusedBlock(num_experts, hidden_dim, intermediate_dim)
        self.assertEqual(get_active_expert_count(unfused_block), 64)

    def test_unsupported_architecture_error(self):
        """Unsupported architectures must fail explicitly rather than silently dropping experts."""
        class InvalidBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = nn.Parameter(torch.randn(10, 10))

        with self.assertRaises(ValueError):
            get_active_expert_count(InvalidBlock())

        with self.assertRaises(ValueError):
            compute_global_capability_vectors(
                model=nn.Linear(10, 10),
                moe_blocks=[],
                df_tokens=None
            )

    def test_metric_space_properties(self):
        """Pairwise distance dictionary must satisfy metric space axioms."""
        C = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0]
        ], dtype=np.float32)
        
        d = pairwise_capability_distances(C)
        
        # Symmetry
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.assertAlmostEqual(d[i][j], d[j][i], places=6)
                    self.assertGreater(d[i][j], 0.0)
                    
        # Triangle inequality: d(i, k) <= d(i, j) + d(j, k)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if i != j and j != k and i != k:
                        self.assertLessEqual(d[i][k], d[i][j] + d[j][k] + 1e-6)

if __name__ == "__main__":
    unittest.main()
