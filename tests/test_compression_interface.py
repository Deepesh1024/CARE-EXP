"""
tests/test_compression_interface.py
===================================
Comprehensive test suite for the unified MoE Compression OOP interface.

Tests:
1. Interface conformance across all six methods + random control.
2. Strict input validation and invalid configuration detection.
3. Target expert counts and per-layer compression dimensions.
4. Router dimensions and post-compression forward pass finiteness.
5. Expert identity tracking (active_to_original).
6. Deterministic behavior for fixed seeds.
7. Fidelity reporting (Sub-MoE, HC-SMoE, MC-SMoE partial/proxy status).
8. Serialization, strict checkpoint reload, and forward logit parity.
9. Fail-fast validation rejecting silent fallbacks.
10. Step 1 nomination parity between Static-K5 and Adaptive.
"""

import sys
import os
import unittest
import copy
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.compressors import (
    BaseMoECompressor,
    BaseMoEModelAdapter,
    OLMoEModelAdapter,
    CompressionResult,
    CAREStaticCompressor,
    CAREAdaptiveCompressor,
    SubMoECompressor,
    REAPPruneCompressor,
    HCSMoECompressor,
    MCSMoECompressor,
    RandomControlCompressor,
    get_compressor,
    COMPRESSOR_REGISTRY,
)


# ==============================================================================
# Multi-Layer Mock MoE Model
# ==============================================================================

class MockFusedExperts(nn.Module):
    def __init__(self, num_experts: int, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.num_experts = num_experts
        object.__setattr__(self, "intermediate_dim", intermediate_dim)
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_dim, hidden_dim) * 0.05
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_dim, intermediate_dim) * 0.05
        )


class MockMoEBlock(nn.Module):
    def __init__(self, num_experts: int = 8, hidden_dim: int = 16, intermediate_dim: int = 32, top_k: int = 2):
        super().__init__()
        self.num_experts_per_tok = top_k
        self.experts = MockFusedExperts(num_experts, hidden_dim, intermediate_dim)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, h = x.shape
        flat_x = x.reshape(-1, h)
        logits = self.gate(flat_x)
        weights = F.softmax(logits, dim=-1)
        topk_weights, selected = torch.topk(weights, min(self.num_experts_per_tok, logits.shape[-1]), dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(flat_x)
        d_ff = self.intermediate_dim
        for t in range(flat_x.shape[0]):
            for k in range(topk_weights.shape[-1]):
                eid = selected[t, k].item()
                w = topk_weights[t, k]
                gu = self.experts.gate_up_proj[eid]
                g = F.silu(gu[:d_ff] @ flat_x[t])
                u = gu[d_ff:] @ flat_x[t]
                down = self.experts.down_proj[eid]
                out[t] += w * (down @ (g * u))
        return out.reshape(bsz, seq, h)


class MockMultiLayerMoEModel(nn.Module):
    def __init__(self, num_layers: int = 2, num_experts: int = 8, hidden_dim: int = 16, intermediate_dim: int = 32):
        super().__init__()
        self.config = type("Config", (), {
            "num_experts": num_experts,
            "num_local_experts": num_experts,
            "num_experts_per_tok": 2,
            "hidden_size": hidden_dim,
            "vocab_size": 100,
        })()
        self.embedding = nn.Embedding(100, hidden_dim)
        self.layers = nn.ModuleList([
            MockMoEBlock(num_experts, hidden_dim, intermediate_dim) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, 100, bias=False)

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False, **kwargs):
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        elif input_ids.ndim == 3:
            input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        x = self.embedding(input_ids)
        all_hidden = [x]
        for layer in self.layers:
            x = layer(x)
            all_hidden.append(x)
        logits = self.head(x)
        class Out: pass
        out = Out()
        out.logits = logits
        if output_hidden_states:
            out.hidden_states = all_hidden
        return out


def make_test_model(num_layers: int = 2, num_experts: int = 8, seed: int = 42) -> MockMultiLayerMoEModel:
    torch.manual_seed(seed)
    return MockMultiLayerMoEModel(num_layers=num_layers, num_experts=num_experts)


def make_mock_calibration_batches(seq_len: int = 8, count: int = 8, seed: int = 42):
    torch.manual_seed(seed)
    batches = []
    for _ in range(count):
        batches.append({
            "input_ids": torch.randint(0, 100, (seq_len,)),
            "attention_mask": torch.ones(seq_len, dtype=torch.long)
        })
    return batches


def make_mock_probe_dataframe(num_axes: int = 10, samples_per_axis: int = 2, seq_len: int = 8, seed: int = 42):
    torch.manual_seed(seed)
    rows = []
    for axis in range(num_axes):
        for _ in range(samples_per_axis):
            rows.append({
                "axis_idx": axis,
                "input_ids": torch.randint(0, 100, (seq_len,)).tolist(),
                "attention_mask": [1] * seq_len
            })
    return pd.DataFrame(rows)


# ==============================================================================
# Test Cases
# ==============================================================================

class TestMoECompressionInterface(unittest.TestCase):

    def setUp(self):
        self.cal_batches = make_mock_calibration_batches()
        self.probe_df = make_mock_probe_dataframe()

    def _get_compressor_and_data(self, method: str):
        if method in ["care-static", "care_static", "care-adaptive", "care_adaptive"]:
            compressor = get_compressor(method, config={
                "eval_batches": self.cal_batches,
                "samples_per_axis": 2,
                "candidate_pool_size": 5,
            })
            return compressor, self.probe_df
        elif method in ["random-control", "random"]:
            compressor = get_compressor(method, config={
                "eval_batches": self.cal_batches,
                "candidate_k": 5,
            })
            return compressor, self.cal_batches
        elif method in ["reap-prune", "reap"]:
            compressor = get_compressor(method, config={"metric": "reap"})
            return compressor, self.cal_batches
        else:
            compressor = get_compressor(method)
            return compressor, self.cal_batches

    def test_01_interface_conformance(self):
        """Every compressor in COMPRESSOR_REGISTRY must inherit from BaseMoECompressor."""
        unique_classes = set(COMPRESSOR_REGISTRY.values())
        for cls in unique_classes:
            compressor = cls()
            self.assertIsInstance(compressor, BaseMoECompressor)
            self.assertTrue(hasattr(compressor, "compress"))
            self.assertTrue(hasattr(compressor, "validate_inputs"))
            self.assertIsInstance(compressor.METHOD_ID, str)
            self.assertIsInstance(compressor.IMPLEMENTATION_VERSION, str)
            self.assertIsInstance(compressor.DEFAULT_FIDELITY_STATUS, str)

    def test_02_input_validation(self):
        """validate_inputs must strictly catch invalid expert counts and invalid models."""
        compressor = CAREStaticCompressor()
        model = make_test_model(num_experts=8)

        # Valid
        cur, tgt = compressor.validate_inputs(model, 6)
        self.assertEqual((cur, tgt), (8, 6))

        # Default target_experts (cur - 1)
        cur, tgt = compressor.validate_inputs(model, None)
        self.assertEqual((cur, tgt), (8, 7))

        # Target > Current -> ValueError
        with self.assertRaises(ValueError):
            compressor.validate_inputs(model, 10)

        # Target < 1 -> ValueError
        with self.assertRaises(ValueError):
            compressor.validate_inputs(model, 0)

        # Non-integer target -> TypeError
        with self.assertRaises(TypeError):
            compressor.validate_inputs(model, "six")

        # Non-MoE model -> ValueError
        linear = nn.Linear(10, 10)
        with self.assertRaises(ValueError):
            compressor.validate_inputs(linear, 5)

    def test_03_compression_result_structure(self):
        """CompressionResult must contain all required fields across all 6 methods + random control."""
        methods = ["care-static", "care-adaptive", "sub-moe", "reap-prune", "hc-smoe", "mc-smoe", "random-control"]
        for method in methods:
            model = make_test_model(num_layers=2, num_experts=8)
            compressor, cal_data = self._get_compressor_and_data(method)
            result = compressor.compress(model, calibration_data=cal_data, target_experts=6, seed=42)

            self.assertIsInstance(result, CompressionResult)
            self.assertEqual(result.experts_before, 8)
            self.assertEqual(result.experts_after, 6)
            self.assertIsInstance(result.method_id, str)
            self.assertIsInstance(result.implementation_version, str)
            self.assertIsInstance(result.fidelity_status, str)
            self.assertIsInstance(result.per_layer_details, dict)
            self.assertEqual(len(result.per_layer_details), 2)
            self.assertGreaterEqual(result.runtime_s, 0.0)

    def test_04_target_expert_counts_and_dimensions(self):
        """Assert exact tensor dimensions after compression across all layers."""
        methods = ["care-static", "care-adaptive", "sub-moe", "reap-prune", "hc-smoe", "mc-smoe", "random-control"]
        target_k = 5

        for method in methods:
            model = make_test_model(num_layers=2, num_experts=8)
            compressor, cal_data = self._get_compressor_and_data(method)
            result = compressor.compress(model, calibration_data=cal_data, target_experts=target_k, seed=42)

            self.assertEqual(result.model.config.num_experts, target_k)
            self.assertEqual(result.model.config.num_local_experts, target_k)

            for layer in result.model.layers:
                self.assertEqual(layer.experts.gate_up_proj.shape[0], target_k)
                self.assertEqual(layer.experts.down_proj.shape[0], target_k)
                self.assertEqual(layer.gate.weight.shape[0], target_k)

    def test_05_forward_pass_finite_logits(self):
        """Compressed model must execute a forward pass yielding strictly finite logits."""
        methods = ["care-static", "care-adaptive", "sub-moe", "reap-prune", "hc-smoe", "mc-smoe", "random-control"]
        dummy_ids = torch.randint(0, 100, (2, 8))

        for method in methods:
            model = make_test_model(num_layers=2, num_experts=8)
            compressor, cal_data = self._get_compressor_and_data(method)
            result = compressor.compress(model, calibration_data=cal_data, target_experts=6, seed=42)

            with torch.no_grad():
                out = result.model(dummy_ids)
                logits = out.logits
                self.assertEqual(logits.shape, (2, 8, 100))
                self.assertTrue(torch.isfinite(logits).all(), f"Method {method} produced non-finite logits!")

    def test_06_active_to_original_identity_tracking(self):
        """CARE-Static, CARE-Adaptive, and Random-Control must return a valid active_to_original map."""
        for method in ["care-static", "care-adaptive", "random-control"]:
            model = make_test_model(num_layers=2, num_experts=8)
            compressor, cal_data = self._get_compressor_and_data(method)
            result = compressor.compress(model, calibration_data=cal_data, target_experts=5, seed=42)

            mapping = result.active_to_original
            self.assertIsNotNone(mapping, f"Mapping missing for {method}")
            self.assertEqual(len(mapping), 5)
            for local_id, orig_id in mapping.items():
                self.assertTrue(0 <= local_id < 5)
                self.assertTrue(0 <= orig_id < 8)
            self.assertEqual(len(set(mapping.values())), 5)

    def test_07_deterministic_behavior(self):
        """Two runs with the identical seed must produce identical selections and parameters."""
        for method in ["care-static", "care-adaptive", "random-control"]:
            model1 = make_test_model(num_layers=2, num_experts=8, seed=123)
            comp1, data1 = self._get_compressor_and_data(method)
            res1 = comp1.compress(model1, calibration_data=data1, target_experts=6, seed=999)

            model2 = make_test_model(num_layers=2, num_experts=8, seed=123)
            comp2, data2 = self._get_compressor_and_data(method)
            res2 = comp2.compress(model2, calibration_data=data2, target_experts=6, seed=999)

            w1 = res1.model.layers[0].experts.gate_up_proj.data
            w2 = res2.model.layers[0].experts.gate_up_proj.data
            self.assertTrue(torch.allclose(w1, w2, atol=1e-6), f"{method} not deterministic across runs!")

    def test_08_fidelity_reporting(self):
        """Fidelity status and warnings must accurately reflect algorithmic fidelity."""
        self.assertEqual(CAREStaticCompressor().DEFAULT_FIDELITY_STATUS, "FAITHFUL")
        self.assertEqual(CAREAdaptiveCompressor().DEFAULT_FIDELITY_STATUS, "FAITHFUL")
        self.assertEqual(SubMoECompressor().DEFAULT_FIDELITY_STATUS, "PARTIAL (router-weight clustering proxy)")
        self.assertEqual(HCSMoECompressor().DEFAULT_FIDELITY_STATUS, "PARTIAL (router-proxy)")
        self.assertEqual(MCSMoECompressor().DEFAULT_FIDELITY_STATUS, "PARTIAL (Stages 1–4 only)")
        self.assertEqual(RandomControlCompressor().DEFAULT_FIDELITY_STATUS, "CONTROL (random baseline)")

        # Warnings check for partial baselines
        model_sub = make_test_model(num_layers=2, num_experts=8)
        res_sub = SubMoECompressor().compress(model_sub, calibration_data=None, target_experts=6)
        self.assertTrue(any("router weights as a proxy" in w for w in res_sub.warnings))

        model_hc = make_test_model(num_layers=2, num_experts=8)
        res_hc = HCSMoECompressor().compress(model_hc, calibration_data=None, target_experts=6)
        self.assertTrue(any("router weight cosine" in w for w in res_hc.warnings))

        model_mc = make_test_model(num_layers=2, num_experts=8)
        res_mc = MCSMoECompressor().compress(model_mc, calibration_data=None, target_experts=6)
        self.assertTrue(any("Stage 5" in w for w in res_mc.warnings))

    def test_09_serialization_and_strict_checkpoint_reload_parity(self):
        """
        Compressed model must save and reload strictly without ignore_mismatched_sizes,
        maintaining exact numerical logit parity (max diff == 0).
        """
        methods = ["care-static", "care-adaptive", "sub-moe", "reap-prune", "hc-smoe", "mc-smoe", "random-control"]
        dummy_ids = torch.randint(0, 100, (2, 8))

        for method in methods:
            model = make_test_model(num_layers=2, num_experts=8)
            compressor, cal_data = self._get_compressor_and_data(method)
            result = compressor.compress(model, calibration_data=cal_data, target_experts=6, seed=42)

            with torch.no_grad():
                logits_before = result.model(dummy_ids).logits

            buf = io.BytesIO()
            torch.save(result.model.state_dict(), buf)
            buf.seek(0)

            reloaded_model = MockMultiLayerMoEModel(num_layers=2, num_experts=6)
            reloaded_model.load_state_dict(torch.load(buf), strict=True)
            reloaded_model.eval()

            with torch.no_grad():
                logits_after = reloaded_model(dummy_ids).logits

            max_diff = (logits_before - logits_after).abs().max().item()
            self.assertAlmostEqual(max_diff, 0.0, places=5,
                                  msg=f"Method {method} failed reload logit parity check! max_diff={max_diff}")

    def test_10_fail_fast_on_missing_calibration(self):
        """CARE-Adaptive and CARE-Static must fail fast with ValueError when calibration data is missing."""
        model = make_test_model(num_layers=2, num_experts=8)
        with self.assertRaises(ValueError):
            CAREAdaptiveCompressor().compress(model, calibration_data=None, target_experts=6)

        with self.assertRaises(ValueError):
            CAREStaticCompressor().compress(model, calibration_data=None, target_experts=6)

        with self.assertRaises(ValueError):
            REAPPruneCompressor().compress(model, calibration_data=None, target_experts=6)

    def test_11_step1_nomination_parity_static_k5_vs_adaptive(self):
        """With identical inputs at step 1, Static-K5 and Adaptive must evaluate the same candidate pool."""
        model1 = make_test_model(num_layers=2, num_experts=8, seed=42)
        model2 = make_test_model(num_layers=2, num_experts=8, seed=42)

        comp_static = CAREStaticCompressor(config={
            "candidate_pool_size": 5,
            "eval_batches": self.cal_batches,
            "samples_per_axis": 2
        })
        comp_adapt = CAREAdaptiveCompressor(config={
            "candidate_pool_size": 5,
            "eval_batches": self.cal_batches,
            "samples_per_axis": 2
        })

        res_static = comp_static.compress(model1, calibration_data=self.probe_df, target_experts=7, seed=42)
        res_adapt = comp_adapt.compress(model2, calibration_data=self.probe_df, target_experts=7, seed=42)

        # Both must select the exact same pair at step 1
        static_step1_pair = res_static.trajectory[0]["selected_original_pair"]
        adapt_step1_pair = res_adapt.trajectory[0]["selected_original_pair"]
        self.assertEqual(static_step1_pair, adapt_step1_pair,
                         f"Step 1 selection mismatch: Static {static_step1_pair} vs Adaptive {adapt_step1_pair}")


if __name__ == "__main__":
    unittest.main()
