"""
tests/test_physical_merge_engine.py
===================================
Forensic test suite verifying PhysicalMergeEngine transactional integrity.

Verifies:
1. Bitwise identical restoration (weights, biases, router weights) after trial merges.
2. Bitwise identical forward logit parity before snapshot vs after restore.
3. Integrity of active_to_original mapping and original_to_cluster through multiple snapshot/restore cycles.
4. Correct physical excision and parameter shrinking upon permanent merge commit.
5. Random-K candidate evaluation execution without NameError or state corruption.
6. Support for both fused (OLMoE) and unfused MoE architectures.
"""

import sys
import os
import unittest
import copy
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.care_com_v21.core import PhysicalMergeEngine
from tests.test_compression_interface import MockMultiLayerMoEModel, make_test_model, make_mock_calibration_batches
from benchmark.compressors.random_control import RandomControlCompressor
from benchmark.compressors.care_adaptive import CAREAdaptiveCompressor
from benchmark.compressors.care_static import CAREStaticCompressor


class MockUnfusedExpert(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MockUnfusedMoEBlock(nn.Module):
    def __init__(self, num_experts: int = 8, hidden_dim: int = 16, intermediate_dim: int = 32, top_k: int = 2):
        super().__init__()
        self.num_experts_per_tok = top_k
        self.experts = nn.ModuleList([MockUnfusedExpert(hidden_dim, intermediate_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, h = x.shape
        flat_x = x.reshape(-1, h)
        logits = self.gate(flat_x)
        weights = F.softmax(logits, dim=-1)
        topk_weights, selected = torch.topk(weights, min(self.num_experts_per_tok, logits.shape[-1]), dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(flat_x)
        for t in range(flat_x.shape[0]):
            for k in range(topk_weights.shape[-1]):
                eid = selected[t, k].item()
                w = topk_weights[t, k]
                out[t] += w * self.experts[eid](flat_x[t:t+1]).squeeze(0)
        return out.reshape(bsz, seq, h)


class MockUnfusedMoEModel(nn.Module):
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
            MockUnfusedMoEBlock(num_experts, hidden_dim, intermediate_dim) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, 100, bias=False)

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False, **kwargs):
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
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


def hash_model_state(model: nn.Module) -> str:
    """Computes SHA-256 hash of all model parameter bytes."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.data.cpu().numpy().tobytes())
    return hasher.hexdigest()


class TestPhysicalMergeEngineIntegrity(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_01_fused_snapshot_restore_bitwise_equality(self):
        """
        In fused models (OLMoE architecture), snapshot(i, j) followed by merge and restore
        must leave all parameter tensors bitwise identical (hash match & zero diff).
        """
        model = make_test_model(num_layers=2, num_experts=8, seed=42)
        engine = PhysicalMergeEngine(model)

        hash_before = hash_model_state(model)
        map_before = copy.deepcopy(engine.active_to_original)
        cluster_before = copy.deepcopy(engine.original_to_cluster)

        dummy_input = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            logits_before = model(dummy_input).logits.clone()

        # Simulate 5 trial candidate evaluations
        candidates = [(0, 1), (2, 3), (1, 4), (0, 5), (3, 6)]
        for i, j in candidates:
            engine.snapshot(i, j)
            engine.merge_experts(i, j)
            # Intermediate state must have 7 experts
            self.assertEqual(engine.current_num_experts, 7)
            engine.restore()
            # Post-restore state must have 8 experts
            self.assertEqual(engine.current_num_experts, 8)

            # Assert bitwise identical hash after EACH trial
            hash_after_trial = hash_model_state(model)
            self.assertEqual(hash_before, hash_after_trial, f"Hash mismatch after trial merge ({i}, {j}) and restore!")

        # Verify logits after all trials
        with torch.no_grad():
            logits_after = model(dummy_input).logits

        max_diff = (logits_before - logits_after).abs().max().item()
        self.assertEqual(max_diff, 0.0, f"Logit difference after restore: {max_diff}")
        self.assertEqual(engine.active_to_original, map_before)
        self.assertEqual(engine.original_to_cluster, cluster_before)

    def test_02_unfused_snapshot_restore_bitwise_equality(self):
        """
        In unfused models (ModuleList architecture), snapshot(i, j) followed by merge and restore
        must restore all module parameters and list structure bitwise identical.
        """
        model = MockUnfusedMoEModel(num_layers=2, num_experts=8)
        engine = PhysicalMergeEngine(model)

        hash_before = hash_model_state(model)
        map_before = copy.deepcopy(engine.active_to_original)

        dummy_input = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            logits_before = model(dummy_input).logits.clone()

        candidates = [(0, 1), (2, 4), (3, 5)]
        for i, j in candidates:
            engine.snapshot(i, j)
            engine.merge_experts(i, j)
            self.assertEqual(engine.current_num_experts, 7)
            engine.restore()
            self.assertEqual(engine.current_num_experts, 8)

            hash_after_trial = hash_model_state(model)
            self.assertEqual(hash_before, hash_after_trial, f"Unfused model hash mismatch after trial ({i}, {j})!")

        with torch.no_grad():
            logits_after = model(dummy_input).logits

        max_diff = (logits_before - logits_after).abs().max().item()
        self.assertEqual(max_diff, 0.0)
        self.assertEqual(engine.active_to_original, map_before)

    def test_03_physical_removal_and_identity_tracking(self):
        """
        Permanent merge_experts commit must shrink tensors by exactly 1,
        properly remove the excised ID, and maintain cluster membership.
        """
        model = make_test_model(num_layers=2, num_experts=8, seed=42)
        engine = PhysicalMergeEngine(model)

        # Merge local (1, 3): keeping 1, excising 3
        orig_1 = engine.get_original_id(1)
        orig_3 = engine.get_original_id(3)

        engine.merge_experts(1, 3)

        self.assertEqual(engine.current_num_experts, 7)
        self.assertEqual(len(engine.active_to_original), 7)
        self.assertNotIn(orig_3, engine.active_to_original)
        self.assertIn(orig_1, engine.active_to_original)

        # Cluster of surviving expert must contain both orig_1 and orig_3
        cluster_survivor = engine.get_cluster(1)
        self.assertIn(orig_1, cluster_survivor)
        self.assertIn(orig_3, cluster_survivor)

        # Tensor shapes must be exactly 7
        for block in engine.moe_blocks:
            self.assertEqual(block.experts.gate_up_proj.shape[0], 7)
            self.assertEqual(block.experts.down_proj.shape[0], 7)
            self.assertEqual(block.gate.weight.shape[0], 7)

    def test_04_random_k5_candidate_evaluation_execution(self):
        """
        RandomControlCompressor must execute K=5 candidate evaluation without NameError (NumPy)
        and achieve valid compression to target expert count.
        """
        model = make_test_model(num_layers=2, num_experts=8, seed=42)
        cal_batches = make_mock_calibration_batches(seq_len=8, count=4)

        compressor = RandomControlCompressor(config={
            "candidate_k": 5,
            "eval_batches": cal_batches,
        })

        result = compressor.compress(model, calibration_data=cal_batches, target_experts=6, seed=42)

        self.assertEqual(result.experts_before, 8)
        self.assertEqual(result.experts_after, 6)
        self.assertEqual(len(result.trajectory), 2)  # 8 -> 7 -> 6 (2 steps)
        for step_rec in result.trajectory:
            self.assertIn("candidate_pool", step_rec)
            self.assertGreaterEqual(len(step_rec["candidate_pool"]), 1)
            self.assertTrue(np.isfinite(step_rec["marginal_damage"]))


if __name__ == "__main__":
    unittest.main()
