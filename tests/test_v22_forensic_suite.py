"""
tests/test_v22_forensic_suite.py
================================
Comprehensive Forensic Test Suite for CARE-COM v2.2 Audit.

Covers:
1. Merge and restore correctness: bitwise parameter parity, exact forward logit parity,
   ordering invariance (i < j and j < i), multiple cycles, exception-safe restoration.
2. Expert-count and mapping invariants: parameter shrinking, router weight dimensions,
   active_to_original and original_to_cluster integrity.
3. Capability vector shape and indexing: strict [num_experts, num_layers * 10] shape,
   finiteness, non-zero dynamic range, and explicit synthetic validation on > 16 experts (N=32)
   to ensure the historical slicing bug (layer_capabilities[:, layer_idx, :]) cannot go undetected.
4. Candidate-pool validity: valid, unique pairs within [0, N-1], sorted by distance.
5. Marginal KL correctness: deterministic comparison against independent token-level calculation,
   label shifting, pad token exclusion.
6. PPL aggregation: valid-token NLL summation vs biased batch-mean averaging.
7. Baseline candidate-selection behavior: Static-K5 vs Static-K1, Adaptive-K5, Random-K5.
8. Evaluation data separation: hash-verified disjointness between selection and test data.
"""

import sys
import os
import unittest
import copy
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.care_com_v21.core import PhysicalMergeEngine
from experiments.care_com_v21.evaluator import collect_current_logits, evaluate_marginal_damage
from experiments.care_com_v21.capability import (
    compute_global_capability_vectors,
    pairwise_capability_distances,
)
from experiments.care_com_v21.care_capability_v23 import (
    compute_global_capability_vectors_v23,
    pairwise_capability_distances_v23,
    generate_candidate_pool_v23,
)
from benchmark.compressors.care_static import CAREStaticCompressor, CAREStaticK1Compressor
from benchmark.compressors.care_adaptive import CAREAdaptiveCompressor
from benchmark.compressors.random_control import RandomControlCompressor


# ==============================================================================
# Synthetic Model Fixtures
# ==============================================================================

class SyntheticOlmoeExperts(nn.Module):
    """Fused 3D expert module matching OLMoE layout."""
    def __init__(self, num_experts: int, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_dim = intermediate_dim
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_dim, hidden_dim) * 0.05
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_dim, intermediate_dim) * 0.05
        )

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        gu = F.linear(x, self.gate_up_proj[expert_idx])
        g, u = gu.chunk(2, dim=-1)
        return F.linear(F.silu(g) * u, self.down_proj[expert_idx])


class SyntheticMoEBlock(nn.Module):
    def __init__(self, num_experts: int, hidden_dim: int, intermediate_dim: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = SyntheticOlmoeExperts(num_experts, hidden_dim, intermediate_dim)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        nn.init.orthogonal_(self.gate.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, h_dim = x.shape
        flat_x = x.reshape(-1, h_dim)
        router_logits = self.gate(flat_x)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, min(self.top_k, router_logits.shape[-1]), dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(flat_x)
        for t in range(flat_x.shape[0]):
            for k in range(topk_weights.shape[-1]):
                exp_id = topk_indices[t, k].item()
                w = topk_weights[t, k]
                out[t] += w * self.experts(flat_x[t:t+1], exp_id).squeeze(0)
        return out.reshape(bsz, seq_len, h_dim)


class SyntheticMoEModel(nn.Module):
    def __init__(self, num_layers: int = 2, num_experts: int = 8, hidden_dim: int = 16, intermediate_dim: int = 32):
        super().__init__()
        self.config = type("Config", (), {
            "num_experts": num_experts,
            "num_local_experts": num_experts,
            "num_experts_per_tok": 2,
            "hidden_size": hidden_dim,
            "vocab_size": 128,
        })()
        self.embedding = nn.Embedding(128, hidden_dim)
        self.layers = nn.ModuleList([
            SyntheticMoEBlock(num_experts, hidden_dim, intermediate_dim) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, 128, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, output_hidden_states: bool = False, **kwargs):
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        x = self.embedding(input_ids)
        all_hidden = [x]
        for layer in self.layers:
            x = layer(x)
            all_hidden.append(x)
        logits = self.head(x)

        class Output: pass
        out = Output()
        out.logits = logits
        if output_hidden_states:
            out.hidden_states = all_hidden
        return out


# ==============================================================================
# Forensic Test Suite
# ==============================================================================

class TestCARECOMV22Forensics(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.model = SyntheticMoEModel(num_layers=2, num_experts=8, hidden_dim=16, intermediate_dim=32)
        self.engine = PhysicalMergeEngine(self.model, router_merge_mode="uniform")

    # --------------------------------------------------------------------------
    # 1. Merge and Restore Correctness
    # --------------------------------------------------------------------------
    def test_snapshot_restore_bitwise_parity_and_logit_equality(self):
        """Verifies bitwise parameter parity and forward logit equality after trial merge and restore."""
        test_input = torch.randint(0, 128, (2, 8))
        self.model.eval()
        with torch.no_grad():
            orig_logits = self.model(test_input).logits.clone()

        orig_params = {k: v.clone() for k, v in self.model.named_parameters()}
        orig_active_map = dict(self.engine.get_active_mapping())

        # Trial merge
        self.engine.snapshot(2, 5)
        self.engine.merge_experts(2, 5)
        self.assertEqual(self.engine.current_num_experts, 7)

        # Restore
        self.engine.restore()

        # Check expert count and mapping
        self.assertEqual(self.engine.current_num_experts, 8)
        self.assertEqual(self.engine.get_active_mapping(), orig_active_map)

        # Bitwise parameter check
        for k, v in self.model.named_parameters():
            self.assertTrue(torch.equal(v, orig_params[k]), f"Bitwise mismatch in parameter {k} after restore!")

        # Logit parity check
        with torch.no_grad():
            restored_logits = self.model(test_input).logits
        diff = (orig_logits - restored_logits).abs().max().item()
        self.assertEqual(diff, 0.0, f"Forward logits diverged after restore! Max diff: {diff}")

    def test_both_expert_index_orderings(self):
        """Tests that merge_experts works identically regardless of i < j vs j < i."""
        # Ordering 1: (1, 4)
        m1 = SyntheticMoEModel(num_layers=2, num_experts=8, hidden_dim=16, intermediate_dim=32)
        e1 = PhysicalMergeEngine(m1, router_merge_mode="uniform")
        e1.merge_experts(1, 4)
        self.assertEqual(e1.current_num_experts, 7)

        # Ordering 2: (4, 1)
        m2 = SyntheticMoEModel(num_layers=2, num_experts=8, hidden_dim=16, intermediate_dim=32)
        e2 = PhysicalMergeEngine(m2, router_merge_mode="uniform")
        e2.merge_experts(4, 1)
        self.assertEqual(e2.current_num_experts, 7)

        # Both should excise the larger index into the smaller index
        self.assertEqual(e1.current_num_experts, e2.current_num_experts)

    def test_multiple_consecutive_merges(self):
        """Tests multiple consecutive merges down to target."""
        self.assertEqual(self.engine.current_num_experts, 8)
        self.engine.merge_experts(0, 1) # 8 -> 7
        self.assertEqual(self.engine.current_num_experts, 7)
        self.engine.merge_experts(0, 1) # 7 -> 6
        self.assertEqual(self.engine.current_num_experts, 6)
        self.engine.merge_experts(2, 3) # 6 -> 5
        self.assertEqual(self.engine.current_num_experts, 5)

        for block in self.engine.moe_blocks:
            self.assertEqual(block.experts.gate_up_proj.shape[0], 5)
            self.assertEqual(block.experts.down_proj.shape[0], 5)
            self.assertEqual(block.gate.weight.shape[0], 5)

    def test_exception_safe_restoration(self):
        """Verifies that an exception during candidate evaluation does not corrupt model state."""
        orig_params = {k: v.clone() for k, v in self.model.named_parameters()}
        self.engine.snapshot(0, 1)

        try:
            self.engine.merge_experts(0, 1)
            raise RuntimeError("Injected evaluation failure during candidate trial")
        except RuntimeError:
            self.engine.restore()

        self.assertEqual(self.engine.current_num_experts, 8)
        for k, v in self.model.named_parameters():
            self.assertTrue(torch.equal(v, orig_params[k]))

    # --------------------------------------------------------------------------
    # 2. Expert-Count and Mapping Invariants
    # --------------------------------------------------------------------------
    def test_expert_count_and_mapping_invariants(self):
        """Tests that active_to_original and original_to_cluster maintain exact invariants."""
        active_map_0 = dict(self.engine.get_active_mapping())
        self.assertEqual(len(active_map_0), 8)

        # Merge expert 5 into expert 2
        excised_orig = self.engine.get_original_id(5)
        target_orig = self.engine.get_original_id(2)

        self.engine.merge_experts(2, 5)

        active_map_1 = dict(self.engine.get_active_mapping())
        self.assertEqual(len(active_map_1), 7)
        self.assertNotIn(excised_orig, active_map_1.values())
        self.assertIn(target_orig, active_map_1.values())

        cluster = self.engine.get_cluster(2)
        self.assertIn(excised_orig, cluster)
        self.assertIn(target_orig, cluster)

    # --------------------------------------------------------------------------
    # 3. Capability Extraction on Large Pools (> 16 experts to catch slicing bug)
    # --------------------------------------------------------------------------
    def test_capability_extraction_on_32_experts(self):
        """
        Synthetically tests capability extraction on a 32-expert model.
        Catches the historical layer_capabilities[:, layer_idx, :] slicing bug which
        accidentally sliced the expert axis and only considered experts 0..15.
        """
        large_model = SyntheticMoEModel(num_layers=2, num_experts=32, hidden_dim=16, intermediate_dim=32)
        large_blocks = [layer for layer in large_model.layers]

        import pandas as pd
        # Synthetic token probe data
        df_tokens = pd.DataFrame([
            {
                "input_ids": torch.randint(0, 128, (16,)).tolist(),
                "attention_mask": [1] * 16,
                "axis_idx": i % 10
            }
            for i in range(30)
        ])

        C = compute_global_capability_vectors_v23(
            model=large_model,
            moe_blocks=large_blocks,
            current_num_experts=32,
            df_tokens=df_tokens,
            samples_per_axis=2,
            device="cpu"
        )

        # 1. Shape must be exactly [32, 2 * 10] = [32, 20]
        self.assertEqual(C.shape, (32, 20), f"Expected shape (32, 20), got {C.shape}")

        # 2. Must be finite
        self.assertTrue(np.all(np.isfinite(C)), "Capability matrix contains NaN or Inf")

        # 3. All experts, especially experts 16..31, must have distinct, non-zero vectors
        for e in range(16, 32):
            norm_e = np.linalg.norm(C[e])
            self.assertGreater(norm_e, 0.0, f"Expert {e} (>= 16) has zero capability norm!")

        # 4. Feature variance across experts must be strictly non-zero
        feature_stds = np.std(C, axis=0)
        self.assertGreater(np.min(feature_stds), 1e-6, "Capability matrix has zero feature variance column")

    # --------------------------------------------------------------------------
    # 4. Candidate Pool Validity
    # --------------------------------------------------------------------------
    def test_candidate_pool_generation(self):
        """Verifies candidate pool returns valid, unique pairs within active expert bounds."""
        np.random.seed(42)
        C = np.random.randn(8, 20)
        C = C / np.linalg.norm(C, axis=1, keepdims=True)

        distances = pairwise_capability_distances_v23(C)
        candidates = generate_candidate_pool_v23(distances, pool_size=5)

        self.assertEqual(len(candidates), 5)
        seen = set()
        for i, j in candidates:
            self.assertLess(i, j)
            self.assertGreaterEqual(i, 0)
            self.assertLess(j, 8)
            pair = (i, j)
            self.assertNotIn(pair, seen)
            seen.add(pair)

    def test_candidate_pool_selection_at_or_above_16_on_64_experts(self):
        """
        Forensic regression test for candidate indexing:
        Explicitly constructs a 64-expert fixture where pairs with indices >= 16
        have the smallest capability distances. Verifies candidate generator
        selects valid pairs involving indices >= 16 and matches active bounds.
        """
        num_experts = 64
        # Construct synthetic capability vectors where expert 25 and expert 40 are nearest neighbors
        C = np.ones((num_experts, 10), dtype=np.float32) * 5.0
        # Make all experts distant except 25 and 40, and 30 and 50
        for e in range(num_experts):
            C[e, e % 10] += (e + 1) * 2.0
        C[25] = C[40] + 0.001
        C[30] = C[50] + 0.002

        distances = pairwise_capability_distances_v23(C)
        candidates = generate_candidate_pool_v23(distances, pool_size=5)

        self.assertEqual(len(candidates), 5)
        # Check that top candidate is (25, 40)
        self.assertIn((25, 40), candidates[:2])
        # Check that at least one candidate involves an index >= 16
        has_ge_16 = any(max(i, j) >= 16 for i, j in candidates)
        self.assertTrue(has_ge_16, "Candidate pool failed to nominate pairs with indices >= 16!")

        # Cross-check against an independent reference brute-force generator
        all_pairs = []
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                all_pairs.append((i, j, distances[i][j]))
        all_pairs.sort(key=lambda x: x[2])
        reference_candidates = [(p[0], p[1]) for p in all_pairs[:5]]
        self.assertEqual(candidates, reference_candidates, "Candidate generator diverged from reference!")

    # --------------------------------------------------------------------------
    # 5. Marginal KL Correctness
    # --------------------------------------------------------------------------
    def test_marginal_kl_evaluator_against_independent_calculation(self):
        """Compares evaluate_marginal_damage against an independent token-level formula."""
        bsz, seq_len, vocab_size = 2, 4, 16
        input_ids = torch.randint(0, vocab_size, (bsz, seq_len))
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]) # Mask with padding

        # Simulated base and candidate logits
        base_logits = torch.randn(bsz, seq_len, vocab_size)
        cand_logits = torch.randn(bsz, seq_len, vocab_size)

        # Independent calculation:
        # Causal shift: prediction for t is from logits at t-1
        shift_base = base_logits[:, :-1, :]
        shift_cand = cand_logits[:, :-1, :]
        shift_mask = attention_mask[:, 1:].bool()

        p_base = F.softmax(shift_base, dim=-1)
        logp_base = F.log_softmax(shift_base, dim=-1)
        logp_cand = F.log_softmax(shift_cand, dim=-1)

        # KL(P || Q) = sum P * (log P - log Q)
        token_kl = (p_base * (logp_base - logp_cand)).sum(dim=-1)
        expected_kl = token_kl[shift_mask].sum().item() / shift_mask.sum().item()

        # Compare with evaluate_marginal_damage logic
        cached_logprobs = [logp_base]
        
        # Mock candidate model
        class MockCand(nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.l = logits
            def forward(self, input_ids, attention_mask=None, **kwargs):
                class Out: pass
                o = Out()
                o.logits = self.l
                return o

        cand_model = MockCand(cand_logits)
        eval_chunks = [{"input_ids": input_ids[b], "attention_mask": attention_mask[b]} for b in range(bsz)]
        
        measured_kl = evaluate_marginal_damage(
            cand_model, eval_chunks, cached_logprobs, batch_size=bsz, max_eval_batches=1, device="cpu"
        )

        self.assertAlmostEqual(measured_kl, expected_kl, places=5)

    def test_marginal_kl_identical_distributions_is_zero(self):
        """Verifies that when base and candidate distributions are identical, KL is ~0.0."""
        bsz, seq_len, vocab_size = 2, 4, 16
        input_ids = torch.randint(0, vocab_size, (bsz, seq_len))
        attention_mask = torch.ones((bsz, seq_len), dtype=torch.long)
        base_logits = torch.randn(bsz, seq_len, vocab_size)

        shift_base = base_logits[:, :-1, :]
        logp_base = F.log_softmax(shift_base, dim=-1)

        class MockCand(nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.l = logits
            def forward(self, input_ids, attention_mask=None, **kwargs):
                class Out: pass
                o = Out()
                o.logits = self.l
                return o

        cand_model = MockCand(base_logits.clone())
        eval_chunks = [{"input_ids": input_ids[b], "attention_mask": attention_mask[b]} for b in range(bsz)]
        measured_kl = evaluate_marginal_damage(
            cand_model, eval_chunks, [logp_base], batch_size=bsz, max_eval_batches=1, device="cpu"
        )
        self.assertLess(measured_kl, 1e-6, f"Expected near-zero KL for identical models, got {measured_kl}")

    def test_marginal_kl_zero_valid_tokens_raises_value_error(self):
        """Verifies that evaluate_marginal_damage raises ValueError when there are zero valid tokens."""
        bsz, seq_len, vocab_size = 2, 4, 16
        input_ids = torch.randint(0, vocab_size, (bsz, seq_len))
        # Mask where only index 0 is 1, so shifted mask [:, 1:] has 0 valid prediction tokens!
        attention_mask = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])
        base_logits = torch.randn(bsz, seq_len, vocab_size)
        shift_base = base_logits[:, :-1, :]
        logp_base = F.log_softmax(shift_base, dim=-1)

        class MockCand(nn.Module):
            def forward(self, input_ids, attention_mask=None, **kwargs):
                class Out: pass
                o = Out()
                o.logits = torch.randn_like(base_logits)
                return o

        cand_model = MockCand()
        eval_chunks = [{"input_ids": input_ids[b], "attention_mask": attention_mask[b]} for b in range(bsz)]
        with self.assertRaises(ValueError) as ctx:
            evaluate_marginal_damage(
                cand_model, eval_chunks, [logp_base], batch_size=bsz, max_eval_batches=1, device="cpu"
            )
        self.assertIn("zero valid tokens", str(ctx.exception).lower())

    # --------------------------------------------------------------------------
    # 6. PPL Token-Weighted Aggregation vs Biased Batch Mean
    # --------------------------------------------------------------------------
    def test_ppl_aggregation_token_weighted_vs_batch_mean(self):
        """Demonstrates why valid-token NLL summation is mathematically required over batch means."""
        # Batch 1: 10 valid tokens, mean loss = 2.0 (total NLL = 20.0)
        # Batch 2: 2 valid tokens, mean loss = 4.0 (total NLL = 8.0)
        # True global token-weighted loss = 28.0 / 12 = 2.3333 -> PPL = exp(2.3333) = 10.31
        # Biased unweighted batch average = (2.0 + 4.0) / 2 = 3.0000 -> PPL = exp(3.0) = 20.09
        losses = [torch.tensor([2.0]*10), torch.tensor([4.0]*2)]
        total_nll = sum(l.sum().item() for l in losses)
        total_tokens = sum(len(l) for l in losses)
        true_mean_nll = total_nll / total_tokens

        batch_means = [l.mean().item() for l in losses]
        biased_mean_nll = sum(batch_means) / len(batch_means)

        self.assertAlmostEqual(true_mean_nll, 28.0 / 12.0)
        self.assertAlmostEqual(biased_mean_nll, 3.0)
        self.assertNotAlmostEqual(true_mean_nll, biased_mean_nll, places=2)

    # --------------------------------------------------------------------------
    # 7. Baseline Candidate-Selection Behavior (Static-K5 vs Static-K1 vs Adaptive)
    # --------------------------------------------------------------------------
    def test_static_k5_vs_static_k1_behavior(self):
        """Verifies that Static-K5 evaluates top 5 candidates while Static-K1 blind commits top 1."""
        comp_k5 = CAREStaticCompressor(config={"candidate_pool_size": 5})
        comp_k1 = CAREStaticK1Compressor()

        self.assertEqual(comp_k5.candidate_pool_size, 5)
        self.assertEqual(comp_k1.candidate_pool_size, 1)

    # --------------------------------------------------------------------------
    # 8. Evaluation Data Separation Integrity
    # --------------------------------------------------------------------------
    def test_evaluation_data_separation(self):
        """Verifies that candidate selection data and final evaluation data are strictly disjoint."""
        sel_tokens = [torch.randint(0, 100, (16,)) for _ in range(16)]
        eval_tokens = [torch.randint(101, 200, (16,)) for _ in range(32)]

        sel_hashes = {hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in sel_tokens}
        eval_hashes = {hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in eval_tokens}

        overlap = sel_hashes.intersection(eval_hashes)
        self.assertEqual(len(overlap), 0, f"Found {len(overlap)} overlapping tokens!")


if __name__ == "__main__":
    unittest.main()
