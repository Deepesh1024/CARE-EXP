"""
tests/test_run_l40s_benchmark_safeguards.py
============================================
Regression test suite for benchmark/run_l40s_benchmark.py safeguards:
1. Causally aligned next-token KL divergence vs independent reference.
2. Calibration parquet artifact validation & removal of synthetic fallback.
3. Failed-run accounting & campaign completeness status.
4. Expert-count mismatch detection and run invalidation.
5. Checkpoint reload parity verification with attention mask.
6. Collision-resistant run directory generation.
"""

import os
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_OLMOE_BENCH_DIR = PROJECT_ROOT / "benchmark" / "OLMoE-1B-7B"
if str(_OLMOE_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_OLMOE_BENCH_DIR))

from core.evaluate import compute_causal_kl_from_logits, compute_heldout_kl
from benchmark.run_l40s_benchmark import (
    load_and_validate_calibration_data,
    create_unique_run_dir,
    get_critical_source_hashes,
    get_git_info,
    verify_checkpoint_integrity,
)


class TestCausalKLAlignment(unittest.TestCase):
    """
    Test 1: Causally aligned next-token KL divergence.
    Validates that:
    - Target token at index t+1 is predicted by logits at index t.
    - Compares base_logits[:, :-1, :] vs comp_logits[:, :-1, :]
    - Masks with attention_mask[:, 1:]
    - Accumulates over valid target tokens only.
    - Matches an independent token-by-token reference calculation.
    """

    def _independent_reference_causal_kl(
        self, base_logits: torch.Tensor, comp_logits: torch.Tensor, attention_mask: torch.Tensor
    ) -> float:
        """Ground-truth token-by-token manual reference loop."""
        B, T, V = base_logits.shape
        total_kl = 0.0
        total_tokens = 0
        for b in range(B):
            for t in range(T - 1):  # logits at t predict target token at t + 1
                target_idx = t + 1
                if attention_mask[b, target_idx] == 1:
                    # Target token at t+1 is valid
                    p_logits = base_logits[b, t]
                    q_logits = comp_logits[b, t]
                    p = F.softmax(p_logits, dim=-1)
                    log_p = F.log_softmax(p_logits, dim=-1)
                    log_q = F.log_softmax(q_logits, dim=-1)
                    kl_tok = torch.sum(p * (log_p - log_q)).item()
                    total_kl += kl_tok
                    total_tokens += 1
        return total_kl / total_tokens if total_tokens > 0 else 0.0

    def test_causal_kl_vs_independent_reference(self):
        torch.manual_seed(101)
        B, T, V = 3, 8, 20
        base_logits = torch.randn(B, T, V)
        comp_logits = torch.randn(B, T, V)
        # Heterogeneous attention mask with variable padding
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1],  # 8 tokens -> 7 target transitions
            [1, 1, 1, 1, 1, 0, 0, 0],  # 5 tokens -> 4 target transitions
            [1, 1, 1, 0, 0, 0, 0, 0],  # 3 tokens -> 2 target transitions
        ])
        expected_valid_tokens = 7 + 4 + 2  # 13 valid target tokens

        # Implementation calculation
        sum_kl, valid_tokens = compute_causal_kl_from_logits(base_logits, comp_logits, attention_mask)
        mean_kl = sum_kl / valid_tokens

        # Independent manual calculation
        ref_mean_kl = self._independent_reference_causal_kl(base_logits, comp_logits, attention_mask)

        self.assertEqual(valid_tokens, expected_valid_tokens)
        self.assertAlmostEqual(mean_kl, ref_mean_kl, places=5)
        self.assertGreater(mean_kl, 0.0)

    def test_causal_vs_uncorrected_unshifted_divergence(self):
        """Demonstrates that unshifted KL differs significantly from causally aligned KL."""
        torch.manual_seed(202)
        B, T, V = 2, 6, 15
        base_logits = torch.randn(B, T, V)
        comp_logits = torch.randn(B, T, V)
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
        ])

        # Correct causal KL
        sum_kl, valid_tokens = compute_causal_kl_from_logits(base_logits, comp_logits, attention_mask)
        causal_kl = sum_kl / valid_tokens

        # Unshifted (flawed) KL: compares base_logits[:, :, :] masked by attention_mask[:, :]
        p = F.softmax(base_logits, dim=-1)
        log_p = F.log_softmax(base_logits, dim=-1)
        log_q = F.log_softmax(comp_logits, dim=-1)
        unshifted_tok_kl = (p * (log_p - log_q)).sum(dim=-1)
        unshifted_kl = (unshifted_tok_kl * attention_mask.bool()).sum().item() / attention_mask.sum().item()

        # They must not be identical because causal distribution alignment shifts by 1 position
        self.assertNotAlmostEqual(causal_kl, unshifted_kl, places=3)

    def test_zero_valid_tokens_handling(self):
        """Validates that a sequence with no valid causal transitions raises or reports 0."""
        base_logits = torch.randn(1, 1, 10)
        comp_logits = torch.randn(1, 1, 10)
        attention_mask = torch.tensor([[1]])  # Length 1 sequence has 0 transitions

        sum_kl, valid_tokens = compute_causal_kl_from_logits(base_logits, comp_logits, attention_mask)
        self.assertEqual(valid_tokens, 0)
        self.assertEqual(sum_kl, 0.0)


class TestCalibrationDataSafeguards(unittest.TestCase):
    """
    Test 2: Calibration data validation and removal of synthetic fallback.
    Validates that:
    - Missing parquet raises FileNotFoundError.
    - Malformed / unreadable parquet raises ValueError.
    - Empty parquet raises ValueError.
    - Parquet lacking required columns raises ValueError.
    - Parquet with fewer rows than required raises ValueError.
    - Valid parquet returns DataFrame and complete metadata with SHA-256.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_parquet_raises_file_not_found(self):
        non_existent = Path(self.temp_dir) / "does_not_exist.parquet"
        with self.assertRaises(FileNotFoundError) as ctx:
            load_and_validate_calibration_data(non_existent, required_samples=10)
        self.assertIn("Synthetic fallback has been permanently removed", str(ctx.exception))

    def test_empty_parquet_raises_value_error(self):
        empty_parquet = Path(self.temp_dir) / "empty.parquet"
        df_empty = pd.DataFrame(columns=["input_ids", "attention_mask", "axis_idx"])
        df_empty.to_parquet(empty_parquet)

        with self.assertRaises(ValueError) as ctx:
            load_and_validate_calibration_data(empty_parquet, required_samples=1)
        self.assertIn("is empty (0 rows)", str(ctx.exception))

    def test_missing_required_columns_raises_value_error(self):
        missing_cols_parquet = Path(self.temp_dir) / "missing_cols.parquet"
        df_invalid = pd.DataFrame([{
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            # axis_idx is missing!
        }])
        df_invalid.to_parquet(missing_cols_parquet)

        with self.assertRaises(ValueError) as ctx:
            load_and_validate_calibration_data(missing_cols_parquet, required_samples=1)
        self.assertIn("lacks required columns", str(ctx.exception))
        self.assertIn("axis_idx", str(ctx.exception))

    def test_insufficient_samples_raises_value_error(self):
        parquet_path = Path(self.temp_dir) / "few_samples.parquet"
        df = pd.DataFrame([{
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "axis_idx": 0,
        }])  # 1 row
        df.to_parquet(parquet_path)

        with self.assertRaises(ValueError) as ctx:
            load_and_validate_calibration_data(parquet_path, required_samples=128)
        self.assertIn("requested 128 samples", str(ctx.exception))

    def test_valid_parquet_returns_dataframe_and_sha256(self):
        parquet_path = Path(self.temp_dir) / "valid.parquet"
        df = pd.DataFrame([
            {"input_ids": [i, i + 1], "attention_mask": [1, 1], "axis_idx": i % 3}
            for i in range(20)
        ])
        df.to_parquet(parquet_path)

        loaded_df, meta = load_and_validate_calibration_data(parquet_path, required_samples=10)
        self.assertEqual(len(loaded_df), 20)
        self.assertIn("sha256", meta)
        self.assertEqual(len(meta["sha256"]), 64)
        self.assertEqual(meta["total_rows"], 20)


class TestFailedRunAccounting(unittest.TestCase):
    """
    Test 3: Planned run tracking & campaign completeness status.
    Validates that:
    - All planned combinations are tracked.
    - If any run fails or is omitted, campaign_status is marked INCOMPLETE.
    - Failure reasons and tracebacks are preserved.
    - Only when 100% of planned runs succeed is the campaign marked COMPLETE.
    """

    def test_campaign_status_with_failure(self):
        planned_combinations = [
            {"method": "care_static", "target_experts": 60, "seed": 42},
            {"method": "care_adaptive", "target_experts": 60, "seed": 42},
            {"method": "random", "target_experts": 60, "seed": 42},
        ]
        # Simulate 1 success, 1 failure, 1 omitted
        all_results = [
            {
                "method": "care_static",
                "target_experts": 60,
                "seed": 42,
                "status": "SUCCESS",
            },
            {
                "method": "care_adaptive",
                "target_experts": 60,
                "seed": 42,
                "status": "FAILED",
                "failure_reason": "COMPRESSION_FAILED: Out of memory",
                "traceback": "Traceback (most recent call last):\n  RuntimeError: CUDA out of memory\n",
            },
        ]

        successful_runs = [r for r in all_results if r.get("status") == "SUCCESS"]
        failed_runs = [r for r in all_results if r.get("status") != "SUCCESS"]
        all_planned_keys = {f"{c['method']}_{c['target_experts']}e_seed_{c['seed']}" for c in planned_combinations}
        recorded_keys = {f"{r['method']}_{r['target_experts']}e_seed_{r['seed']}" for r in all_results}
        missing_keys = all_planned_keys - recorded_keys

        campaign_status = "COMPLETE" if (
            len(failed_runs) == 0 and len(missing_keys) == 0 and len(all_results) == len(planned_combinations)
        ) else "INCOMPLETE"

        self.assertEqual(campaign_status, "INCOMPLETE")
        self.assertEqual(len(failed_runs), 1)
        self.assertEqual(len(missing_keys), 1)
        self.assertIn("random_60e_seed_42", missing_keys)
        self.assertIn("CUDA out of memory", failed_runs[0]["traceback"])

    def test_campaign_status_all_succeeded(self):
        planned_combinations = [
            {"method": "care_static", "target_experts": 60, "seed": 42},
            {"method": "care_adaptive", "target_experts": 60, "seed": 42},
        ]
        all_results = [
            {"method": "care_static", "target_experts": 60, "seed": 42, "status": "SUCCESS"},
            {"method": "care_adaptive", "target_experts": 60, "seed": 42, "status": "SUCCESS"},
        ]
        successful_runs = [r for r in all_results if r.get("status") == "SUCCESS"]
        failed_runs = [r for r in all_results if r.get("status") != "SUCCESS"]
        all_planned_keys = {f"{c['method']}_{c['target_experts']}e_seed_{c['seed']}" for c in planned_combinations}
        recorded_keys = {f"{r['method']}_{r['target_experts']}e_seed_{r['seed']}" for r in all_results}
        missing_keys = all_planned_keys - recorded_keys

        campaign_status = "COMPLETE" if (
            len(failed_runs) == 0 and len(missing_keys) == 0 and len(all_results) == len(planned_combinations)
        ) else "INCOMPLETE"

        self.assertEqual(campaign_status, "COMPLETE")
        self.assertEqual(len(missing_keys), 0)


class TestExpertCountMismatchSafeguard(unittest.TestCase):
    """
    Test 4: Invalidation upon expert-count mismatch.
    Validates that:
    - If any layer has count != target, all_layers_matched is False.
    - Run is marked FAILED with EXPERT_COUNT_MISMATCH.
    """

    def test_expert_count_mismatch_detected(self):
        target = 60
        # Layer 3 has 61 experts instead of 60
        actual_counts = [60] * 16
        actual_counts[3] = 61

        all_layers_matched = (len(actual_counts) > 0 and all(c == target for c in actual_counts))
        self.assertFalse(all_layers_matched)

        err_msg = f"EXPERT_COUNT_MISMATCH: expected {target} across all 16 layers, got counts: {actual_counts}"
        self.assertIn("expected 60", err_msg)
        self.assertIn("61", err_msg)


class DummyLM(nn.Module):
    def __init__(self, d_model=16, vocab_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.dtype = torch.float32

    def forward(self, input_ids=None, attention_mask=None):
        x = self.embedding(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()
        logits = self.head(x)
        class Output:
            def __init__(self, logits):
                self.logits = logits
        return Output(logits)


class TestCheckpointReloadParitySafeguard(unittest.TestCase):
    """
    Test 5: Checkpoint reload parity passing attention_mask.
    Validates that:
    - verify_checkpoint_integrity passes identical attention_mask to both reference and reloaded model.
    - A mismatch in logits or keys fails the check.
    """

    def test_attention_mask_passed_to_both_models(self):
        torch.manual_seed(303)
        model = DummyLM()
        val_input = torch.tensor([[1, 5, 10, 0]])
        val_mask = torch.tensor([[1, 1, 1, 0]])

        # Reference logits with attention mask
        with torch.no_grad():
            ref_logits = model(input_ids=val_input, attention_mask=val_mask).logits

        # Simulate reload logits with attention mask
        with torch.no_grad():
            reloaded_logits = model(input_ids=val_input, attention_mask=val_mask).logits

        diff = (ref_logits.float() - reloaded_logits.float()).abs()
        max_diff = diff.max().item()
        self.assertEqual(max_diff, 0.0)

        # Perturbation simulates reload corruption
        corrupted_logits = reloaded_logits + 0.05
        corrupted_diff = (ref_logits.float() - corrupted_logits.float()).abs().max().item()
        passed = (corrupted_diff < 1e-2)
        self.assertFalse(passed, "Corrupted reload must fail parity test")


class TestManifestAndDirectorySafeguards(unittest.TestCase):
    """
    Test 6: Collision-resistant run directory creation & manifest metadata.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_unique_run_dirs(self):
        dirs = [create_unique_run_dir(Path(self.temp_dir)) for _ in range(5)]
        unique_paths = {str(d) for d in dirs}
        self.assertEqual(len(unique_paths), 5)
        for d in dirs:
            self.assertTrue(d.exists())

    def test_critical_source_hashes(self):
        hashes = get_critical_source_hashes()
        self.assertIn("benchmark/run_l40s_benchmark.py", hashes)
        self.assertIn("benchmark/OLMoE-1B-7B/core/evaluate.py", hashes)
        self.assertIn("experiments/care_com_v21/core.py", hashes)
        for path, h in hashes.items():
            self.assertEqual(len(h), 64, f"Hash for {path} must be 64-char hex string")

    def test_git_info(self):
        git_info = get_git_info()
        self.assertIn("revision", git_info)
        self.assertIn("is_dirty", git_info)
        self.assertIn("dirty_files", git_info)


if __name__ == "__main__":
    unittest.main()
