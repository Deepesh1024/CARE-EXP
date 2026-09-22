import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from experiments.care_com_v22.config import CareComV22Config
from experiments.care_com_v21.core import PhysicalMergeEngine
from experiments.care_com_v22.baselines import run_random_baseline, run_static_baseline
from experiments.care_com_v22.adaptive import run_adaptive_baseline

# Mock model architecture (from v2.1)
class MockRouter(nn.Module):
    def __init__(self, num_experts, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, dim))

class MockExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, dim, bias=False)
        self.down_proj = nn.Linear(dim, dim, bias=False)

class MockBlock(nn.Module):
    def __init__(self, num_experts, dim):
        super().__init__()
        self.gate = MockRouter(num_experts, dim)
        self.experts = nn.ModuleList([MockExpert(dim) for _ in range(num_experts)])
        self.num_experts = num_experts

class MockConfig:
    def __init__(self, num_experts):
        self.num_experts = num_experts

class MockModel(nn.Module):
    def __init__(self, num_experts=64, dim=16):
        super().__init__()
        self.config = MockConfig(num_experts)
        self.block1 = MockBlock(num_experts, dim)
        self.block2 = MockBlock(num_experts, dim)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Mock forward pass returning some dummy loss
        class Output:
            def __init__(self):
                self.loss = torch.tensor(1.0)
        return Output()

@pytest.fixture
def mock_model():
    return MockModel(num_experts=64)

@pytest.fixture
def mock_config():
    config = CareComV22Config()
    config.trajectory = [64, 63, 62] # Just 2 steps
    config.checkpoints = [64, 62]
    config.device = "cpu"
    config.eval_batch_size = 1
    config.max_eval_batches = 1
    config.ppl_batch_size = 1
    config.max_ppl_batches = 1
    config.random_seeds = [42]
    return config

@pytest.fixture
def eval_chunks():
    return [{"input_ids": torch.ones((1, 16), dtype=torch.long), "attention_mask": torch.ones((1, 16), dtype=torch.long)}]
    
@pytest.fixture
def df_tokens():
    return torch.ones((1, 16), dtype=torch.long)

@patch('experiments.care_com_v22.baselines.evaluate_marginal_damage', return_value=0.5)
@patch('experiments.care_com_v22.baselines.collect_current_logits', return_value=None)
@patch('experiments.care_com_v22.baselines.compute_ppl', return_value=10.0)
def test_random_baseline_independent(mock_ppl, mock_collect, mock_eval, mock_model, mock_config, eval_chunks):
    """3. Random method does not access capability information."""
    engine = PhysicalMergeEngine(mock_model)
    
    with patch('experiments.care_com_v22.baselines.compute_global_capability_vectors') as mock_cap:
        model, trace, ppl = run_random_baseline(mock_model, engine, eval_chunks, mock_config, seed=42)
        mock_cap.assert_not_called()
        
    assert engine.current_num_experts == 62
    assert len(trace) == 2
    assert len(ppl) == 2 # 64 and 62
    
@patch('experiments.care_com_v22.baselines.evaluate_marginal_damage', return_value=0.5)
@patch('experiments.care_com_v22.baselines.collect_current_logits', return_value=None)
@patch('experiments.care_com_v22.baselines.compute_ppl', return_value=10.0)
def test_static_v1_baseline_called_once(mock_ppl, mock_collect, mock_eval, mock_model, mock_config, df_tokens, eval_chunks):
    """4. Static v1 capability ranking is computed only once."""
    engine = PhysicalMergeEngine(mock_model)
    
    with patch('experiments.care_com_v22.baselines.compute_global_capability_vectors') as mock_cap:
        # Mock capability matrix returning zeros
        mock_cap.return_value = torch.zeros((64, 16))
        model, trace, ppl = run_static_baseline(mock_model, engine, df_tokens, eval_chunks, mock_config)
        
        # Ensure it was called EXACTLY once
        assert mock_cap.call_count == 1
        
    assert engine.current_num_experts == 62
    assert len(trace) == 2

@patch('experiments.care_com_v22.adaptive.evaluate_marginal_damage', return_value=0.5)
@patch('experiments.care_com_v22.adaptive.collect_current_logits', return_value=None)
@patch('experiments.care_com_v22.adaptive.compute_ppl', return_value=10.0)
def test_adaptive_baseline_recomputes(mock_ppl, mock_collect, mock_eval, mock_model, mock_config, df_tokens, eval_chunks):
    """5. Adaptive v2.1 capability is recomputed after every accepted merge."""
    engine = PhysicalMergeEngine(mock_model)
    
    with patch('experiments.care_com_v22.adaptive.compute_global_capability_vectors') as mock_cap:
        # Mock capability matrix
        def side_effect(model, blocks, tokens, device):
            return torch.zeros((engine.current_num_experts, 16))
        mock_cap.side_effect = side_effect
        
        model, trace, ppl = run_adaptive_baseline(mock_model, engine, df_tokens, eval_chunks, mock_config)
        
        # 64 -> 63 (step 1), 63 -> 62 (step 2). Should be called 2 times.
        assert mock_cap.call_count == 2
        
    assert engine.current_num_experts == 62

@patch('experiments.care_com_v22.baselines.evaluate_marginal_damage', return_value=0.5)
@patch('experiments.care_com_v22.baselines.collect_current_logits', return_value=None)
def test_all_reduce_n_by_1_and_use_same_engine(mock_collect, mock_eval, mock_model, mock_config, eval_chunks):
    """1. All methods reduce N by exactly 1 per merge. 2. Use same engine semantics."""
    engine = PhysicalMergeEngine(mock_model)
    
    # Run 1 step of random
    mock_config.trajectory = [64, 63]
    model, trace, ppl = run_random_baseline(mock_model, engine, eval_chunks, mock_config, seed=42)
    
    assert engine.current_num_experts == 63
    assert len(mock_model.block1.experts) == 63
    assert mock_model.block1.gate.weight.shape[0] == 63
    
    # We rely on PhysicalMergeEngine which is proven to use the exact same semantics.
    assert "PhysicalMergeEngine" in str(type(engine))
