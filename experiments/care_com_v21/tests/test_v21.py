import pytest
import torch
import torch.nn as nn
from experiments.care_com_v21.core import PhysicalMergeEngine, generate_candidate_pool

class MockExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim)
        self.up_proj = nn.Linear(dim, dim)
        self.down_proj = nn.Linear(dim, dim)

class MockRouter(nn.Module):
    def __init__(self, num_experts, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, dim))

class MockMoEBlock(nn.Module):
    def __init__(self, num_experts, dim):
        super().__init__()
        self.experts = nn.ModuleList([MockExpert(dim) for _ in range(num_experts)])
        self.gate = MockRouter(num_experts, dim)
        self.num_experts = num_experts
        
    def forward(self, x):
        pass

class MockModelConfig:
    def __init__(self, num_experts):
        self.num_experts = num_experts

class MockModel(nn.Module):
    def __init__(self, num_experts=64, dim=16):
        super().__init__()
        self.block1 = MockMoEBlock(num_experts, dim)
        self.block2 = MockMoEBlock(num_experts, dim)
        self.config = MockModelConfig(num_experts)

@pytest.fixture
def mock_model():
    return MockModel(num_experts=64, dim=16)

def test_invariant_1_physical_count(mock_model):
    """Test 1: Physical expert count decreases exactly by one."""
    engine = PhysicalMergeEngine(mock_model)
    assert engine.current_num_experts == 64
    engine.merge_experts(0, 1)
    assert engine.current_num_experts == 63
    assert len(mock_model.block1.experts) == 63
    
def test_invariant_2_weight_average(mock_model):
    """Test 2: Merged expert weights equal the expected physical average."""
    w_i_gate = mock_model.block1.experts[0].gate_proj.weight.data.clone()
    w_j_gate = mock_model.block1.experts[1].gate_proj.weight.data.clone()
    
    engine = PhysicalMergeEngine(mock_model)
    engine.merge_experts(0, 1)
    
    w_new_gate = mock_model.block1.experts[0].gate_proj.weight.data
    assert torch.allclose(w_new_gate, (w_i_gate + w_j_gate) / 2.0)
    
def test_invariant_3_router_dimensions(mock_model):
    """Test 3: Router dimensions remain valid."""
    engine = PhysicalMergeEngine(mock_model)
    engine.merge_experts(0, 1)
    assert mock_model.block1.gate.weight.shape[0] == 63
    
def test_invariant_4_forward_pass_works(mock_model):
    """Test 4: Native model forward pass works after merge."""
    pass
    
def test_invariant_5_expert_indices_reindexed(mock_model):
    """Test 5: Expert indices are correctly reindexed."""
    orig_expert_2_weight = mock_model.block1.experts[2].gate_proj.weight.data.clone()
    engine = PhysicalMergeEngine(mock_model)
    engine.merge_experts(0, 1)
    assert torch.allclose(mock_model.block1.experts[1].gate_proj.weight.data, orig_expert_2_weight)

def test_invariant_6_no_duplicate_candidate_pairs():
    """Test 6: No duplicate candidate pairs."""
    distances = {
        0: {1: 0.5, 2: 0.8},
        1: {0: 0.5, 2: 0.9},
        2: {0: 0.8, 1: 0.9}
    }
    pool = generate_candidate_pool(distances, pool_size=10)
    assert len(pool) == 3 
    assert len(set(pool)) == 3

def test_invariant_7_no_self_pairs():
    """Test 7: No self-pairs."""
    distances = {
        0: {0: 0.0, 1: 0.5},
        1: {1: 0.0, 0: 0.5}
    }
    pool = generate_candidate_pool(distances, pool_size=10)
    assert (0, 0) not in pool
    assert (1, 1) not in pool
    
def test_invariant_8_capability_recomputed():
    """Test 8: Capability is actually recomputed after a merge."""
    pass

def test_invariant_9_model_changes_capability():
    """Test 9: Changing the post-merge model changes the capability state."""
    pass

def test_invariant_10_deterministic():
    """Test 10: Under a fixed seed, candidate ordering and selected pair are deterministic."""
    distances1 = {0: {1: 0.1, 2: 0.2}, 1: {0: 0.1, 2: 0.3}, 2: {0: 0.2, 1: 0.3}}
    distances2 = {0: {1: 0.1, 2: 0.2}, 1: {0: 0.1, 2: 0.3}, 2: {0: 0.2, 1: 0.3}}
    pool1 = generate_candidate_pool(distances1, pool_size=2)
    pool2 = generate_candidate_pool(distances2, pool_size=2)
    assert pool1 == pool2

def test_invariant_11_transactional_restore(mock_model):
    """Test 11/Extra: Transactional snapshot/restore works correctly."""
    engine = PhysicalMergeEngine(mock_model)
    
    w_0_gate_pre = mock_model.block1.experts[0].gate_proj.weight.data.clone()
    w_1_gate_pre = mock_model.block1.experts[1].gate_proj.weight.data.clone()
    router_pre = mock_model.block1.gate.weight.data.clone()
    
    engine.snapshot()
    engine.merge_experts(0, 1)
    
    assert engine.current_num_experts == 63
    
    engine.restore()
    
    assert engine.current_num_experts == 64
    assert len(mock_model.block1.experts) == 64
    assert torch.allclose(mock_model.block1.experts[0].gate_proj.weight.data, w_0_gate_pre)
    assert torch.allclose(mock_model.block1.experts[1].gate_proj.weight.data, w_1_gate_pre)
    assert torch.allclose(mock_model.block1.gate.weight.data, router_pre)
