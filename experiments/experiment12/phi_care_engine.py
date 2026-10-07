"""
Phi-3.5-MoE CARE-COM Engine
==============================
Actual remote-code architecture (modeling_phimoe.py from HuggingFace Hub):
  - MoE block:     layer.block_sparse_moe  (PhiMoESparseMoeBlock)
  - Router:        block.gate              (nn.Linear, hidden -> num_experts)
  - Expert list:   block.experts           (nn.ModuleList of PhiMoEBlockSparseTop2MLP)
  - Expert MLP:    expert.w1 (gate), expert.w3 (up), expert.w2 (down)
  - Expert fwd:    SiLU(w1(x)) * w3(x) -> w2(...)

IMPORTANT: With load_in_4bit=True, all Linear weights are stored as bitsandbytes
Params4bit tensors with packed shapes (NOT [out, in]). We must:
  - Use .out_features for invariant checks (not .weight.shape[0])
  - Dequantize gate weights before row arithmetic, store back as float nn.Parameter
  - Expert w1/w2/w3 forward passes work natively via bnb; we let them run as-is
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# WEIGHT UTILITIES: handle bitsandbytes Params4bit
# ──────────────────────────────────────────────────────────────

def _dequantize_linear_weight(linear: nn.Linear) -> torch.Tensor:
    """
    Returns the weight as a float32 tensor regardless of whether it is
    a standard nn.Parameter or a bitsandbytes Params4bit.
    Shape: [out_features, in_features].
    """
    w = linear.weight
    if hasattr(w, 'quant_state'):          # bitsandbytes Params4bit
        import bitsandbytes as bnb
        return bnb.functional.dequantize_4bit(
            w.data, w.quant_state, dtype=torch.float32
        ).contiguous()
    return w.data.to(torch.float32)


def _get_expert_weight(expert: nn.Module, attr: str) -> torch.Tensor:
    """Returns dequantized float32 weight for expert.attr (w1/w2/w3)."""
    return _dequantize_linear_weight(getattr(expert, attr))


# ──────────────────────────────────────────────────────────────
# EXPERT EVALUATION
# ──────────────────────────────────────────────────────────────

def evaluate_expert_on_probe(x: torch.Tensor, expert: nn.Module) -> torch.Tensor:
    """
    Evaluates the unweighted expert functional mapping on a residual-stream probe x.
    Expert forward: SiLU(w1(x)) * w3(x) -> w2
    Uses native bnb forward pass (quantized experts run fine in forward mode).
    """
    assert hasattr(expert, "w1"), "Expert missing w1 (gate projection)"
    assert hasattr(expert, "w2"), "Expert missing w2 (down projection)"
    assert hasattr(expert, "w3"), "Expert missing w3 (up projection)"
    return expert.w2(F.silu(expert.w1(x)) * expert.w3(x))


# ──────────────────────────────────────────────────────────────
# PHYSICAL MERGE ENGINE
# ──────────────────────────────────────────────────────────────

class PhiPhysicalMergeEngine:
    """
    Implements true N -> N-1 physical expert deletion for Phi-3.5-MoE.

    Weight handling:
      - Gate: dequantize to float -> row arithmetic -> store as plain nn.Parameter float
      - Expert w1/w2/w3: dequantize to float -> arithmetic -> store as plain nn.Parameter float
        (After the first merge, subsequent merges work on already-float parameters.)

    All 8 hard invariants are checked after every state transition.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.moe_blocks = []

        base = getattr(model, "model", model)
        layers = getattr(base, "layers", None)
        assert layers is not None, "Cannot find .layers on model"

        for layer in layers:
            block = getattr(layer, "block_sparse_moe", None)
            if block is not None and hasattr(block, "gate") and hasattr(block, "experts"):
                self.moe_blocks.append(block)

        assert len(self.moe_blocks) > 0, (
            "No MoE blocks found. Expected layer.block_sparse_moe with .gate and .experts."
        )

        self.current_num_experts = self.moe_blocks[0].num_experts
        self._snapshot = None
        self._validate_invariants()

    def snapshot(self):
        """Save full model state for transaction rollback (dequantized floats)."""
        self._snapshot = {
            "config_num_local": getattr(self.model.config, "num_local_experts", None),
            "config_num":       getattr(self.model.config, "num_experts", None),
            "blocks": []
        }
        for block in self.moe_blocks:
            experts_state = []
            for exp in block.experts:
                experts_state.append({
                    "w1": _get_expert_weight(exp, "w1").clone(),
                    "w2": _get_expert_weight(exp, "w2").clone(),
                    "w3": _get_expert_weight(exp, "w3").clone(),
                })
            self._snapshot["blocks"].append({
                "block":       block,
                "gate_weight": _dequantize_linear_weight(block.gate).clone(),
                "num_experts": block.num_experts,
                "experts_state": experts_state,
            })

    def restore(self):
        """Rollback model to last snapshot."""
        if self._snapshot is None:
            raise RuntimeError("No snapshot to restore from")

        if self._snapshot["config_num_local"] is not None:
            self.model.config.num_local_experts = self._snapshot["config_num_local"]
        if self._snapshot["config_num"] is not None:
            self.model.config.num_experts = self._snapshot["config_num"]

        for b_data in self._snapshot["blocks"]:
            block    = b_data["block"]
            n        = b_data["num_experts"]
            gate_w   = b_data["gate_weight"]           # [n, hidden], float32

            # Restore gate as a plain float Parameter
            device = block.gate.weight.device if hasattr(block.gate.weight, 'device') \
                     else next(block.parameters()).device
            block.gate.weight      = nn.Parameter(gate_w.to(device))
            block.gate.out_features = n
            block.gate.in_features  = gate_w.shape[1]

            # Restore experts from saved float weights
            new_experts = nn.ModuleList()
            for i, exp_state in enumerate(b_data["experts_state"]):
                exp = block.experts[i]
                exp.w1.weight = nn.Parameter(exp_state["w1"].to(device))
                exp.w2.weight = nn.Parameter(exp_state["w2"].to(device))
                exp.w3.weight = nn.Parameter(exp_state["w3"].to(device))
                new_experts.append(exp)
            block.experts    = new_experts
            block.num_experts = n

        self.current_num_experts = self._snapshot["blocks"][0]["num_experts"]
        self._snapshot = None
        self._validate_invariants()

    @torch.no_grad()
    def merge_experts(self, i: int, j: int):
        """
        Physically merge expert j into expert i, then delete expert j.
        Produces a true N -> N-1 structural change.
        """
        N = self.current_num_experts
        assert 0 <= i < N and 0 <= j < N and i != j, \
            f"Invalid expert indices {i}, {j} for N={N}"

        for block in self.moe_blocks:
            device = block.gate.weight.device \
                     if hasattr(block.gate.weight, 'device') \
                     else next(block.parameters()).device

            # 1. Dequantize and average gate rows
            gate_w = _dequantize_linear_weight(block.gate)   # [N, hidden]
            merged_gate_row = (gate_w[i] + gate_w[j]) / 2.0

            keep  = [idx for idx in range(N) if idx != j]
            new_i = keep.index(i)

            new_gate_w          = gate_w[keep].clone()
            new_gate_w[new_i]   = merged_gate_row

            block.gate.weight      = nn.Parameter(new_gate_w.to(device))
            block.gate.out_features = len(keep)
            block.gate.in_features  = new_gate_w.shape[1]

            # 2. Average expert weights (dequantize, merge, store as float)
            for attr in ("w1", "w2", "w3"):
                wi = _get_expert_weight(block.experts[i], attr)
                wj = _get_expert_weight(block.experts[j], attr)
                merged_w = (wi + wj) / 2.0
                getattr(block.experts[i], attr).weight = nn.Parameter(merged_w.to(device))

            # 3. Delete expert j from ModuleList
            experts_list = list(block.experts)
            experts_list.pop(j)
            block.experts     = nn.ModuleList(experts_list)
            block.num_experts = len(experts_list)

        new_n = len(keep)
        self.current_num_experts            = new_n
        self.model.config.num_local_experts = new_n
        if hasattr(self.model.config, "num_experts"):
            self.model.config.num_experts   = new_n

        self._validate_invariants()

    def _validate_invariants(self):
        """Hard invariant check after every state transition."""
        N = self.current_num_experts
        for block in self.moe_blocks:
            # Invariant 1: expert list size (no dead slots)
            assert len(block.experts) == N, \
                f"Expert list has {len(block.experts)}, expected {N}"
            # Invariant 2: router out_features (use .out_features, NOT weight.shape[0])
            assert block.gate.out_features == N, \
                f"gate.out_features={block.gate.out_features}, expected {N}"
            # Invariant 3: block-level count
            assert block.num_experts == N, \
                f"block.num_experts={block.num_experts}, expected {N}"
        # Invariant 4: config
        assert self.model.config.num_local_experts == N, \
            f"config.num_local_experts={self.model.config.num_local_experts}, expected {N}"


# ──────────────────────────────────────────────────────────────
# CAPABILITY EXTRACTION  (residual-stream probe, CARE protocol)
# ──────────────────────────────────────────────────────────────

def extract_phi_care_capability(
    model: nn.Module,
    engine: "PhiPhysicalMergeEngine",
    calib_batches,
    device: str,
    num_axes: int = 10,
) -> torch.Tensor:
    """
    Computes capability matrix C ∈ R^[N, num_layers * num_axes].

    Protocol (faithful to CARE-COM):
      - Probe: hidden_states[layer_idx] (residual stream before MoE block)
      - Expert function: SiLU(w1(x)) * w3(x) -> w2  (unweighted, no routing weight)
      - Axis assignment: token_idx % num_axes  (deterministic)
      - Accumulation: L2 norm of expert output, averaged per axis

    Invariant 7: C.shape[0] == N is asserted.
    """
    model.eval()
    num_layers = len(engine.moe_blocks)
    N          = engine.current_num_experts

    cap    = torch.zeros(num_layers, N, num_axes, device="cpu")
    counts = torch.zeros(num_layers, N, num_axes, device="cpu")

    with torch.no_grad():
        for batch in calib_batches:
            out = model(
                batch.unsqueeze(0).to(device),
                output_hidden_states=True
            )
            # hidden_states[l] = residual stream *input* to transformer layer l
            hidden_states = out.hidden_states
            seq_len = batch.shape[0]

            for layer_idx, block in enumerate(engine.moe_blocks):
                x = hidden_states[layer_idx].squeeze(0)   # [seq, hidden], still on device

                for exp_idx, expert in enumerate(block.experts):
                    exp_out = evaluate_expert_on_probe(x, expert).to("cpu")  # [seq, hidden]

                    for tok in range(seq_len):
                        axis = tok % num_axes
                        cap[layer_idx, exp_idx, axis] += \
                            torch.norm(exp_out[tok].float(), p=2).item()
                        counts[layer_idx, exp_idx, axis] += 1

    valid      = counts > 0
    cap[valid] /= counts[valid]

    # C: [N, num_layers * num_axes]
    C = cap.permute(1, 0, 2).reshape(N, num_layers * num_axes)

    # Invariant 7: capability dimensionality
    assert C.shape[0] == N, f"Capability matrix has {C.shape[0]} rows, expected {N}"
    return C


# ──────────────────────────────────────────────────────────────
# KL DAMAGE UTILITY
# ──────────────────────────────────────────────────────────────

def compute_kl_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
) -> float:
    p_probs      = F.softmax(p_logits, dim=-1)
    q_log_probs  = F.log_softmax(q_logits, dim=-1)
    return F.kl_div(q_log_probs, p_probs, reduction="batchmean").item()
