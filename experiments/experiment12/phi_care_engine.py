"""
Phi-3.5-MoE CARE-COM Engine  (memory-safe revision)
=====================================================
Actual remote-code architecture:
  - MoE block:   layer.block_sparse_moe  (PhiMoESparseMoeBlock)
  - Router:      block.gate              (nn.Linear, hidden -> num_experts)
  - Experts:     block.experts           (nn.ModuleList of PhiMoEBlockSparseTop2MLP)
  - Expert fwd:  SiLU(w1(x)) * w3(x) -> w2

GPU memory contract
-------------------
Snapshot  : saves packed int8 to CPU   — zero GPU overhead
Merge     : dequantize one layer at a time → average → re-quantize to 4-bit
            net GPU change ≈ 0 (freed j cancels converted i)
Restore   : mutate packed data in-place → re-insert j module reference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# LOW-LEVEL WEIGHT UTILITIES
# ──────────────────────────────────────────────────────────────

def _is_4bit(param) -> bool:
    return hasattr(param, "quant_state")


def _dequantize_to_float(param, device) -> torch.Tensor:
    """Dequantize a Linear weight (Params4bit or plain) to float32 on device."""
    w = param.weight
    if _is_4bit(w):
        import bitsandbytes as bnb
        return bnb.functional.dequantize_4bit(
            w.data, w.quant_state
        ).to(torch.float32).contiguous()
    return w.data.to(torch.float32)


def _requantize_inplace(linear: nn.Module, merged_fp: torch.Tensor) -> None:
    """
    Re-quantize merged_fp (float32 GPU tensor) back to 4-bit and mutate
    linear.weight in-place so no new Parameter objects are created.
    Falls back to bfloat16 plain parameter if quantization fails.
    """
    w = linear.weight
    if _is_4bit(w):
        try:
            import bitsandbytes as bnb
            qs = w.quant_state
            blocksize = getattr(qs, "blocksize", 64)
            quant_type = getattr(qs, "quant_type", "fp4")
            new_packed, new_qs = bnb.functional.quantize_4bit(
                merged_fp.to(torch.float16),
                blocksize=blocksize,
                quant_type=quant_type,
            )
            w.data = new_packed
            w.quant_state = new_qs
            return
        except Exception:
            pass  # fall through to plain-float fallback
    # Plain float or fallback
    linear.weight = nn.Parameter(merged_fp.to(torch.bfloat16))


def _save_expert_packed_cpu(expert: nn.Module) -> dict:
    """Save expert weights as packed int8 on CPU. Zero GPU overhead."""
    state = {}
    for attr in ("w1", "w2", "w3"):
        w = getattr(expert, attr).weight
        if _is_4bit(w):
            state[attr] = {
                "packed_cpu": w.data.cpu(),   # just a CPU copy of the packed int8
                "quant_state": w.quant_state, # small GPU tensors - kept alive by reference
                "is_4bit": True,
                "out_f": getattr(expert, attr).out_features,
                "in_f": getattr(expert, attr).in_features,
            }
        else:
            state[attr] = {
                "data_cpu": w.data.cpu(),
                "is_4bit": False,
                "out_f": getattr(expert, attr).out_features,
                "in_f": getattr(expert, attr).in_features,
            }
    return state


def _restore_expert_from_saved(expert: nn.Module, state: dict, device) -> None:
    """Restore expert weights from a CPU-saved state dict."""
    for attr in ("w1", "w2", "w3"):
        s = state[attr]
        linear = getattr(expert, attr)
        if s["is_4bit"]:
            w = linear.weight
            # Mutate packed data back in-place
            w.data = s["packed_cpu"].to(device)
            w.quant_state = s["quant_state"]
        else:
            linear.weight = nn.Parameter(s["data_cpu"].to(device))
        linear.out_features = s["out_f"]
        linear.in_features = s["in_f"]


# ──────────────────────────────────────────────────────────────
# EXPERT EVALUATION  (forward probe — no weight dequantization)
# ──────────────────────────────────────────────────────────────

def evaluate_expert_on_probe(x: torch.Tensor, expert: nn.Module) -> torch.Tensor:
    """
    Expert forward: SiLU(w1(x)) * w3(x) -> w2
    Uses native bnb forward — works for both Params4bit and plain nn.Linear.
    """
    return expert.w2(F.silu(expert.w1(x)) * expert.w3(x))


# ──────────────────────────────────────────────────────────────
# PHYSICAL MERGE ENGINE
# ──────────────────────────────────────────────────────────────

class PhiPhysicalMergeEngine:
    """
    True N -> N-1 physical expert deletion for Phi-3.5-MoE.

    snapshot(i, j):  Saves only experts i and j (packed int8 to CPU — zero GPU overhead).
    merge_experts:   Sequential per-layer dequant → average → re-quantize.  Net GPU ≈ 0.
    restore():       Mutates weights back in-place, re-inserts j module reference.
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

    # ── Snapshot ──────────────────────────────────────────────

    def snapshot(self, i: int, j: int):
        """
        Minimal snapshot of experts i and j.
        GPU overhead: zero  (packed int8 copied to CPU, no dequantization).
        """
        torch.cuda.empty_cache()   # reclaim reserved-but-free GPU memory

        N = self.current_num_experts
        assert 0 <= i < N and 0 <= j < N and i != j

        self._snapshot = {
            "i": i, "j": j,
            "config_num_local": getattr(self.model.config, "num_local_experts", None),
            "config_num":       getattr(self.model.config, "num_experts", None),
            "blocks": [],
        }

        for block in self.moe_blocks:
            # Gate: [N, hidden] — dequantize to CPU float (256 KB for N=16, hidden=4096)
            device = block.gate.weight.device if not _is_4bit(block.gate.weight) \
                     else block.gate.weight.data.device
            if _is_4bit(block.gate.weight):
                import bitsandbytes as bnb
                gate_float = bnb.functional.dequantize_4bit(
                    block.gate.weight.data, block.gate.weight.quant_state
                ).to(torch.float32).cpu()
            else:
                gate_float = block.gate.weight.data.cpu().float()

            self._snapshot["blocks"].append({
                "block":          block,
                "exp_i_module":   block.experts[i],
                "exp_i_state":    _save_expert_packed_cpu(block.experts[i]),
                "exp_j_module":   block.experts[j],   # reference only; weights untouched
                "gate_float_cpu": gate_float,
                "gate_in_f":      block.gate.in_features,
                "num_experts":    block.num_experts,
            })

    # ── Restore ───────────────────────────────────────────────

    def restore(self):
        """Rollback to snapshot. Mutates weights in-place; re-inserts j module."""
        if self._snapshot is None:
            raise RuntimeError("No snapshot to restore from")

        i = self._snapshot["i"]
        j = self._snapshot["j"]

        if self._snapshot["config_num_local"] is not None:
            self.model.config.num_local_experts = self._snapshot["config_num_local"]
        if self._snapshot["config_num"] is not None:
            self.model.config.num_experts       = self._snapshot["config_num"]

        for b_data in self._snapshot["blocks"]:
            block  = b_data["block"]
            n      = b_data["num_experts"]
            device = next(iter(block.parameters())).device

            # 1. Restore gate
            gate_w = b_data["gate_float_cpu"].to(device)
            block.gate.weight      = nn.Parameter(gate_w)
            block.gate.out_features = n
            block.gate.in_features  = b_data["gate_in_f"]

            # 2. Restore expert i weights in-place
            _restore_expert_from_saved(b_data["exp_i_module"], b_data["exp_i_state"], device)

            # 3. Re-insert expert j at original position j
            #    (module is still alive via exp_j_module reference)
            experts_list = list(block.experts)
            experts_list.insert(j, b_data["exp_j_module"])
            block.experts     = nn.ModuleList(experts_list)
            block.num_experts = n

        self.current_num_experts = self._snapshot["blocks"][0]["num_experts"]
        self._snapshot = None
        torch.cuda.empty_cache()
        self._validate_invariants()

    # ── Merge ─────────────────────────────────────────────────

    @torch.no_grad()
    def merge_experts(self, i: int, j: int):
        """
        Physical N -> N-1 merge.
        Processes one block at a time to bound peak GPU usage.
        Each block: dequant i+j per matrix (sequential) → average → re-quantize.
        Net GPU memory change ≈ 0 (freed j ≈ re-quantized i).
        """
        N = self.current_num_experts
        assert 0 <= i < N and 0 <= j < N and i != j, \
            f"Invalid expert indices {i},{j} for N={N}"

        keep = [idx for idx in range(N) if idx != j]
        new_i = keep.index(i)

        for block in self.moe_blocks:
            device = next(iter(block.parameters())).device

            # --- Gate ---
            if _is_4bit(block.gate.weight):
                import bitsandbytes as bnb
                gate_w = bnb.functional.dequantize_4bit(
                    block.gate.weight.data, block.gate.weight.quant_state
                ).to(torch.float32)
            else:
                gate_w = block.gate.weight.data.float()

            merged_gate_row = (gate_w[i] + gate_w[j]) / 2.0
            new_gate = gate_w[keep].clone()
            new_gate[new_i] = merged_gate_row
            block.gate.weight      = nn.Parameter(new_gate.to(device))
            block.gate.out_features = len(keep)
            block.gate.in_features  = new_gate.shape[1]
            del gate_w, new_gate

            # --- Expert weights (sequential per attribute to bound GPU peak) ---
            for attr in ("w1", "w2", "w3"):
                fi = _dequantize_to_float(getattr(block.experts[i], attr), device)
                fj = _dequantize_to_float(getattr(block.experts[j], attr), device)
                merged = (fi + fj) / 2.0
                del fi, fj
                _requantize_inplace(getattr(block.experts[i], attr), merged)
                del merged

            # --- Delete expert j ---
            experts_list = list(block.experts)
            experts_list.pop(j)
            block.experts     = nn.ModuleList(experts_list)
            block.num_experts = len(experts_list)

            torch.cuda.empty_cache()  # reclaim freed j weights immediately

        self.current_num_experts            = len(keep)
        self.model.config.num_local_experts = len(keep)
        if hasattr(self.model.config, "num_experts"):
            self.model.config.num_experts   = len(keep)

        self._validate_invariants()

    # ── Invariants ────────────────────────────────────────────

    def _validate_invariants(self):
        N = self.current_num_experts
        for block in self.moe_blocks:
            assert len(block.experts) == N, \
                f"Expert list has {len(block.experts)}, expected {N}"
            assert block.gate.out_features == N, \
                f"gate.out_features={block.gate.out_features}, expected {N}"
            assert block.num_experts == N, \
                f"block.num_experts={block.num_experts}, expected {N}"
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
    Probe: hidden_states[l] (residual stream before MoE block l).
    Expert function: SiLU(w1(x)) * w3(x) -> w2 (no routing weight).
    All intermediate tensors moved to CPU to avoid accumulating GPU allocations.
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
                output_hidden_states=True,
            )
            hidden_states = out.hidden_states   # tuple: [seq, hidden] per layer
            seq_len = batch.shape[0]

            for layer_idx, block in enumerate(engine.moe_blocks):
                x = hidden_states[layer_idx].squeeze(0)   # [seq, hidden] on device

                for exp_idx, expert in enumerate(block.experts):
                    exp_out = evaluate_expert_on_probe(x, expert).cpu()  # [seq, hidden]
                    for tok in range(seq_len):
                        axis = tok % num_axes
                        cap[layer_idx, exp_idx, axis] += \
                            exp_out[tok].float().norm(p=2).item()
                        counts[layer_idx, exp_idx, axis] += 1

            # Free hidden states and model output immediately
            del out, hidden_states
            torch.cuda.empty_cache()

    valid      = counts > 0
    cap[valid] /= counts[valid]

    C = cap.permute(1, 0, 2).reshape(N, num_layers * num_axes)
    assert C.shape[0] == N, f"Capability matrix has {C.shape[0]} rows, expected {N}"
    return C


# ──────────────────────────────────────────────────────────────
# KL DAMAGE UTILITY
# ──────────────────────────────────────────────────────────────

def compute_kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p_probs     = F.softmax(p_logits, dim=-1)
    q_log_probs = F.log_softmax(q_logits, dim=-1)
    return F.kl_div(q_log_probs, p_probs, reduction="batchmean").item()
