# Master MoE Systems Benchmark: Router Inspection & Top-K Modulation Analysis

**Date:** October 15, 2026  
**Target Architecture:** `Qwen/Qwen3-30B-A3B`  
**Location:** [`care_large_moe_systems/master_benchmark/router_inspection_and_topk.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/router_inspection_and_topk.md)

---

## 1. Architectural Router Audit

In the installed Hugging Face Transformers version (5.19.0), the MoE gating and dispatch mechanism for `Qwen3-30B-A3B` consists of:
- **Router Module:** `Qwen3MoeTopKRouter`
- **Sparse MoE Block:** `Qwen3MoeSparseMoeBlock`
- **Expert Container:** `Qwen3MoeExperts`

### Gating Implementation:
```python
class Qwen3MoeTopKRouter(nn.Module):

  def forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(
        hidden_states, self.weight
    )  # (seq_len, num_experts)
    router_probs = torch.nn.functional.softmax(
        router_logits, dtype=torch.float, dim=-1
    )
    router_top_value, router_indices = torch.topk(
        router_probs, self.top_k, dim=-1
    )

    if self.norm_topk_prob:
      router_top_value /= router_top_value.sum(dim=-1, keepdim=True)

    router_top_value = router_top_value.to(router_logits.dtype)
    return router_logits, router_top_value, router_indices
```

### Expert Execution Implementation:
```python
class Qwen3MoeExperts(nn.Module):

  def forward(self, hidden_states, top_k_index, top_k_weights):
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
      expert_mask = torch.nn.functional.one_hot(
          top_k_index, num_classes=self.num_experts + 1
      )
      expert_mask = expert_mask.permute(2, 1, 0)
      expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
      expert_idx = expert_idx[0]
      if expert_idx == self.num_experts:
        continue
      top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
      current_state = hidden_states[token_idx]
      gate, up = nn.functional.linear(
          current_state, self.gate_up_proj[expert_idx]
      ).chunk(2, dim=-1)
      current_hidden_states = self.act_fn(gate) * up
      current_hidden_states = nn.functional.linear(
          current_hidden_states, self.down_proj[expert_idx]
      )
      current_hidden_states = (
          current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
      )
      final_hidden_states.index_add_(
          0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
      )

    return final_hidden_states
```

---

## 2. Key Scientific Findings on Top-K Modulation

1. **Top-K Is a Modifiable Hyperparameter Without Checkpoint Rebuilding:**
   - `self.top_k` is stored as an instance attribute on `Qwen3MoeTopKRouter`.
   - Modifying `block.gate.top_k = k` across all 48 decoder layers safely modulates inference top-$k$ without modifying any tensor weights.
2. **Strict Invariant Validity ($1 \le k \le E$):**
   - The tensor operations `torch.topk(router_probs, self.top_k, dim=-1)` and `torch.nn.functional.one_hot(top_k_index, ...)` accept any integer $k \in [1, E]$.
   - For our target evaluations ($k \in \{8, 6, 4, 2\}$), the condition $k \le E$ is strictly satisfied for all available expert counts ($E \in \{128, 112, 96, 90, 80, 64\}$).
3. **Automatic Probability Renormalization:**
   - Because `self.norm_topk_prob = True`, the chosen top-$k$ weights are automatically normalized such that $\sum_{m=1}^k w_m = 1.0$.
   - Reducing $k$ from 8 to 4 or 2 preserves output magnitude scaling and prevents artificial activation suppression.
4. **Disambiguation of Candidate Budget $K$ vs. Inference Top-$k$:**
   - **Candidate Budget ($K = 5$):** Compression-time hyperparameter in CARE-COM determining how many candidate merge pairs are evaluated via Oracle KL per sequential step.
   - **Inference Top-$k$ ($k \in \{8, 6, 4, 2\}$):** Serving-time hyperparameter determining how many experts are dynamically routed to per token.
   - These are strictly separated in our benchmark schema.

---

## 3. The Three Complementary Experiments Protocol

```
                     ┌────────────────────────────────────────────────────────┐
                     │          PRIMARY MODEL: Qwen3-30B-A3B (128E)          │
                     └───────────────────────────┬────────────────────────────┘
                                                 │
            ┌────────────────────────────────────┼────────────────────────────────────┐
            ▼                                    ▼                                    ▼
┌───────────────────────────────┐  ┌───────────────────────────────┐  ┌───────────────────────────────┐
│         EXPERIMENT 1          │  │         EXPERIMENT 2          │  │         EXPERIMENT 3          │
│    Compression Comparison     │  │      Routing Comparison       │  │    Combined Configurations    │
│   (Expert Population Sweep)   │  │   (Fixed Architecture 128E)   │  │  (Joint E and k Optimization) │
├───────────────────────────────┤  ├───────────────────────────────┤  ├───────────────────────────────┤
│ • E ∈ {128, 112, 96, 80, 64}  │  │ • E = 128 (Uncompressed)      │  │ • E = 80 (GPU-Resident)       │
│ • k = 8 (Constant native topk)│  │ • k ∈ {8, 6, 4, 2}            │  │   k ∈ {8, 6, 4, 2}            │
│ • Compares CARE, REAP, SubMoE │  │ • Tests pure router sparsity  │  │ • E = 64 (High-Speed)         │
│   and Random at matched top-k │  │   without weight reduction    │  │   k ∈ {8, 6, 4, 2}            │
└───────────────────────────────┘  └───────────────────────────────┘  └───────────────────────────────┘
```

1. **Experiment 1 (Pure Compression at Matched $k=8$):**
   - Holds active parameters per token constant (~4.86B params).
   - Measures how shrinking the available expert pool from 128 down to 64 affects quality (WikiText-2 PPL, GSM8K, MATH-500) and storage/VRAM.
2. **Experiment 2 (Pure Routing Sparsity at Fixed $E=128$):**
   - Keeps total parameters constant (32.05B params).
   - Reduces active parameters per token from 4.86B ($k=8$) down to 3.50B ($k=2$).
   - Isolates the pure algorithmic damage of discarding lower-probability expert activations.
3. **Experiment 3 (Joint Optimization / Combined Frontiers):**
   - Evaluates whether a compressed model (e.g., CARE-80) operating at $k=4$ provides a superior quality-throughput Pareto trade-off compared to the uncompressed baseline operating at $k=4$.
