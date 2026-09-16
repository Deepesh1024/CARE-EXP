# Experiment 7C — Validity Audit

**Date:** 2026-09-16  
**Auditor:** Automated pipeline inspection + CSV verification  
**Status:** COMPLETE

---

## Primary Result Being Audited

| Statistic | Value |
|---|---|
| Spearman ρ (C_mutual vs residual) | -0.0671 |
| p-value | 7.91 × 10⁻¹ |
| 95% Bootstrap CI | [-0.56, +0.47] |
| N | 18 candidate pairs |

---

## Audit Summary Table

| Audit | Topic | Status | Key Finding |
|---|---|---|---|
| 1 | Activation Tensor | ✅ PASS | Shape (29, 1024, 262144), float32, Layer 8. All 29 experts present. |
| 2 | Common Probe Alignment | ⚠️ CONCERN | Deterministic single-pass confirmed. **Concern: ~87.5% of token positions per expert are zero (unrouted), included in similarity computation.** |
| 3 | Padding Handling | ⚠️ CONCERN | `attention_mask` not passed in phase6 forward call. Padded tokens may be routed and recorded. Minor severity. |
| 4 | Post-Activation Semantics | ✅ PASS | `act_fn(gate_proj(x)) * up_proj(x)` exactly matches `OlmoeMLP.forward()`. Before down_proj. Not residual stream. |
| 5 | Signature Statistics | ✅ PASS | Magnitudes realistic (−11 to +10.5). Expert 45 high-frequency (40.5% nonzero). No overflow. |
| 6 | Pair 10 Outlier | ✅ PASS | pair_10 (Experts 7,32) at 3.8σ. Both directions agree. Consistent with genuine high functional overlap. Not a pipeline artifact. |
| 7 | Cosine Similarity Distribution | ⚠️ CONCERN | Full 1024×1024 matrices not available locally. Low absolute C_mutual values (0.001–0.035) expected given sparse routing. |
| 8 | Primary Result Reproduction | ✅ PASS | ρ = −0.0671, p = 7.9143 × 10⁻¹ exactly reproduced from CSV data. |
| 9 | Normalization | ✅ PASS | L2 normalization over 262144-dim token vector per neuron. Exactly implements cosine similarity. Epsilon prevents zero-division for inactive neurons. |
| 10 | Data Join | ✅ PASS | All 18 pairs joined by `pair_id`. `residual = D_actual_KL − D_pred` verified arithmetically for all rows. |
| 11 | 7B/7C Protocol Alignment | ⚠️ CONCERN | Same Wikitext dataset, split, seed, token count. Minor: `attention_mask` omitted in extraction forward call. |

---

## Detailed Findings

### Audit 1 — Activation Tensor ✅ PASS
- Shape: `(29, 1024, 262144)` — 29 unique experts, 1024 intermediate neurons, 262144 tokens
- Dtype: `float32` (overflow-safe; float16 was rejected during development due to OLMoE outlier activations)
- Layer: 8 (TARGET_LAYER_IDX = 8, as specified)
- All 29 unique expert IDs from the 18 candidate pairs are present: `{1, 4, 5, 6, 7, 9, 10, 11, 13, 14, 19, 21, 22, 24, 27, 31, 32, 35, 36, 39, 43, 45, 47, 49, 53, 56, 58, 59, 62}`
- NaN/Inf not verifiable locally (29 GB file on VM), but VM diagnostics reported bounded finite magnitudes for all 29 experts

### Audit 2 — Common Probe Alignment ⚠️ CONCERN
**Confirmed correct:**
- Single model loaded once
- Single `prepare_wikitext_eval_batches` call with `max_tokens=262144`
- `RANDOM_SEED = 42` with `torch.manual_seed` and `np.random.seed`
- `random.Random(42).shuffle(texts)` in evaluation utility (deterministic)
- All 29 experts written within the **same sequence of forward passes** via `moe_block_hook`
- `global_token_offset` is a single shared counter → expert signatures at position `t` correspond to the same token

**Unresolved concern:**  
With top-k=8 out of 64 experts, each expert receives approximately 12.5% of tokens. The remaining ~87.5% of token positions remain at 0.0 (initialization). Cosine similarity is computed over all 262,144 positions, including these zero positions.

Mathematical consequence: If neuron `k` in expert `i` has value 0 at position `t` (because expert `i` was not routed to token `t`), and neuron `l` in expert `j` also has value 0 at position `t`, neither contributes to the inner product numerator. The L2 norm accounts for this correctly — zero entries neither inflate nor deflate the cosine similarity. **The mathematics are correct.**

Semantic concern: The cosine similarity includes "non-response" positions where neither expert was active. Whether shared non-response is informative about functional geometry is philosophically unresolved. It is not a computational defect.

### Audit 3 — Padding Token Handling ⚠️ CONCERN (minor)
Phase 6 calls `model(input_ids=input_ids)` without passing `attention_mask`. OLMoE uses causal attention (no cross-attention), so the impact of missing `attention_mask` on hidden states is limited. However, padded token positions within sequences may be routed to experts, and if so, their post-activation values are written to the memmap. This is a minor contamination affecting a subset of the ~12.5% routed positions.

### Audit 4 — Post-Activation Semantics ✅ PASS
```python
# Our extraction hook:
gate = expert.act_fn(expert.gate_proj(routed_h))   # [n, ffn_dim=1024]
up   = expert.up_proj(routed_h)                    # [n, ffn_dim=1024]
post_act = gate * up                               # [n, ffn_dim=1024]

# OlmoeMLP.forward (HuggingFace source):
def forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```
`post_act` is exactly the intermediate tensor passed to `down_proj`. Captured before down-projection. Not the residual stream. Shape confirmed as `[n, 1024]` by debug script.

### Audit 5 — Signature Statistics ✅ PASS
VM-reported diagnostics for visible experts:
- Expert 45: 108,713,201 nonzero (~40.5% of 268M slots), min=−9.5, max=+7.44
- Expert 62: 64,120,330 nonzero (~23.9%), min=−4.94, max=+4.38
- Expert 35: 7,972,864 nonzero (~3.0%), min=−6.0, max=+4.09

Magnitude range of −11 to +10.5 is consistent with post-SiLU×up_proj in a 1B-parameter model. No float32 overflow possible. Non-zero fractions vary substantially by routing frequency (expected, as OLMoE routing is non-uniform).

### Audit 6 — Pair 10 Outlier ✅ PASS
- pair_10 (Expert 7 ↔ Expert 32): C_mutual = 0.1175 (3.8σ above mean)
- Second highest: pair_17 at 0.0359 — pair_10 is 3.3× higher
- Both directions strongly agree: C_i→j = 0.122, C_j→i = 0.113
- 7B features corroborate genuine overlap: Jaccard_Overlap = 0.106 (highest in dataset), Activation_Similarity = 0.002
- Residual = −0.0017 (near zero): CARE-COM correctly predicted this pair, no anomalous damage
- **Conclusion: This is a legitimate functional similarity phenomenon, not a pipeline artifact.** Experts 7 and 32 are genuinely functionally redundant, which is a known phenomenon in sparse MoE models.

### Audit 7 — Similarity Distribution ⚠️ CONCERN
C_mutual summary statistics:
| Statistic | Value |
|---|---|
| mean | 0.019 |
| median | 0.014 |
| std | 0.027 |
| min | 0.0007 |
| max | 0.1175 |
| p25 | 0.006 |
| p75 | 0.022 |

Full 1024×1024 matrices not available locally. The very low absolute values (most pairs ~0.001–0.035) are expected given sparse routing — most neuron pairs have dissimilar functional responses across the token space. The metric is internally consistent and directionally concordant.

### Audit 8 — Primary Result Reproduction ✅ PASS
Spearman ρ = −0.0671, p = 7.9143 × 10⁻¹ reproduced exactly from the saved CSV data. Bootstrap CI [−0.56, +0.47] spans zero with large width. Consistent with VM output.

### Audit 9 — Normalization ✅ PASS
L2 normalization along `axis=1` (over 262144 token positions) per neuron. Epsilon = 1e-10 prevents zero-division for never-activated neurons (which correctly contribute 0 to similarity). Pre-specified as Protocol B. Implemented correctly.

### Audit 10 — Data Join ✅ PASS
All 18 pairs joined by `pair_id` (not row index). `residual = D_actual_KL − D_pred` verified to floating-point precision for all 18 rows. Expert IDs consistent across all three tables.

### Audit 11 — 7B/7C Protocol Alignment ⚠️ CONCERN (minor)
| Attribute | 7B D_actual | 7C Activation Probe |
|---|---|---|
| Dataset | Salesforce/wikitext-2-raw-v1 | Same |
| Split | train | Same |
| Token count | 262144 | Same |
| Shuffle seed | random.Random(42) | Same |
| attention_mask | Passed to model, used for KL masking | NOT passed to model |
| Padding handling | Padding excluded from KL metric | Padding may be routed and recorded |

The attention_mask omission in phase6 is a minor protocol divergence. It does not affect the overall Wikitext distribution from which tokens are drawn.

---

## Final Classification

### ⚠️ C — AMBIGUOUS

The pipeline is technically correct in all verifiable respects. The primary statistic is reproducible, the activation semantics are verified, and the data join is arithmetically sound.

Two unresolved concerns prevent a strong **VALID NULL** classification:

**Concern A (Sparse routing zeros):** ~87.5% of token positions per expert are zero (unrouted). Including these in cosine similarity computation is mathematically correct but semantically ambiguous — it compares "non-response" positions alongside genuine functional responses. This does not invalidate the result but limits the interpretability of C_mutual as a measure of *functional* similarity.

**Concern B (Padding contamination, minor):** The absence of `attention_mask` in phase6 may allow padded token activations to enter the memmap. Impact is limited to the routed fraction (~12.5%) of padding-containing positions.

### What This Means

These concerns would tend to **add noise** to C_mutual (reducing its discriminative power), meaning the true correlation could be anywhere in the range ~[−0.07, some modest positive value]. The null result does **not** rule out that a more carefully constructed mutual coverage metric — one restricted to tokens where at least one expert is activated, or restricted to mutually activated tokens — might find a weak signal.

### Pre-Specified Verdict

> *"The pre-specified 7C neuron-cloud mutual-coverage hypothesis is not supported by the N=18 experiment."*

Per the pre-specified failure criterion, the null result is accepted. No post-hoc statistic search was performed.

The classification is **C (AMBIGUOUS)** rather than **A (VALID NULL)** because two methodological concerns about the metric definition (not the statistics) remain unresolved. However, these concerns do not reverse the conclusion — they merely reduce confidence in a **strong rejection** of the hypothesis.

---

## Minimum Necessary Repair (for Future Reference Only)

If 7C is revisited in a future experiment, the minimum repair would be:

1. **Sparse routing**: Compute cosine similarity only over tokens where **both** experts were actually activated (mutual activation support). This is a stricter and more semantically valid comparison.

2. **Padding**: Pass `attention_mask` to the extraction forward call and filter padded positions before writing to memmap.

These repairs would constitute a new pre-registered experiment (7D), not a post-hoc rescue of 7C.
