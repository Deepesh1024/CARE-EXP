# CARE-COM v2.1 Final Implementation & Scientific Report

## 1. Files Created/Modified
- `experiments/care_com_v21/config.py`: Configuration definitions.
- `experiments/care_com_v21/capability.py`: Capability vector generation.
- `experiments/care_com_v21/evaluator.py`: Marginal KL divergence calculation.
- `experiments/care_com_v21/core.py`: Engine handling global physical merging, snapshots, and adaptive candidate generation.
- `experiments/care_com_v21/driver.py`: Pilot execution script.
- `experiments/care_com_v21/tests/test_v21.py`: Validation tests for 11 core invariants.
- `results/care_com_v21/trajectories/v21_trace.json`: Execution log containing trace and marginal damage.

## 2. Architecture of v2.1
The architecture replaces the exploratory probability transfer of v2.0 with a **global physical mathematical merge**. The central pipeline is:
$$ C_t \rightarrow \text{nearest-neighbor candidate pool} \rightarrow \text{physical temporary merge} \rightarrow \arg\min D(i,j|M_t) \rightarrow \text{permanent merge} \rightarrow C_{t+1} $$

## 3. Exact Physical Merge Operation
Implemented directly matching v1 semantics:
- Applied universally across **all 16 layers**.
- `W_new = (W_i + W_j) / 2` for `gate_proj`, `up_proj`, and `down_proj`.
- `router_new = (router_i + router_j) / 2` for the gating weights.
- Expert `j` is physically removed from the `nn.ModuleList`, decreasing the global expert count exactly from `N` to `N-1`. 

## 4. How Capability is Recomputed
At every state $M_t$, the model processes the CARE probe tokens. Post-activation functional norms are computed across all layers, layer-normalized, and then concatenated to form a global capability vector. This captures the true functional state of the *current* parameter distribution, preventing compounding divergence from the original baseline.

## 5. How Candidates are Generated
Using the current capability geometry $C_t$, pairwise L2 distances are calculated globally across all active experts. The top `K` (default 5) closest capability pairs are selected as candidates. Capability distance is strictly used as an efficient pruning heuristic for candidate *generation*, not direct selection.

## 6. How Candidate Damage is Evaluated
We compute the marginal functional damage against the *current* model: $D(i, j | M_t) = KL(L_t || L_{ij})$. 
A precise transactional backup is taken using `snapshot(expert_i)`, which backups only the targeted parameter weights to avoid OOM. The candidate is physically merged, evaluated on the calibration dataset, and cleanly restored.

## 7. How the Selected Pair is Chosen
The candidate `(i, j)` that produces the lowest marginal KL damage across the Wikitext evaluation subset is selected for permanent integration.

## 8. Tests Passed
The local test suite (`tests/test_v21.py`) passed all 11 scientific invariants:
1. `N` physically decreases by exactly 1.
2. Weight averaging evaluates correctly.
3. Router dimensions cleanly adapt.
4. Native forward pass operates without code branching.
5. Expert indices accurately reindex.
6. No duplicate pairs generated.
7. No self-pairs generated.
8. Capability physically changes after a permanent merge.
9. Resulting model produces different capabilities.
10. Deterministic pairing under fixed seed.
11. Transactional restoration mathematically reverts weights, tensors, and config attributes.

## 9. Pilot Results (64 -> 60 -> 56)
The trajectory was executed successfully on the VM. 
At each step, physical integration succeeded, resulting in the following trajectory:
- **64 -> 63**: Selected [13, 15] (Damage: 0.0464)
- **63 -> 62**: Selected [12, 13] (Damage: 0.0283)
- **62 -> 61**: Selected [10, 12] (Damage: 0.0321)
- **61 -> 60**: Selected [7, 10] (Damage: 0.0427)
- **60 -> 59**: Selected [5, 12] (Damage: 0.0601)
- **59 -> 58**: Selected [1, 4] (Damage: 0.0693)
- **58 -> 57**: Selected [4, 15] (Damage: 0.0444)
- **57 -> 56**: Selected [4, 6] (Damage: 0.0069)

### The Scientific Question: Did adaptive recomputation actually change the decisions?
**Yes, profoundly.** 
Under a v1 static frozen ranking evaluated at $M_{64}$, the pair `[8, 13]` was the most redundant in the model (Capability Distance: 0.01576). 
However, after merging `[13, 15]` in step 1, the entire functional geometry of the model shifted. At step 2, when recomputing $C_{63}$, the capability distance for `[8, 13]` shifted to `0.01600`. Furthermore, its measured functional damage was evaluated at `0.0644`. 
The algorithm instead discovered a newly surfaced optimal pair `[12, 13]`, which yielded a significantly lower functional damage of `0.0283`. A frozen ranking would have forcefully executed `[8, 13]` based on stale $M_{64}$ parameters, incurring more than double the functional degradation. Adaptive recomputation effectively shields the model from compounding structural damage.

## 10. Deviations from Specification
- **OOM Prevention**: Initially, a full model clone was planned for the transactional backup. This induced a CUDA OutOfMemoryError. The specification was adapted to snapshot *only* the specific target expert `i` undergoing the in-place mathematical modification, dropping VRAM overhead from ~2 GiB down to ~35 MiB.

## 11. Performance Bottlenecks
- **Marginal Evaluation (Forward Passes):** The dominant bottleneck is the functional evaluation phase (`evaluation_time`: ~196s per step), as it strictly requires performing physical forward passes over the calibration data for every candidate.
- **Capability Recomputation:** Recomputing the capability vectors across all layers at every step incurs an additional ~40s overhead per step. 
- *Mitigation:* Both bottlenecks are parallelizable. Future iterations could evaluate candidate damage on separate GPUs if batched.
