# CARE-COM Master MoE Systems & Inference Benchmark: Final Scientific Report

**Model:** `Qwen/Qwen3-30B-A3B` (48 layers, 128 initial experts per layer, top-8 active)  
**Hardware Platform:** NVIDIA L40S GPU (48 GB GDDR6 VRAM, PCIe Gen4), 188 GB Host CPU RAM  
**Benchmark Suite:** Canonical WikiText-2 (Full Evaluation: 32 chunks × 1024 tokens = 32,768 tokens; Pilot Evaluation: 4 chunks × 512 tokens = 2,048 tokens)  
**Canonical Datasets:**
- Master Results Table: [`master_benchmark_results.csv`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/master_benchmark_results.csv)
- Method Inventory: [`method_checkpoint_inventory.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/method_checkpoint_inventory.md)
- Compression Costs: [`compression_costs.csv`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/compression_costs.csv)
- Router & Top-K Semantics: [`router_inspection_and_topk.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/router_inspection_and_topk.md)
- Machine-Readable Records: [`full_wikitext_master_results.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/full_wikitext_master_results.json) & [`pilot_topk_results.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/pilot_topk_results.json)

---

## 1. Executive Summary & Verification Outcomes

### 1.1 Validation of Reconstruction vs. Recorded CARE Implementation
A critical concern addressed in this benchmark is whether reconstructing compressed checkpoints via tensor averaging faithfully reproduces the recorded CARE merge implementation.
1. **Recorded Implementation Audit:** Inspection of the authoritative merge engine ([`benchmark/Qwen3-30B-A3B/run_benchmark.py`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark/Qwen3-30B-A3B/run_benchmark.py#L134-L162)) and the merge operator specification ([`results/exp5/merge_operator_spec.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/results/exp5/merge_operator_spec.md#L11-L33)) confirms that CARE physically merges candidate expert pairs $(i, j)$ into a single surviving expert via arithmetic mean:
   $$\mathbf{W}_{\text{merged}} = \frac{1}{2}(\mathbf{W}_i + \mathbf{W}_j)$$
   applied uniformly across `gate_up_proj`, `down_proj`, and the router gate weights `gate.weight`.
2. **Deterministic Bit-Level Fidelity:** Replaying the 48 recorded merge steps from [`audit/QWEN3_CARE_COM_RESULTS.json`](file:///home/ubuntu/DEEPESH/audit/QWEN3_CARE_COM_RESULTS.json) on pristine Hugging Face weights yielded:
   - **Absolute PPL Delta:** **`0.000000`** ($19.929230$ reconstructed vs. $19.929230$ Exp 13 reference).
   - **Logit Equivalence:** Max absolute difference = `0.000000` across all test sequences.
   - **Tensor Fingerprints:** Exact SHA-256 hash match on layers 0, 24, and 47.
   The reconstruction pipeline is mathematically and empirically validated.

### 1.2 The Systems Regime Cliff (Regime A vs. Regime B)
The benchmark establishes that the primary operational bottleneck for massive MoE architectures on single workstation GPUs is the **VRAM threshold boundary**:
- **Regime B (CPU-Offloaded Streaming, $E > 80$):** At 128 experts, Qwen3-30B requires 59.70 GB in BF16, exceeding the 48 GB VRAM capacity of the L40S. Execution requires dynamic PCIe layer streaming, constraining GPU VRAM allocation to 5.89 GB and capping autoregressive decode throughput at **0.68 tok/s** (inter-token latency 1470 ms).
- **Regime A (Pure GPU-Resident, $E \le 80$):** At 80 experts, the model footprint shrinks to 39.44 GB, fitting fully within the 48 GB VRAM (36.62 GB allocated). This eliminates the PCIe streaming bottleneck, boosting sustained decode throughput to **19.29–19.68 tok/s** (**28.4× speedup**) and prefill throughput from 851 tok/s to **12,689–19,556 tok/s** (**14.9× to 23.0× speedup**).
- **Scientific Conclusion:** MoE expert reduction produces a **step-function systems speedup** at hardware boundary crossings, rather than gradual compute reductions.

---

## 2. Comparative Benchmark Results

### 2.1 Experiment 1: Expert Count Scaling ($E \in \{128, 112, 96, 80, 64, 60\}$, $k=8$)
Full canonical WikiText-2 perplexity (32 chunks × 1024 tokens = 32,768 tokens) across matched expert counts:

| Method | Experts ($E$) | Active ($k$) | Regime | Model Size | WikiText-2 PPL | $\Delta$ PPL | Rel $\Delta$ (%) | Decode (tok/s) | Peak VRAM | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dense Baseline** | **128** | 8 | Regime B | 59.70 GB | **9.5715** | 0.0000 | 0.00% | 0.68 | 5.89 GB | Verified HF |
| **CARE-Adaptive** | **112** | 8 | Regime B | 52.95 GB | **11.1902** | +1.6187 | +16.91% | 0.76 | 5.89 GB | Verified |
| **REAP** | **112** | 8 | Regime B | 52.95 GB | **11.2011** | +1.6296 | +17.03% | 0.76 | 5.89 GB | Verified |
| **Sub-MoE** | **112** | 8 | Regime B | 52.95 GB | **11.8771** | +2.3056 | +24.09% | 0.76 | 5.89 GB | Verified |
| **Random Merging** | **112** | 8 | Regime B | 52.95 GB | **12.7422** | +3.1707 | +33.13% | 0.76 | 5.89 GB | Verified |
| **REAP** | **96** | 8 | Regime B | 46.19 GB | **13.1724** | +3.6009 | +37.62% | 0.85 | 5.89 GB | Verified |
| **Sub-MoE** | **96** | 8 | Regime B | 46.19 GB | **14.3825** | +4.8110 | +50.26% | 0.85 | 5.89 GB | Verified |
| **CARE-Adaptive** | **96** | 8 | Regime B | 46.19 GB | **14.8101** | +5.2386 | +54.73% | 0.85 | 5.89 GB | Verified |
| **Random Merging** | **96** | 8 | Regime B | 46.19 GB | **26.3504** | +16.7789 | +175.30% | 0.85 | 5.89 GB | Verified |
| **REAP** | **80** | 8 | **Regime A** | 39.44 GB | **16.1392** | +6.5677 | +68.62% | **20.03** | 40.53 GB | Reasoning 0% |
| **CARE-Adaptive** | **80** | 8 | **Regime A** | 39.44 GB | **19.9292** | +10.3577 | +108.21% | **19.29** | 40.53 GB | Verified |
| **Random Merging** | **80** | 8 | **Regime A** | 39.44 GB | **47.6214** | +38.0499 | +397.53% | **20.44** | 40.53 GB | Verified |
| **Sub-MoE** | **80** | 8 | **Regime A** | 39.44 GB | **90.7231** | +81.1516 | +847.85% | **20.32** | 40.53 GB | Severe Degr. |
| **REAP** | **64** | 8 | **Regime A** | 32.69 GB | **22.0596** | +12.4881 | +130.47% | **20.49** | 33.45 GB | Verified |
| **CARE-Adaptive** | **64** | 8 | **Regime A** | 32.69 GB | **31.3531** | +21.7816 | +227.57% | **20.37** | 33.45 GB | Verified |
| **Sub-MoE** | **64** | 8 | **Regime A** | 32.69 GB | **176.9348** | +167.3633 | +1748.56% | **20.62** | 33.45 GB | Collapsed |
| **Random Merging** | **64** | 8 | **Regime A** | 32.69 GB | **189.8233** | +180.2518 | +1883.22% | **20.75** | 33.45 GB | Collapsed |
| **REAP** | **60** | 8 | **Regime A** | 31.00 GB | **23.9455** | +14.3740 | +150.18% | **20.58** | 31.72 GB | Reasoning 0% |
| **CARE-Adaptive** | **60** | 8 | **Regime A** | 31.00 GB | **33.5058** | +23.9343 | +250.06% | **20.45** | 31.72 GB | Verified |
| **Sub-MoE** | **60** | 8 | **Regime A** | 31.00 GB | **203.6507** | +194.0792 | +2027.68% | **20.70** | 31.72 GB | Collapsed |
| **Random Merging** | **60** | 8 | **Regime A** | 31.00 GB | **213.4241** | +203.8526 | +2129.79% | **20.83** | 31.72 GB | Collapsed |
| **HC-SMoE** | 128 | 8 | N/A | N/A | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | Unimplemented |
| **MC-SMoE** | 128 | 8 | N/A | N/A | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | Unimplemented |

---

### 2.2 Experiments 2 & 3: Top-K Sparsity Control ($k \in \{8, 6, 4, 2\}$)
To disentangle **expert population size ($E$)** from **inference routing sparsity ($k$)**, we benchmarked top-$k$ modulation across both systems regimes on the identical test tokens.

#### A. Full Canonical Evaluation (32,768 tokens)
| Configuration | Population ($E$) | Active ($k$) | Regime | WikiText-2 PPL | $\Delta$ PPL vs Baseline | Prefill Throughput | Decode Throughput | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline-128** | 128 | 8 | Regime B | **9.5715** | 0.0000 | 851.4 tok/s | 0.68 tok/s | 5.89 GB |
| **Baseline-128** | 128 | 6 | Regime B | **9.8657** | +0.2942 | 864.1 tok/s | 0.70 tok/s | 5.89 GB |
| **Baseline-128** | 128 | 4 | Regime B | **11.4332** | +1.8617 | 878.4 tok/s | 0.73 tok/s | 5.89 GB |
| **Baseline-128** | 128 | 2 | Regime B | **52.6092** | +43.0377 | 884.8 tok/s | 0.75 tok/s | 5.89 GB |
| **CARE-80** | 80 | 8 | **Regime A** | **19.9292** | +10.3577 | **12,689.6 tok/s** | **19.29 tok/s** | 40.53 GB |
| **CARE-80** | 80 | 6 | **Regime A** | **20.3620** | +10.7905 | **14,449.9 tok/s** | **19.46 tok/s** | 40.53 GB |
| **CARE-80** | 80 | 4 | **Regime A** | **24.4178** | +14.8463 | **16,751.2 tok/s** | **19.42 tok/s** | 40.53 GB |
| **CARE-80** | 80 | 2 | **Regime A** | **114.2377** | +104.6662 | **19,556.3 tok/s** | **19.68 tok/s** | 40.53 GB |

#### B. Pilot Evaluation (2,048 tokens)
| Configuration | Population ($E$) | Active ($k$) | Regime | Pilot PPL | Latency | Tokens / Sec | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline-128 (Pilot)** | 128 | 8 | Regime B | **11.5793** | 26.82s | 76.4 tok/s | 5.89 GB |
| **Baseline-128 (Pilot)** | 128 | 6 | Regime B | **11.8077** | 18.18s | 112.7 tok/s | 5.89 GB |
| **Baseline-128 (Pilot)** | 128 | 4 | Regime B | **14.0768** | 18.17s | 112.7 tok/s | 5.89 GB |
| **Baseline-128 (Pilot)** | 128 | 2 | Regime B | **75.5065** | 18.11s | 113.1 tok/s | 5.89 GB |
| **CARE-80 (Pilot)** | 80 | 8 | **Regime A** | **28.5473** | 0.980s | **2,089.8 tok/s** | 36.96 GB |
| **CARE-80 (Pilot)** | 80 | 6 | **Regime A** | **28.9969** | 0.512s | **4,000.0 tok/s** | 36.96 GB |
| **CARE-80 (Pilot)** | 80 | 4 | **Regime A** | **35.1138** | 0.481s | **4,257.8 tok/s** | 36.96 GB |
| **CARE-80 (Pilot)** | 80 | 2 | **Regime A** | **155.5357** | 0.441s | **4,644.0 tok/s** | 36.96 GB |

---

## 3. Visualizations & Graphical Analysis

### Figure 1: Perplexity vs. Available Expert Count ($k=8$)
Shows the comparative scaling of CARE-Adaptive, REAP, Sub-MoE, and Random Merging across $E \in [60, 128]$ on the canonical 32k-token WikiText-2 benchmark.

![Perplexity vs Experts](/Users/deepeshkumarjha/.gemini/antigravity-ide/brain/8d6d78d9-760d-4f2e-bbee-c8d7644d95c5/qwen3_ppl_vs_experts_k8.png)

### Figure 2: Perplexity vs. Top-K Routing Sparsity
Shows the control experiment comparing active routing sparsity ($k \in [2, 8]$) on both the full 32k-token dataset and the 2k-token pilot sweep for Baseline 128E and CARE 80E.

![Perplexity vs Top-K](/Users/deepeshkumarjha/.gemini/antigravity-ide/brain/8d6d78d9-760d-4f2e-bbee-c8d7644d95c5/qwen3_ppl_vs_topk.png)

### Figure 3: Quality-Efficiency Pareto Frontiers
Demonstrates the dual Pareto frontiers: Panel A illustrates PPL vs. Checkpoint Footprint (GB); Panel B illustrates PPL vs. Sustained Autoregressive Decode Throughput (tok/s), showcasing the discontinuous jump between Regime B (CPU streaming) and Regime A (pure GPU).

![Pareto Frontiers](/Users/deepeshkumarjha/.gemini/antigravity-ide/brain/8d6d78d9-760d-4f2e-bbee-c8d7644d95c5/qwen3_pareto_frontiers.png)

### Figure 4: Prefill Throughput vs. Top-K Sparsity Across Regimes
Contrasts the prefill token processing rate between GPU-resident CARE-80 and offloaded Baseline-128 as a function of top-$k$.

![Prefill Throughput vs Top-K](/Users/deepeshkumarjha/.gemini/antigravity-ide/brain/8d6d78d9-760d-4f2e-bbee-c8d7644d95c5/qwen3_topk_throughput.png)

---

## 4. Scientific Insights & Engineering Trade-Offs

### 4.1 Sparsity Decoupling: Population ($E$) vs. Active Routing ($k$)
1. **Quality Retention:** Moderately reducing active top-$k$ from 8 to 6 incurs negligible perplexity degradation (+0.29 PPL on Baseline, +0.43 PPL on CARE-80). However, dropping below $k=4$ causes non-linear degradation, and $k=2$ triggers severe perplexity collapse (PPL $> 52$ on Baseline, $> 114$ on CARE-80).
2. **Computational Dynamics:**
   - Reducing $k$ reduces arithmetic FLOPs per token by $\sim 25\%$ (from $k=8$ to $k=6$), increasing prefill throughput from 12,689 to 14,449 tok/s on GPU.
   - However, reducing $k$ **does not reduce model storage, VRAM allocation, or PCIe transfer size**. The model remains identically sized in memory.
   - Reducing $E$, conversely, directly shrinks physical parameter footprint ($32.05\text{B} \to 21.18\text{B}$), enabling the model to fit on chip and delivering the $28.4\times$ speedup.

### 4.2 Comparative Method Analysis
1. **CARE-Adaptive (Geometric Merging):**
   - Retains smooth, monotonic degradation down to 60 experts ($33.51$ PPL).
   - Preserves reasoning capabilities (22% GSM8K at 80E) by consolidating complementary expert capabilities rather than dropping weights.
   - **Trade-off:** High compression time ($21,148\text{ s} \approx 5.87\text{ hours}$ for 80E) due to candidate evaluation and subspace recalculation.
2. **REAP (Activation Pruning):**
   - Yields the lowest WikiText-2 language modeling perplexity across all pruned configurations ($16.14$ at 80E, $23.95$ at 60E).
   - Extremely rapid compression time ($171.2\text{ s} \approx 2.85\text{ minutes}$).
   - **Critical Vulnerability:** Suffers **complete reasoning collapse** ($0.0\%$ GSM8K and $0.0\%$ MATH-500) because pruning router-infrequent experts eradicates specialized reasoning circuits.
3. **Sub-MoE (Clustering):**
   - Competitive at mild compression ($11.88$ at 112E, $14.38$ at 96E), but suffers catastrophic divergence at 80 experts ($90.72$ PPL) and collapses completely below 80E ($203.65$ PPL at 60E).
4. **Random Merging:**
   - Degrades rapidly past 96 experts ($47.62$ at 80E, $213.42$ at 60E), demonstrating that geometry-guided candidate selection is essential for deep compression.
5. **HC-SMoE & MC-SMoE Absence:**
   - Neither hierarchical clustering (HC-SMoE) nor multi-criteria SMoE (MC-SMoE) has an existing implementation for Qwen3-30B-A3B in the repository. They are documented strictly as `Unimplemented` to maintain audit integrity.

---

## 5. Artifact Summary

| File | Type | Description |
| :--- | :---: | :--- |
| [`master_benchmark_results.csv`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/master_benchmark_results.csv) | CSV | Canonical results table with all verified measurements across methods, top-$k$, and regimes |
| [`compression_costs.csv`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/compression_costs.csv) | CSV | Compression wall-clock times, peak VRAM, and evaluation pass overheads |
| [`method_checkpoint_inventory.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/method_checkpoint_inventory.md) | Markdown | Comprehensive audit of all 7 candidate methods on Qwen3-30B |
| [`router_inspection_and_topk.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/router_inspection_and_topk.md) | Markdown | Source analysis of `Qwen3MoeTopKRouter`, probability renormalization, and FLOP semantics |
| [`full_wikitext_master_results.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/full_wikitext_master_results.json) | JSON | Machine-readable measurements for full 32k-token WikiText-2 evaluations |
| [`pilot_topk_results.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/pilot_topk_results.json) | JSON | Machine-readable measurements for 2k-token pilot top-$k$ sweeps |
| [`plots/qwen3_ppl_vs_experts_k8.png`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/plots/qwen3_ppl_vs_experts_k8.png) | PNG | Perplexity vs. expert count plot |
| [`plots/qwen3_ppl_vs_topk.png`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/plots/qwen3_ppl_vs_topk.png) | PNG | Perplexity vs. top-$k$ routing sparsity plot |
| [`plots/qwen3_pareto_frontiers.png`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/plots/qwen3_pareto_frontiers.png) | PNG | Quality-storage and quality-throughput Pareto frontiers |
| [`plots/qwen3_topk_throughput.png`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/master_benchmark/plots/qwen3_topk_throughput.png) | PNG | Prefill throughput vs. top-$k$ plot |
