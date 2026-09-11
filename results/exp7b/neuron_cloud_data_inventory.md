# Neuron Cloud Data Inventory

==================================================
## 1. FIND ALL ACTIVATION-RELATED ARTIFACTS
==================================================

After a comprehensive search of the `CARE-EXP` repository across all experiments, results, and intermediate artifacts, **no raw activation tensors exist**. The only activation-related artifacts are derived aggregate statistics from Experiment 7A:

- **`results/exp7a/sensitivity/A_n.npy`**
  - **File Type**: Numpy Array (`.npy`)
  - **File Size**: 262,272 bytes (~256 KB)
  - **Experiment**: Exp 7A (Neuron-Level Sensitivity)
  - **Contents**: Derived aggregated statistic (mean absolute activation per neuron)
  - **Shape**: `(64, 1024)`
  - **Dtype**: `float32`
  - **Layer**: Target central layer (Layer 8)
  - **Experts**: All 64 experts
  - **Neuron Dimension**: All 1024 neurons
  - **Samples**: Aggregated scalar (0 raw sequences/tokens retained)
  - **Dataset Source**: ARC subset
  - **Masks/Positions**: Lost during aggregation
  - **Routing Information**: Not stored

- **`results/exp7a/sensitivity/T_K.npy`**
  - **Contents**: Taylor Sensitivity metric per neuron. Shape `(64, 1024)`. Size 256 KB.
- **`results/exp7a/sensitivity/W_n.npy`**
  - **Contents**: Derived weight statistic (downstream weight norms). Shape `(64, 1024)`. Size 256 KB.

*Note: Source code like `test_grad.py` and `phase4_intervention.py` briefly capture activations during runtime using hooks (`_captured_activations`), but they are immediately discarded from RAM after processing and never saved to disk.*

==================================================
## 2. DETERMINE WHETHER WE ALREADY HAVE THE REQUIRED NEURON FUNCTIONAL SIGNATURES
==================================================

We **DO NOT** have the required functional neuron signatures.

**A. Do we already have per-neuron activation vectors?**
No. Only scalar aggregates exist.
**B. Are they from the SAME probe/input set across experts?**
N/A (No activation vectors exist).
**C. Which MoE layer(s) are available?**
Only Layer 8 statistics are stored.
**D. Which experts are available?**
Statistics exist for all 64 experts.
**E. Are all 1024 neurons available per expert?**
Yes, statistics exist for all 1024 neurons.
**F. How many probe examples/tokens are available?**
0 token-level activations are preserved.
**G. Are activations pre-activation or post-activation?**
N/A.
**H. Are they before or after expert routing?**
N/A.
**I. Are they expert-local FFN activations?**
N/A.
**J. Can they be directly compared across experts?**
No, scalar averages cannot form high-dimensional functional signatures.

==================================================
## 3. CHECK THE PROBE DATA
==================================================

The following existing datasets/probe sets were identified in `results/exp7a/data/`:

- **`D_proxy.pt`, `D_final.pt`, `D_validation.pt`**
  - **Dataset Name**: AI2 ARC Challenge
  - **Split**: Validation/Test partitions
  - **Usage**: Used for Capability-K ($D_{discover}$, $D_{validate}$) evaluation.
  - **Wikitext KL evaluation**: The Wikitext dataset used for $D_{actual\_KL}$ is NOT stored on disk; it is dynamically loaded via HuggingFace (`Salesforce/wikitext`, `train` split) during Phase 2. 

**Conclusion**: The capability-specific residual $R_K(i,j)$ probes exist (`D_proxy.pt`), but the Merge-KL residual $R_{KL}(i,j)$ probes (Wikitext) are not stored locally as artifacts. Neither dataset has associated saved activations.

==================================================
## 4. CHECK DOWNSTREAM WEIGHT INFORMATION
==================================================

- **Availability**: **Yes**. The exact statistic $q_{i,k} = ||W_{down}[:,k]||_2$ has already been computed and saved.
- **Path**: `results/exp7a/sensitivity/W_n.npy`
- **Shape**: `(64, 1024)`
- **Mapping**: Unambiguously maps to expert $i$, neuron $k$.
- **Model Loading**: **Not required**. It can be read directly from the numpy array without loading the full model.

==================================================
## 5. CHECK EXISTING NEURON-LEVEL EXPERIMENTS
==================================================

- **7A/7A.51 generated activation data?** No, only evaluated interventions dynamically.
- **Taylor sensitivity data stored?** Yes (`T_K.npy`).
- **Neuron ranking data stored?** Yes (`sampled_neurons.csv`).
- **Neuron embeddings/signatures stored?** **No.**
- **Similarity matrices stored?** **No.**
- **Clustering performed?** **No.**
- **Intervention data exists?** Yes, `results/exp7a_51/intervention/intervention_results.csv` stores ARC impacts of zeroing out specific neurons.

*Conclusion: All existing artifacts are derived/Taylor/intervention metrics, not raw activation signatures.*

==================================================
## 6. CHECK WHETHER THE DATA IS SUFFICIENT
==================================================

**OPTION C: NOT AVAILABLE**

We need a fresh activation extraction pass.
**Missing Information:**
- Per-token raw activations (post-activation function) for all $64 \times 1024$ neurons.
- Activations must be recorded over an explicit $M$-token probe set (either a fixed Wikitext partition for $R_{KL}$ or ARC for $R_K$).

==================================================
## 7. ESTIMATE CPU FEASIBILITY
==================================================

**Activation Signature Tensor:**
Assuming we use $M = 10,000$ probe tokens:
Shape = $64 \text{ experts} \times 1024 \text{ neurons} \times 10,000 \text{ tokens}$ = 655,360,000 elements.
- `float32`: **~2.6 GB**
- `float16` / `bfloat16`: **~1.3 GB**

**Similarity Matrix:**
Pairwise similarities across all neurons ($65,536 \times 65,536$):
Total elements = 4,294,967,296.
- `float32`: **~17.1 GB**
- `float16`: **~8.5 GB**

**Conclusion:** 
Constructing the cross-expert similarity cloud is **fully feasible** on an Intel i7 with 32 GB RAM. The 17 GB similarity matrix and the 2.6 GB activation tensor easily fit within 32 GB of system memory. However, the *extraction* of the activations requires a GPU; the laptop can comfortably perform the CPU-bound pairwise similarity and clustering.

==================================================
## 8. DERN OVERLAP CHECK
==================================================

A grep search for `DERN` throughout the repository yielded **no functional overlap** or explicit implementation notes. There is no existing code or methodology in the CARE-EXP repository that currently performs expert decomposition, cross-expert matching, or segment recombination.

==================================================
## 9. RECOMMENDED NEXT STEP
==================================================

**NOT READY — fresh activation extraction required**
