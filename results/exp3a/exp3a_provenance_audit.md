# EXPERIMENT 3A: PROVENANCE AUDIT

**FINAL PROVENANCE STATUS:** VERIFIED (WITH MAJOR CIRCULARITY WARNING)

## 1. Exact Model
- **Model Name:** OLMoE-1B-7B
- **HuggingFace Revision:** `allenai/OLMoE-1B-7B-0924` (Commit `28e3a93d2ad14e075e10d2258f79d75b92f00626` as per Exp 1 metadata).
- **Architecture:** Mixture-of-Experts
- **Number of MoE Layers:** 16
- **Number of Experts per Layer:** 64
- **Layers Used:** `first` (layer 0), `middle` (layer 8), `last` (layer 15), and `aggregated` (mean affinity across the three layers).

## 2. Calibration Dataset
- **Dataset:** `allenai/c4`
- **Split:** `train`
- **Seed:** 42
- **Number of Sequences:** Target was 256 or 512 sequences; filtering restricted it to `Seq_Len=512`.
- **Tokenization:** Base `allenai/OLMoE-1B-7B-0924` tokenizer. 
- **Exact Cached File:** Pre-computed in Experiment 1; 3A reads `output.json` directly and filters by `Seq_Len == 512`.

## 3. Oracle Methodology
- **Exact Definition of Oracle KL:** Kullback-Leibler Divergence between the logits of the original unmerged model and the logits of the surgically merged model.
- **Merge Operation:** `UniformAverage` of weights.
- **KL Direction:** $KL(P_{\text{orig}} \parallel P_{\text{merged}})$.
- **Reduction:** Averaged across all tokens in the batch.
- **Exact Expert Parameters Modified:** `mlp.experts[i]` and `mlp.experts[j]`. Both are replaced with `(W_i + W_j) / 2`.

## 4. 3A Matrix Contents (CRITICAL FINDING)
- **What it contains:** The 3A graph is **NOT** built from true Oracle KL. 
- **The Process:** Phase 1 uses a frozen XGBoost surrogate (`XGBoost_C.pkl` from Experiment 2) to predict the Oracle KL using 11 pre-merge features (e.g., Weight Distance, Routing Similarity, Usage Frequency).
- **Conclusion:** The community structure is fundamentally derived from surrogate predictions, not the true Oracle KL. However, the communities are later validated using the true Oracle KL. 

## 5. Graph Construction
- **Affinity Equation:** $\text{Affinity}(i,j) = \exp\!\left(-\frac{\text{Predicted KL}(i,j)}{\text{Median}(\text{Predicted KL})}\right)$
- **Normalization:** By the median of Predicted KL for that layer.
- **kNN Value:** $k=3, 5, 8$ (with $k=8$ used as the primary setting).
- **Mutual-kNN:** Yes. Edges only exist if $i \in \text{Top-}k(j)$ and $j \in \text{Top-}k(i)$.
- **Weighting:** The graph is initially built with weights (Affinity), but **Community Detection (Louvain) was run on an UNWEIGHTED (binary) version of the graph.**
- **Directed/Undirected:** Undirected.
- **Self-Edge Handling:** Self-affinity is set to $-\infty$ during Top-$k$ selection, so no self-edges exist.

## 6. Community Detection
- **Algorithm:** Louvain (`community_louvain.best_partition`).
- **Resolution Parameter:** Default (1.0).
- **Random Seed:** Set globally via `utils.set_global_seed()` (Seed = 42).
- **Number of Communities:** Dynamically discovered. (e.g., 7 communities at $k=8$ for the aggregated layer).
- **Multiple Runs:** No, a single deterministic run was used.

## 7. Report Lineage
```
Model (OLMoE-1B-7B) -> Exp 1 Calibration (C4)
       ↓
Exp 1 True Oracle KL Measurements
       ↓
Exp 2 XGBoost Surrogate Training (on Weight/Routing features)
       ↓
Exp 3A Surrogate Inference (Predicted KL)
       ↓
Affinity Transformation (Exp(-Predicted_KL / Median))
       ↓
Mutual-kNN Graph Construction (k=8)
       ↓
Louvain Community Detection (Unweighted)
       ↓
Validation vs TRUE Exp 1 Oracle KL (Within vs Between)
       ↓
Reported Statistics (Mann-Whitney U, Cohen's d, Silhouette)
```

## 8. Model Mismatch Resolution
- **Issue:** Some documentation referenced `Mistral-7B-Instruct`.
- **Resolution:** A code audit reveals that `phase6_report.py` contains a hardcoded string: `r.append("- All analyses are performed on Mistral-7B-Instruct...")`. The actual execution loaded `allenai/OLMoE-1B-7B-0924` from `config.py` and the `output.json` data confirms the target model is OLMoE. **This was a copy-paste error in the report generator.**

## 9. Inconsistencies Identified
1. **Circularity:** The graph is constructed from surrogate predictions, but validated on the true labels that the surrogate was trained to predict.
2. **Unweighted Louvain:** The report discusses affinity graphs, but the community detection explicitly binarized the graph (`G_binary.add_edges_from(G.edges())`) before running Louvain. 

## 10. Credentials
- No fabricated titles or affiliations were found in the report headers.

---
**Verdict:** Provenance is VERIFIED. The data lineage is intact, though the methodology contains significant circularity that must be accounted for in Null testing.
