# Figure Captions

**Figure 1 (Method Overview):**
Comparison of traditional static expert merging (left) versus the Sequential Functional Intervention framework of CARE-COM (right). Unlike static methods that permanently freeze the expert topology, CARE-COM adaptively recomputes the capability state $C_t$ after every physical merge to accurately reflect the shifting functional landscape.

**Figure 2 (Geometry vs Damage):**
Correlation between capability distance $d_C(i,j)$ and actual functional intervention damage (Oracle KL divergence). While capability geometry provides a strong global predictive signal (Spearman $\rho=0.7657$), substantial variance remains at the local level.

**Figure 3 (Similarity vs Consequence):**
Distribution of actual functional damage for 186 candidate pairs with nearly identical capability distances ($\epsilon < 1e^{-4}$). The significant variance (up to 4.6$\times$) demonstrates that capability similarity is informative but insufficient to uniquely determine the optimal intervention, necessitating explicit functional evaluation.

**Figure 4 (Static vs Adaptive):**
Controlled ablation comparing static ranking against adaptive recomputation on OLMoE-1B-7B. Adaptive recomputation consistently reduces cumulative functional divergence and improves end-to-end language modeling perplexity across all compression targets.

**Figure 5 (External Benchmark):**
Language modeling performance (WikiText-2 Perplexity) under aggressive expert consolidation. CARE-Adaptive remains highly competitive with state-of-the-art structural pruning methods (Sub-MoE) while substantially outperforming routing-based merging baselines (HC-SMoE, M-SMoE) and Random merging.

**Figure 6 (Efficiency):**
Computational efficiency at the 48-expert target (25% parameter reduction). CARE-Adaptive evaluates ~30% fewer exact candidates than CARE-Static due to the dynamically updating topology, resulting in an 80% reduction in wall-clock compression time.
