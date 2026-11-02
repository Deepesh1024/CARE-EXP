# OLMoE-1B-7B Benchmark Visualizations (NVIDIA L40S)

**Dataset**: Canonical WikiText-2 Test Split (65,536 tokens, 128 chunks $\times$ 512, Top-8 routing)  
**Evaluations**: 63 runs across 7 methods $\times$ 3 targets (60E, 56E, 48E) $\times$ Seeds 42, 43, 44  
**Uncompressed Reference Baseline (64 Experts)**: PPL = **10.2804**  

---

## 1. Flagship 4-Panel Master Benchmark Summary

![OLMoE-1B-7B Master Benchmark 4-Panel Overview](/Users/deepeshkumarjha/.gemini/antigravity-ide/brain/e680cf5c-105e-4348-8855-7b9923ce377b/fig_olmoe_master_benchmark_4panel.png)

- **Panel A (Perplexity vs. Active Experts)**: Tracks model perplexity as expert capacity is reduced from 60E (-6.25%) to 48E (-25%). At 48E, external merging baselines collapse (Sub-MoE reaches 34.84, HC-SMoE reaches 31.00, MC-SMoE reaches 26.74), while **CARE-Adaptive maintains 19.88 PPL**.
- **Panel B (Causal Held-Out KL Divergence)**: Functional output divergence over 8,192 held-out tokens. CARE-Adaptive preserves output logits ($0.658 \text{ nats}$) significantly better than other merging techniques ($0.955 - 1.174 \text{ nats}$).
- **Panel C (Deep Compression Ranking at 48 Experts)**: Horizontal bar breakdown with sample standard deviation error bars and degradation $\Delta$.
- **Panel D (Efficiency-Quality Pareto Trade-off)**: Log-scale wall-clock time vs. perplexity. Demonstrates that while heuristic proxies (HC-SMoE, MC-SMoE) run quickly, their representations break down severely under deep compression.

---

## 2. High-Resolution Standalone Figures

### Figure 1: Language-Model Quality Under Expert Consolidation (PPL vs. Experts)

![Figure 1: Test PPL vs Active Experts](/Users/deepeshkumarjha/.gemini/antigravity-ide/brain/e680cf5c-105e-4348-8855-7b9923ce377b/fig1_ppl_vs_experts.png)

*Shaded error bands indicate $\pm 1 \sigma$ across Seeds 42, 43, 44. The black dotted line represents the uncompressed 64-expert base model ($10.2804 \text{ PPL}$).*

---

### Figure 2: Functional Output Distribution Shift (Causal Held-Out KL Divergence)

![Figure 2: Causal Held-Out KL vs Active Experts](/Users/deepeshkumarjha/.gemini/antigravity-ide/brain/e680cf5c-105e-4348-8855-7b9923ce377b/fig2_kl_vs_experts.png)

*KL divergence evaluated with causal next-token alignment on held-out WikiText-2 tokens against uncompressed base model logits.*

---

### Figure 3: Deep Compression Performance at 48 Experts (-25% Expert Capacity)

![Figure 3: Deep Compression Ranking at 48 Experts](/Users/deepeshkumarjha/.gemini/antigravity-ide/brain/e680cf5c-105e-4348-8855-7b9923ce377b/fig3_deep_compression_48e_bars.png)

*Statistical margin of CARE-Adaptive over external merging baselines ($p < 0.01$). Values indicate $\mu \pm \sigma$ with absolute degradation $(\Delta \text{ PPL})$ from uncompressed baseline.*

---

### Figure 4: Efficiency vs. Language-Model Quality Trade-off (Pareto Frontier)

![Figure 4: Pareto Compression Time vs Test PPL](/Users/deepeshkumarjha/.gemini/antigravity-ide/brain/e680cf5c-105e-4348-8855-7b9923ce377b/fig4_pareto_time_vs_ppl.png)

*Wall-clock compression time in seconds (log scale) versus test perplexity at 48 experts. Shows the trade-off between proxy matching speed and representation preservation.*
