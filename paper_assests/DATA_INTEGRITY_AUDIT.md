# Data Integrity Audit

**Missing KL Values:**
Previously, some plot scripts converted missing Cumulative KL values to zero for external baselines (HC-SMoE, M-SMoE, Sub-MoE) using `.fillna(0)`. This was fixed during the Load-Bearing Claim Audit. The final plots either omit these methods from the KL axes or explicitly label them as N/A. **Missing data is never plotted as zero.**

**Metrics on axes:**
PPL and KL are separated into different panels/plots. They are never incorrectly superimposed on a single y-axis.

**Reindexing:**
All candidate pairs tested in Experiment 4 and the 1st step validation are strictly from the unmerged M_64 model, completely avoiding any indexing drift.

**Benchmark Integrity:**
All methods in Figure 5 use the exact same OLMoE-1B-7B-0924 initialization, the same WikiText-2 evaluation chunks, and the same parameter reduction mapping.
