# Literature Audit

Relevant Works:
- **Sub-MoE** (Sub-Network for MoE): Competitor baseline. Reduces parameters but relies on structural pruning heuristics.
- **HC-SMoE** (Hierarchical Clustering for MoE): Competitor baseline. Performs one-shot clustering.
- **M-SMoE** (Merging SMoE): Competitor baseline. Uses static routing representations to merge.
- **REAP / REAM**: Routing-aware pruning/merging techniques.

Crucially, existing literature primarily formulates expert merging as a **static one-shot clustering** problem. Our central premise—that *physical interventions fundamentally alter the functional topology and require sequential adaptive recomputation*—is a novel framing that challenges the prevailing static paradigm.
