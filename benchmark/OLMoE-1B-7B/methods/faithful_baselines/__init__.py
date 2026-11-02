"""
benchmark/OLMoE-1B-7B/methods/faithful_baselines/__init__.py
============================================================
Faithful, primary-source reproductions of MoE baseline algorithms:
- SubMoEEngine: AAAI 2026 (Experts Union Decomposition via Joint SVD)
- REAPPruneEngine, REAPSaliencyObserver: Cerebras Research 2024 (Router-weighted activation pruning)
- HCSMoEEngine: PMLR / ICML 2025 (Hierarchical clustering with average/ward linkage)
- MCSMoEEngine: OpenReview / ICLR 2024 (Hungarian bipartite neuron permutation matching)
"""

from .sub_moe_engine import SubMoEEngine
from .reap_prune_engine import REAPPruneEngine, REAPSaliencyObserver
from .hc_smoe_engine import HCSMoEEngine
from .mc_smoe_engine import MCSMoEEngine

__all__ = [
    "SubMoEEngine",
    "REAPPruneEngine",
    "REAPSaliencyObserver",
    "HCSMoEEngine",
    "MCSMoEEngine",
]
