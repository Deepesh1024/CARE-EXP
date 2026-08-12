"""
CARE-MoE Experiment 4 — Feature Provenance Audit
==================================================
Full audit of all 11 local features before execution.

Per the specification (§4 + user correction §3,4):
- Audit exact formulas from Exp 2 implementation
- Flag any feature that depends on experts OUTSIDE pair (i,j)
- Produce FEATURE_PROVENANCE table saved to results/exp4/feature_provenance.json
- Do NOT silently modify feature definitions
- Do NOT substitute alternative features

Feature classification:
  pair_local    = computed solely from pair (i,j) data in output.json
  global_stats  = uses aggregated statistics over all pairs containing expert i or j
                  (FLAGGED — see per-feature notes)
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    LOCAL_FEATURES,
    ORIGINAL_FEATURES,
    NEW_DESCRIPTORS,
    FORBIDDEN_FEATURES,
    EPSILON,
    FEATURE_PROVENANCE_PATH,
)


# ══════════════════════════════════════════════════════════
# Feature Provenance Table
# ══════════════════════════════════════════════════════════

FEATURE_PROVENANCE = [
    # ── Original 7 features (pair-local, stored directly in output.json) ──
    {
        "feature": "Weight_Distance",
        "source": "output.json (Exp 1 Oracle benchmark)",
        "formula": "L2 norm of (W_i - W_j) normalized by parameter count",
        "calibration": "Seq_Len=512, Layer=middle (Exp 3B configuration)",
        "locality": "pair_local",
        "depends_on_experts_outside_pair": False,
        "uses_global_statistics": False,
        "can_contain_target_info": False,
        "can_contain_postmerge_info": False,
        "flagged": False,
        "flag_reason": None,
        "exp2_formula_source": "phase1_descriptors.py — stored directly in output.json",
    },
    {
        "feature": "Weight_Cosine",
        "source": "output.json (Exp 1 Oracle benchmark)",
        "formula": "1 - cosine_similarity(W_i.flatten(), W_j.flatten())",
        "calibration": "Seq_Len=512, Layer=middle",
        "locality": "pair_local",
        "depends_on_experts_outside_pair": False,
        "uses_global_statistics": False,
        "can_contain_target_info": False,
        "can_contain_postmerge_info": False,
        "flagged": False,
        "flag_reason": None,
        "exp2_formula_source": "phase1_descriptors.py — stored directly in output.json",
    },
    {
        "feature": "Activation_Similarity",
        "source": "output.json (Exp 1 Oracle benchmark)",
        "formula": "Pearson correlation of mean activation vectors of experts i and j "
                   "on calibration data",
        "calibration": "Seq_Len=512, Layer=middle",
        "locality": "pair_local",
        "depends_on_experts_outside_pair": False,
        "uses_global_statistics": False,
        "can_contain_target_info": False,
        "can_contain_postmerge_info": False,
        "flagged": False,
        "flag_reason": None,
        "exp2_formula_source": "phase1_descriptors.py — stored directly in output.json",
    },
    {
        "feature": "Output_Similarity",
        "source": "output.json (Exp 1 Oracle benchmark)",
        "formula": "Pearson correlation of expert output vectors on calibration tokens",
        "calibration": "Seq_Len=512, Layer=middle",
        "locality": "pair_local",
        "depends_on_experts_outside_pair": False,
        "uses_global_statistics": False,
        "can_contain_target_info": False,
        "can_contain_postmerge_info": False,
        "flagged": False,
        "flag_reason": None,
        "exp2_formula_source": "phase1_descriptors.py — stored directly in output.json",
    },
    {
        "feature": "Routing_Similarity",
        "source": "output.json (Exp 1 Oracle benchmark)",
        "formula": "Pearson correlation of router probability vectors P_i and P_j "
                   "across calibration tokens",
        "calibration": "Seq_Len=512, Layer=middle",
        "locality": "pair_local",
        "depends_on_experts_outside_pair": False,
        "uses_global_statistics": False,
        "can_contain_target_info": False,
        "can_contain_postmerge_info": False,
        "flagged": False,
        "flag_reason": None,
        "exp2_formula_source": "phase1_descriptors.py — stored directly in output.json",
    },
    {
        "feature": "Usage_Frequency",
        "source": "output.json (Exp 1 Oracle benchmark)",
        "formula": "|A∪B| / N = fraction of calibration tokens processed by "
                   "expert i OR expert j (union routing set)",
        "calibration": "Seq_Len=512, Layer=middle",
        "locality": "pair_local",
        "depends_on_experts_outside_pair": False,
        "uses_global_statistics": False,
        "can_contain_target_info": False,
        "can_contain_postmerge_info": False,
        "flagged": False,
        "flag_reason": None,
        "exp2_formula_source": "phase1_descriptors.py — stored directly in output.json",
    },
    {
        "feature": "Jaccard_Overlap",
        "source": "output.json (Exp 1 Oracle benchmark)",
        "formula": "|A∩B| / |A∪B| = Jaccard coefficient of token routing sets",
        "calibration": "Seq_Len=512, Layer=middle",
        "locality": "pair_local",
        "depends_on_experts_outside_pair": False,
        "uses_global_statistics": False,
        "can_contain_target_info": False,
        "can_contain_postmerge_info": False,
        "flagged": False,
        "flag_reason": None,
        "exp2_formula_source": "phase1_descriptors.py — stored directly in output.json",
    },
    # ── 4 new CARE descriptors (Exp 2) ──
    {
        "feature": "Usage_Asymmetry",
        "source": "Exp 2 phase1_descriptors.py::compute_usage_asymmetry",
        "formula": "|ū_i - ū_j| where ū_i = mean(Usage_Frequency) "
                   "over ALL 63 pairs containing expert i",
        "calibration": "Seq_Len=512, Layer=middle",
        "locality": "global_stats",
        "depends_on_experts_outside_pair": True,
        "uses_global_statistics": True,
        "can_contain_target_info": False,
        "can_contain_postmerge_info": False,
        "flagged": True,
        "flag_reason": (
            "FLAGGED: per-expert marginal usage ū_i is computed over all pairs "
            "containing expert i, i.e., 63 pairs including cross-fold pairs. "
            "This is pre-merge routing information (no Oracle KL content), "
            "but does use information from experts outside pair (i,j). "
            "Feature is RETAINED unchanged per spec (do not silently replace). "
            "Implementation: per-expert stats computed from all 2016 pairs "
            "using only pre-merge routing data."
        ),
        "exp2_formula_source": "phase1_descriptors.py lines 131-170",
    },
    {
        "feature": "Routing_JSD_Proxy",
        "source": "Exp 2 phase1_descriptors.py::compute_routing_jsd_proxy",
        "formula": "(1 - Routing_Similarity(i,j)) × (1 - Jaccard_Overlap(i,j))",
        "calibration": "Seq_Len=512, Layer=middle",
        "locality": "pair_local",
        "depends_on_experts_outside_pair": False,
        "uses_global_statistics": False,
        "can_contain_target_info": False,
        "can_contain_postmerge_info": False,
        "flagged": False,
        "flag_reason": None,
        "exp2_formula_source": "phase1_descriptors.py lines 177-215",
    },
    {
        "feature": "Routing_NPMI_Proxy",
        "source": "Exp 2 phase1_descriptors.py::compute_routing_npmi_proxy",
        "formula": (
            "NPMI(i,j) = log(P_ij / (P_i * P_j)) / (-log(P_ij)), "
            "P_i ≈ ū_i / (3 * global_mean), "
            "P_j ≈ ū_j / (3 * global_mean), "
            "P_ij ≈ Jaccard_Overlap * Usage_Frequency, "
            "where ū_i = mean Usage_Frequency over all pairs containing expert i, "
            "global_mean = mean ū_i across all 64 experts"
        ),
        "calibration": "Seq_Len=512, Layer=middle",
        "locality": "global_stats",
        "depends_on_experts_outside_pair": True,
        "uses_global_statistics": True,
        "can_contain_target_info": False,
        "can_contain_postmerge_info": False,
        "flagged": True,
        "flag_reason": (
            "FLAGGED: uses per-expert marginal usage ū_i and global mean usage "
            "computed over all 64 experts. These are pre-merge routing statistics "
            "(no Oracle KL content) but depend on experts outside pair (i,j). "
            "Feature is RETAINED unchanged per spec."
        ),
        "exp2_formula_source": "phase1_descriptors.py lines 222-288",
    },
    {
        "feature": "Specialization_Diff",
        "source": "Exp 2 phase1_descriptors.py::compute_specialization_diff",
        "formula": (
            "|1/(ū_i + ε) - 1/(ū_j + ε)| "
            "where ū_i = mean Usage_Frequency over all pairs containing expert i"
        ),
        "calibration": "Seq_Len=512, Layer=middle",
        "locality": "global_stats",
        "depends_on_experts_outside_pair": True,
        "uses_global_statistics": True,
        "can_contain_target_info": False,
        "can_contain_postmerge_info": False,
        "flagged": True,
        "flag_reason": (
            "FLAGGED: per-expert specialization score 1/(ū_i+ε) uses ū_i computed "
            "over all 63 pairs containing expert i. Pre-merge routing info only "
            "(no Oracle KL content). Feature is RETAINED unchanged per spec."
        ),
        "exp2_formula_source": "phase1_descriptors.py lines 295-342",
    },
]

# Summary counts
_n_flagged = sum(1 for f in FEATURE_PROVENANCE if f["flagged"])
_n_pair_local = sum(1 for f in FEATURE_PROVENANCE if not f["flagged"])


def print_provenance_summary() -> None:
    """Print the feature provenance table to stdout."""
    print("=" * 80)
    print("FEATURE PROVENANCE AUDIT")
    print("=" * 80)
    print(f"{'Feature':<28} {'Locality':<14} {'Flagged':<9} {'Target?':<9} {'Post-merge?'}")
    print("-" * 80)
    for p in FEATURE_PROVENANCE:
        flag = "⚠ YES" if p["flagged"] else "  no"
        target = "YES" if p["can_contain_target_info"] else "no"
        postmerge = "YES" if p["can_contain_postmerge_info"] else "no"
        print(f"  {p['feature']:<26} {p['locality']:<14} {flag:<9} {target:<9} {postmerge}")
    print("-" * 80)
    print(f"  Total: {len(FEATURE_PROVENANCE)} features | "
          f"{_n_pair_local} pair-local | {_n_flagged} global-stats (flagged)")
    print()
    print("FLAGGED FEATURES (use global routing statistics, NOT Oracle KL):")
    for p in FEATURE_PROVENANCE:
        if p["flagged"]:
            print(f"  {p['feature']}: {p['flag_reason'][:120]}...")
    print()
    print("  Decision: ALL flagged features RETAINED (no silent replacement).")
    print("  Per-expert stats computed from all 2016 pairs using pre-merge")
    print("  routing data only. No Oracle KL enters feature computation.")
    print("=" * 80)


def save_provenance(results_dir: str) -> None:
    """Save provenance table to disk."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "feature_provenance.json")
    output = {
        "n_features": len(FEATURE_PROVENANCE),
        "n_pair_local": _n_pair_local,
        "n_flagged_global_stats": _n_flagged,
        "features": FEATURE_PROVENANCE,
        "decision": (
            "All 11 features retained without modification. "
            "Flagged features (Usage_Asymmetry, Routing_NPMI_Proxy, Specialization_Diff) "
            "use pre-merge routing statistics aggregated across all pairs. "
            "No Oracle KL information enters feature computation."
        ),
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[provenance] Saved → {path}")


def validate_feature_list() -> None:
    """Assert that provenance covers exactly LOCAL_FEATURES, in order."""
    provenance_names = [p["feature"] for p in FEATURE_PROVENANCE]
    assert provenance_names == list(LOCAL_FEATURES), (
        f"Provenance feature list mismatch.\n"
        f"Expected: {list(LOCAL_FEATURES)}\n"
        f"Got:      {provenance_names}"
    )
    # Assert no forbidden features
    for p in FEATURE_PROVENANCE:
        assert p["feature"] not in FORBIDDEN_FEATURES, (
            f"Feature '{p['feature']}' is in FORBIDDEN_FEATURES list!"
        )
    print("[provenance] Feature list validated ✓")


if __name__ == "__main__":
    validate_feature_list()
    print_provenance_summary()
