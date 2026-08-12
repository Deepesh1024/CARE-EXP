"""
CARE-MoE Experiment 4 — Leakage Assertions
============================================
Hard FAIL conditions from Specification §10 + §24.

All checks are automated assertions.
ANY failure → experiment STOPS immediately.
Do NOT produce scientific conclusions if any check fails.
"""

import numpy as np


def assert_no_train_test_overlap(
    train_experts: list, test_experts: list,
    context: str = "",
) -> None:
    """Rule 1: No test expert in training set."""
    train_set = set(train_experts)
    test_set = set(test_experts)
    overlap = train_set & test_set
    assert not overlap, (
        f"LEAKAGE FAIL [{context}]: test experts {sorted(overlap)} "
        f"appear in training set."
    )


def assert_no_test_test_in_mds(
    D_train: np.ndarray,
    train_experts: list,
    D_full: np.ndarray,
    context: str = "",
) -> None:
    """Rule 2: test-test Oracle distances NOT used to fit training MDS.

    Verifies D_train == D_full[train×train].
    """
    train_idx = np.array(train_experts, dtype=int)
    expected = D_full[np.ix_(train_idx, train_idx)]
    assert np.allclose(D_train, expected, atol=1e-8), (
        f"LEAKAGE FAIL [{context}]: training MDS distance matrix does not match "
        f"D_full[train×train]. Possible test-test contamination."
    )


def assert_no_test_test_in_oos(
    d_to_train: np.ndarray,
    train_experts: list,
    test_experts: list,
    test_expert_id: int,
    D_full: np.ndarray,
    context: str = "",
) -> None:
    """Rule 3: test-test distances NOT used to optimize test-expert coordinate.

    Verifies d_to_train == D_full[test_expert_id, train_experts].
    """
    train_idx = np.array(train_experts, dtype=int)
    expected = D_full[test_expert_id, train_idx]
    assert np.allclose(d_to_train, expected, atol=1e-8), (
        f"LEAKAGE FAIL [{context}]: OOS distances for expert {test_expert_id} "
        f"do not match D_full[test, train]. Possible test-test contamination."
    )

    # Ensure test-test distances not accidentally included
    test_idx = set(test_experts)
    n_train = len(train_experts)
    assert len(d_to_train) == n_train, (
        f"LEAKAGE FAIL [{context}]: d_to_train length {len(d_to_train)} "
        f"!= n_train {n_train}. Test-test distances may be included."
    )


def assert_no_target_in_features(
    feature_names: list,
    forbidden_names: list,
    context: str = "",
) -> None:
    """Rule 4: Oracle KL and post-merge features not in feature list."""
    for feat in feature_names:
        assert feat not in forbidden_names, (
            f"LEAKAGE FAIL [{context}]: forbidden feature '{feat}' "
            f"appears in feature list."
        )


def assert_model_retrained(
    model_trained_on_fold: bool,
    context: str = "",
) -> None:
    """Rule 5 / Rule 6: Model trained from scratch on this fold."""
    assert model_trained_on_fold, (
        f"LEAKAGE FAIL [{context}]: model was not retrained for this fold. "
        f"Must not reuse a model trained on different experts."
    )


def assert_q_not_tuned_on_exp4(q: int, expected_q: int, context: str = "") -> None:
    """Rule 7: q must equal the pre-registered value from Exp 3B."""
    assert q == expected_q, (
        f"LEAKAGE FAIL [{context}]: q={q} != expected {expected_q}. "
        f"q must not be selected based on Experiment 4 results."
    )


def assert_hyperparams_unchanged(
    params: dict, reference_params: dict, context: str = ""
) -> None:
    """Rule 8: hyperparameters unchanged after pilot."""
    for key in reference_params:
        assert key in params, (
            f"LEAKAGE FAIL [{context}]: hyperparameter '{key}' missing."
        )
        assert params[key] == reference_params[key], (
            f"LEAKAGE FAIL [{context}]: hyperparameter '{key}' changed: "
            f"{reference_params[key]} → {params[key]}."
        )


def assert_fold_assignment_reproducible(
    splits_path: str, splits_loaded: list, context: str = ""
) -> None:
    """Rule 9: fold assignments must be reproducible from frozen file."""
    import json, os
    assert os.path.exists(splits_path), (
        f"LEAKAGE FAIL [{context}]: cv_splits.json not found."
    )
    with open(splits_path) as f:
        splits_disk = json.load(f)

    for p_idx in range(len(splits_loaded)):
        for f_idx in range(len(splits_loaded[p_idx]["folds"])):
            disk_test = sorted(splits_disk[p_idx]["folds"][f_idx]["test_experts"])
            mem_test = sorted(splits_loaded[p_idx]["folds"][f_idx]["test_experts"])
            assert disk_test == mem_test, (
                f"LEAKAGE FAIL [{context}]: fold assignment mismatch at "
                f"partition {p_idx} fold {f_idx}."
            )


def assert_identical_folds_abc(
    train_a: list, test_a: list,
    train_b: list, test_b: list,
    train_c: list, test_c: list,
    context: str = "",
) -> None:
    """Rule 10: Models A, B, C use identical train/test experts."""
    assert sorted(train_a) == sorted(train_b) == sorted(train_c), (
        f"LEAKAGE FAIL [{context}]: train experts differ between models A/B/C."
    )
    assert sorted(test_a) == sorted(test_b) == sorted(test_c), (
        f"LEAKAGE FAIL [{context}]: test experts differ between models A/B/C."
    )


def assert_no_topology_features(feature_names: list, context: str = "") -> None:
    """Spec §21: No topology/graph features."""
    forbidden_keywords = [
        "community", "louvain", "degree", "centrality", "pagerank",
        "density", "knn", "graph_dist", "topology", "embedding_graph",
    ]
    for feat in feature_names:
        for kw in forbidden_keywords:
            assert kw.lower() not in feat.lower(), (
                f"LEAKAGE FAIL [{context}]: topology feature '{feat}' detected "
                f"(keyword: '{kw}'). Topology features are excluded."
            )


def assert_mds_output_dimension(Z: np.ndarray, q: int, context: str = "") -> None:
    """Pilot check: MDS output must have exactly q dimensions."""
    assert Z.ndim == 2 and Z.shape[1] == q, (
        f"INTEGRITY FAIL [{context}]: MDS output shape {Z.shape}, "
        f"expected (n, {q})."
    )


def assert_no_nan_inf(arr: np.ndarray, name: str, context: str = "") -> None:
    """Pilot check: no NaN or Inf in arrays."""
    assert not np.any(np.isnan(arr)), (
        f"INTEGRITY FAIL [{context}]: NaN detected in {name}."
    )
    assert not np.any(np.isinf(arr)), (
        f"INTEGRITY FAIL [{context}]: Inf detected in {name}."
    )


def assert_feature_target_alignment(
    y_fold: np.ndarray,
    pi: np.ndarray,
    pj: np.ndarray,
    D_oracle: np.ndarray,
    context: str = "",
) -> None:
    """Verify feature-target index alignment for all pairs in fold."""
    for k in range(len(y_fold)):
        i, j = int(pi[k]), int(pj[k])
        expected = float(D_oracle[i, j])
        actual = float(y_fold[k])
        assert abs(expected - actual) < 1e-6, (
            f"INTEGRITY FAIL [{context}]: target mismatch at pair ({i},{j}): "
            f"expected {expected:.8f}, got {actual:.8f}."
        )


def run_all_leakage_checks(
    *,
    train_experts: list,
    test_experts: list,
    D_train: np.ndarray,
    D_full: np.ndarray,
    feature_names: list,
    forbidden_features: list,
    q: int,
    expected_q: int,
    xgb_params: dict,
    xgb_ref_params: dict,
    splits_path: str,
    splits: list,
    Z_train: np.ndarray,
    y_test: np.ndarray,
    pi_test: np.ndarray,
    pj_test: np.ndarray,
    context: str = "",
) -> None:
    """Run all leakage checks for a fold. Raises AssertionError on any failure."""
    assert_no_train_test_overlap(train_experts, test_experts, context)
    assert_no_test_test_in_mds(D_train, train_experts, D_full, context)
    assert_no_target_in_features(feature_names, forbidden_features, context)
    assert_q_not_tuned_on_exp4(q, expected_q, context)
    assert_hyperparams_unchanged(xgb_params, xgb_ref_params, context)
    assert_fold_assignment_reproducible(splits_path, splits, context)
    assert_no_topology_features(feature_names, context)
    assert_mds_output_dimension(Z_train, expected_q, context)
    assert_no_nan_inf(Z_train, "Z_train", context)
    assert_feature_target_alignment(y_test, pi_test, pj_test, D_full, context)
    print(f"[leakage_checks] All checks PASSED for {context}")
