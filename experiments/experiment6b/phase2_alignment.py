"""
EXPERIMENT 6B — TASK 2 + TASK 3 + TASK 4:
  FUNCTIONAL SPACE ALIGNMENT, TRAJECTORIES, AND DISPLACEMENT
============================================================================
TASK 2: Build functional-space alignment pipeline (Procrustes).
TASK 3: Generate historical functional trajectories (aligned MDS).
TASK 4: Calculate historical displacement vectors.

CRITICAL REQUIREMENT:
  MDS has translation, rotation, and reflection ambiguity.
  We MUST align embeddings via orthogonal Procrustes before computing
  displacement. C_i(T2) - C_i(T1) is INVALID without alignment.

METHODOLOGY:
  1. For each checkpoint+layer, construct the functional distance matrix
     from existing Exp3C oracle_distance.csv.
  2. Run metric SMACOF MDS at each q value.
  3. Select a reference checkpoint (T100 — most data, full coverage).
  4. Align all other checkpoint embeddings to the reference using
     orthogonal Procrustes (scipy.spatial.procrustes).
  5. Quantify alignment residual, stress, pairwise-distance preservation.
  6. Compute aligned displacement vectors DeltaC_i.
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import MDS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    N_EXPERTS, LAYERS, CHECKPOINTS, CHECKPOINT_ORDER,
    Q_VALUES, Q_PRIMARY,
    EXP3C_RESULTS_DIR,
    RESULTS_DIR, EMBEDDINGS_DIR, METRICS_DIR, PLOTS_DIR,
    SMACOF_MAX_ITER, SMACOF_N_INIT, SMACOF_EPS,
    RANDOM_SEED, SAMPLED_PAIRS,
    ensure_dirs, mark_task, is_task_completed,
)


# ══════════════════════════════════════════════════════════
# LOAD ORACLE DISTANCE MATRICES
# ══════════════════════════════════════════════════════════

def load_oracle_matrix(ckpt_name, layer):
    """Load the Exp3C oracle distance matrix for a checkpoint+layer."""
    csv_path = os.path.join(EXP3C_RESULTS_DIR, ckpt_name, layer, "oracle_distance.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Oracle matrix not found: {csv_path}")
    mat = pd.read_csv(csv_path, header=None).values
    return mat


def get_common_pairs(layer):
    """Find expert pairs consistently evaluated across all checkpoints."""
    all_pairs = []
    for ckpt_name in CHECKPOINT_ORDER:
        mat = load_oracle_matrix(ckpt_name, layer)
        pairs = set()
        for i in range(mat.shape[0]):
            for j in range(i + 1, mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    pairs.add((i, j))
        all_pairs.append(pairs)
    common = all_pairs[0]
    for s in all_pairs[1:]:
        common = common.intersection(s)
    return sorted(list(common))


def build_symmetric_matrix_from_pairs(pairs, values, n=N_EXPERTS):
    """Build a symmetric distance matrix from pair-value mappings."""
    D = np.zeros((n, n))
    for (i, j), v in zip(pairs, values):
        D[i, j] = v
        D[j, i] = v
    return D


# ══════════════════════════════════════════════════════════
# MDS EMBEDDING
# ══════════════════════════════════════════════════════════

def run_mds(D, q, seed=RANDOM_SEED):
    """Run metric SMACOF MDS on a (possibly sparse) distance matrix.

    For sparse matrices (early checkpoints with only 384 pairs),
    we use weighted SMACOF where missing entries get weight=0.
    """
    n = D.shape[0]
    # Create weight matrix: 1 where we have data, 0 where NaN
    W = (~np.isnan(D)).astype(float)
    np.fill_diagonal(W, 0)

    # Replace NaN with 0 for MDS (weight=0 means ignored)
    D_clean = np.nan_to_num(D, nan=0.0)

    # Ensure symmetry
    D_clean = (D_clean + D_clean.T) / 2
    W = (W + W.T) / 2
    W = np.minimum(W, 1.0)

    best_Z = None
    best_stress = np.inf

    for init_idx in range(SMACOF_N_INIT):
        init_seed = (seed * 10000 + init_idx) % (2**31 - 1)
        rng = np.random.RandomState(init_seed)
        Z0 = rng.randn(n, q) * 0.01

        # Weighted SMACOF iteration
        Z = Z0.copy()
        for iteration in range(SMACOF_MAX_ITER):
            # Compute current embedding distances
            D_embed = squareform(pdist(Z))

            # Compute stress (weighted)
            residuals = W * (D_clean - D_embed)
            stress = np.sum(residuals ** 2)

            if stress < best_stress:
                best_stress = stress
                best_Z = Z.copy()

            # Guttman transform (weighted)
            B = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j and D_embed[i, j] > 1e-10 and W[i, j] > 0:
                        B[i, j] = -W[i, j] * D_clean[i, j] / D_embed[i, j]
            np.fill_diagonal(B, -B.sum(axis=1))

            V = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        V[i, j] = -W[i, j]
                np.fill_diagonal(V, 0)
            np.fill_diagonal(V, -V.sum(axis=1))

            # Regularize V for inversion
            V_reg = V + np.eye(n) * 1e-8
            try:
                Z_new = np.linalg.solve(V_reg, B @ Z)
            except np.linalg.LinAlgError:
                Z_new = np.linalg.lstsq(V_reg, B @ Z, rcond=None)[0]

            # Check convergence
            change = np.sqrt(np.sum((Z_new - Z) ** 2))
            Z = Z_new
            if change < SMACOF_EPS:
                break

    return best_Z.astype(np.float64), float(best_stress)


def run_mds_full(D, q, seed=RANDOM_SEED):
    """Run standard metric SMACOF for full (non-sparse) matrices."""
    try:
        mds = MDS(
            n_components=q,
            metric=True,
            dissimilarity="precomputed",
            max_iter=SMACOF_MAX_ITER,
            n_init=SMACOF_N_INIT,
            eps=SMACOF_EPS,
            random_state=seed,
            n_jobs=1,
            normalized_stress=False,
        )
    except TypeError:
        mds = MDS(
            n_components=q,
            metric=True,
            dissimilarity="precomputed",
            max_iter=SMACOF_MAX_ITER,
            n_init=SMACOF_N_INIT,
            eps=SMACOF_EPS,
            random_state=seed,
            n_jobs=1,
        )
    Z = mds.fit_transform(D.astype(np.float64))
    return Z.astype(np.float64), float(mds.stress_)


# ══════════════════════════════════════════════════════════
# PROCRUSTES ALIGNMENT
# ══════════════════════════════════════════════════════════

def align_to_reference(Z_target, Z_reference, common_expert_ids=None):
    """Align Z_target to Z_reference using orthogonal Procrustes.

    If common_expert_ids is provided, alignment is computed using only
    those experts, then applied to all experts.

    Returns:
        Z_aligned: aligned coordinates
        disparity: Procrustes disparity (alignment residual)
    """
    if common_expert_ids is not None:
        # Use only common experts for alignment computation
        Z_ref_sub = Z_reference[common_expert_ids]
        Z_tgt_sub = Z_target[common_expert_ids]
    else:
        Z_ref_sub = Z_reference
        Z_tgt_sub = Z_target

    # scipy.spatial.procrustes handles centering, scaling, rotation
    # But we want orthogonal Procrustes WITHOUT scaling
    # (functional distances have absolute meaning)

    # Center both
    mu_ref = Z_ref_sub.mean(axis=0)
    mu_tgt = Z_tgt_sub.mean(axis=0)

    Z_ref_c = Z_ref_sub - mu_ref
    Z_tgt_c = Z_tgt_sub - mu_tgt

    # Find optimal rotation: R = argmin ||Z_tgt_c @ R - Z_ref_c||^2
    # Solution: SVD of Z_ref_c^T @ Z_tgt_c
    M = Z_ref_c.T @ Z_tgt_c
    U, S, Vt = np.linalg.svd(M)
    # R = V @ U^T (handles reflections via det check)
    d = np.linalg.det(U @ Vt)
    D_sign = np.eye(len(S))
    D_sign[-1, -1] = np.sign(d)
    R = (Vt.T @ D_sign @ U.T)

    # Apply to ALL experts in Z_target
    Z_aligned = (Z_target - mu_tgt) @ R + mu_ref

    # Compute disparity on common experts
    Z_aligned_sub = Z_aligned[common_expert_ids] if common_expert_ids is not None else Z_aligned
    disparity = float(np.sqrt(np.mean(np.sum((Z_aligned_sub - Z_ref_sub) ** 2, axis=1))))

    return Z_aligned, disparity, R


# ══════════════════════════════════════════════════════════
# TASK 2: ALIGNMENT PIPELINE
# ══════════════════════════════════════════════════════════

def run_task2():
    """Build functional-space alignment pipeline.

    For each layer × q:
      1. Load oracle distance matrices from all checkpoints
      2. Run MDS embedding
      3. Align to reference (checkpoint_100)
      4. Save aligned embeddings
      5. Generate alignment quality report
    """
    if is_task_completed("task2_alignment"):
        print("[TASK 2] Already completed. Skipping.")
        return

    print("\n" + "=" * 70)
    print("TASK 2: FUNCTIONAL SPACE ALIGNMENT PIPELINE")
    print("=" * 70)
    mark_task("task2_alignment", "running")

    alignment_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "reference_checkpoint": "checkpoint_100",
        "method": "Orthogonal Procrustes (no scaling)",
        "results": {},
    }

    for layer in LAYERS:
        print(f"\n--- Layer: {layer} ---")
        common_pairs = get_common_pairs(layer)
        print(f"  Common pairs across all checkpoints: {len(common_pairs)}")

        # Get expert IDs present in common pairs
        common_experts = sorted(set(
            [p[0] for p in common_pairs] + [p[1] for p in common_pairs]
        ))
        print(f"  Common experts: {len(common_experts)}")

        alignment_report["results"][layer] = {}

        for q in Q_VALUES:
            print(f"\n  q = {q}:")
            embeddings = {}
            stresses = {}

            for ckpt_name in CHECKPOINT_ORDER:
                D = load_oracle_matrix(ckpt_name, layer)

                # Check if full or sparse
                n_valid = np.sum(~np.isnan(D)) - N_EXPERTS  # subtract diagonal
                n_valid = n_valid // 2  # upper triangle

                if n_valid >= N_EXPERTS * (N_EXPERTS - 1) // 2 * 0.9:
                    # Nearly full matrix — use standard MDS
                    D_sym = (D + D.T) / 2
                    np.fill_diagonal(D_sym, 0)
                    D_sym = np.nan_to_num(D_sym, nan=0.0)
                    Z, stress = run_mds_full(D_sym, q)
                else:
                    # Sparse — use weighted SMACOF
                    Z, stress = run_mds(D, q)

                embeddings[ckpt_name] = Z
                stresses[ckpt_name] = stress
                print(f"    {ckpt_name}: stress={stress:.4f}, "
                      f"pairs={n_valid}")

            # Align all to reference (checkpoint_100)
            ref_Z = embeddings["checkpoint_100"]
            aligned = {}
            alignment_quality = {}

            for ckpt_name in CHECKPOINT_ORDER:
                if ckpt_name == "checkpoint_100":
                    aligned[ckpt_name] = ref_Z
                    alignment_quality[ckpt_name] = {
                        "disparity": 0.0,
                        "is_reference": True,
                    }
                else:
                    Z_aligned, disparity, R = align_to_reference(
                        embeddings[ckpt_name], ref_Z,
                        common_expert_ids=common_experts
                    )
                    aligned[ckpt_name] = Z_aligned
                    alignment_quality[ckpt_name] = {
                        "disparity": disparity,
                        "is_reference": False,
                    }
                    print(f"    Aligned {ckpt_name}: disparity={disparity:.6f}")

            # Verify pairwise-distance preservation
            for ckpt_name in CHECKPOINT_ORDER:
                D_original = load_oracle_matrix(ckpt_name, layer)
                D_embed = squareform(pdist(aligned[ckpt_name]))

                # Compare on common pairs only
                orig_vals = []
                embed_vals = []
                for (i, j) in common_pairs:
                    if not np.isnan(D_original[i, j]):
                        orig_vals.append(D_original[i, j])
                        embed_vals.append(D_embed[i, j])

                if len(orig_vals) > 2:
                    rho, _ = spearmanr(orig_vals, embed_vals)
                    pr, _ = pearsonr(orig_vals, embed_vals)
                else:
                    rho, pr = 0.0, 0.0

                alignment_quality[ckpt_name]["spearman_preservation"] = float(rho)
                alignment_quality[ckpt_name]["pearson_preservation"] = float(pr)
                alignment_quality[ckpt_name]["stress"] = stresses[ckpt_name]

            # Save aligned embeddings
            for ckpt_name in CHECKPOINT_ORDER:
                out_dir = os.path.join(EMBEDDINGS_DIR, f"q{q}")
                np.save(
                    os.path.join(out_dir, f"{ckpt_name}_{layer}_aligned.npy"),
                    aligned[ckpt_name].astype(np.float32)
                )

            alignment_report["results"][layer][f"q{q}"] = {
                "common_pairs": len(common_pairs),
                "common_experts": len(common_experts),
                "checkpoints": alignment_quality,
            }

    # Save alignment report
    with open(os.path.join(RESULTS_DIR, "functional_alignment_report.json"), "w") as f:
        json.dump(alignment_report, f, indent=2)

    _generate_alignment_markdown(alignment_report)

    mark_task("task2_alignment", "completed")
    print("\n[TASK 2] ALIGNMENT PIPELINE COMPLETE")


def _generate_alignment_markdown(report):
    """Generate functional_alignment_report.md"""
    md = ["# Experiment 6B — Functional Alignment Report\n\n"]
    md.append(f"**Generated:** {report['timestamp']}\n")
    md.append(f"**Reference checkpoint:** {report['reference_checkpoint']}\n")
    md.append(f"**Alignment method:** {report['method']}\n\n")

    for layer in LAYERS:
        md.append(f"## Layer: {layer}\n\n")
        layer_data = report["results"][layer]
        for q_key, q_data in layer_data.items():
            md.append(f"### {q_key} (common pairs: {q_data['common_pairs']}, "
                      f"common experts: {q_data['common_experts']})\n\n")
            md.append("| Checkpoint | Disparity | Stress | Spearman ρ | Pearson r |\n")
            md.append("|---|---|---|---|---|\n")
            for ckpt_name in CHECKPOINT_ORDER:
                aq = q_data["checkpoints"][ckpt_name]
                ref = " (REF)" if aq.get("is_reference") else ""
                md.append(
                    f"| {ckpt_name}{ref} | {aq['disparity']:.6f} | "
                    f"{aq.get('stress', 0):.4f} | "
                    f"{aq.get('spearman_preservation', 0):.4f} | "
                    f"{aq.get('pearson_preservation', 0):.4f} |\n"
                )
            md.append("\n")

    with open(os.path.join(RESULTS_DIR, "functional_alignment_report.md"), "w") as f:
        f.write("".join(md))


# ══════════════════════════════════════════════════════════
# TASK 3 + 4: TRAJECTORIES AND DISPLACEMENT
# ══════════════════════════════════════════════════════════

def run_task3_4():
    """Compute functional trajectories and displacement vectors.

    For each layer × q × expert:
      C_i(T) = aligned MDS position
      DeltaC_i = C_i(T_b) - C_i(T_a)
      magnitude_i = ||DeltaC_i||
      direction_i = DeltaC_i / ||DeltaC_i|| (when magnitude > 0)
      displacement_rate_i = DeltaC_i / DeltaTrainingSteps
    """
    if is_task_completed("task3_4_displacement"):
        print("[TASK 3+4] Already completed. Skipping.")
        return

    print("\n" + "=" * 70)
    print("TASK 3+4: FUNCTIONAL TRAJECTORIES & DISPLACEMENT")
    print("=" * 70)
    mark_task("task3_4_displacement", "running")

    all_results = {}

    for layer in LAYERS:
        print(f"\n--- Layer: {layer} ---")
        common_pairs = get_common_pairs(layer)
        all_results[layer] = {}

        for q in Q_VALUES:
            print(f"  q = {q}:")

            # Load aligned embeddings
            embeddings = {}
            for ckpt_name in CHECKPOINT_ORDER:
                path = os.path.join(EMBEDDINGS_DIR, f"q{q}",
                                   f"{ckpt_name}_{layer}_aligned.npy")
                embeddings[ckpt_name] = np.load(path)

            # Compute trajectories and displacements
            transitions = []
            for t_idx in range(len(CHECKPOINT_ORDER) - 1):
                ckpt_a = CHECKPOINT_ORDER[t_idx]
                ckpt_b = CHECKPOINT_ORDER[t_idx + 1]
                step_a = CHECKPOINTS[ckpt_a]["actual_step"]
                step_b = CHECKPOINTS[ckpt_b]["actual_step"]
                delta_steps = step_b - step_a

                Z_a = embeddings[ckpt_a]
                Z_b = embeddings[ckpt_b]

                transition_data = {
                    "transition": f"{ckpt_a} -> {ckpt_b}",
                    "step_a": step_a,
                    "step_b": step_b,
                    "delta_steps": delta_steps,
                    "experts": {},
                }

                magnitudes = []
                for expert_id in range(N_EXPERTS):
                    C_a = Z_a[expert_id]
                    C_b = Z_b[expert_id]
                    delta_C = C_b - C_a
                    magnitude = float(np.linalg.norm(delta_C))
                    magnitudes.append(magnitude)

                    if magnitude > 1e-12:
                        direction = delta_C / magnitude
                    else:
                        direction = np.zeros(q)

                    displacement_rate = delta_C / delta_steps

                    transition_data["experts"][expert_id] = {
                        "C_a": C_a.tolist(),
                        "C_b": C_b.tolist(),
                        "delta_C": delta_C.tolist(),
                        "magnitude": magnitude,
                        "direction": direction.tolist(),
                        "displacement_rate": displacement_rate.tolist(),
                    }

                transition_data["magnitude_stats"] = {
                    "mean": float(np.mean(magnitudes)),
                    "std": float(np.std(magnitudes)),
                    "median": float(np.median(magnitudes)),
                    "min": float(np.min(magnitudes)),
                    "max": float(np.max(magnitudes)),
                    "n_zero": int(sum(1 for m in magnitudes if m < 1e-12)),
                }

                transitions.append(transition_data)
                print(f"    {ckpt_a} -> {ckpt_b}: "
                      f"mean_mag={np.mean(magnitudes):.6f}, "
                      f"max_mag={np.max(magnitudes):.6f}")

            # Save per-expert trajectories
            trajectory_data = {}
            for expert_id in range(N_EXPERTS):
                traj = []
                for ckpt_name in CHECKPOINT_ORDER:
                    traj.append({
                        "checkpoint": ckpt_name,
                        "step": CHECKPOINTS[ckpt_name]["actual_step"],
                        "position": embeddings[ckpt_name][expert_id].tolist(),
                    })
                trajectory_data[expert_id] = traj

            all_results[layer][f"q{q}"] = {
                "transitions": transitions,
                "trajectories": trajectory_data,
            }

            # Save displacement vectors as numpy
            for t_idx, transition in enumerate(transitions):
                ckpt_a = CHECKPOINT_ORDER[t_idx]
                ckpt_b = CHECKPOINT_ORDER[t_idx + 1]
                delta_C_matrix = np.array([
                    transition["experts"][i]["delta_C"]
                    for i in range(N_EXPERTS)
                ])
                np.save(
                    os.path.join(EMBEDDINGS_DIR, f"q{q}",
                                f"deltaC_{layer}_{ckpt_a}_to_{ckpt_b}.npy"),
                    delta_C_matrix.astype(np.float32)
                )

    # Save full results
    # (trajectories are large; save summary metrics separately)
    summary = {}
    for layer in LAYERS:
        summary[layer] = {}
        for q_key, q_data in all_results[layer].items():
            summary[layer][q_key] = {
                "transitions": [
                    {
                        "transition": t["transition"],
                        "delta_steps": t["delta_steps"],
                        "magnitude_stats": t["magnitude_stats"],
                    }
                    for t in q_data["transitions"]
                ]
            }

    with open(os.path.join(METRICS_DIR, "displacement_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _generate_trajectory_markdown(summary)

    mark_task("task3_4_displacement", "completed")
    print("\n[TASK 3+4] COMPLETE")


def _generate_trajectory_markdown(summary):
    """Generate checkpoint_trajectory_analysis.md"""
    md = ["# Experiment 6B — Checkpoint Trajectory Analysis\n\n"]
    md.append("## Functional Displacement Magnitude Summary\n\n")

    for layer in LAYERS:
        md.append(f"### Layer: {layer}\n\n")
        for q_key, q_data in summary[layer].items():
            md.append(f"#### {q_key}\n\n")
            md.append("| Transition | ΔSteps | Mean |Δ| | Std |Δ| | Max |Δ| | Zero-motion |\n")
            md.append("|---|---|---|---|---|---|\n")
            for t in q_data["transitions"]:
                ms = t["magnitude_stats"]
                md.append(
                    f"| {t['transition']} | {t['delta_steps']:,} | "
                    f"{ms['mean']:.6f} | {ms['std']:.6f} | "
                    f"{ms['max']:.6f} | {ms['n_zero']} |\n"
                )
            md.append("\n")

    with open(os.path.join(RESULTS_DIR, "checkpoint_trajectory_analysis.md"), "w") as f:
        f.write("".join(md))


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    ensure_dirs()
    run_task2()
    run_task3_4()
