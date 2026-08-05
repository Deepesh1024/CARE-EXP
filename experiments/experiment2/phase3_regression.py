"""
CARE-MoE Experiment 2 — Phase 3: Combined CARE Model
=======================================================
Train exactly the same 4 model families as Experiment 1.5 on
the augmented feature set (original 7 + 4 new descriptors).

Models: LinearRegression, Ridge, LASSO, XGBoost
Variants:
  A : 11 features (original + new)
  B : 12 features (A + Relative_Depth)
  C : 23 features (B + 11 interaction terms)

Identical hyperparameters as Experiment 1.5.

Produces:
  results/exp2/models/{model}_{variant}.pkl
  results/exp2/metrics.json
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    _TREE_ENGINE = "XGBoost"
except ImportError:
    try:
        from lightgbm import LGBMRegressor as XGBRegressor
        _TREE_ENGINE = "LightGBM"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor as XGBRegressor
        _TREE_ENGINE = "HistGradientBoosting"

from config import (
    ALL_FEATURES,
    TARGET,
    LAYER_DEPTH_MAP,
    TRAIN_PARQUET,
    TEST_PARQUET,
    MODELS_DIR,
    METRICS_PATH,
    RANDOM_SEED,
    RIDGE_ALPHA,
    LASSO_ALPHA,
    LASSO_MAX_ITER,
    XGBOOST_PARAMS,
)
from utils import (
    set_global_seed,
    ensure_dirs,
    save_pickle,
    save_json,
)


# ──────────────────────────────────────────────
# Feature Variant Construction
# ──────────────────────────────────────────────
def build_feature_variants(df: pd.DataFrame):
    """Build three feature sets from the augmented DataFrame.

    Returns dict of {"A": np.ndarray, "B": np.ndarray, "C": np.ndarray}
    and a parallel dict of column-name lists.
    """
    # Variant A: All 11 features (original + new)
    cols_a = list(ALL_FEATURES)
    X_a = df[cols_a].values

    # Variant B: A + Relative_Depth
    if "Relative_Depth" not in df.columns:
        df = df.copy()
        df["Relative_Depth"] = df["Layer"].map(LAYER_DEPTH_MAP)
    cols_b = cols_a + ["Relative_Depth"]
    X_b = df[cols_b].values

    # Variant C: B + interaction terms (feature × depth)
    depth = df["Relative_Depth"].values.reshape(-1, 1)
    interaction_cols = [f"{f}_x_depth" for f in ALL_FEATURES]
    interactions = df[ALL_FEATURES].values * depth
    X_c = np.hstack([X_b, interactions])
    cols_c = cols_b + interaction_cols

    variants = {"A": X_a, "B": X_b, "C": X_c}
    col_names = {"A": cols_a, "B": cols_b, "C": cols_c}

    for k, v in variants.items():
        print(f"[Phase 3] Variant {k}: {v.shape[1]} features, {v.shape[0]} samples")

    return variants, col_names


# ──────────────────────────────────────────────
# Model Factories
# ──────────────────────────────────────────────
def _build_models():
    """Return an ordered dict of (name, estimator) pairs."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED),
        "LASSO": Lasso(
            alpha=LASSO_ALPHA,
            max_iter=LASSO_MAX_ITER,
            random_state=RANDOM_SEED,
        ),
        _TREE_ENGINE: XGBRegressor(**XGBOOST_PARAMS),
    }


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all five evaluation metrics."""
    sp_rho, _ = spearmanr(y_true, y_pred)
    pe_r, _ = pearsonr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {
        "Spearman": float(sp_rho),
        "Pearson": float(pe_r),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


# ──────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────
def train_and_evaluate(train_variants, test_variants,
                        y_train, y_test, col_names):
    """Train every model × variant combination."""
    all_metrics = []
    trained_models = {}
    predictions = {}

    for variant in ("A", "B", "C"):
        X_tr = train_variants[variant]
        X_te = test_variants[variant]
        models = _build_models()

        for name, model in models.items():
            key = f"{name}_{variant}"
            print(f"  Training {key} ...", end=" ", flush=True)

            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            metrics = evaluate(y_test, y_pred)
            metrics["Model"] = name
            metrics["Variant"] = variant
            metrics["N_Features"] = X_tr.shape[1]

            all_metrics.append(metrics)
            trained_models[key] = model
            predictions[key] = y_pred

            print(f"Spearman={metrics['Spearman']:+.4f}  R²={metrics['R2']:.4f}")

    return all_metrics, trained_models, predictions


# ──────────────────────────────────────────────
# Linearization Gap
# ──────────────────────────────────────────────
def compute_linearization_gap(all_metrics: list) -> dict:
    """Compute Δ = ρ(best tree) - ρ(best linear)."""
    linear_names = {"LinearRegression", "Ridge", "LASSO"}

    linear_results = [m for m in all_metrics if m["Model"] in linear_names]
    tree_results = [m for m in all_metrics if m["Model"] not in linear_names]

    best_linear = max(linear_results, key=lambda m: m["Spearman"])
    best_tree = max(tree_results, key=lambda m: m["Spearman"])

    gap = best_tree["Spearman"] - best_linear["Spearman"]

    summary = {
        "best_linear_model": f"{best_linear['Model']}_{best_linear['Variant']}",
        "best_linear_spearman": best_linear["Spearman"],
        "best_linear_r2": best_linear["R2"],
        "best_tree_model": f"{best_tree['Model']}_{best_tree['Variant']}",
        "best_tree_spearman": best_tree["Spearman"],
        "best_tree_r2": best_tree["R2"],
        "linearization_gap": gap,
    }

    print("\n" + "=" * 60)
    print("LINEARIZATION GAP (Experiment 2)")
    print("=" * 60)
    print(f"  Best Linear  : {summary['best_linear_model']}  "
          f"ρ = {summary['best_linear_spearman']:+.4f}  "
          f"R² = {summary['best_linear_r2']:.4f}")
    print(f"  Best Tree    : {summary['best_tree_model']}  "
          f"ρ = {summary['best_tree_spearman']:+.4f}  "
          f"R² = {summary['best_tree_r2']:.4f}")
    print(f"  Δ_gap        : {gap:+.4f}")
    print("=" * 60)

    return summary


def main():
    set_global_seed()
    ensure_dirs()

    print("=" * 70)
    print("PHASE 3 — COMBINED CARE MODEL")
    print("=" * 70)
    print(f"[Phase 3] Tree engine: {_TREE_ENGINE}")

    # Load Phase 1 artifacts
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    y_train = train_df[TARGET].values
    y_test = test_df[TARGET].values

    print(f"[Phase 3] Loaded train={len(train_df):,}  test={len(test_df):,}")

    # Build feature variants
    train_variants, train_cols = build_feature_variants(train_df)
    test_variants, test_cols = build_feature_variants(test_df)

    # Train and evaluate
    all_metrics, trained_models, predictions = train_and_evaluate(
        train_variants, test_variants, y_train, y_test, train_cols
    )

    # Linearization Gap
    gap_summary = compute_linearization_gap(all_metrics)

    # Performance table
    perf_df = pd.DataFrame(all_metrics)
    perf_df = perf_df[["Model", "Variant", "N_Features", "Spearman",
                        "Pearson", "MAE", "RMSE", "R2"]]
    perf_table = perf_df.sort_values("Spearman", ascending=False)
    print("\n" + perf_table.to_markdown(index=False, floatfmt=".4f"))

    # Save models
    import os
    for key, model in trained_models.items():
        path = os.path.join(MODELS_DIR, f"{key}.pkl")
        save_pickle(model, path)

    # Save metrics
    pred_arrays = {k: v.tolist() for k, v in predictions.items()}
    output = {
        "performance": all_metrics,
        "linearization_gap": gap_summary,
        "predictions": pred_arrays,
        "y_test": y_test.tolist(),
    }
    save_json(output, METRICS_PATH)

    print(f"\n[Phase 3] All artifacts saved.")
    return all_metrics, gap_summary, trained_models, predictions


if __name__ == "__main__":
    main()
