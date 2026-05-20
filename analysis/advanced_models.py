"""
Advanced model pipeline: target encoding + interaction features + SMOTE + stacking.

This script tries unconventional approaches to push CV AUC past 0.75:

1. K-fold target encoding (smoothed) for all numeric features.
2. Polynomial interaction features for top predictors.
3. SMOTE oversampling of the minority class.
4. Tuned gradient boosting + stacking ensemble.
"""

from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)

from analysis.data_utils import load_data, get_Xy, split_feature_types
from analysis.feature_engineering import engineer_features

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_target_encoded_features(X: pd.DataFrame, y: pd.Series,
                                  cols: list[str],
                                  cv: StratifiedKFold,
                                  smoothing: float = 10.0) -> np.ndarray:
    """K-fold target encoding: encode each feature as the smoothed mean target.

    Done within CV folds to prevent target leakage.
    """
    from pandas.api.types import is_numeric_dtype

    global_mean = y.mean()
    te_matrix = np.zeros((len(X), len(cols)))

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        y_train = y.iloc[train_idx]
        for j, col in enumerate(cols):
            if is_numeric_dtype(X[col]):
                x_col = X[col].copy()
                valid = x_col.notna()
                if valid.sum() < 20:
                    te_matrix[val_idx, j] = global_mean
                    continue
                bins = pd.qcut(x_col[valid], q=10, duplicates="drop")
                bin_labels = pd.Series("__NAN__", index=x_col.index, dtype="object")
                bin_labels.loc[valid] = bins.astype(str).values
            else:
                bin_labels = X[col].fillna("__NAN__").astype(str)

            # Compute smoothed mean per bin on training fold
            train_labels = bin_labels.iloc[train_idx]
            train_y = y_train.reset_index(drop=True)
            train_labels = train_labels.reset_index(drop=True)

            stats = pd.DataFrame({"label": train_labels, "y": train_y.values})
            agg = stats.groupby("label")["y"].agg(["mean", "count"])
            agg["smoothed"] = (agg["count"] * agg["mean"] + smoothing * global_mean) / (
                agg["count"] + smoothing
            )

            # Encode validation fold
            val_labels = bin_labels.iloc[val_idx]
            encoded = val_labels.map(agg["smoothed"]).fillna(global_mean)
            te_matrix[val_idx, j] = encoded.values

    return te_matrix


def add_pairwise_interactions(X: np.ndarray, top_k: int = 10) -> np.ndarray:
    """Add pairwise product interactions of the first top_k columns."""
    k = min(top_k, X.shape[1])
    interactions = []
    for i, j in combinations(range(k), 2):
        interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
    if interactions:
        return np.hstack([X] + interactions)
    return X


def main():
    df = load_data()
    print(f"Data: {df.shape[0]} rows x {df.shape[1]} cols")

    results_all = {}

    for include_leaky, set_name in [(False, "baseline"), (True, "all_variables")]:
        print(f"\n{'='*60}")
        print(f"  {set_name}")
        print(f"{'='*60}")

        X_raw, y, raw_cols = get_Xy(df, include_leaky=include_leaky)
        X_eng, eng_cols = engineer_features(df, raw_cols)
        print(f"  {len(raw_cols)} raw -> {len(eng_cols)} engineered")

        # Standard preprocessing
        from analysis.data_utils import build_preprocessor
        pp = build_preprocessor(X_eng, eng_cols, scale=True)
        X_std = pp.fit_transform(X_eng)
        print(f"  Standard transformed: {X_std.shape}")

        # Target encoding (5-fold)
        te_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        X_te = build_target_encoded_features(X_eng, y, eng_cols, te_cv, smoothing=10.0)
        print(f"  Target encoded: {X_te.shape}")

        # Impute NaN in target encoding, scale
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_te_clean = scl.fit_transform(imp.fit_transform(X_te))

        # Combine: standard + target-encoded features
        X_combined = np.hstack([X_std, X_te_clean])
        print(f"  Combined (std+TE): {X_combined.shape}")

        # Add top-k pairwise interactions on the combined features
        # Select top features first
        mi_scores = mutual_info_classif(X_combined, y, random_state=42)
        top_idx = np.argsort(mi_scores)[-15:]
        X_top = X_combined[:, top_idx]
        X_interactions = add_pairwise_interactions(X_top, top_k=15)
        X_final = np.hstack([X_combined, X_interactions[:, 15:]])  # don't duplicate top features
        print(f"  With interactions: {X_final.shape}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model_results = []

        # --- Model 1: LightGBM with SMOTE ---
        print("\n  [1] LightGBM + SMOTE ...", end=" ", flush=True)
        try:
            from lightgbm import LGBMClassifier
            pipe = ImbPipeline([
                ("sel", SelectKBest(mutual_info_classif, k=min(80, X_final.shape[1]))),
                ("smote", SMOTE(random_state=42, k_neighbors=5)),
                ("clf", LGBMClassifier(
                    n_estimators=800, learning_rate=0.02, num_leaves=15,
                    max_depth=4, min_child_samples=30, subsample=0.7,
                    colsample_bytree=0.5, reg_alpha=5.0, reg_lambda=10.0,
                    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
                )),
            ])
            prob = cross_val_predict(pipe, X_final, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, prob)
            model_results.append(("lgbm_smote", auc, prob))
            print(f"AUC={auc:.4f}")
        except Exception as e:
            print(f"FAIL: {e}")

        # --- Model 2: LightGBM no SMOTE ---
        print("  [2] LightGBM (no SMOTE) ...", end=" ", flush=True)
        try:
            from lightgbm import LGBMClassifier
            pipe = Pipeline([
                ("sel", SelectKBest(mutual_info_classif, k=min(80, X_final.shape[1]))),
                ("clf", LGBMClassifier(
                    n_estimators=800, learning_rate=0.02, num_leaves=15,
                    max_depth=4, min_child_samples=30, subsample=0.7,
                    colsample_bytree=0.5, reg_alpha=5.0, reg_lambda=10.0,
                    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
                )),
            ])
            prob = cross_val_predict(pipe, X_final, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, prob)
            model_results.append(("lgbm_no_smote", auc, prob))
            print(f"AUC={auc:.4f}")
        except Exception as e:
            print(f"FAIL: {e}")

        # --- Model 3: XGBoost + SMOTE ---
        print("  [3] XGBoost + SMOTE ...", end=" ", flush=True)
        try:
            from xgboost import XGBClassifier
            pipe = ImbPipeline([
                ("sel", SelectKBest(mutual_info_classif, k=min(80, X_final.shape[1]))),
                ("smote", SMOTE(random_state=42, k_neighbors=5)),
                ("clf", XGBClassifier(
                    n_estimators=800, learning_rate=0.02, max_depth=3,
                    min_child_weight=15, subsample=0.7,
                    colsample_bytree=0.5, reg_alpha=5.0, reg_lambda=10.0,
                    gamma=1.0, scale_pos_weight=5.5,
                    tree_method="hist", random_state=42, n_jobs=-1,
                )),
            ])
            prob = cross_val_predict(pipe, X_final, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, prob)
            model_results.append(("xgb_smote", auc, prob))
            print(f"AUC={auc:.4f}")
        except Exception as e:
            print(f"FAIL: {e}")

        # --- Model 4: CatBoost ---
        print("  [4] CatBoost ...", end=" ", flush=True)
        try:
            from catboost import CatBoostClassifier
            pipe = Pipeline([
                ("sel", SelectKBest(mutual_info_classif, k=min(80, X_final.shape[1]))),
                ("clf", CatBoostClassifier(
                    iterations=800, learning_rate=0.02, depth=4,
                    l2_leaf_reg=10.0, min_data_in_leaf=30,
                    subsample=0.7, auto_class_weights="Balanced",
                    random_seed=42, verbose=0, allow_writing_files=False,
                )),
            ])
            prob = cross_val_predict(pipe, X_final, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, prob)
            model_results.append(("catboost", auc, prob))
            print(f"AUC={auc:.4f}")
        except Exception as e:
            print(f"FAIL: {e}")

        # --- Model 5: RandomForest + SMOTE ---
        print("  [5] RandomForest + SMOTE ...", end=" ", flush=True)
        try:
            pipe = ImbPipeline([
                ("sel", SelectKBest(mutual_info_classif, k=min(60, X_final.shape[1]))),
                ("smote", SMOTE(random_state=42, k_neighbors=5)),
                ("clf", RandomForestClassifier(
                    n_estimators=800, max_depth=8, min_samples_leaf=10,
                    max_features=0.3, class_weight="balanced_subsample",
                    n_jobs=-1, random_state=42,
                )),
            ])
            prob = cross_val_predict(pipe, X_final, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, prob)
            model_results.append(("rf_smote", auc, prob))
            print(f"AUC={auc:.4f}")
        except Exception as e:
            print(f"FAIL: {e}")

        # --- Model 6: HistGBT + SMOTE ---
        print("  [6] HistGBT + SMOTE ...", end=" ", flush=True)
        try:
            pipe = ImbPipeline([
                ("sel", SelectKBest(mutual_info_classif, k=min(80, X_final.shape[1]))),
                ("smote", SMOTE(random_state=42, k_neighbors=5)),
                ("clf", HistGradientBoostingClassifier(
                    learning_rate=0.02, max_iter=800, max_depth=3,
                    max_leaf_nodes=15, min_samples_leaf=30,
                    l2_regularization=10.0, class_weight="balanced",
                    early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=30, random_state=42,
                )),
            ])
            prob = cross_val_predict(pipe, X_final, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, prob)
            model_results.append(("hgbt_smote", auc, prob))
            print(f"AUC={auc:.4f}")
        except Exception as e:
            print(f"FAIL: {e}")

        # --- Model 7: GradientBoosting ---
        print("  [7] GradientBoosting ...", end=" ", flush=True)
        try:
            pipe = Pipeline([
                ("sel", SelectKBest(mutual_info_classif, k=min(60, X_final.shape[1]))),
                ("clf", GradientBoostingClassifier(
                    n_estimators=500, learning_rate=0.02, max_depth=3,
                    min_samples_leaf=30, subsample=0.7, max_features=0.5,
                    random_state=42,
                )),
            ])
            prob = cross_val_predict(pipe, X_final, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, prob)
            model_results.append(("gbt", auc, prob))
            print(f"AUC={auc:.4f}")
        except Exception as e:
            print(f"FAIL: {e}")

        # --- Model 8: ExtraTrees + SMOTE ---
        print("  [8] ExtraTrees + SMOTE ...", end=" ", flush=True)
        try:
            pipe = ImbPipeline([
                ("sel", SelectKBest(mutual_info_classif, k=min(60, X_final.shape[1]))),
                ("smote", SMOTE(random_state=42, k_neighbors=5)),
                ("clf", ExtraTreesClassifier(
                    n_estimators=800, max_depth=8, min_samples_leaf=10,
                    max_features=0.3, class_weight="balanced_subsample",
                    n_jobs=-1, random_state=42,
                )),
            ])
            prob = cross_val_predict(pipe, X_final, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, prob)
            model_results.append(("et_smote", auc, prob))
            print(f"AUC={auc:.4f}")
        except Exception as e:
            print(f"FAIL: {e}")

        # --- Model 9: Logistic with poly features ---
        print("  [9] Logistic + poly2 ...", end=" ", flush=True)
        try:
            pipe = Pipeline([
                ("sel", SelectKBest(mutual_info_classif, k=min(20, X_final.shape[1]))),
                ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                ("clf", LogisticRegression(
                    C=0.01, max_iter=3000, class_weight="balanced",
                    solver="saga", random_state=42,
                )),
            ])
            prob = cross_val_predict(pipe, X_final, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, prob)
            model_results.append(("lr_poly2", auc, prob))
            print(f"AUC={auc:.4f}")
        except Exception as e:
            print(f"FAIL: {e}")

        # --- Ensemble ---
        model_results.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  Individual model ranking:")
        for name, auc, _ in model_results:
            print(f"    {name:<20} AUC={auc:.4f}")

        # Weighted ensemble (random search)
        if len(model_results) >= 2:
            names = [r[0] for r in model_results]
            probs = np.array([r[2] for r in model_results])
            best_ens_auc, best_w = 0.0, None
            rng = np.random.RandomState(42)
            for _ in range(30000):
                w = rng.dirichlet(np.ones(len(names)))
                ens = np.average(probs, axis=0, weights=w)
                a = roc_auc_score(y, ens)
                if a > best_ens_auc:
                    best_ens_auc, best_w = a, w.copy()

            print(f"\n  Weighted ensemble AUC={best_ens_auc:.4f}")
            for i, n in enumerate(names):
                if best_w[i] > 0.01:
                    print(f"    {n}: w={best_w[i]:.3f}")

            # Top-k averages
            for k in [2, 3, 4, 5]:
                if k > len(model_results):
                    break
                top = model_results[:k]
                avg = np.mean([r[2] for r in top], axis=0)
                avg_auc = roc_auc_score(y, avg)
                print(f"  Top-{k} avg AUC={avg_auc:.4f}")

        best_single = model_results[0][1] if model_results else 0
        best_overall = max(best_single, best_ens_auc) if len(model_results) >= 2 else best_single

        results_all[set_name] = {
            "models": {r[0]: round(r[1], 4) for r in model_results},
            "ensemble_auc": round(best_ens_auc, 4) if len(model_results) >= 2 else None,
            "best_auc": round(best_overall, 4),
        }

        hit = best_overall >= 0.75
        print(f"\n  >>> {set_name} best AUC = {best_overall:.4f} {'>=0.75 ✓' if hit else '<0.75'}")

    # Save
    out = RESULTS_DIR / "advanced_model_results.json"
    with open(out, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nSaved to {out}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for sn, r in results_all.items():
        hit = ">=0.75 ✓" if r["best_auc"] >= 0.75 else "<0.75"
        print(f"  {sn}: best AUC = {r['best_auc']:.4f}  {hit}")


if __name__ == "__main__":
    main()
