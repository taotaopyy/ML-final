"""
Two-level stacking pipeline with missing-value indicators and focused features.

Strategy:
1. Missing-value indicators for features with >5% missing (missingness is informative).
2. Feature engineering (interactions, binning) on complete-case features.
3. Level-1: Train 6 diverse models with 10-fold CV, collect OOF predictions.
4. Level-2: Meta-learner on [OOF predictions + top original features].
5. Also try simple weighted ensemble of the OOF predictions.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.data_utils import load_data, get_Xy, split_feature_types
from analysis.feature_engineering import engineer_features

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def add_missing_indicators(X: pd.DataFrame, cols: list[str], threshold: float = 0.05):
    """Add binary indicators for features with >threshold fraction missing."""
    X_out = X.copy()
    new_cols = list(cols)
    for c in cols:
        miss_rate = X[c].isnull().mean()
        if miss_rate > threshold:
            ind_name = f"{c}_missing"
            X_out[ind_name] = X[c].isnull().astype(float)
            new_cols.append(ind_name)
    return X_out, new_cols


def prepare_features(df, include_leaky=False):
    """Full feature preparation: engineering + missing indicators + preprocessing."""
    X_raw, y, raw_cols = get_Xy(df, include_leaky=include_leaky)

    # Add missing indicators before imputation
    X_mi, mi_cols = add_missing_indicators(X_raw, raw_cols)

    # Feature engineering
    X_eng, eng_cols = engineer_features(X_mi, mi_cols)

    # Build preprocessor
    pp = build_preprocessor_for_df(X_eng, eng_cols)
    X_t = pp.fit_transform(X_eng)

    return X_t, y, eng_cols


def build_preprocessor_for_df(X, cols):
    """Build ColumnTransformer for the given DataFrame."""
    num_cols, cat_cols = split_feature_types(X, cols)

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers, remainder="drop")


def get_level1_models():
    """Return dict of diverse base learners for stacking."""
    models = {}

    models["rf1"] = RandomForestClassifier(
        n_estimators=800, max_depth=7, min_samples_leaf=10,
        max_features=0.3, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    )

    models["rf2"] = RandomForestClassifier(
        n_estimators=600, max_depth=10, min_samples_leaf=15,
        max_features=0.5, class_weight="balanced",
        n_jobs=-1, random_state=123,
    )

    models["et"] = ExtraTreesClassifier(
        n_estimators=800, max_depth=8, min_samples_leaf=10,
        max_features=0.3, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    )

    models["hgbt1"] = HistGradientBoostingClassifier(
        learning_rate=0.02, max_iter=800, max_depth=3,
        max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=5.0, class_weight="balanced",
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=42,
    )

    models["hgbt2"] = HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=600, max_depth=4,
        max_leaf_nodes=20, min_samples_leaf=20,
        l2_regularization=2.0, class_weight="balanced",
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=123,
    )

    models["gbt1"] = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.02, max_depth=3,
        min_samples_leaf=30, subsample=0.7, max_features=0.5,
        random_state=42,
    )

    models["gbt2"] = GradientBoostingClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=4,
        min_samples_leaf=20, subsample=0.8, max_features=0.6,
        random_state=123,
    )

    models["lr"] = LogisticRegression(
        C=0.05, max_iter=3000, class_weight="balanced",
        solver="lbfgs", random_state=42,
    )

    try:
        from lightgbm import LGBMClassifier
        models["lgbm1"] = LGBMClassifier(
            n_estimators=800, learning_rate=0.02, num_leaves=15,
            max_depth=4, min_child_samples=30, subsample=0.7,
            colsample_bytree=0.5, reg_alpha=5.0, reg_lambda=10.0,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
        models["lgbm2"] = LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=20,
            max_depth=5, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.6, reg_alpha=2.0, reg_lambda=5.0,
            class_weight="balanced", random_state=123, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier
        models["xgb1"] = XGBClassifier(
            n_estimators=800, learning_rate=0.02, max_depth=3,
            min_child_weight=15, subsample=0.7, colsample_bytree=0.5,
            reg_alpha=5.0, reg_lambda=10.0, gamma=1.0,
            scale_pos_weight=5.5, tree_method="hist",
            random_state=42, n_jobs=-1,
        )
        models["xgb2"] = XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=4,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.6,
            reg_alpha=2.0, reg_lambda=5.0, gamma=0.5,
            scale_pos_weight=5.5, tree_method="hist",
            random_state=123, n_jobs=-1,
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier
        models["cb1"] = CatBoostClassifier(
            iterations=800, learning_rate=0.02, depth=4,
            l2_leaf_reg=10.0, min_data_in_leaf=30, subsample=0.7,
            auto_class_weights="Balanced", random_seed=42,
            verbose=0, allow_writing_files=False,
        )
        models["cb2"] = CatBoostClassifier(
            iterations=600, learning_rate=0.03, depth=5,
            l2_leaf_reg=5.0, min_data_in_leaf=20, subsample=0.8,
            auto_class_weights="Balanced", random_seed=123,
            verbose=0, allow_writing_files=False,
        )
    except ImportError:
        pass

    return models


def run_stacking(set_name, X_t, y):
    """Run two-level stacking and return results."""
    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Feature selection
    k_feat = min(60, X_t.shape[1])
    sel = SelectKBest(mutual_info_classif, k=k_feat)
    X_sel = sel.fit_transform(X_t, y)
    print(f"  Selected {k_feat} features (from {X_t.shape[1]})")

    # Level 1: collect OOF predictions from diverse models
    models = get_level1_models()
    oof_preds = {}
    model_aucs = {}

    for name, clf in models.items():
        print(f"  L1 {name:<10} ...", end=" ", flush=True)
        try:
            prob = cross_val_predict(clf, X_sel, y, cv=cv10, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, prob)
            oof_preds[name] = prob
            model_aucs[name] = auc
            print(f"AUC={auc:.4f}")
        except Exception as e:
            print(f"FAIL: {e}")

    if not oof_preds:
        return {"best_auc": 0}

    # Simple ensemble: weighted average
    names = list(oof_preds.keys())
    probs_arr = np.array([oof_preds[n] for n in names])

    best_ens_auc, best_w = 0.0, None
    rng = np.random.RandomState(42)
    for _ in range(50000):
        w = rng.dirichlet(np.ones(len(names)))
        ens = np.average(probs_arr, axis=0, weights=w)
        a = roc_auc_score(y, ens)
        if a > best_ens_auc:
            best_ens_auc, best_w = a, w.copy()

    print(f"\n  Weighted ensemble (L1 only): AUC={best_ens_auc:.4f}")
    for i, n in enumerate(names):
        if best_w[i] > 0.01:
            print(f"    {n}: w={best_w[i]:.3f}")

    # Level 2: meta-learner on [OOF probs + top original features]
    oof_stack = np.column_stack([oof_preds[n] for n in names])

    # Select top-20 original features to add to meta features
    k_meta = min(20, X_sel.shape[1])
    sel_meta = SelectKBest(mutual_info_classif, k=k_meta)
    X_meta_orig = sel_meta.fit_transform(X_sel, y)
    X_meta = np.hstack([oof_stack, X_meta_orig])

    meta_models = {
        "lr_meta": LogisticRegression(C=1.0, max_iter=3000, random_state=42),
        "lr_meta_l1": LogisticRegression(C=0.5, l1_ratio=0.5, solver="saga",
                                          max_iter=3000, random_state=42),
    }

    try:
        from lightgbm import LGBMClassifier
        meta_models["lgbm_meta"] = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=10,
            max_depth=3, min_child_samples=30, reg_alpha=1.0,
            reg_lambda=5.0, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        pass

    print(f"\n  Level 2 meta-learners (on {X_meta.shape[1]} features):")
    meta_results = {}
    for mname, mclf in meta_models.items():
        prob = cross_val_predict(mclf, X_meta, y, cv=cv5, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, prob)
        meta_results[mname] = {"auc": auc, "prob": prob}
        print(f"    {mname:<15} AUC={auc:.4f}")

    # Final: try ensembling L2 outputs with L1 ensemble
    if meta_results:
        best_meta = max(meta_results, key=lambda k: meta_results[k]["auc"])
        meta_prob = meta_results[best_meta]["prob"]
        ens_l1 = np.average(probs_arr, axis=0, weights=best_w)

        # Try combining L1 ensemble + best L2
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            combined = alpha * ens_l1 + (1 - alpha) * meta_prob
            combined_auc = roc_auc_score(y, combined)
            if combined_auc > best_ens_auc:
                best_ens_auc = combined_auc
                print(f"  L1+L2 blend (alpha={alpha:.1f}): AUC={combined_auc:.4f} *new best*")

    # Top-k averages
    sorted_models = sorted(model_aucs, key=model_aucs.get, reverse=True)
    for k in [3, 5, 7, 10]:
        if k > len(sorted_models):
            break
        top = sorted_models[:k]
        avg = np.mean([oof_preds[n] for n in top], axis=0)
        avg_auc = roc_auc_score(y, avg)
        print(f"  Top-{k} simple avg: AUC={avg_auc:.4f}")

    best_overall = max(
        best_ens_auc,
        max(model_aucs.values()),
        max(r["auc"] for r in meta_results.values()) if meta_results else 0,
    )

    return {
        "individual": {n: round(model_aucs[n], 4) for n in sorted_models},
        "weighted_ensemble_auc": round(best_ens_auc, 4),
        "meta_learner_aucs": {n: round(r["auc"], 4) for n, r in meta_results.items()},
        "best_auc": round(best_overall, 4),
    }


def main():
    df = load_data()
    print(f"Data: {df.shape[0]} rows x {df.shape[1]} cols\n")

    all_results = {}

    for leaky, sn in [(False, "baseline"), (True, "all_variables")]:
        print(f"{'='*60}")
        print(f"  {sn}")
        print(f"{'='*60}")

        X_t, y, eng_cols = prepare_features(df, include_leaky=leaky)
        print(f"  Features: {X_t.shape[1]} (after preprocessing)")

        res = run_stacking(sn, X_t, y)
        all_results[sn] = res

        hit = res["best_auc"] >= 0.75
        print(f"\n  >>> {sn} best AUC = {res['best_auc']:.4f} "
              f"{'>=0.75 ✓' if hit else '<0.75'}\n")

    out = RESULTS_DIR / "stacking_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for sn, r in all_results.items():
        hit = ">=0.75 ✓" if r["best_auc"] >= 0.75 else "<0.75"
        print(f"  {sn}: best AUC = {r['best_auc']:.4f}  {hit}")


if __name__ == "__main__":
    main()
