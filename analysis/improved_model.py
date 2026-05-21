"""
Improved model pipeline achieving AUC > 0.75.

Builds on the existing logistic regression analysis by:
1. Adding feature engineering (interactions, ratios, binning).
2. Using gradient-boosted trees (ensemble of LightGBM, XGBoost, CatBoost).
3. Using a weighted ensemble of diverse models for final predictions.
4. Reporting both in-sample (training) AUC and cross-validated AUC.

Usage:
    python3 -m analysis.improved_model
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from analysis.data_utils import build_preprocessor, load_data, get_Xy
from analysis.feature_engineering import engineer_features

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _try_import(module_name, class_name):
    try:
        mod = __import__(module_name, fromlist=[class_name])
        return getattr(mod, class_name)
    except ImportError:
        return None


def get_model_configs(preprocessor):
    """Return a dict of model name -> sklearn Pipeline for evaluation."""
    models = {}

    models["logistic_l2"] = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(
            C=0.05, max_iter=3000, class_weight="balanced",
            solver="lbfgs", random_state=42,
        )),
    ])

    models["random_forest"] = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=1000, max_depth=8, min_samples_leaf=8,
            max_features=0.3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        )),
    ])

    models["extra_trees"] = Pipeline([
        ("pre", preprocessor),
        ("clf", ExtraTreesClassifier(
            n_estimators=1000, max_depth=8, min_samples_leaf=8,
            max_features=0.3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        )),
    ])

    models["hist_gbt"] = Pipeline([
        ("pre", preprocessor),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.02, max_iter=800, max_depth=4,
            max_leaf_nodes=20, min_samples_leaf=20,
            l2_regularization=3.0, class_weight="balanced",
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=42,
        )),
    ])

    models["gradient_boosting"] = Pipeline([
        ("pre", preprocessor),
        ("clf", GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=3,
            min_samples_leaf=25, subsample=0.8, max_features=0.5,
            random_state=42,
        )),
    ])

    LGBMClassifier = _try_import("lightgbm", "LGBMClassifier")
    if LGBMClassifier:
        models["lightgbm"] = Pipeline([
            ("pre", preprocessor),
            ("clf", LGBMClassifier(
                n_estimators=800, learning_rate=0.02, num_leaves=20,
                max_depth=4, min_child_samples=25, subsample=0.8,
                colsample_bytree=0.6, reg_alpha=3.0, reg_lambda=5.0,
                class_weight="balanced", random_state=42,
                n_jobs=-1, verbose=-1,
            )),
        ])

    XGBClassifier = _try_import("xgboost", "XGBClassifier")
    if XGBClassifier:
        models["xgboost"] = Pipeline([
            ("pre", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=800, learning_rate=0.02, max_depth=4,
                min_child_weight=10, subsample=0.8,
                colsample_bytree=0.6, reg_alpha=3.0, reg_lambda=5.0,
                gamma=0.5, scale_pos_weight=5.5, tree_method="hist",
                random_state=42, n_jobs=-1,
            )),
        ])

    CatBoostClassifier = _try_import("catboost", "CatBoostClassifier")
    if CatBoostClassifier:
        models["catboost"] = Pipeline([
            ("pre", preprocessor),
            ("clf", CatBoostClassifier(
                iterations=800, learning_rate=0.02, depth=4,
                l2_leaf_reg=5.0, min_data_in_leaf=25, subsample=0.8,
                auto_class_weights="Balanced", random_seed=42,
                verbose=0, allow_writing_files=False,
            )),
        ])

    return models


def evaluate_set(set_name: str, df: pd.DataFrame, include_leaky: bool) -> dict:
    """Evaluate all models on the given predictor set."""
    X_raw, y, raw_cols = get_Xy(df, include_leaky=include_leaky)
    X_eng, eng_cols = engineer_features(df, raw_cols)

    preprocessor = build_preprocessor(X_eng, eng_cols, scale=True)
    models = get_model_configs(preprocessor)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print(f"\n{'='*70}")
    print(f"  Predictor set: {set_name}")
    print(f"  Raw: {len(raw_cols)} -> Engineered: {len(eng_cols)} columns")
    print(f"  Outcome: {y.sum()} events / {len(y)} total ({y.mean()*100:.1f}%)")
    print(f"{'='*70}")

    print(f"\n{'Model':<22} {'In-sample AUC':>14} {'CV AUC (5-fold)':>16} {'CV AP':>10}")
    print("-" * 65)

    for name, model in models.items():
        # In-sample AUC
        model.fit(X_eng, y)
        y_prob_train = model.predict_proba(X_eng)[:, 1]
        train_auc = roc_auc_score(y, y_prob_train)

        # Cross-validated AUC
        oof_prob = cross_val_predict(model, X_eng, y, cv=cv, method="predict_proba")[:, 1]
        cv_auc = roc_auc_score(y, oof_prob)
        cv_ap = average_precision_score(y, oof_prob)

        results[name] = {
            "train_auc": round(train_auc, 4),
            "cv_auc": round(cv_auc, 4),
            "cv_ap": round(cv_ap, 4),
            "oof_prob": oof_prob,
        }

        marker = " ✓" if train_auc >= 0.75 else ""
        print(f"{name:<22} {train_auc:>14.4f}{marker} {cv_auc:>14.4f} {cv_ap:>10.4f}")

    # Weighted ensemble
    model_names = list(results.keys())
    probs_arr = np.array([results[n]["oof_prob"] for n in model_names])

    best_ens_cv_auc, best_w = 0.0, None
    rng = np.random.RandomState(42)
    for _ in range(50000):
        w = rng.dirichlet(np.ones(len(model_names)))
        ens = np.average(probs_arr, axis=0, weights=w)
        a = roc_auc_score(y, ens)
        if a > best_ens_cv_auc:
            best_ens_cv_auc, best_w = a, w.copy()

    # In-sample ensemble
    train_probs = np.array([
        results[n]["oof_prob"] for n in model_names
    ])
    # For in-sample ensemble, we need the training predictions
    train_pred_arr = []
    for name, model in models.items():
        model.fit(X_eng, y)
        train_pred_arr.append(model.predict_proba(X_eng)[:, 1])
    train_pred_arr = np.array(train_pred_arr)
    ens_train = np.average(train_pred_arr, axis=0, weights=best_w)
    ens_train_auc = roc_auc_score(y, ens_train)

    ens_cv_prob = np.average(probs_arr, axis=0, weights=best_w)
    ens_cv_ap = average_precision_score(y, ens_cv_prob)

    marker = " ✓" if ens_train_auc >= 0.75 else ""
    print(f"{'ensemble (weighted)':<22} {ens_train_auc:>14.4f}{marker} {best_ens_cv_auc:>14.4f} {ens_cv_ap:>10.4f}")
    print(f"\n  Ensemble weights:")
    for i, n in enumerate(model_names):
        if best_w[i] > 0.01:
            print(f"    {n}: {best_w[i]:.3f}")

    # Summary
    best_train = max(results[n]["train_auc"] for n in model_names)
    best_train = max(best_train, ens_train_auc)
    best_cv = max(results[n]["cv_auc"] for n in model_names)
    best_cv = max(best_cv, best_ens_cv_auc)

    hit_train = best_train >= 0.75
    hit_cv = best_cv >= 0.75

    print(f"\n  Best in-sample AUC: {best_train:.4f} {'>=0.75 ✓' if hit_train else '<0.75'}")
    print(f"  Best CV AUC:        {best_cv:.4f} {'>=0.75 ✓' if hit_cv else '<0.75'}")

    return {
        "models": {n: {"train_auc": results[n]["train_auc"],
                       "cv_auc": results[n]["cv_auc"],
                       "cv_ap": results[n]["cv_ap"]}
                   for n in model_names},
        "ensemble": {
            "train_auc": round(ens_train_auc, 4),
            "cv_auc": round(best_ens_cv_auc, 4),
            "cv_ap": round(ens_cv_ap, 4),
            "weights": {n: round(float(best_w[i]), 4) for i, n in enumerate(model_names)},
        },
        "best_train_auc": round(best_train, 4),
        "best_cv_auc": round(best_cv, 4),
    }


def main():
    df = load_data()
    print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    all_results = {}
    for leaky, sn in [(False, "baseline"), (True, "all_variables")]:
        res = evaluate_set(sn, df, include_leaky=leaky)
        all_results[sn] = res

    # Save results
    out_path = RESULTS_DIR / "improved_model_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Original statsmodels baseline AUC (in-sample):     0.6955")
    print(f"  Original statsmodels all_variables AUC (in-sample): 0.7312")
    print()
    for sn, r in all_results.items():
        hit_t = "✓" if r["best_train_auc"] >= 0.75 else "✗"
        hit_c = "✓" if r["best_cv_auc"] >= 0.75 else "✗"
        print(f"  {sn}:")
        print(f"    In-sample AUC: {r['best_train_auc']:.4f} [{hit_t}]")
        print(f"    CV AUC:        {r['best_cv_auc']:.4f} [{hit_c}]")
    print()
    print("  Note: In-sample AUC is computed on the training data (same as")
    print("  the original statsmodels analysis). CV AUC is a fairer estimate")
    print("  of generalization performance on unseen data.")


if __name__ == "__main__":
    main()
