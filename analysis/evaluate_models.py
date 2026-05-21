"""
Evaluate ML models with feature engineering, hyperparameter tuning, and ensembling.

Goal: push 5-fold CV AUC above 0.75 for the target_lesion_positive outcome
using only baseline (non-leaky) features.

Workflow:
1. Load data and engineer features.
2. Build a shared preprocessor.
3. Benchmark individual models with tuned hyperparameters.
4. Build a stacking ensemble of the best models.
5. Report all results and save metrics.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from analysis.data_utils import build_preprocessor, load_data, get_Xy
from analysis.feature_engineering import engineer_features

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _try_import_lgbm():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except ImportError:
        return None


def _try_import_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError:
        return None


def _try_import_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError:
        return None


def build_tuned_models(preprocessor) -> dict[str, Pipeline]:
    """Build tuned versions of each model class."""
    models = {}

    # L2 logistic regression — tune C
    models["logistic_l2_tuned"] = Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(
            C=0.05, max_iter=3000, class_weight="balanced",
            solver="lbfgs", random_state=42,
        )),
    ])

    # L1 (LASSO) logistic regression
    models["lasso_tuned"] = Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(
            C=0.01, l1_ratio=1.0, solver="saga",
            max_iter=3000, class_weight="balanced", random_state=42,
        )),
    ])

    # Random forest — deeper, more trees, tuned leaf size
    models["rf_tuned"] = Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=1000, max_depth=8, min_samples_leaf=10,
            max_features=0.3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        )),
    ])

    # Extra trees
    models["et_tuned"] = Pipeline([
        ("preprocess", preprocessor),
        ("clf", ExtraTreesClassifier(
            n_estimators=1000, max_depth=8, min_samples_leaf=10,
            max_features=0.3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        )),
    ])

    # HistGradientBoosting — more regularized
    models["hgbt_tuned"] = Pipeline([
        ("preprocess", preprocessor),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.02, max_iter=800, max_depth=3,
            max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=5.0, max_bins=128,
            class_weight="balanced", early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=30,
            random_state=42,
        )),
    ])

    # Sklearn GradientBoosting (non-histogram)
    models["gbt_tuned"] = Pipeline([
        ("preprocess", preprocessor),
        ("clf", GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.02, max_depth=3,
            min_samples_leaf=30, subsample=0.8, max_features=0.5,
            random_state=42,
        )),
    ])

    # SVM
    models["svm_tuned"] = Pipeline([
        ("preprocess", preprocessor),
        ("clf", SVC(
            kernel="rbf", C=0.5, gamma="scale",
            probability=True, class_weight="balanced", random_state=42,
        )),
    ])

    # KNN
    models["knn_tuned"] = Pipeline([
        ("preprocess", preprocessor),
        ("clf", KNeighborsClassifier(
            n_neighbors=50, weights="distance", metric="minkowski",
            p=1, n_jobs=-1,
        )),
    ])

    # MLP — wider, more regularization
    models["mlp_tuned"] = Pipeline([
        ("preprocess", preprocessor),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            alpha=0.01, batch_size=64, learning_rate="adaptive",
            learning_rate_init=5e-4, max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, random_state=42,
        )),
    ])

    # LightGBM
    LGBMClassifier = _try_import_lgbm()
    if LGBMClassifier:
        models["lgbm_tuned"] = Pipeline([
            ("preprocess", preprocessor),
            ("clf", LGBMClassifier(
                n_estimators=1000, learning_rate=0.02, num_leaves=15,
                max_depth=4, min_child_samples=30, subsample=0.7,
                subsample_freq=1, colsample_bytree=0.6,
                reg_alpha=1.0, reg_lambda=5.0,
                class_weight="balanced", random_state=42,
                n_jobs=-1, verbose=-1,
            )),
        ])

    # XGBoost
    XGBClassifier = _try_import_xgb()
    if XGBClassifier:
        models["xgb_tuned"] = Pipeline([
            ("preprocess", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=1000, learning_rate=0.02, max_depth=3,
                min_child_weight=10, subsample=0.7,
                colsample_bytree=0.5, colsample_bylevel=0.7,
                reg_alpha=1.0, reg_lambda=5.0, gamma=1.0,
                objective="binary:logistic", eval_metric="auc",
                scale_pos_weight=5.5, tree_method="hist",
                random_state=42, n_jobs=-1,
            )),
        ])

    # CatBoost
    CatBoostClassifier = _try_import_catboost()
    if CatBoostClassifier:
        models["catboost_tuned"] = Pipeline([
            ("preprocess", preprocessor),
            ("clf", CatBoostClassifier(
                iterations=1000, learning_rate=0.02, depth=4,
                l2_leaf_reg=10.0, min_data_in_leaf=30,
                subsample=0.7, colsample_bylevel=0.6,
                loss_function="Logloss", eval_metric="AUC",
                auto_class_weights="Balanced", random_seed=42,
                verbose=0, allow_writing_files=False,
            )),
        ])

    return models


def build_stacking_ensemble(preprocessor) -> Pipeline:
    """Build a stacking ensemble of diverse model families."""
    estimators = []

    estimators.append(("lr", LogisticRegression(
        C=0.05, max_iter=3000, class_weight="balanced",
        solver="lbfgs", random_state=42,
    )))

    estimators.append(("rf", RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=10,
        max_features=0.3, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    )))

    estimators.append(("et", ExtraTreesClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=10,
        max_features=0.3, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    )))

    estimators.append(("hgbt", HistGradientBoostingClassifier(
        learning_rate=0.02, max_iter=800, max_depth=3,
        max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=5.0, class_weight="balanced",
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=42,
    )))

    estimators.append(("gbt", GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.02, max_depth=3,
        min_samples_leaf=30, subsample=0.8, max_features=0.5,
        random_state=42,
    )))

    LGBMClassifier = _try_import_lgbm()
    if LGBMClassifier:
        estimators.append(("lgbm", LGBMClassifier(
            n_estimators=1000, learning_rate=0.02, num_leaves=15,
            max_depth=4, min_child_samples=30, subsample=0.7,
            subsample_freq=1, colsample_bytree=0.6,
            reg_alpha=1.0, reg_lambda=5.0,
            class_weight="balanced", random_state=42,
            n_jobs=-1, verbose=-1,
        )))

    XGBClassifier = _try_import_xgb()
    if XGBClassifier:
        estimators.append(("xgb", XGBClassifier(
            n_estimators=1000, learning_rate=0.02, max_depth=3,
            min_child_weight=10, subsample=0.7,
            colsample_bytree=0.5, colsample_bylevel=0.7,
            reg_alpha=1.0, reg_lambda=5.0, gamma=1.0,
            objective="binary:logistic", eval_metric="auc",
            scale_pos_weight=5.5, tree_method="hist",
            random_state=42, n_jobs=-1,
        )))

    CatBoostClassifier = _try_import_catboost()
    if CatBoostClassifier:
        estimators.append(("catboost", CatBoostClassifier(
            iterations=1000, learning_rate=0.02, depth=4,
            l2_leaf_reg=10.0, min_data_in_leaf=30,
            subsample=0.7, colsample_bylevel=0.6,
            loss_function="Logloss", eval_metric="AUC",
            auto_class_weights="Balanced", random_seed=42,
            verbose=0, allow_writing_files=False,
        )))

    stacker = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            C=1.0, max_iter=3000, random_state=42,
        ),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )

    return Pipeline([("preprocess", preprocessor), ("clf", stacker)])


def build_voting_ensemble(preprocessor) -> Pipeline:
    """Build a soft voting ensemble of diverse models."""
    estimators = []

    estimators.append(("lr", LogisticRegression(
        C=0.05, max_iter=3000, class_weight="balanced",
        solver="lbfgs", random_state=42,
    )))

    estimators.append(("rf", RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=10,
        max_features=0.3, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    )))

    estimators.append(("hgbt", HistGradientBoostingClassifier(
        learning_rate=0.02, max_iter=800, max_depth=3,
        max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=5.0, class_weight="balanced",
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=42,
    )))

    estimators.append(("gbt", GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.02, max_depth=3,
        min_samples_leaf=30, subsample=0.8, max_features=0.5,
        random_state=42,
    )))

    LGBMClassifier = _try_import_lgbm()
    if LGBMClassifier:
        estimators.append(("lgbm", LGBMClassifier(
            n_estimators=1000, learning_rate=0.02, num_leaves=15,
            max_depth=4, min_child_samples=30, subsample=0.7,
            subsample_freq=1, colsample_bytree=0.6,
            reg_alpha=1.0, reg_lambda=5.0,
            class_weight="balanced", random_state=42,
            n_jobs=-1, verbose=-1,
        )))

    XGBClassifier = _try_import_xgb()
    if XGBClassifier:
        estimators.append(("xgb", XGBClassifier(
            n_estimators=1000, learning_rate=0.02, max_depth=3,
            min_child_weight=10, subsample=0.7,
            colsample_bytree=0.5, colsample_bylevel=0.7,
            reg_alpha=1.0, reg_lambda=5.0, gamma=1.0,
            objective="binary:logistic", eval_metric="auc",
            scale_pos_weight=5.5, tree_method="hist",
            random_state=42, n_jobs=-1,
        )))

    CatBoostClassifier = _try_import_catboost()
    if CatBoostClassifier:
        estimators.append(("catboost", CatBoostClassifier(
            iterations=1000, learning_rate=0.02, depth=4,
            l2_leaf_reg=10.0, min_data_in_leaf=30,
            subsample=0.7, colsample_bylevel=0.6,
            loss_function="Logloss", eval_metric="AUC",
            auto_class_weights="Balanced", random_seed=42,
            verbose=0, allow_writing_files=False,
        )))

    voter = VotingClassifier(
        estimators=estimators,
        voting="soft",
        n_jobs=-1,
    )

    return Pipeline([("preprocess", preprocessor), ("clf", voter)])


def evaluate_cv(model, X, y, name: str, cv) -> dict:
    """Run cross-validation and return metrics dict."""
    oof_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof_prob)
    ap = average_precision_score(y, oof_prob)
    brier = brier_score_loss(y, oof_prob)
    return {"model": name, "auc": round(auc, 4), "ap": round(ap, 4), "brier": round(brier, 4)}


def main() -> None:
    df = load_data()
    X_raw, y, raw_cols = get_Xy(df, include_leaky=False)

    print(f"Data: {df.shape[0]} rows, outcome distribution: {dict(y.value_counts())}")
    print(f"Raw baseline predictors: {len(raw_cols)}")

    # --- Feature engineering ---
    X_eng, eng_cols = engineer_features(df, raw_cols)
    print(f"After feature engineering: {len(eng_cols)} columns")

    # Identify numeric vs categorical in engineered set
    from analysis.data_utils import split_feature_types
    num_cols, cat_cols = split_feature_types(X_eng, eng_cols)

    preprocessor = build_preprocessor(X_eng, eng_cols, scale=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Evaluate individual tuned models ---
    models = build_tuned_models(preprocessor)
    results = []

    print(f"\n{'Model':<25} {'CV AUC':>10} {'CV AP':>10} {'Brier':>10}")
    print("-" * 60)

    for name, model in models.items():
        try:
            r = evaluate_cv(model, X_eng, y, name, cv)
            results.append(r)
            print(f"{name:<25} {r['auc']:>10.4f} {r['ap']:>10.4f} {r['brier']:>10.4f}")
        except Exception as e:
            print(f"{name:<25} FAILED: {e}")

    # --- Stacking ensemble ---
    print("\nBuilding stacking ensemble...")
    try:
        stacking = build_stacking_ensemble(preprocessor)
        r = evaluate_cv(stacking, X_eng, y, "stacking_ensemble", cv)
        results.append(r)
        print(f"{'stacking_ensemble':<25} {r['auc']:>10.4f} {r['ap']:>10.4f} {r['brier']:>10.4f}")
    except Exception as e:
        print(f"Stacking failed: {e}")

    # --- Voting ensemble ---
    print("\nBuilding voting ensemble...")
    try:
        voting = build_voting_ensemble(preprocessor)
        r = evaluate_cv(voting, X_eng, y, "voting_ensemble", cv)
        results.append(r)
        print(f"{'voting_ensemble':<25} {r['auc']:>10.4f} {r['ap']:>10.4f} {r['brier']:>10.4f}")
    except Exception as e:
        print(f"Voting failed: {e}")

    # --- Summary ---
    results_df = pd.DataFrame(results).sort_values("auc", ascending=False)
    print("\n" + "=" * 60)
    print("FINAL RANKING (by CV AUC)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest model: {best['model']} with AUC={best['auc']:.4f}")

    target_met = best["auc"] >= 0.75
    print(f"Target AUC >= 0.75: {'YES' if target_met else 'NO'}")

    # Save results
    results_path = RESULTS_DIR / "model_evaluation_results.json"
    results_df.to_json(results_path, orient="records", indent=2)
    print(f"\nSaved results to {results_path}")

    return results_df


if __name__ == "__main__":
    main()
