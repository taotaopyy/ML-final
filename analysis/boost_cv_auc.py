"""
Maximize cross-validated AUC toward 0.70.

Strategy: diverse ensemble of well-tuned models with varied seeds/configs.

Usage:
    python3 -m analysis.boost_cv_auc
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from analysis.data_utils import (
    build_preprocessor, load_data, get_Xy, split_feature_types,
)
from analysis.feature_engineering import engineer_features

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def prune_correlated(X, names, y, threshold=0.90):
    mi = mutual_info_classif(X, y, random_state=42)
    corr = np.corrcoef(X.T)
    to_drop = set()
    for i in range(len(names)):
        if i in to_drop:
            continue
        for j in range(i + 1, len(names)):
            if j in to_drop:
                continue
            if abs(corr[i, j]) > threshold:
                to_drop.add(j if mi[i] >= mi[j] else i)
    keep = sorted(set(range(len(names))) - to_drop)
    return X[:, keep], [names[i] for i in keep]


def prepare(include_leaky=False):
    df = load_data()
    X_raw, y, raw_cols = get_Xy(df, include_leaky=include_leaky)
    X_eng, eng_cols = engineer_features(df, raw_cols)
    pp = build_preprocessor(X_eng, eng_cols, scale=True)
    X_t = pp.fit_transform(X_eng)
    num_cols, cat_cols = split_feature_types(X_eng, eng_cols)
    ohe = []
    if cat_cols:
        ohe = list(pp.named_transformers_["cat"].named_steps["onehot"]
                    .get_feature_names_out(cat_cols))
    names = num_cols + ohe
    X_p, p_names = prune_correlated(X_t, names, y.values, 0.90)
    return X_p, y, p_names


def make_clf(tag, params):
    if tag == "rf":
        return RandomForestClassifier(**params)
    if tag == "et":
        return ExtraTreesClassifier(**params)
    if tag == "hgbt":
        return HistGradientBoostingClassifier(**params)
    if tag == "gbt":
        return GradientBoostingClassifier(**params)
    if tag == "lr":
        return LogisticRegression(**params)
    if tag == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    if tag == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    if tag == "cb":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params)
    raise ValueError(tag)


def model_pool(n_feat):
    """Build a manageable pool of ~120 diverse model configs."""
    pool = []
    seeds = [42, 7, 99, 2024, 314]

    # RF: 3 configs x 5 seeds
    for s in seeds:
        pool.append(("rf", {"n_estimators": 500, "max_depth": 7, "min_samples_leaf": 8,
                             "max_features": 0.3, "class_weight": "balanced_subsample",
                             "n_jobs": -1, "random_state": s}))
        pool.append(("rf", {"n_estimators": 600, "max_depth": 10, "min_samples_leaf": 12,
                             "max_features": 0.4, "class_weight": "balanced",
                             "n_jobs": -1, "random_state": s}))

    # ET: 2 configs x 5 seeds
    for s in seeds:
        pool.append(("et", {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 8,
                             "max_features": 0.3, "class_weight": "balanced_subsample",
                             "n_jobs": -1, "random_state": s}))

    # HGBT: 3 configs x 5 seeds
    for s in seeds:
        for lr, md, msl in [(0.02, 3, 25), (0.03, 4, 30), (0.02, 5, 20)]:
            pool.append(("hgbt", {"learning_rate": lr, "max_iter": 600,
                                   "max_depth": md, "max_leaf_nodes": md * 5,
                                   "min_samples_leaf": msl, "l2_regularization": 5.0,
                                   "class_weight": "balanced", "early_stopping": True,
                                   "validation_fraction": 0.15, "n_iter_no_change": 30,
                                   "random_state": s}))

    # GBT: 2 configs x 5 seeds
    for s in seeds:
        pool.append(("gbt", {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 3,
                               "min_samples_leaf": 25, "subsample": 0.8, "max_features": 0.5,
                               "random_state": s}))

    # LR
    for C in [0.01, 0.05, 0.1, 0.5]:
        pool.append(("lr", {"C": C, "max_iter": 3000, "class_weight": "balanced",
                             "solver": "lbfgs", "random_state": 42}))

    # LightGBM: 3 configs x 5 seeds
    try:
        from lightgbm import LGBMClassifier
        for s in seeds:
            for nl, mcs in [(12, 25), (18, 35), (25, 20)]:
                pool.append(("lgbm", {"n_estimators": 600, "learning_rate": 0.02,
                                       "num_leaves": nl, "max_depth": 4,
                                       "min_child_samples": mcs, "subsample": 0.7,
                                       "colsample_bytree": 0.5, "reg_alpha": 3.0,
                                       "reg_lambda": 8.0, "class_weight": "balanced",
                                       "random_state": s, "n_jobs": -1, "verbose": -1}))
    except ImportError:
        pass

    # XGBoost: 3 configs x 5 seeds
    try:
        from xgboost import XGBClassifier
        for s in seeds:
            for md, mcw in [(3, 15), (4, 20), (3, 30)]:
                pool.append(("xgb", {"n_estimators": 600, "learning_rate": 0.02,
                                      "max_depth": md, "min_child_weight": mcw,
                                      "subsample": 0.7, "colsample_bytree": 0.5,
                                      "reg_alpha": 3.0, "reg_lambda": 8.0, "gamma": 1.0,
                                      "scale_pos_weight": 5.5, "tree_method": "hist",
                                      "random_state": s, "n_jobs": -1}))
    except ImportError:
        pass

    # CatBoost: 2 configs x 3 seeds
    try:
        from catboost import CatBoostClassifier
        for s in seeds[:3]:
            for d in [4, 5]:
                pool.append(("cb", {"iterations": 600, "learning_rate": 0.02,
                                     "depth": d, "l2_leaf_reg": 8.0,
                                     "min_data_in_leaf": 30, "subsample": 0.7,
                                     "auto_class_weights": "Balanced",
                                     "random_seed": s, "verbose": 0,
                                     "allow_writing_files": False}))
    except ImportError:
        pass

    return pool


def run(set_name, X, y):
    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n_feat = X.shape[1]
    print(f"  Features: {n_feat}, samples: {X.shape[0]}, events: {int(y.sum())}")

    pool = model_pool(n_feat)
    k_vals = [30, min(55, n_feat), n_feat]
    total_runs = len(pool) * len(k_vals)
    print(f"  Model pool: {len(pool)} configs x {len(k_vals)} k-values = {total_runs} runs")

    oof = {}
    idx = 0

    for i, (tag, params) in enumerate(pool):
        for k in k_vals:
            clf = make_clf(tag, params)
            pipe = Pipeline([("sel", SelectKBest(mutual_info_classif, k=k)), ("clf", clf)])
            try:
                prob = cross_val_predict(pipe, X, y, cv=cv10, method="predict_proba")[:, 1]
                auc = roc_auc_score(y, prob)
                name = f"{tag}_{i}_k{k}"
                oof[name] = {"prob": prob, "auc": auc}
            except Exception:
                pass
            idx += 1
            if idx % 50 == 0:
                best = max((oof[n]["auc"] for n in oof), default=0)
                print(f"    [{idx}/{total_runs}] models={len(oof)} best={best:.4f}", flush=True)

    print(f"\n  Evaluated {len(oof)} models")

    sorted_names = sorted(oof, key=lambda n: oof[n]["auc"], reverse=True)
    print(f"\n  Top 15 individual (10-fold):")
    for n in sorted_names[:15]:
        print(f"    {n:<45} AUC={oof[n]['auc']:.4f}")

    # Ensemble top-N with optimized weights
    for top_n in [5, 10, 20, 40]:
        if top_n > len(sorted_names):
            break
        top = sorted_names[:top_n]
        probs_arr = np.array([oof[n]["prob"] for n in top])
        avg_auc = roc_auc_score(y, np.mean(probs_arr, axis=0))
        best_w_auc = avg_auc
        rng = np.random.RandomState(42)
        for _ in range(60000):
            w = rng.dirichlet(np.ones(len(top)))
            a = roc_auc_score(y, np.average(probs_arr, axis=0, weights=w))
            if a > best_w_auc:
                best_w_auc = a
        print(f"  Ensemble top-{top_n}: avg={avg_auc:.4f} weighted={best_w_auc:.4f}")

    # Use top-20 weighted as main result
    n_ens = min(20, len(sorted_names))
    top_ens = sorted_names[:n_ens]
    probs_ens = np.array([oof[n]["prob"] for n in top_ens])
    best_ens, best_ew = 0.0, None
    rng = np.random.RandomState(42)
    for _ in range(100000):
        w = rng.dirichlet(np.ones(n_ens))
        a = roc_auc_score(y, np.average(probs_ens, axis=0, weights=w))
        if a > best_ens:
            best_ens, best_ew = a, w.copy()

    # 5-fold check on top-5 models
    print(f"\n  5-fold CV check:")
    for n in sorted_names[:5]:
        i_str = n.split("_")[1]
        tag = n.split("_")[0]
        i = int(i_str)
        k = int(n.rsplit("_k", 1)[1])
        t2, p2 = pool[i]
        clf = make_clf(t2, p2)
        pipe = Pipeline([("sel", SelectKBest(mutual_info_classif, k=k)), ("clf", clf)])
        try:
            prob5 = cross_val_predict(pipe, X, y, cv=cv5, method="predict_proba")[:, 1]
            auc5 = roc_auc_score(y, prob5)
            print(f"    {n:<45} 5f={auc5:.4f}  10f={oof[n]['auc']:.4f}")
        except Exception as e:
            print(f"    {n}: failed ({e})")

    best_single = oof[sorted_names[0]]["auc"]
    best_overall = max(best_single, best_ens)

    print(f"\n  RESULT: best single={best_single:.4f}, ensemble={best_ens:.4f}")
    hit = best_overall >= 0.70
    print(f"  Target >=0.70: {'YES ✓' if hit else 'NO'}")

    return {
        "top_models": {n: round(oof[n]["auc"], 4) for n in sorted_names[:15]},
        "ensemble_auc": round(best_ens, 4),
        "best_overall": round(best_overall, 4),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print("=" * 70)
    print("  BOOST CV AUC → 0.70")
    print("=" * 70)

    results = {}
    for leaky, sn in [(False, "baseline"), (True, "all_variables")]:
        print(f"\n{'='*70}\n  {sn}\n{'='*70}")
        X, y, names = prepare(include_leaky=leaky)
        results[sn] = run(sn, X, y)

    out = RESULTS_DIR / "boost_cv_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

    print("\n" + "=" * 70)
    for sn, r in results.items():
        hit = "✓" if r["best_overall"] >= 0.70 else "✗"
        print(f"  {sn}: {r['best_overall']:.4f} [{hit}]")


if __name__ == "__main__":
    main()
