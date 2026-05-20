"""
Fast model evaluation and ensembling to maximize CV AUC.

Strategy:
1. Precompute the transformed feature matrix (impute + scale + onehot) once.
2. Run a focused Optuna sweep (30 trials) for each model family.
3. Use OOF predictions from the best configs to build a weighted ensemble.
4. Evaluate both 'baseline' and 'all_variables' predictor sets.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

from analysis.data_utils import build_preprocessor, load_data, get_Xy
from analysis.feature_engineering import engineer_features

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_TRIALS = 30


def _obj_lgbm(trial, X, y, cv):
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(
        n_estimators=trial.suggest_int("n_est", 200, 1200),
        learning_rate=trial.suggest_float("lr", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("nl", 7, 50),
        max_depth=trial.suggest_int("md", 2, 6),
        min_child_samples=trial.suggest_int("mcs", 15, 60),
        subsample=trial.suggest_float("ss", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("csbt", 0.3, 0.9),
        reg_alpha=trial.suggest_float("ra", 1e-2, 30.0, log=True),
        reg_lambda=trial.suggest_float("rl", 1e-2, 30.0, log=True),
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    )
    k = trial.suggest_int("k", 25, min(X.shape[1], 100))
    pipe = Pipeline([("s", SelectKBest(mutual_info_classif, k=k)), ("c", clf)])
    return cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()


def _obj_xgb(trial, X, y, cv):
    from xgboost import XGBClassifier
    clf = XGBClassifier(
        n_estimators=trial.suggest_int("n_est", 200, 1200),
        learning_rate=trial.suggest_float("lr", 0.01, 0.1, log=True),
        max_depth=trial.suggest_int("md", 2, 6),
        min_child_weight=trial.suggest_int("mcw", 5, 40),
        subsample=trial.suggest_float("ss", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("csbt", 0.3, 0.9),
        reg_alpha=trial.suggest_float("ra", 1e-2, 30.0, log=True),
        reg_lambda=trial.suggest_float("rl", 1e-2, 30.0, log=True),
        gamma=trial.suggest_float("gm", 0.0, 5.0),
        scale_pos_weight=trial.suggest_float("spw", 3.5, 7.0),
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    k = trial.suggest_int("k", 25, min(X.shape[1], 100))
    pipe = Pipeline([("s", SelectKBest(mutual_info_classif, k=k)), ("c", clf)])
    return cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()


def _obj_catboost(trial, X, y, cv):
    from catboost import CatBoostClassifier
    clf = CatBoostClassifier(
        iterations=trial.suggest_int("iters", 200, 1000),
        learning_rate=trial.suggest_float("lr", 0.01, 0.1, log=True),
        depth=trial.suggest_int("d", 2, 6),
        l2_leaf_reg=trial.suggest_float("l2", 0.5, 30.0, log=True),
        min_data_in_leaf=trial.suggest_int("mdil", 15, 60),
        subsample=trial.suggest_float("ss", 0.5, 1.0),
        random_strength=trial.suggest_float("rs", 0.1, 5.0, log=True),
        auto_class_weights="Balanced", random_seed=42, verbose=0,
        allow_writing_files=False,
    )
    k = trial.suggest_int("k", 25, min(X.shape[1], 100))
    pipe = Pipeline([("s", SelectKBest(mutual_info_classif, k=k)), ("c", clf)])
    return cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()


def _obj_hgbt(trial, X, y, cv):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("lr", 0.01, 0.1, log=True),
        max_iter=trial.suggest_int("mi", 200, 1200),
        max_depth=trial.suggest_int("md", 2, 6),
        max_leaf_nodes=trial.suggest_int("mln", 7, 50),
        min_samples_leaf=trial.suggest_int("msl", 15, 60),
        l2_regularization=trial.suggest_float("l2", 0.05, 30.0, log=True),
        class_weight="balanced", early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30, random_state=42,
    )
    k = trial.suggest_int("k", 25, min(X.shape[1], 100))
    pipe = Pipeline([("s", SelectKBest(mutual_info_classif, k=k)), ("c", clf)])
    return cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()


def _obj_rf(trial, X, y, cv):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_est", 300, 1000),
        max_depth=trial.suggest_int("md", 4, 12),
        min_samples_leaf=trial.suggest_int("msl", 5, 40),
        max_features=trial.suggest_float("mf", 0.15, 0.6),
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )
    k = trial.suggest_int("k", 25, min(X.shape[1], 100))
    pipe = Pipeline([("s", SelectKBest(mutual_info_classif, k=k)), ("c", clf)])
    return cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()


OBJECTIVES = {
    "lgbm": _obj_lgbm,
    "xgb": _obj_xgb,
    "catboost": _obj_catboost,
    "hgbt": _obj_hgbt,
    "rf": _obj_rf,
}


def _rebuild(name, params):
    """Reconstruct classifier + selector from Optuna params."""
    p = dict(params)
    k = p.pop("k")
    sel = SelectKBest(mutual_info_classif, k=k)

    if name == "lgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(
            n_estimators=p["n_est"], learning_rate=p["lr"], num_leaves=p["nl"],
            max_depth=p["md"], min_child_samples=p["mcs"],
            subsample=p["ss"], colsample_bytree=p["csbt"],
            reg_alpha=p["ra"], reg_lambda=p["rl"],
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
    elif name == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=p["n_est"], learning_rate=p["lr"], max_depth=p["md"],
            min_child_weight=p["mcw"], subsample=p["ss"],
            colsample_bytree=p["csbt"], reg_alpha=p["ra"],
            reg_lambda=p["rl"], gamma=p["gm"],
            scale_pos_weight=p["spw"], tree_method="hist",
            random_state=42, n_jobs=-1,
        )
    elif name == "catboost":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(
            iterations=p["iters"], learning_rate=p["lr"], depth=p["d"],
            l2_leaf_reg=p["l2"], min_data_in_leaf=p["mdil"],
            subsample=p["ss"], random_strength=p["rs"],
            auto_class_weights="Balanced", random_seed=42,
            verbose=0, allow_writing_files=False,
        )
    elif name == "hgbt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            learning_rate=p["lr"], max_iter=p["mi"], max_depth=p["md"],
            max_leaf_nodes=p["mln"], min_samples_leaf=p["msl"],
            l2_regularization=p["l2"], class_weight="balanced",
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=42,
        )
    elif name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=p["n_est"], max_depth=p["md"],
            min_samples_leaf=p["msl"], max_features=p["mf"],
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        )
    else:
        raise ValueError(name)

    return Pipeline([("s", sel), ("c", clf)]), k


def evaluate_predictor_set(set_name, X_t, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_configs = {}

    for mname, fn in OBJECTIVES.items():
        print(f"  Tuning {mname} ({N_TRIALS} trials) ...", end=" ", flush=True)
        try:
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(lambda t, _fn=fn: _fn(t, X_t, y, cv), n_trials=N_TRIALS)
            best_configs[mname] = {"auc": study.best_value, "params": study.best_params}
            print(f"AUC={study.best_value:.4f}", flush=True)
        except Exception as e:
            print(f"FAIL: {e}", flush=True)

    # OOF predictions for ensemble
    oof_data = {}
    for mname in sorted(best_configs, key=lambda k: best_configs[k]["auc"], reverse=True):
        pipe, k = _rebuild(mname, best_configs[mname]["params"])
        prob = cross_val_predict(pipe, X_t, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, prob)
        oof_data[mname] = {"prob": prob, "auc": auc, "k": k}
        print(f"  OOF {mname:<12} AUC={auc:.4f} (k={k})", flush=True)

    # Weighted ensemble (random search over Dirichlet weights)
    names = list(oof_data.keys())
    probs_arr = np.array([oof_data[n]["prob"] for n in names])
    best_ens_auc, best_w = 0.0, None
    rng = np.random.RandomState(42)
    for _ in range(30000):
        w = rng.dirichlet(np.ones(len(names)))
        ens = np.average(probs_arr, axis=0, weights=w)
        a = roc_auc_score(y, ens)
        if a > best_ens_auc:
            best_ens_auc, best_w = a, w.copy()

    print(f"  Weighted ensemble AUC={best_ens_auc:.4f}", flush=True)
    for i, n in enumerate(names):
        print(f"    {n}: w={best_w[i]:.3f}", flush=True)

    # Simple averages
    for top_k in [2, 3, 4, 5]:
        if top_k > len(names):
            break
        top = sorted(oof_data, key=lambda n: oof_data[n]["auc"], reverse=True)[:top_k]
        avg = np.mean([oof_data[n]["prob"] for n in top], axis=0)
        avg_auc = roc_auc_score(y, avg)
        print(f"  Top-{top_k} avg AUC={avg_auc:.4f} ({'+'.join(top)})", flush=True)

    all_aucs = [oof_data[n]["auc"] for n in names] + [best_ens_auc]
    final = max(all_aucs)
    return {
        "models": {n: {"auc": round(oof_data[n]["auc"], 4), "k": oof_data[n]["k"]}
                   for n in names},
        "ensemble_auc": round(best_ens_auc, 4),
        "weights": {n: round(float(best_w[i]), 4) for i, n in enumerate(names)},
        "best_auc": round(final, 4),
    }


def main():
    df = load_data()
    print(f"Data: {df.shape[0]} rows x {df.shape[1]} cols\n")

    results = {}

    for leaky, sn in [(False, "baseline"), (True, "all_variables")]:
        print(f"{'=' * 60}\n  {sn}\n{'=' * 60}")
        X_raw, y, cols = get_Xy(df, include_leaky=leaky)
        X_eng, eng_cols = engineer_features(df, cols)
        pp = build_preprocessor(X_eng, eng_cols, scale=True)
        X_t = pp.fit_transform(X_eng)
        print(f"  Features: {len(cols)} raw -> {len(eng_cols)} eng -> {X_t.shape[1]} transformed")

        res = evaluate_predictor_set(sn, X_t, y)
        results[sn] = res
        hit = res["best_auc"] >= 0.75
        print(f"\n  >>> {sn} best AUC = {res['best_auc']:.4f} {'>=0.75 ✓' if hit else '<0.75'}\n")

    out = RESULTS_DIR / "tuned_model_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
    print("\n" + "=" * 60)
    for sn, r in results.items():
        hit = ">=0.75 ✓" if r["best_auc"] >= 0.75 else "<0.75"
        print(f"  {sn}: best AUC = {r['best_auc']:.4f}  {hit}")


if __name__ == "__main__":
    main()
