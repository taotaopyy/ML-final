"""
Numerical evaluation metrics.

Everything here works directly on `(y_true, y_prob)` arrays — i.e. no model
is required. Use it on out-of-fold predictions from cross-validation, or on
a held-out test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    """Return a flat dict of common binary-classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    return {
        "auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)),
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "youden_j": sens + spec - 1 if not (np.isnan(sens) or np.isnan(spec)) else np.nan,
    }


def best_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, criterion: str = "youden"
) -> dict:
    """Find the threshold that maximises a given criterion.

    criterion :
        "youden"  — maximise Youden's J = Sens + Spec - 1 (default)
        "f1"      — maximise F1 score
        "balanced_accuracy" — maximise (Sens + Spec) / 2
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    fpr, tpr, thrs = roc_curve(y_true, y_prob)

    if criterion == "youden":
        scores = tpr - fpr
        best_idx = int(np.argmax(scores))
        return {
            "criterion": criterion,
            "threshold": float(thrs[best_idx]),
            "score": float(scores[best_idx]),
            "sensitivity": float(tpr[best_idx]),
            "specificity": float(1 - fpr[best_idx]),
        }

    # f1 / balanced_accuracy — scan unique probability values.
    candidates = np.unique(np.concatenate([[0.0], y_prob, [1.0]]))
    best = {"threshold": 0.5, "score": -np.inf}
    for thr in candidates:
        m = compute_metrics(y_true, y_prob, threshold=float(thr))
        score = m["f1"] if criterion == "f1" else m["balanced_accuracy"]
        if score > best["score"]:
            best = {
                "criterion": criterion,
                "threshold": float(thr),
                "score": float(score),
                "sensitivity": m["recall_sensitivity"],
                "specificity": m["specificity"],
            }
    return best


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    """Percentile-bootstrap confidence interval for ROC-AUC."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(y_true[idx], y_prob[idx])
    aucs = aucs[~np.isnan(aucs)]
    lo, hi = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "auc_ci_low": float(lo),
        "auc_ci_high": float(hi),
        "n_boot": int(len(aucs)),
        "alpha": alpha,
    }


def summarize_models(oof_dict: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Build a comparison table from {model_name: (y_true, y_prob), ...}."""
    rows = []
    for name, (y_true, y_prob) in oof_dict.items():
        m = compute_metrics(y_true, y_prob)
        ci = bootstrap_auc_ci(y_true, y_prob, n_boot=500)
        rows.append({"model": name, **m, "auc_ci_low": ci["auc_ci_low"],
                     "auc_ci_high": ci["auc_ci_high"]})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
