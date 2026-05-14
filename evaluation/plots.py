"""
Plotting helpers for model evaluation.

All functions take `(y_true, y_prob)` arrays and an optional `ax` argument so
they can be composed into multi-panel figures. They `return ax` for chaining.
"""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc as _auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_roc(y_true, y_prob, *, ax=None, label: str | None = None):
    """Plot a single ROC curve. AUC is appended to the label."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    lbl = f"{label or 'model'} (AUC={auc:.3f})"
    ax.plot(fpr, tpr, label=lbl)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


def plot_pr(y_true, y_prob, *, ax=None, label: str | None = None):
    """Precision–recall curve."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    lbl = f"{label or 'model'} (AP={ap:.3f})"
    ax.plot(rec, prec, label=lbl)
    base = np.mean(y_true)
    ax.axhline(base, color="k", ls="--", alpha=0.4, label=f"prevalence={base:.3f}")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision–Recall curve")
    ax.legend(loc="lower left")
    return ax


def plot_calibration(y_true, y_prob, *, ax=None, n_bins: int = 10,
                     strategy: str = "quantile", label: str | None = None):
    """Reliability diagram + simple histogram of predicted probabilities."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins,
                                             strategy=strategy)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", label=label or "model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(f"Calibration (n_bins={n_bins}, strategy={strategy})")
    ax.legend(loc="upper left")
    return ax


def plot_confusion_matrix(y_true, y_pred, *, ax=None, labels=("0", "1"),
                          normalize: bool = False):
    """Annotated confusion matrix (counts or row-normalised)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = cm.astype(float)
    if normalize:
        disp = disp / disp.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(disp, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fmt = ".2f" if normalize else "d"
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(disp[i, j], fmt), ha="center", va="center",
                    color="white" if disp[i, j] > disp.max() / 2 else "black")
    ax.set_title("Confusion matrix" + (" (normalized)" if normalize else ""))
    return ax


def plot_models_roc(oof_dict: Mapping[str, tuple[np.ndarray, np.ndarray]], *,
                    ax=None):
    """Overlay ROC curves for several models. `oof_dict = {name: (y, p)}`."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for name, (y_true, y_prob) in oof_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={_auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC comparison")
    ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_models_pr(oof_dict: Mapping[str, tuple[np.ndarray, np.ndarray]], *,
                   ax=None):
    """Overlay PR curves for several models."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for name, (y_true, y_prob) in oof_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR comparison")
    ax.legend(loc="lower left", fontsize=8)
    return ax


def save_fig(fig, path, *, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
