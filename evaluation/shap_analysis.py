"""
SHAP explainability wrappers.

Works directly on a fitted `sklearn.Pipeline` of the form
    preprocessor -> classifier
which is exactly what `ml.models.MODEL_REGISTRY[*](preprocessor)` produces.

The wrappers:
1. Apply the pipeline's `preprocess` step to X to get the transformed matrix
   the classifier actually sees.
2. Pick the appropriate SHAP explainer for the classifier type:
       - tree-based (RF, ExtraTrees, GBT, LightGBM, XGBoost, CatBoost) -> TreeExplainer
       - linear (LogisticRegression)                                   -> LinearExplainer
       - everything else (SVM, KNN, MLP)                               -> KernelExplainer (slow)
3. Return SHAP values plus the transformed feature names so you can plot
   them with the standard `shap.summary_plot` etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.base import is_classifier


_TREE_CLASSES = (
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "LGBMClassifier",
    "XGBClassifier",
    "CatBoostClassifier",
)
_LINEAR_CLASSES = ("LogisticRegression",)


def _transform_X(pipeline, X) -> tuple[np.ndarray, list[str]]:
    """Apply the preprocessor step and return (X_transformed, feature_names)."""
    pre = pipeline.named_steps["preprocess"]
    Xt = pre.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(Xt.shape[1])]
    return np.asarray(Xt, dtype=float), names


def _pick_explainer(clf, Xt: np.ndarray, background_size: int = 100):
    """Return a SHAP explainer best suited to the classifier."""
    name = clf.__class__.__name__
    if name in _TREE_CLASSES:
        try:
            return shap.TreeExplainer(clf), "tree"
        except Exception:
            pass
    if name in _LINEAR_CLASSES:
        try:
            return shap.LinearExplainer(clf, Xt), "linear"
        except Exception:
            pass
    # Generic fallback — slow.
    bg = shap.sample(Xt, min(background_size, len(Xt)), random_state=42)
    return shap.KernelExplainer(clf.predict_proba, bg), "kernel"


def compute_shap(pipeline, X, *, max_samples: int = 1000,
                 background_size: int = 100, random_state: int = 42):
    """Compute SHAP values for a fitted Pipeline on (a sample of) X.

    Returns
    -------
    dict with keys:
        shap_values     : np.ndarray, shape (n, n_features) — for the positive class
        X_transformed   : np.ndarray, the preprocessed matrix used
        feature_names   : list[str]
        explainer_kind  : "tree" / "linear" / "kernel"
        sampled_index   : pd.Index, which rows of X were used
    """
    if not is_classifier(pipeline.named_steps["clf"]):
        raise ValueError("Pipeline 'clf' step is not a classifier.")

    if len(X) > max_samples:
        sampled = X.sample(n=max_samples, random_state=random_state)
    else:
        sampled = X

    clf = pipeline.named_steps["clf"]
    Xt, names = _transform_X(pipeline, sampled)
    explainer, kind = _pick_explainer(clf, Xt, background_size=background_size)

    sv = explainer.shap_values(Xt)
    # Different libraries return slightly different shapes; normalise to
    # (n_samples, n_features) for the positive class.
    sv_arr = np.asarray(sv)
    if sv_arr.ndim == 3:
        # (n_classes, n, n_features) — pick class 1
        sv_arr = sv_arr[1] if sv_arr.shape[0] == 2 else sv_arr[-1]
    elif isinstance(sv, list) and len(sv) == 2:
        sv_arr = np.asarray(sv[1])

    return {
        "shap_values": sv_arr,
        "X_transformed": Xt,
        "feature_names": names,
        "explainer_kind": kind,
        "sampled_index": sampled.index,
    }


def top_feature_importance(shap_out: dict, top_k: int = 20) -> pd.DataFrame:
    """Mean |SHAP value| per feature — a model-agnostic importance ranking."""
    sv = np.asarray(shap_out["shap_values"])
    mean_abs = np.abs(sv).mean(axis=0)
    return (
        pd.DataFrame({"feature": shap_out["feature_names"],
                      "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def plot_summary(shap_out: dict, *, plot_type: str = "dot", max_display: int = 20,
                 show: bool = False, save_path: str | None = None):
    """Wrapper around `shap.summary_plot` with a save_path option."""
    import matplotlib.pyplot as plt

    shap.summary_plot(
        shap_out["shap_values"],
        features=shap_out["X_transformed"],
        feature_names=shap_out["feature_names"],
        plot_type=plot_type,
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if not show:
        plt.close(fig)
    return fig


def plot_bar(shap_out: dict, *, max_display: int = 20,
             show: bool = False, save_path: str | None = None):
    """Bar plot of mean |SHAP| importance."""
    return plot_summary(shap_out, plot_type="bar", max_display=max_display,
                        show=show, save_path=save_path)


def plot_dependence(shap_out: dict, feature: str, *,
                    interaction_index: str | int | None = "auto",
                    show: bool = False, save_path: str | None = None):
    """SHAP dependence plot for a single feature."""
    import matplotlib.pyplot as plt

    shap.dependence_plot(
        feature,
        shap_out["shap_values"],
        shap_out["X_transformed"],
        feature_names=shap_out["feature_names"],
        interaction_index=interaction_index,
        show=False,
    )
    fig = plt.gcf()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if not show:
        plt.close(fig)
    return fig
