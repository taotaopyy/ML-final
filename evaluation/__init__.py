"""Model evaluation and SHAP explainability toolkit.

Submodules:
    metrics         — point / threshold / bootstrap CI metrics.
    plots           — ROC, PR, calibration, confusion matrix, model comparison.
    shap_analysis   — SHAP wrappers for tree / linear / kernel explainers.
"""

from . import metrics, plots, shap_analysis  # noqa: F401

__all__ = ["metrics", "plots", "shap_analysis"]
