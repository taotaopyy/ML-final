"""Data preprocessing toolkit.

Submodules:
    loaders       — read raw CSV + light cleaning (whitespace, empty cols).
    inspect       — missingness / dtype / cardinality / outlier reports.
    cleaning      — drop high-missing cols/rows, clip outliers, log-skewed.
    transformers  — preprocessor builders (standard / robust / minmax / knn-impute).
    imbalance     — SMOTE / undersampling helpers (requires imbalanced-learn).
    splits        — train / val / test split helpers.

This folder is independent of `analysis/`, `ml/`, and `evaluation/`. It can
fully replace the minimal preprocessing in `ml/data_utils.py` if you need
more flexibility, or be used alongside it.
"""

from . import cleaning, imbalance, inspect, loaders, splits, transformers  # noqa: F401

__all__ = ["loaders", "inspect", "cleaning", "transformers", "imbalance", "splits"]
