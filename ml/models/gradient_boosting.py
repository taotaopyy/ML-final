"""Gradient-boosted trees (sklearn HistGradientBoostingClassifier).

HistGBT is fast, handles missing values natively, and is usually a strong
baseline for tabular data. We keep imputation in the preprocessor for
consistency with the other models, but HistGBT could swallow NaNs on its own.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline


def build(preprocessor) -> Pipeline:
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=400,
        max_depth=None,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])
