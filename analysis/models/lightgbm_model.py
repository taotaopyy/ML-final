"""LightGBM gradient-boosted trees.

Fast, accurate, and the go-to baseline for tabular competitions.
`class_weight="balanced"` handles the 15% positive class.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline


def build(preprocessor) -> Pipeline:
    clf = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])
