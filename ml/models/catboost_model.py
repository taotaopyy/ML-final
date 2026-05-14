"""CatBoost gradient-boosted trees.

`auto_class_weights="Balanced"` mirrors `class_weight="balanced"` in sklearn.
`verbose=0` keeps the console quiet during CV.
"""

from __future__ import annotations

from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline


def build(preprocessor) -> Pipeline:
    clf = CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])
