"""Plain L2-regularised logistic regression."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build(preprocessor) -> Pipeline:
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])
