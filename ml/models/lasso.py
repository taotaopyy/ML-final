"""L1-regularised (LASSO) logistic regression.

LASSO shrinks weak coefficients to exactly zero, so it doubles as a built-in
variable selector. Useful here because the plaque-burden / plaque-volume
features are heavily collinear.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build(preprocessor) -> Pipeline:
    clf = LogisticRegression(
        penalty="l1",
        C=0.1,
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])
