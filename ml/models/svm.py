"""Support vector machine (RBF kernel).

`probability=True` enables Platt scaling so we can report ROC-AUC.
SVMs are slow on large feature sets; if this becomes a bottleneck, switch to
LinearSVC + CalibratedClassifierCV.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def build(preprocessor) -> Pipeline:
    clf = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])
