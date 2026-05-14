"""Extremely Randomised Trees (ExtraTrees).

Similar API to RandomForest but splits are chosen at random thresholds, which
often reduces variance further. Good cheap addition to the ensemble lineup.
"""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline


def build(preprocessor) -> Pipeline:
    clf = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])
