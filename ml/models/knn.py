"""K-nearest neighbours classifier.

KNN needs scaled features (handled by the shared preprocessor) and is a useful
sanity-check baseline. It rarely wins on noisy tabular medical data, but it's
cheap to include.
"""

from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def build(preprocessor) -> Pipeline:
    clf = KNeighborsClassifier(
        n_neighbors=25,
        weights="distance",
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])
