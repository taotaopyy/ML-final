"""Registry of available ML model builders.

Each module exposes a `build(preprocessor)` function that returns a sklearn
Pipeline. Add a new model by writing a new file in this folder and registering
it in MODEL_REGISTRY below.
"""

from __future__ import annotations

from typing import Callable

from sklearn.pipeline import Pipeline

from . import (
    gradient_boosting,
    knn,
    lasso,
    logistic,
    mlp,
    random_forest,
    svm,
)

# name -> build_fn(preprocessor) -> Pipeline
MODEL_REGISTRY: dict[str, Callable[[object], Pipeline]] = {
    "logistic_l2": logistic.build,
    "logistic_l1_lasso": lasso.build,
    "random_forest": random_forest.build,
    "gradient_boosting": gradient_boosting.build,
    "svm_rbf": svm.build,
    "knn": knn.build,
    "mlp": mlp.build,
}

__all__ = ["MODEL_REGISTRY"]
