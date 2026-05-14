"""Registry of available ML model builders.

Each module exposes a `build(preprocessor)` function that returns a sklearn
Pipeline. Add a new model by writing a new file in this folder and registering
it in MODEL_REGISTRY below.

Optional dependencies (lightgbm / xgboost / catboost) are imported lazily so
the rest of the registry still works if any of them is missing.
"""

from __future__ import annotations

from typing import Callable

from sklearn.pipeline import Pipeline

from . import (
    extra_trees,
    gradient_boosting,
    knn,
    lasso,
    logistic,
    mlp,
    random_forest,
    svm,
)

MODEL_REGISTRY: dict[str, Callable[[object], Pipeline]] = {
    "logistic_l2": logistic.build,
    "logistic_l1_lasso": lasso.build,
    "random_forest": random_forest.build,
    "extra_trees": extra_trees.build,
    "gradient_boosting": gradient_boosting.build,
    "svm_rbf": svm.build,
    "knn": knn.build,
    "mlp": mlp.build,
}

# Optional boosting backends — register only if the package is installed.
try:
    from . import lightgbm_model

    MODEL_REGISTRY["lightgbm"] = lightgbm_model.build
except ImportError:
    pass

try:
    from . import xgboost_model

    MODEL_REGISTRY["xgboost"] = xgboost_model.build
except ImportError:
    pass

try:
    from . import catboost_model

    MODEL_REGISTRY["catboost"] = catboost_model.build
except ImportError:
    pass


__all__ = ["MODEL_REGISTRY"]
