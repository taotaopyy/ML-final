"""Multi-layer perceptron (small neural network).

Kept small on purpose: tabular medical datasets with ~2k rows rarely benefit
from deep networks, so a 2-hidden-layer MLP is plenty.
"""

from __future__ import annotations

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


def build(preprocessor) -> Pipeline:
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-3,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])
