"""XGBoost gradient-boosted trees.

`scale_pos_weight` ≈ #neg / #pos handles the class imbalance (about 5.5 here).
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

POS_WEIGHT = 1868 / 340  # baseline class balance in this dataset (~5.5)


def build(preprocessor) -> Pipeline:
    clf = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=POS_WEIGHT,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])
