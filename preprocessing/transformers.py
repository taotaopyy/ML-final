"""
Preprocessor builders — sklearn ColumnTransformer factories with several
variants for different downstream models.

All builders return a `ColumnTransformer` that takes a raw DataFrame and
outputs a numeric matrix ready to feed into any sklearn-compatible model.

Numeric pipeline options:
    impute   : "median" (default) / "mean" / "most_frequent" / "knn"
    scale    : "standard" / "robust" / "minmax" / None
    transform: optional "log1p" or "yeo-johnson" before scaling

Categorical pipeline:
    most_frequent imputation + one-hot encoding (ignore unknown levels).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
import numpy as np


def split_feature_types(
    df: pd.DataFrame, cols: list[str]
) -> tuple[list[str], list[str]]:
    """Numeric vs non-numeric columns (works with object/str/pyarrow dtypes)."""
    num_cols = [c for c in cols if is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if c not in num_cols]
    return num_cols, cat_cols


def _make_imputer(strategy: str, knn_neighbors: int = 5):
    if strategy == "knn":
        return KNNImputer(n_neighbors=knn_neighbors)
    if strategy in ("median", "mean", "most_frequent"):
        return SimpleImputer(strategy=strategy)
    raise ValueError(f"unknown impute strategy: {strategy}")


def _make_scaler(kind: Optional[str]):
    if kind is None:
        return None
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    if kind == "minmax":
        return MinMaxScaler()
    raise ValueError(f"unknown scaler: {kind}")


def _make_transform(kind: Optional[str]):
    if kind is None:
        return None
    if kind == "log1p":
        return FunctionTransformer(np.log1p, validate=False)
    if kind in ("yeo-johnson", "yeojohnson"):
        return PowerTransformer(method="yeo-johnson", standardize=False)
    raise ValueError(f"unknown numeric transform: {kind}")


def _ohe():
    """OneHotEncoder compatible with both old and new sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    df: pd.DataFrame,
    cols: list[str],
    *,
    impute: str = "median",
    scale: Optional[str] = "standard",
    numeric_transform: Optional[str] = None,
    knn_neighbors: int = 5,
) -> ColumnTransformer:
    """Configurable preprocessor.

    Parameters
    ----------
    df, cols :
        DataFrame and the columns to use as predictors.
    impute :
        "median" / "mean" / "most_frequent" / "knn" (numeric only;
        categorical always uses most_frequent).
    scale :
        "standard" / "robust" / "minmax" / None.
    numeric_transform :
        None / "log1p" / "yeo-johnson" — applied before scaling.

    Returns
    -------
    sklearn ColumnTransformer.
    """
    num_cols, cat_cols = split_feature_types(df, cols)

    num_steps: list[tuple[str, object]] = [("impute", _make_imputer(impute, knn_neighbors))]
    if numeric_transform is not None:
        num_steps.append(("transform", _make_transform(numeric_transform)))
    scaler = _make_scaler(scale)
    if scaler is not None:
        num_steps.append(("scale", scaler))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")),
         ("onehot", _ohe())]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


# --- Convenience presets ---------------------------------------------------


def standard_preprocessor(df: pd.DataFrame, cols: list[str]) -> ColumnTransformer:
    """Median impute + StandardScaler — good default for linear / SVM / NN."""
    return build_preprocessor(df, cols, impute="median", scale="standard")


def robust_preprocessor(df: pd.DataFrame, cols: list[str]) -> ColumnTransformer:
    """Median impute + RobustScaler — heavy-tailed / outlier-prone data."""
    return build_preprocessor(df, cols, impute="median", scale="robust")


def tree_preprocessor(df: pd.DataFrame, cols: list[str]) -> ColumnTransformer:
    """Median impute, no scaling — trees don't need it."""
    return build_preprocessor(df, cols, impute="median", scale=None)


def knn_imputed_preprocessor(df: pd.DataFrame, cols: list[str]) -> ColumnTransformer:
    """KNN impute + StandardScaler — better when missingness is informative."""
    return build_preprocessor(df, cols, impute="knn", scale="standard")
