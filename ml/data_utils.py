"""
Shared data loading and preprocessing utilities.

All ML models in `analysis/models/*` build a sklearn Pipeline of the form
    [preprocessor] -> [classifier]
where `preprocessor` is the ColumnTransformer returned by `build_preprocessor`.

Keeping the preprocessor here (instead of inside each model file) means every
model sees the exact same X matrix, so differences in CV scores are due to the
classifier, not to feature engineering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "all_variable_final.csv"

OUTCOME = "target_lesion_positive"
ID_COLS = {"uid"}
LEAKY_PREFIXES = ("postop_", "followup_", "diff_")


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Read the CSV, strip column whitespace, drop the trailing empty column."""
    df = pd.read_csv(path or DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    return df


def get_predictor_columns(
    df: pd.DataFrame,
    include_leaky: bool = False,
) -> list[str]:
    """Return the list of candidate predictor columns.

    Set `include_leaky=True` to keep `postop_*`, `followup_*`, `diff_*`
    (sensitivity analysis only — these are measured after the outcome window).
    """
    cols = [c for c in df.columns if c != OUTCOME and c not in ID_COLS]
    if not include_leaky:
        cols = [c for c in cols if not c.startswith(LEAKY_PREFIXES)]
    return cols


def split_feature_types(
    df: pd.DataFrame, cols: list[str]
) -> tuple[list[str], list[str]]:
    """Numeric columns vs object/categorical columns.

    Uses pandas' `is_numeric_dtype` so we catch both legacy `object` columns
    and the newer pandas-native `string` / `str` dtype.
    """
    from pandas.api.types import is_numeric_dtype

    num_cols = [c for c in cols if is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(
    df: pd.DataFrame,
    cols: list[str],
    *,
    scale: bool = True,
) -> ColumnTransformer:
    """Build the shared preprocessing ColumnTransformer.

    Numeric  : median imputation (+ optional StandardScaler).
    Object   : most_frequent imputation + one-hot encoding (ignore unknown).

    `scale=True` is required for linear / distance / NN models (logistic, SVM,
    KNN, MLP). Tree-based models don't need scaling but it doesn't hurt them
    either; `scale=False` is offered as an opt-out for slightly faster trees.
    """
    num_cols, cat_cols = split_feature_types(df, cols)

    num_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    # `sparse_output` was renamed from `sparse` in scikit-learn 1.2.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers, remainder="drop")


def get_Xy(
    df: pd.DataFrame, include_leaky: bool = False
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Convenience: return (X, y, predictor_columns)."""
    cols = get_predictor_columns(df, include_leaky=include_leaky)
    X = df[cols].copy()
    y = df[OUTCOME].astype(int)
    return X, y, cols
