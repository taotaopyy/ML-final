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
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    include_leaky: bool = False,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, predictor_columns).

    Parameters
    ----------
    df :
        DataFrame from `load_data()`.
    columns :
        Explicit list of column names to use as predictors. If provided, this
        overrides the default selection (any names not present in `df`, the
        outcome column, or `uid` are dropped, and a warning is raised if
        anything was filtered out).
    include_leaky :
        When `columns is None`, whether to keep `postop_* / followup_* / diff_*`
        variables. Ignored when `columns` is given (you fully control the set).
    """
    if columns is None:
        cols = get_predictor_columns(df, include_leaky=include_leaky)
    else:
        # Validate user-provided list.
        requested = list(dict.fromkeys(columns))  # dedupe, keep order
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise KeyError(
                f"{len(missing)} requested column(s) not in df: {missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
            )
        bad = [c for c in requested if c == OUTCOME or c in ID_COLS]
        cols = [c for c in requested if c not in bad]
        if bad:
            import warnings

            warnings.warn(
                f"Dropped {len(bad)} column(s) from your list because they are "
                f"the outcome or an id: {bad}",
                stacklevel=2,
            )

    X = df[cols].copy()
    y = df[OUTCOME].astype(int)
    return X, y, cols


def find_columns(
    df: pd.DataFrame,
    *,
    prefix: str | tuple[str, ...] | None = None,
    contains: str | None = None,
    regex: str | None = None,
    exclude_leaky: bool = True,
) -> list[str]:
    """Helper to discover predictor columns by name pattern.

    Examples
    --------
    >>> find_columns(df, prefix="baseline_")
    >>> find_columns(df, contains="plaque")
    >>> find_columns(df, regex=r"^(history_|baseline_).*")
    """
    cols = [c for c in df.columns if c != OUTCOME and c not in ID_COLS]
    if exclude_leaky:
        cols = [c for c in cols if not c.startswith(LEAKY_PREFIXES)]
    if prefix is not None:
        cols = [c for c in cols if c.startswith(prefix)]
    if contains is not None:
        cols = [c for c in cols if contains in c]
    if regex is not None:
        import re

        pat = re.compile(regex)
        cols = [c for c in cols if pat.search(c)]
    return cols
