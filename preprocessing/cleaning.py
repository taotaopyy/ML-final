"""
DataFrame-level cleaning operations.

All functions are pure (return new DataFrames) and never modify the input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def drop_high_missing_columns(
    df: pd.DataFrame, threshold: float = 0.4, exclude: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns whose missing rate is strictly greater than `threshold`.

    Returns (cleaned_df, dropped_columns).
    """
    exclude = set(exclude or [])
    rates = df.isna().mean()
    drop_cols = [c for c, r in rates.items() if r > threshold and c not in exclude]
    return df.drop(columns=drop_cols), drop_cols


def drop_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with 0 or 1 unique non-NaN values."""
    drop_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    return df.drop(columns=drop_cols), drop_cols


def drop_high_missing_rows(
    df: pd.DataFrame, threshold: float = 0.5
) -> tuple[pd.DataFrame, int]:
    """Drop rows whose missing rate is strictly greater than `threshold`."""
    rates = df.isna().mean(axis=1)
    keep = rates <= threshold
    return df.loc[keep].copy(), int((~keep).sum())


def clip_outliers_iqr(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    *,
    iqr_mult: float = 1.5,
) -> pd.DataFrame:
    """Winsorise numeric columns to the IQR fences."""
    df = df.copy()
    num_cols = cols or [c for c in df.columns if is_numeric_dtype(df[c])]
    for c in num_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
        df[c] = x.clip(lower=lo, upper=hi)
    return df


def winsorize_quantiles(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Clip numeric columns at the given lower/upper quantiles."""
    df = df.copy()
    num_cols = cols or [c for c in df.columns if is_numeric_dtype(df[c])]
    for c in num_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        lo, hi = x.quantile(lower), x.quantile(upper)
        df[c] = x.clip(lower=lo, upper=hi)
    return df


def log_transform_skewed(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    *,
    skew_threshold: float = 1.0,
    offset: float = 1.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply log1p(x + offset - min(x, 0)) to highly right-skewed columns.

    Only columns with skewness > `skew_threshold` and all values >= 0 are
    transformed; the rest are left as-is. Returns the transformed list of
    columns alongside the modified frame.
    """
    df = df.copy()
    num_cols = cols or [c for c in df.columns if is_numeric_dtype(df[c])]
    transformed = []
    for c in num_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.dropna().empty:
            continue
        if x.min() < 0:
            continue
        sk = x.skew()
        if sk is not None and sk > skew_threshold:
            df[c] = np.log1p(x + offset)
            transformed.append(c)
    return df, transformed


def deduplicate_rows(df: pd.DataFrame, subset: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """Drop duplicate rows. Returns (df, n_dropped)."""
    before = len(df)
    out = df.drop_duplicates(subset=subset, keep="first")
    return out, before - len(out)
