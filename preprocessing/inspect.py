"""
EDA / sanity-check reports.

All functions return tidy DataFrames so you can save them or filter them
programmatically (no plotting here — see `evaluation.plots` for that).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def overview(df: pd.DataFrame) -> dict:
    """One-shot summary: shape, dtype counts, total missing rate."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "dtype_counts": df.dtypes.astype(str).value_counts().to_dict(),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_rate": float(df.isna().mean().mean()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def missing_report(df: pd.DataFrame, *, sort: bool = True) -> pd.DataFrame:
    """Per-column missingness."""
    miss = df.isna().sum()
    rate = df.isna().mean()
    out = pd.DataFrame(
        {"n_missing": miss.astype(int), "missing_rate": rate.astype(float)}
    )
    out["dtype"] = df.dtypes.astype(str)
    out["n_unique"] = df.nunique(dropna=True).astype(int)
    if sort:
        out = out.sort_values("missing_rate", ascending=False)
    return out.reset_index().rename(columns={"index": "column"})


def cardinality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Number of distinct values per column (helps spot quasi-constant features)."""
    n = len(df)
    out = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "n_unique": df.nunique(dropna=True).values,
        }
    )
    out["unique_rate"] = out["n_unique"] / max(n, 1)
    out["is_constant"] = out["n_unique"] <= 1
    return out.sort_values("n_unique").reset_index(drop=True)


def numeric_summary(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """describe() + skewness / kurtosis for numeric columns."""
    num_cols = cols or [c for c in df.columns if is_numeric_dtype(df[c])]
    sub = df[num_cols]
    desc = sub.describe().T
    desc["skew"] = sub.skew(numeric_only=True)
    desc["kurt"] = sub.kurt(numeric_only=True)
    desc["n_missing"] = sub.isna().sum()
    return desc.reset_index().rename(columns={"index": "column"})


def outlier_report(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    *,
    method: str = "iqr",
    iqr_mult: float = 1.5,
    z_thresh: float = 3.0,
) -> pd.DataFrame:
    """How many extreme values each numeric column has.

    method :
        "iqr" — values outside [Q1 - k*IQR, Q3 + k*IQR]
        "z"   — |z-score| > z_thresh (uses median/MAD for robustness)
    """
    num_cols = cols or [c for c in df.columns if is_numeric_dtype(df[c])]
    rows = []
    for c in num_cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(x) == 0:
            rows.append({"column": c, "n_outliers": 0, "outlier_rate": 0.0,
                         "low": np.nan, "high": np.nan})
            continue
        if method == "iqr":
            q1, q3 = np.quantile(x, [0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
        elif method == "z":
            med = np.median(x)
            mad = np.median(np.abs(x - med)) or 1e-9
            lo, hi = med - z_thresh * 1.4826 * mad, med + z_thresh * 1.4826 * mad
        else:
            raise ValueError(f"unknown method: {method}")
        n_out = int(((x < lo) | (x > hi)).sum())
        rows.append({"column": c, "n_outliers": n_out,
                     "outlier_rate": n_out / len(x), "low": lo, "high": hi})
    return pd.DataFrame(rows).sort_values("outlier_rate", ascending=False).reset_index(drop=True)


def target_balance(y) -> pd.Series:
    """Class distribution helper."""
    y = pd.Series(y)
    counts = y.value_counts(dropna=False)
    rates = y.value_counts(normalize=True, dropna=False)
    return pd.DataFrame({"count": counts, "rate": rates}).sort_index()
