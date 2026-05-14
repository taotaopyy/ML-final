"""
CSV loading with light, opt-in cleaning.

The default settings reproduce `ml.data_utils.load_data()`:
    - strip whitespace from column names
    - drop columns that are entirely empty (the trailing ',' in the source CSV
      creates one such column)

Pass `clean=False` to get an untouched DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "all_variable_final.csv"


def load_csv(
    path: Path | str | None = None,
    *,
    clean: bool = True,
    encoding: str = "utf-8",
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Read a CSV. Defaults to `data/all_variable_final.csv` in this repo.

    Parameters
    ----------
    path :
        Path to the CSV file. If `None`, the project default is used.
    clean :
        If True, strip column whitespace and drop fully empty columns.
    encoding :
        File encoding (default 'utf-8').
    **read_csv_kwargs :
        Forwarded to `pd.read_csv`.
    """
    df = pd.read_csv(path or DEFAULT_CSV, encoding=encoding, **read_csv_kwargs)
    if clean:
        df = clean_columns(df)
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and drop fully empty columns."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    return df


def coerce_numeric(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Coerce specified (or all object-like) columns to numeric.

    Non-numeric strings become NaN (so they can be imputed downstream).
    """
    df = df.copy()
    if cols is None:
        cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
