"""
Train / validation / test split helpers.

Thin wrappers around sklearn that default to stratified splits — important
when the positive rate is only ~15%.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)


def stratified_split(
    X, y,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Single stratified train/test split.

    Returns (X_train, X_test, y_train, y_test).
    """
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def stratified_train_val_test(
    X, y,
    *,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified 3-way split.

    `val_size` and `test_size` are fractions of the full dataset.
    """
    # First peel off the test set.
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Then split the remainder into train + val.
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=rel_val, stratify=y_rest,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def kfold_splitter(n_splits: int = 5, *, shuffle: bool = True, random_state: int = 42):
    """Convenience constructor for a stratified k-fold object."""
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def grouped_stratified_split(
    X: pd.DataFrame, y, group_col: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Patient-level split: ensure all rows from the same `group_col` value
    end up in the same fold. Useful when one patient contributes multiple
    lesions (rows).

    Returns (X_train, X_test, y_train, y_test) with the group column intact.
    """
    groups = X[group_col]
    # GroupShuffleSplit doesn't stratify; combine via a deterministic loop.
    unique_groups = groups.drop_duplicates().reset_index(drop=True)
    # Compute group-level positive rate to stratify on.
    grp_y = y.groupby(groups).max()
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(unique_groups, grp_y.loc[unique_groups]))
    train_groups = set(unique_groups.iloc[train_idx])
    test_groups = set(unique_groups.iloc[test_idx])
    train_mask = groups.isin(train_groups)
    test_mask = groups.isin(test_groups)
    return (
        X.loc[train_mask],
        X.loc[test_mask],
        y.loc[train_mask],
        y.loc[test_mask],
    )
