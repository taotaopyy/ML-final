"""
Class-imbalance handling via resampling.

Wraps a fitted preprocessor + resampler into an `imblearn.pipeline.Pipeline`
that you can use exactly like a regular sklearn Pipeline (with `fit` and
`predict_proba`). Resampling only runs at fit time, so it doesn't leak into
cross-validation or test predictions.

Requires the `imbalanced-learn` package.
"""

from __future__ import annotations

from typing import Literal

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler

    _HAS_IMBLEARN = True
except ImportError:  # pragma: no cover
    _HAS_IMBLEARN = False


def _require():
    if not _HAS_IMBLEARN:
        raise ImportError(
            "imbalanced-learn is not installed. Run `pip install imbalanced-learn`."
        )


def make_resampler(
    strategy: Literal["smote", "random_over", "random_under"] = "smote",
    *,
    sampling_strategy: float | str = "auto",
    random_state: int = 42,
    k_neighbors: int = 5,
):
    """Return an imbalanced-learn resampler.

    sampling_strategy :
        float — desired minority/majority ratio after resampling.
        "auto" / "minority" / "not minority" — see imblearn docs.
    """
    _require()
    if strategy == "smote":
        return SMOTE(sampling_strategy=sampling_strategy,
                     random_state=random_state, k_neighbors=k_neighbors)
    if strategy == "random_over":
        return RandomOverSampler(sampling_strategy=sampling_strategy,
                                 random_state=random_state)
    if strategy == "random_under":
        return RandomUnderSampler(sampling_strategy=sampling_strategy,
                                  random_state=random_state)
    raise ValueError(f"unknown resampling strategy: {strategy}")


def wrap_with_resampling(preprocessor, classifier, *, strategy: str = "smote",
                         sampling_strategy: float | str = "auto",
                         random_state: int = 42):
    """Compose `preprocessor -> resampler -> classifier` as an imblearn Pipeline.

    The result has the standard sklearn `fit / predict / predict_proba`
    interface but applies resampling only during `fit`, which is the
    statistically correct way to use SMOTE etc. inside cross-validation.
    """
    _require()
    return ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("resample", make_resampler(strategy, sampling_strategy=sampling_strategy,
                                        random_state=random_state)),
            ("clf", classifier),
        ]
    )
