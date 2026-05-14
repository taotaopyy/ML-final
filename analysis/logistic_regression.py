"""
Logistic regression analysis: univariate screening -> multivariate model.

Workflow:
1. Load the CSV at data/all_variable_final.csv.
2. Outcome: target_lesion_positive (binary).
3. Run univariate logistic regression for every candidate predictor.
4. Carry variables with p < 0.05 into a multivariable model.
5. Save tidy tables and overall model metrics to results/.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "all_variable_final.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTCOME = "target_lesion_positive"

# Columns that should not be predictors:
# - uid: identifier
# - postop_*, followup_*, diff_*: measured after the index event,
#   so they're not appropriate predictors of target lesion positivity.
#   We still report a sensitivity run that includes them, but the primary
#   model uses only baseline / lesion-imaging features.
LEAKY_PREFIXES = ("postop_", "followup_", "diff_")
ID_COLS = {"uid"}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    # Drop any fully empty columns (the CSV has a trailing comma -> empty col).
    df = df.dropna(axis=1, how="all")
    return df


def split_predictors(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (baseline_predictors, all_predictors_including_post)."""
    cols = [c for c in df.columns if c != OUTCOME and c not in ID_COLS]
    baseline = [c for c in cols if not c.startswith(LEAKY_PREFIXES)]
    return baseline, cols


def prepare_design(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert categorical columns to dummies; coerce others to numeric."""
    X = df[cols].copy()
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dummy_na=False)

    # Drop variables that are constant after coercion.
    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique > 1]
    # Cast everything to float (statsmodels chokes on bool dummy cols).
    X = X.astype(float)
    return X


def fit_univariate(y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        sub = pd.concat([y, X[col]], axis=1).dropna()
        if sub.shape[0] < 30 or sub[col].nunique() < 2:
            continue
        if sub[y.name].nunique() < 2:
            continue
        try:
            xmat = sm.add_constant(sub[[col]], has_constant="add")
            model = sm.Logit(sub[y.name], xmat).fit(disp=False, maxiter=200)
        except Exception as exc:  # singular, separation, etc.
            rows.append(
                {
                    "variable": col,
                    "n": sub.shape[0],
                    "coef": np.nan,
                    "se": np.nan,
                    "OR": np.nan,
                    "OR_CI_low": np.nan,
                    "OR_CI_high": np.nan,
                    "p_value": np.nan,
                    "note": f"fit_failed: {type(exc).__name__}",
                }
            )
            continue

        coef = model.params[col]
        se = model.bse[col]
        p = model.pvalues[col]
        ci_low, ci_high = model.conf_int().loc[col]
        rows.append(
            {
                "variable": col,
                "n": sub.shape[0],
                "coef": coef,
                "se": se,
                "OR": np.exp(coef),
                "OR_CI_low": np.exp(ci_low),
                "OR_CI_high": np.exp(ci_high),
                "p_value": p,
                "note": "",
            }
        )
    out = pd.DataFrame(rows).sort_values("p_value", na_position="last").reset_index(drop=True)
    return out


def fit_multivariable(y: pd.Series, X: pd.DataFrame, predictors: list[str]) -> tuple[pd.DataFrame, dict]:
    sub = pd.concat([y, X[predictors]], axis=1).dropna()
    if sub.empty:
        raise RuntimeError("No complete cases for the multivariable model.")
    xmat = sm.add_constant(sub[predictors], has_constant="add")
    model = sm.Logit(sub[y.name], xmat).fit(disp=False, maxiter=500)
    params = model.params
    ses = model.bse
    pvals = model.pvalues
    ci = model.conf_int()

    table = pd.DataFrame(
        {
            "variable": params.index,
            "coef": params.values,
            "se": ses.values,
            "OR": np.exp(params.values),
            "OR_CI_low": np.exp(ci[0].values),
            "OR_CI_high": np.exp(ci[1].values),
            "p_value": pvals.values,
        }
    )

    preds = model.predict(xmat)
    auc = roc_auc_score(sub[y.name], preds)

    metrics = {
        "n": int(sub.shape[0]),
        "events": int(sub[y.name].sum()),
        "n_predictors": len(predictors),
        "llf": float(model.llf),
        "ll_null": float(model.llnull),
        "pseudo_r2_mcfadden": float(model.prsquared),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "lr_pvalue": float(model.llr_pvalue),
        "auc": float(auc),
    }
    return table.sort_values("p_value").reset_index(drop=True), metrics


def run(predictor_set_name: str, df: pd.DataFrame, cols: list[str]) -> None:
    print(f"\n========== {predictor_set_name} ==========")
    print(f"Candidate columns (raw): {len(cols)}")

    X = prepare_design(df, cols)
    y = df[OUTCOME].astype(int)
    print(f"Design matrix columns after encoding/cleaning: {X.shape[1]}")
    print(f"Outcome distribution: {y.value_counts().to_dict()}")

    uni = fit_univariate(y, X)
    uni_path = RESULTS_DIR / f"univariate_{predictor_set_name}.csv"
    uni.to_csv(uni_path, index=False)
    print(f"Saved univariate table -> {uni_path}")

    sig = uni[(uni["p_value"] < 0.05) & uni["p_value"].notna()]["variable"].tolist()
    print(f"Variables with univariate p<0.05: {len(sig)}")

    if not sig:
        print("No variables met p<0.05 threshold; skipping multivariable fit.")
        return

    try:
        multi, metrics = fit_multivariable(y, X, sig)
    except Exception as exc:
        print(f"Multivariable fit failed with full p<0.05 set: {exc}")
        # Try progressively dropping high-p variables to recover convergence.
        ordered = uni[uni["variable"].isin(sig)].sort_values("p_value")["variable"].tolist()
        for k in range(len(ordered) - 1, 0, -1):
            subset = ordered[:k]
            try:
                multi, metrics = fit_multivariable(y, X, subset)
                print(f"Fallback succeeded with top {k} variables.")
                sig = subset
                break
            except Exception:
                continue
        else:
            raise

    multi_path = RESULTS_DIR / f"multivariable_{predictor_set_name}.csv"
    multi.to_csv(multi_path, index=False)
    metrics_path = RESULTS_DIR / f"multivariable_{predictor_set_name}_metrics.json"
    pd.Series(metrics).to_json(metrics_path, indent=2)
    print(f"Saved multivariable table -> {multi_path}")
    print(f"Saved multivariable metrics -> {metrics_path}")
    print(f"Multivariable summary: {metrics}")
    print("\nTop 15 multivariable terms (by p):")
    print(multi.head(15).to_string(index=False))


def main() -> None:
    df = load_data()
    print(f"Loaded data: {df.shape[0]} rows x {df.shape[1]} columns")
    if OUTCOME not in df.columns:
        raise SystemExit(f"Outcome column '{OUTCOME}' not in data.")

    baseline, all_preds = split_predictors(df)
    # Primary analysis: baseline + lesion imaging features only.
    run("baseline", df, baseline)
    # Sensitivity: include post-op / follow-up / diff variables (interpret with caution).
    run("all_variables", df, all_preds)


if __name__ == "__main__":
    main()
