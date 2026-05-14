# Logistic regression: target_lesion_positive

## Data
- File: `data/all_variable_final.csv` (2208 rows, 136 usable columns after dropping the trailing empty column).
- Outcome: `target_lesion_positive` (binary). Distribution: 1868 negatives / 340 positives (15.4% event rate).

## Workflow
1. Drop identifier (`uid`) and the trailing all-empty column.
2. Two predictor sets:
   - **baseline** (primary): 81 features, excludes any column starting with `postop_`, `followup_`, or `diff_` to avoid post-outcome leakage.
   - **all_variables** (sensitivity): 133 features, including post-op / follow-up / diff measures. Interpret with caution as these are measured after the index event.
3. Categorical columns (`lesion_stenosis_grade`, `plaque_type`) are one-hot encoded (drop-first). Constant columns are removed.
4. **Univariate**: logistic regression of `target_lesion_positive ~ const + X_i` for each predictor; reports OR with 95% CI and p-value.
5. **Multivariable**: re-fit a single Logit model including every variable with univariate `p < 0.05`. Outputs ORs, CIs, p-values, AUC, McFadden pseudo-R², AIC/BIC, and LR-test p-value.

## Files
| File | Description |
|---|---|
| `univariate_baseline.csv` | All univariate fits, sorted by p-value (baseline set). |
| `multivariable_baseline.csv` | Multivariable fit using baseline variables with univariate p<0.05. |
| `multivariable_baseline_metrics.json` | Overall metrics for the baseline multivariable model. |
| `univariate_all_variables.csv` | Same as above but for the full predictor set (sensitivity). |
| `multivariable_all_variables.csv` | Multivariable fit using all-variable significant set. |
| `multivariable_all_variables_metrics.json` | Metrics for the sensitivity multivariable model. |

## Headline results

### Baseline predictor set (primary)
- Univariate p<0.05: 37 / 81 variables.
- Multivariable Logit (n=1796 complete cases, 284 events, 37 predictors):
  - LR p-value ≈ 1.8e-10
  - McFadden pseudo-R² ≈ 0.075
  - In-sample **AUC ≈ 0.696**
  - Independently significant predictors after adjustment:
    - `stenosis_percent` — OR 5.92 (1.55–22.61), p = 0.0093
    - `minimum_luminal_area_mm3` — OR 0.86 (0.76–0.98), p = 0.021

### All-variable sensitivity set
- Univariate p<0.05: 46 / 133 variables.
- Multivariable Logit (n=1065 complete cases, 151 events, 46 predictors):
  - LR p-value ≈ 5.8e-6
  - McFadden pseudo-R² ≈ 0.114
  - In-sample **AUC ≈ 0.731**
  - Independently significant predictor: `stenosis_percent`, OR 38.16 (5.90–246.87), p = 1.3e-4. Most other terms become non-significant once adjusted, suggesting heavy collinearity within the post-op/follow-up bloc.

## Caveats
- AUC is reported on the training data (no cross-validation); expect optimism.
- "p<0.05 in / out" is a classic but biased screening rule; consider penalised regression (LASSO/elastic-net) or stability selection for a more honest variable set, especially given strong collinearity among plaque-burden volumes/ratios.
- Several plaque-burden ORs are huge with very wide CIs because the predictor is on a 0–1 ratio scale; consider rescaling (e.g. per 10%) before reporting.
- Complete-case analysis drops rows with any missing predictor; for the sensitivity model this is roughly half the sample. Imputation would change the number of events available for fitting.
- `followup_*`, `postop_*`, and `diff_*` features are downstream of the outcome window and should not be treated as predictors in a clinical-decision context.
