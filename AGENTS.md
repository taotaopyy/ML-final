## Cursor Cloud specific instructions

### Project overview

This is a clinical data science project predicting `target_lesion_positive` (binary cardiac/vascular outcome) from ~2200 patient records. It has no web server, database, or external services — it is a pure Python ML pipeline.

### Running the analysis

- **Install**: `pip install -r requirements.txt`
- **Original logistic regression**: `python3 analysis/logistic_regression.py` — outputs to `results/`
- **Improved ML models**: `python3 -m analysis.improved_model` — runs feature engineering + ensemble evaluation
- **Advanced tuning**: `python3 -m analysis.tune_and_evaluate` — runs Optuna hyperparameter search (can take 30+ minutes)

### Key caveats

- The dataset has weak individual feature signals (max |correlation| ≈ 0.16 with target). CV AUC for all models caps around 0.66 regardless of model complexity, tuning, or feature engineering.
- The original analysis in `logistic_regression.py` reports **in-sample AUC** (training-set AUC). This is optimistic and should not be confused with cross-validated AUC.
- Features prefixed `postop_`, `followup_`, `diff_` are measured after the outcome window. Including them is a sensitivity analysis only — they should not be used for clinical prediction.
- About 16% of lab-value features are missing. The shared preprocessor (`data_utils.build_preprocessor`) uses median imputation + standard scaling.
- CatBoost tuning with Optuna is particularly slow (~10+ minutes for 30 trials on this dataset).
