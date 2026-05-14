# Target lesion logistic regression

Minimal logistic-regression pipeline for the `target_lesion_positive` outcome:
univariate screen of every candidate predictor → multivariable model on the variables with `p < 0.05`.

## Usage

```bash
pip install -r requirements.txt
python3 analysis/logistic_regression.py
```

Outputs land in `results/` (univariate tables, multivariable tables, JSON metrics,
and a README summarising the headline numbers).
