# Models

Each file in this folder defines **one** machine-learning model and exposes a
single `build(preprocessor)` function that returns a scikit-learn `Pipeline`
of the form `preprocessor -> classifier`.

| File | Classifier | Notes |
|---|---|---|
| `logistic.py` | `LogisticRegression` (L2) | Standard logistic regression baseline. |
| `lasso.py` | `LogisticRegression` (L1) | LASSO — built-in feature selection. |
| `random_forest.py` | `RandomForestClassifier` | 500 trees, balanced class weights. |
| `extra_trees.py` | `ExtraTreesClassifier` | Random-threshold trees; lower variance than RF. |
| `gradient_boosting.py` | `HistGradientBoostingClassifier` | sklearn-native histogram GBT. |
| `lightgbm_model.py` | `LGBMClassifier` (LightGBM) | Fast leaf-wise GBT; usually strong on tabular data. |
| `xgboost_model.py` | `XGBClassifier` (XGBoost) | `scale_pos_weight` set for the ~15% event rate. |
| `catboost_model.py` | `CatBoostClassifier` (CatBoost) | `auto_class_weights="Balanced"`, quiet mode. |
| `svm.py` | `SVC` (RBF, `probability=True`) | Slower; useful for benchmarking. |
| `knn.py` | `KNeighborsClassifier` | Distance-weighted, k=25. |
| `mlp.py` | `MLPClassifier` (64, 32) | Small 2-layer neural net. |

> LightGBM / XGBoost / CatBoost are optional. If their packages aren't installed,
> the registry simply skips them — everything else still works.

All models share the **same preprocessor** built by `analysis.data_utils.build_preprocessor`,
so any score difference between them comes from the classifier alone.

---

## Quick start

```python
from analysis import data_utils
from analysis.models import MODEL_REGISTRY
# Or import a single model directly:
#   from analysis.models import random_forest

# 1. Load the data (target = target_lesion_positive)
df = data_utils.load_data()
X, y, cols = data_utils.get_Xy(df)            # baseline predictors only
# X, y, cols = data_utils.get_Xy(df, include_leaky=True)  # sensitivity analysis

# 2. Build the shared preprocessor (imputation + scaling + one-hot)
preprocessor = data_utils.build_preprocessor(df, cols)

# 3. Build the model you want
model = MODEL_REGISTRY["random_forest"](preprocessor)
# equivalent: model = random_forest.build(preprocessor)

# 4. Fit / predict like any sklearn pipeline
model.fit(X, y)
y_prob = model.predict_proba(X)[:, 1]
y_pred = model.predict(X)
```

---

## Cross-validated evaluation (recommended)

The numbers above are training-set scores; for an honest estimate use CV:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

print("AUC :", roc_auc_score(y, oof_prob))
print("AP  :", average_precision_score(y, oof_prob))
```

---

## Save / load a trained model

```python
import joblib

# After fitting
joblib.dump(model, "results/ml/random_forest.joblib")

# Later, in another script
model = joblib.load("results/ml/random_forest.joblib")
y_prob = model.predict_proba(new_X)[:, 1]
```

The whole `Pipeline` (preprocessor + classifier) is saved, so `new_X` only
needs to be a raw `DataFrame` with the same columns as the original `X` —
imputation, scaling and one-hot encoding are applied automatically.

---

## Hyper-parameter tuning

Every model is a normal sklearn estimator, so `GridSearchCV` / `RandomizedSearchCV`
just works. Use the `clf__` prefix to reach the classifier inside the Pipeline:

```python
from sklearn.model_selection import GridSearchCV

model = MODEL_REGISTRY["random_forest"](preprocessor)
grid = GridSearchCV(
    model,
    param_grid={
        "clf__n_estimators": [300, 500, 800],
        "clf__min_samples_leaf": [1, 5, 10],
    },
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
)
grid.fit(X, y)
print(grid.best_params_, grid.best_score_)
```

---

## Adding a new model

1. Create `analysis/models/my_model.py` with a `build(preprocessor) -> Pipeline` function.
2. Register it in `analysis/models/__init__.py`:

   ```python
   from . import my_model
   MODEL_REGISTRY["my_model"] = my_model.build
   ```

That's it — the rest of the code can use it through `MODEL_REGISTRY["my_model"]`.
