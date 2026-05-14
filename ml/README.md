# `ml/` — 机器学习模型

11 个独立模型 + 数据加载 / 默认预处理 + 一个命令行入口。

```
ml/
├── data_utils.py        # load_data / get_Xy / build_preprocessor / find_columns
├── quickstart.py        # 命令行一键 CV + 保存最优模型
├── models/              # 11 个独立模型，每个一个文件
│   ├── __init__.py      # MODEL_REGISTRY
│   ├── logistic.py / lasso.py
│   ├── random_forest.py / extra_trees.py
│   ├── gradient_boosting.py
│   ├── lightgbm_model.py / xgboost_model.py / catboost_model.py
│   ├── svm.py / knn.py / mlp.py
│   └── README.md        # 模型清单 + 添加新模型的步骤
└── README.md            # 当前文件
```

可用模型 key：`logistic_l2`、`logistic_l1_lasso`、`random_forest`、`extra_trees`、`gradient_boosting`、`lightgbm`、`xgboost`、`catboost`、`svm_rbf`、`knn`、`mlp`。详见 [`models/README.md`](models/README.md)。

---

## 1. 命令行一键运行：`quickstart.py`

```bash
# 默认（lightgbm + 83 个基线变量）
python3 -m ml.quickstart

# 选模型
python3 -m ml.quickstart --model xgboost
python3 -m ml.quickstart --model logistic_l2

# 一次比较所有 11 个模型
python3 -m ml.quickstart --model all

# 自定义输入变量
python3 -m ml.quickstart --columns age_years,sex,stenosis_percent,plaque_type
python3 -m ml.quickstart --columns-file my_features.txt --model xgboost

# 把事后变量（postop_/followup_/diff_）也加进来做敏感性分析
python3 -m ml.quickstart --include-leaky
```

### 流程
1. 加载 `data/all_variable_final.csv`，按命令行参数选预测变量。
2. 用 `data_utils.build_preprocessor` 构造预处理器（中位数填补 + 标准化 + One-Hot）。
3. 跑 **5 折分层交叉验证**，按 AUC / AP / Sens / Spec 评估。
4. 把对比表写到 `results/ml/cv_summary.csv`。
5. 在全量数据上 refit AUC 最高的模型，存为 `results/ml/<best_model>.joblib`，并演示加载并对第一行做预测。

### `--columns-file` 的格式

每行一个列名，`#` 开头当注释，空行忽略：

```
# 临床基础
age_years
sex
# 影像
stenosis_percent
minimum_luminal_area_mm3
plaque_type
ffrct_value
```

---

## 2. Python API

### 2.1 `data_utils.load_data`

```python
from ml import data_utils

df = data_utils.load_data()                # 默认 data/all_variable_final.csv
df = data_utils.load_data("/path/x.csv")   # 任意 CSV
```

自动 strip 列名 + 删除整列为空的列（原始 CSV 末尾有 `,` 会产生一个空列）。

### 2.2 `data_utils.get_Xy` —— 定义 X / y

**默认（剔除事后变量，83 个基线变量）：**

```python
X, y, cols = data_utils.get_Xy(df)
```

**自定义变量列表：**

```python
X, y, cols = data_utils.get_Xy(df, columns=[
    "age_years", "sex", "stenosis_percent",
    "minimum_luminal_area_mm3", "plaque_type", "ffrct_value",
])
```

> 拼错列名会直接抛 `KeyError`；如果不小心放进了 `target_lesion_positive` 或 `uid`，会自动剔除并给一个 warning。

**包含事后变量（敏感性分析）：**

```python
X, y, cols = data_utils.get_Xy(df, include_leaky=True)
```

### 2.3 `data_utils.find_columns` —— 按模式批量选

```python
baseline = data_utils.find_columns(df, prefix="baseline_")       # 26 列
history  = data_utils.find_columns(df, prefix="history_")        # 6 列
plaque   = data_utils.find_columns(df, contains="plaque")        # 9 列
mine     = data_utils.find_columns(df, regex=r"^(history_|baseline_lipo)")

my_cols = list(dict.fromkeys(baseline + plaque + ["age_years", "sex"]))
X, y, cols = data_utils.get_Xy(df, columns=my_cols)
```

默认会排除事后变量；要包含的话加 `exclude_leaky=False`。

### 2.4 `data_utils.build_preprocessor` —— 默认预处理器

```python
preprocessor = data_utils.build_preprocessor(df, cols)              # 含 StandardScaler
preprocessor = data_utils.build_preprocessor(df, cols, scale=False) # 树模型可省略
```

里面做的事：
- 数值列：中位数填补 → 标准化（可关）
- 分类列：众数填补 → One-Hot（`handle_unknown="ignore"`）

> 想要更复杂的预处理（RobustScaler / KNN 插补 / log 变换 / SMOTE 等），用 [`preprocessing/`](../preprocessing/README.md) 里的工具，把它的返回值直接当 `preprocessor` 喂进来即可。

### 2.5 选模型 → fit → predict

```python
from ml.models import MODEL_REGISTRY

model = MODEL_REGISTRY["lightgbm"](preprocessor)   # 或任意 key
model.fit(X, y)
y_prob = model.predict_proba(X)[:, 1]
y_pred = model.predict(X)
```

等价于：

```python
from ml.models import lightgbm_model
model = lightgbm_model.build(preprocessor)
```

### 2.6 交叉验证

训练集 AUC 对树模型几乎都是 1.0，**必须用 CV**：

```python
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
print("AUC :", roc_auc_score(y, oof))
print("AP  :", average_precision_score(y, oof))
```

详细评测（bootstrap CI / Youden 阈值 / 校准曲线 / ROC PR 图 / SHAP）见 [`evaluation/README.md`](../evaluation/README.md)。

### 2.7 调参

`clf__` 前缀指向 Pipeline 里的分类器：

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    model,
    param_grid={
        "clf__n_estimators":  [300, 600, 1000],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__num_leaves":    [15, 31, 63],          # LightGBM
        # "clf__max_depth":   [4, 6, 8],            # XGBoost
        # "clf__C":           [0.01, 0.1, 1, 10],   # logistic / SVM
    },
    scoring="roc_auc", cv=5, n_jobs=-1,
)
grid.fit(X, y)
best_model = grid.best_estimator_
print(grid.best_score_, grid.best_params_)
```

### 2.8 保存 / 加载

```python
import joblib

joblib.dump(model, "results/ml/lightgbm.joblib")    # 整个 Pipeline 一起保存

model = joblib.load("results/ml/lightgbm.joblib")
y_prob = model.predict_proba(new_df)[:, 1]          # new_df 列名要和训练时一致
```

预处理器随模型一起保存，所以新数据不用再手动填补 / 标准化 / One-Hot。

---

## 3. 添加新模型

详见 [`models/README.md`](models/README.md)。简短版：

```python
# ml/models/my_model.py
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

def build(preprocessor):
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05)
    return Pipeline([("preprocess", preprocessor), ("clf", clf)])
```

```python
# ml/models/__init__.py 里增加一行
from . import my_model
MODEL_REGISTRY["my_model"] = my_model.build
```

之后 `MODEL_REGISTRY["my_model"]` 就能用了，`quickstart.py` 也自动支持。

---

## 4. 11 模型 CV 基准（默认 83 个基线变量，5 折分层，阈值 0.5）

| model | AUC | AP | Sens | Spec |
|---|---|---|---|---|
| random_forest | **0.654** | 0.268 | 0.047 | 0.989 |
| extra_trees | 0.653 | 0.259 | 0.200 | 0.907 |
| lightgbm | 0.641 | 0.290 | 0.138 | 0.967 |
| xgboost | 0.638 | 0.284 | 0.124 | 0.973 |
| logistic_l1_lasso | 0.628 | 0.222 | 0.541 | 0.619 |
| catboost | 0.624 | 0.274 | 0.124 | 0.960 |
| gradient_boosting | 0.616 | 0.229 | 0.391 | 0.772 |
| logistic_l2 | 0.612 | 0.212 | 0.494 | 0.635 |
| svm_rbf | 0.607 | 0.210 | 0.000 | 1.000 |
| knn | 0.601 | 0.210 | 0.003 | 1.000 |
| mlp | 0.507 | 0.159 | 0.000 | 0.999 |

阈值 0.5 在 15% 阳性率上对树模型偏保守，所以 Sens 普遍低；可以用 `evaluation.metrics.best_threshold(..., criterion="youden")` 重选阈值，例如 LightGBM 在阈值 0.019 处可达 Sens=0.65 / Spec=0.57。
