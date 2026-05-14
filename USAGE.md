# 使用说明 (USAGE)

本文档说明如何加载、配置和运行本项目里的 ML 模型。所有命令都假设你在 **项目根目录 `/workspace`** 下执行。

> 项目里还有一个统计版的单变量→多变量逻辑回归脚本 `analysis/logistic_regression.py`，它和 ML 部分完全独立，本文档不重复介绍。

---

## 0. 环境准备

只需要做一次：

```bash
cd /workspace
pip install -r requirements.txt
```

`requirements.txt` 包含：`pandas`、`numpy`、`statsmodels`、`scikit-learn`、`lightgbm`、`xgboost`、`catboost`。
其中 LightGBM / XGBoost / CatBoost 是**可选**的——如果缺包，对应模型会自动从注册表里跳过，剩下的模型仍然能用。

---

## 1. 项目结构一览

```
/workspace
├── data/all_variable_final.csv      # 数据
├── analysis/logistic_regression.py  # 统计版（与本文档无关）
└── ml/
    ├── data_utils.py                # 加载 + 预处理 + 自定义列选择
    ├── quickstart.py                # 命令行一键运行
    └── models/                      # 11 个独立模型文件
        ├── logistic.py / lasso.py
        ├── random_forest.py / extra_trees.py
        ├── gradient_boosting.py / lightgbm_model.py
        ├── xgboost_model.py / catboost_model.py
        ├── svm.py / knn.py / mlp.py
        └── __init__.py              # MODEL_REGISTRY
```

`MODEL_REGISTRY` 里所有可选 key：

```
"logistic_l2"          L2 logistic 回归
"logistic_l1_lasso"    L1 logistic（LASSO）
"random_forest"        随机森林
"extra_trees"          ExtraTrees
"gradient_boosting"    sklearn HistGradientBoosting
"lightgbm"             LightGBM
"xgboost"              XGBoost
"catboost"             CatBoost
"svm_rbf"              RBF 核 SVM
"knn"                  K 近邻
"mlp"                  小型 MLP
```

查看当前环境里实际可用的模型：

```python
from ml.models import MODEL_REGISTRY
print(sorted(MODEL_REGISTRY))
```

---

## 2. 最快上手：命令行

```bash
# 默认（LightGBM + 83 个基线变量 + 5 折 CV）
python3 -m ml.quickstart

# 换一个模型
python3 -m ml.quickstart --model xgboost
python3 -m ml.quickstart --model random_forest

# 一次比较所有 11 个模型
python3 -m ml.quickstart --model all

# 自定义输入变量：逗号分隔写在命令行
python3 -m ml.quickstart \
    --model logistic_l2 \
    --columns age_years,sex,stenosis_percent,minimum_luminal_area_mm3,plaque_type,ffrct_value

# 自定义输入变量：从文件读（每行一个列名，# 开头是注释）
python3 -m ml.quickstart --model xgboost --columns-file my_features.txt

# 加入事后变量（敏感性分析）
python3 -m ml.quickstart --include-leaky
```

输出会写到：

- `results/ml/cv_summary.csv` — 每个模型的 AUC / AP / Sens / Spec
- `results/ml/<best_model>.joblib` — AUC 最高的那个模型在全量数据上 refit 后保存的完整 Pipeline

---

## 3. 在 Python / Jupyter 里用（推荐）

### 3.1 加载数据

```python
from ml import data_utils

df = data_utils.load_data()                  # 默认读 data/all_variable_final.csv
# 或者
df = data_utils.load_data("/path/to/other.csv")

print(df.shape)                              # (2208, 136)
print(df["target_lesion_positive"].value_counts())
```

### 3.2 定义 X / y

**方式 A：默认（剔除事后变量）**

```python
X, y, cols = data_utils.get_Xy(df)
# X.shape == (2208, 83)
```

**方式 B：默认 + 包含事后变量（敏感性分析）**

```python
X, y, cols = data_utils.get_Xy(df, include_leaky=True)
# X.shape == (2208, 135)
```

**方式 C：自定义列表**

```python
X, y, cols = data_utils.get_Xy(df, columns=[
    "age_years",
    "sex",
    "stenosis_percent",
    "minimum_luminal_area_mm3",
    "plaque_type",
    "lesion_stenosis_grade",
    "ffrct_value",
])
# 拼错的列名会直接 KeyError；outcome / uid 会自动剔除并 warning
```

**方式 D：按模式批量选**

```python
baseline = data_utils.find_columns(df, prefix="baseline_")          # 26 列
history  = data_utils.find_columns(df, prefix="history_")           # 6 列
plaque   = data_utils.find_columns(df, contains="plaque")           # 9 列
mine     = data_utils.find_columns(df, regex=r"^(history_|baseline_lipo)")

my_cols = list(dict.fromkeys(baseline + plaque + ["age_years", "sex"]))
X, y, cols = data_utils.get_Xy(df, columns=my_cols)
```

### 3.3 构造预处理器（所有模型共用）

```python
preprocessor = data_utils.build_preprocessor(df, cols)
```

里面做了：
- 数值列：中位数填缺失 → 标准化
- 分类列：众数填缺失 → One-Hot
- 树模型不依赖标准化，也可以用 `build_preprocessor(df, cols, scale=False)`

### 3.4 构建并训练模型

```python
from ml.models import MODEL_REGISTRY

model = MODEL_REGISTRY["lightgbm"](preprocessor)   # 选任意一个 key
model.fit(X, y)

y_prob = model.predict_proba(X)[:, 1]              # 阳性概率
y_pred = model.predict(X)                          # 0/1 硬预测
```

或者直接 import 单个模型文件：

```python
from ml.models import lightgbm_model
model = lightgbm_model.build(preprocessor)
```

### 3.5 评估（一定要用交叉验证！）

训练集 AUC 对树模型基本是 1.0，没意义。用 5 折分层 CV：

```python
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

print("AUC :", roc_auc_score(y, oof))
print("AP  :", average_precision_score(y, oof))

y_pred = (oof >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
print(f"Sens={tp/(tp+fn):.3f}  Spec={tn/(tn+fp):.3f}")
```

### 3.6 横向比较所有模型

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from ml.models import MODEL_REGISTRY

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for name, build in MODEL_REGISTRY.items():
    pipe = build(preprocessor)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    rows.append({
        "model": name,
        "auc":   roc_auc_score(y, oof),
        "ap":    average_precision_score(y, oof),
    })

print(pd.DataFrame(rows).sort_values("auc", ascending=False).to_string(index=False))
```

### 3.7 留出测试集

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=42)
model.fit(X_tr, y_tr)
print("test AUC:", roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
```

### 3.8 调参（`clf__` 前缀指向 Pipeline 里的分类器）

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    model,
    param_grid={
        "clf__n_estimators":  [300, 600, 1000],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__num_leaves":    [15, 31, 63],   # LightGBM 专用
        # "clf__max_depth":   [4, 6, 8],     # 树模型通用
        # "clf__C":           [0.01, 0.1, 1, 10],   # logistic / SVM
    },
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
)
grid.fit(X, y)

print(grid.best_score_, grid.best_params_)
best_model = grid.best_estimator_   # 已用最佳参数 refit 过整个 Pipeline
```

### 3.9 保存 / 加载模型

```python
import joblib

joblib.dump(model, "results/ml/lightgbm.joblib")   # 整个 Pipeline 一起保存

# 以后（甚至在另一个脚本里）
model = joblib.load("results/ml/lightgbm.joblib")
y_prob = model.predict_proba(new_df)[:, 1]
```

`new_df` 只要列名和原 `X` 一致即可——预处理（填缺失 / 标准化 / One-Hot）会自动应用。

### 3.10 特征重要性

```python
import pandas as pd

model.fit(X, y)
names = model.named_steps["preprocess"].get_feature_names_out()
imp   = model.named_steps["clf"].feature_importances_     # 树模型
# coefs = model.named_steps["clf"].coef_[0]               # logistic 系数

print(pd.DataFrame({"feature": names, "importance": imp})
        .sort_values("importance", ascending=False)
        .head(20)
        .to_string(index=False))
```

---

## 4. 常用变量组合套餐

```python
# A. 临床 + 病史
clinical = ["age_years", "sex"] + data_utils.find_columns(df, prefix="history_")

# B. 临床 + 影像几何 + 高危斑块特征
imaging = [
    "stenosis_percent", "minimum_luminal_area_mm3", "lesion_range_mm",
    "remodeling_index", "ffrct_value",
    "positive_remodeling_flag", "low_attenuation_plaque_flag",
    "napkin_ring_sign_flag", "spotty_calcification_flag",
    "High-Risk Plaque Flag", "plaque_type", "lesion_stenosis_grade",
]

# C. 基线生化全集
labs = data_utils.find_columns(df, prefix="baseline_")

# 任意组合
X, y, cols = data_utils.get_Xy(df, columns=clinical + imaging)
```

---

## 5. 端到端最小示例（端到端一段就能跑）

```python
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from ml import data_utils
from ml.models import MODEL_REGISTRY

# 1. 加载并选变量（这里用自定义 7 个变量演示）
df = data_utils.load_data()
my_cols = ["age_years", "sex", "stenosis_percent",
           "minimum_luminal_area_mm3", "plaque_type",
           "lesion_stenosis_grade", "ffrct_value"]
X, y, cols = data_utils.get_Xy(df, columns=my_cols)

# 2. 预处理 + 选模型
preprocessor = data_utils.build_preprocessor(df, cols)
model = MODEL_REGISTRY["lightgbm"](preprocessor)   # 换成任意 key

# 3. 5 折 CV 评估
cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
print("CV AUC =", roc_auc_score(y, oof))

# 4. 在全量上 refit 并保存
model.fit(X, y)
joblib.dump(model, "results/ml/my_model.joblib")
```

---

## 6. 注意事项

- **类别不平衡**：阳性率 15.4%，默认阈值 0.5 在树模型上会让 Sens 偏低。若要追求高灵敏度，可以基于 PR 曲线或 Youden 指数另选阈值，或在 `build_preprocessor` 后接 SMOTE 等重采样（需要 `imbalanced-learn`）。
- **训练集 AUC 不可信**：尤其是 LightGBM / XGBoost / CatBoost / 随机森林，训练集 AUC 几乎都是 1.0。**所有性能结论以交叉验证为准**。
- **事后变量**：`postop_*` / `followup_*` / `diff_*` 在主分析里默认剔除；只在做"事后能不能反推"这类敏感性分析时才开 `include_leaky=True`。
- **`baseline_ldl_level ` 末尾有空格**：原始 CSV 列名带尾空格，`load_data()` 会自动 `strip()`。如果你直接用 pandas 读 CSV，记得 `df.columns = [c.strip() for c in df.columns]`。
