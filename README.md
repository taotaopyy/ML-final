# ML-final — `target_lesion_positive` 预测项目

预测 `target_lesion_positive`（靶病变阳性）的端到端工作台，包含：

| 模块 | 用途 |
|---|---|
| [`analysis/`](analysis/README.md) | 统计版逻辑回归（单变量 → 多变量） |
| [`preprocessing/`](preprocessing/README.md) | 数据加载 / 清洗 / 预处理 / 切分 / 不平衡处理 |
| [`ml/`](ml/README.md) | 11 个机器学习模型 + 一键命令行入口 |
| [`evaluation/`](evaluation/README.md) | 数值指标 / 评测图 / SHAP 可解释性 |

四个模块完全解耦——你可以只用其中一个、几个、或者全部串起来。每个模块下都有自己的 README 详细讲它的 API；本文是项目总入口。

---

## 1. 安装

```bash
cd /workspace
pip install -r requirements.txt
```

`requirements.txt` 内容：

```
pandas / numpy / scikit-learn / statsmodels
lightgbm / xgboost / catboost       # 可选；缺包会自动跳过
shap / matplotlib                   # evaluation/ 用
imbalanced-learn                    # preprocessing/imbalance 用
```

---

## 2. 项目结构

```
/workspace
├── data/
│   └── all_variable_final.csv     # 原始数据（2208 行 × 137 列）
│
├── analysis/                       # 统计版分析
│   ├── logistic_regression.py     # 单变量 -> 多变量 Logit
│   └── README.md
│
├── preprocessing/                  # 预处理工具箱
│   ├── loaders.py                 # 加载 + 列名清洗
│   ├── inspect.py                 # EDA 报告（DataFrame）
│   ├── cleaning.py                # 删高缺失列/Winsorize/log 等
│   ├── transformers.py            # 4 个预设 + 完全可配置 builder
│   ├── imbalance.py               # SMOTE / 过采样 / 欠采样
│   ├── splits.py                  # 分层切分 / 患者级切分
│   └── README.md
│
├── ml/                             # 机器学习模型
│   ├── data_utils.py              # 默认数据加载与预处理
│   ├── quickstart.py              # 命令行一键运行
│   ├── models/                    # 11 个独立模型文件
│   │   ├── logistic.py / lasso.py
│   │   ├── random_forest.py / extra_trees.py
│   │   ├── gradient_boosting.py
│   │   ├── lightgbm_model.py / xgboost_model.py / catboost_model.py
│   │   ├── svm.py / knn.py / mlp.py
│   │   ├── __init__.py            # MODEL_REGISTRY
│   │   └── README.md
│   └── README.md
│
├── evaluation/                     # 评测 + SHAP
│   ├── metrics.py                 # AUC / AP / Brier / 阈值搜索 / bootstrap CI
│   ├── plots.py                   # ROC / PR / 校准 / 混淆矩阵 / 多模型对比
│   ├── shap_analysis.py           # 自动选 Tree/Linear/Kernel explainer
│   └── README.md
│
├── results/                        # 所有产出（CSV / JSON / PNG / joblib）
├── requirements.txt
└── README.md  ← 当前文件
```

---

## 3. 30 秒上手：一键命令行

最快的一条路径——`ml/quickstart.py`：5 折交叉验证评估某个模型，并把最优模型保存成 `.joblib`。

```bash
# 默认（LightGBM + 83 个基线变量）
python3 -m ml.quickstart

# 换模型
python3 -m ml.quickstart --model xgboost
python3 -m ml.quickstart --model logistic_l2

# 一次比较所有 11 个模型
python3 -m ml.quickstart --model all

# 自定义输入变量
python3 -m ml.quickstart --columns age_years,stenosis_percent,plaque_type
python3 -m ml.quickstart --columns-file my_features.txt --model xgboost
```

输出：
- `results/ml/cv_summary.csv` — 各模型 5 折 CV 的 AUC / AP / Sens / Spec
- `results/ml/<best_model>.joblib` — AUC 最高的模型，在全量数据上 refit 后保存的完整 Pipeline

所有 `--model` 取值见 [`ml/README.md`](ml/README.md)。

---

## 4. Python API 速览

每个模块的详细文档在它自己的 README，这里给四步走总览：

```python
# Step 1 — 加载 + 清洗（preprocessing/）
from preprocessing import loaders, cleaning, inspect as ins

df = loaders.load_csv()                                       # 默认数据文件
print(ins.overview(df))                                       # 行/列/缺失率/重复行
df, _ = cleaning.drop_high_missing_columns(df, threshold=0.4,
            exclude=["target_lesion_positive", "uid"])

# Step 2 — 定义 X / y（ml/data_utils）
from ml import data_utils

X, y, cols = data_utils.get_Xy(df)                            # 默认 83 个基线变量
# X, y, cols = data_utils.get_Xy(df, columns=["age_years", "stenosis_percent", ...])

# Step 3 — 预处理器 + 模型（preprocessing/ + ml/）
from preprocessing import transformers as tr
from ml.models import MODEL_REGISTRY

preprocessor = tr.standard_preprocessor(df, cols)             # 4 个预设之一
model = MODEL_REGISTRY["lightgbm"](preprocessor)              # 11 个模型任选

# Step 4 — 评测（evaluation/）
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from evaluation import metrics, plots, shap_analysis

oof = cross_val_predict(model, X, y, method="predict_proba",
                        cv=StratifiedKFold(5, shuffle=True, random_state=42))[:, 1]

print(metrics.compute_metrics(y.to_numpy(), oof))             # 全套指标
print(metrics.bootstrap_auc_ci(y.to_numpy(), oof))            # AUC 95% CI
print(metrics.best_threshold(y.to_numpy(), oof, criterion="youden"))

model.fit(X, y)
out = shap_analysis.compute_shap(model, X, max_samples=1000)
shap_analysis.plot_summary(out, save_path="results/evaluation/shap_summary.png")
```

---

## 5. 端到端食谱

下面三个场景是最常见的工作流，复制即可运行。

### 食谱 A：跑统计版逻辑回归（单变量 → 多变量）

```bash
python3 analysis/logistic_regression.py
```

输出落在 `results/`：见 [`analysis/README.md`](analysis/README.md) 里每个 CSV 的列含义。

### 食谱 B：完整 ML 流水线（CV + 保存模型 + SHAP）

```python
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import loaders, cleaning, transformers as tr
from ml import data_utils
from ml.models import MODEL_REGISTRY
from evaluation import metrics, plots, shap_analysis

# 1. 数据
df = loaders.load_csv()
df, _ = cleaning.drop_high_missing_columns(df, threshold=0.4,
            exclude=["target_lesion_positive", "uid"])
X, y, cols = data_utils.get_Xy(df)

# 2. 预处理 + 模型
pre   = tr.standard_preprocessor(df, cols)
model = MODEL_REGISTRY["lightgbm"](pre)

# 3. 5 折 CV
cv  = StratifiedKFold(5, shuffle=True, random_state=42)
oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
print(f"CV AUC = {roc_auc_score(y, oof):.3f}")
print(metrics.bootstrap_auc_ci(y.to_numpy(), oof))

# 4. 评测图四联
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
plots.plot_roc(y, oof,         ax=axes[0, 0], label="lightgbm")
plots.plot_pr(y, oof,          ax=axes[0, 1], label="lightgbm")
plots.plot_calibration(y, oof, ax=axes[1, 0], n_bins=10)
plots.plot_confusion_matrix(y, (oof >= 0.5).astype(int), ax=axes[1, 1])
plots.save_fig(fig, "results/evaluation/diagnostics.png")

# 5. 全量 refit + 保存 + SHAP
model.fit(X, y)
joblib.dump(model, "results/ml/lightgbm.joblib")
out = shap_analysis.compute_shap(model, X, max_samples=1000)
shap_analysis.plot_summary(out, save_path="results/evaluation/shap_summary.png")
print(shap_analysis.top_feature_importance(out, top_k=15))
```

### 食谱 C：自定义变量 + SMOTE 处理不平衡 + 留出测试集

```python
from sklearn.metrics import roc_auc_score
from preprocessing import loaders, transformers as tr, splits, imbalance
from ml.models import MODEL_REGISTRY
from evaluation import metrics

df = loaders.load_csv()

my_cols = [
    "age_years", "sex", "stenosis_percent", "minimum_luminal_area_mm3",
    "plaque_type", "lesion_stenosis_grade", "ffrct_value",
    "High-Risk Plaque Flag",
]
from ml import data_utils
X, y, cols = data_utils.get_Xy(df, columns=my_cols)

X_tr, X_te, y_tr, y_te = splits.stratified_split(X, y, test_size=0.2)

pre = tr.standard_preprocessor(df, cols)
clf = MODEL_REGISTRY["logistic_l2"](pre).named_steps["clf"]
pipe = imbalance.wrap_with_resampling(pre, clf, strategy="smote")

pipe.fit(X_tr, y_tr)
prob = pipe.predict_proba(X_te)[:, 1]
print("test AUC:", roc_auc_score(y_te, prob))
print(metrics.compute_metrics(y_te.to_numpy(), prob))
```

---

## 6. 加载已保存的模型

`results/ml/<name>.joblib` 里保存的是整个 sklearn `Pipeline`（预处理 + 分类器）；加载后可以直接对新数据预测：

```python
import joblib
import pandas as pd

model = joblib.load("results/ml/lightgbm.joblib")

# new_df 是任意 DataFrame，列名和训练时的 X 一致即可
new_df = pd.read_csv("/path/to/new_patients.csv")
new_df.columns = [c.strip() for c in new_df.columns]

y_prob = model.predict_proba(new_df)[:, 1]
```

预处理（缺失值填补、标准化、One-Hot）会自动应用。

---

## 7. 结果摘要

### 统计版（`analysis/logistic_regression.py`）
- 主分析（83 个基线变量，单变量 p<0.05 入选 37 个进多变量）：
  - 多变量模型 LR p ≈ 2e-10，McFadden R² ≈ 0.075，AUC ≈ **0.696**
  - 独立显著：`stenosis_percent`（OR 5.92）、`minimum_luminal_area_mm3`（OR 0.86）

### ML 版（11 模型，默认超参，5 折分层 CV）

| 模型 | AUC | AP | Sens@0.5 | Spec@0.5 |
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

阈值 0.5 在 15% 阳性率上对树模型偏保守，用 `metrics.best_threshold(..., criterion="youden")` 重选后 LightGBM 在阈值 0.019 处达到 Sens=0.65 / Spec=0.57。

---

## 8. 注意事项

- **训练集 AUC 不可信**：树模型几乎都是 1.0，所有结论以 CV / 留出测试集为准。
- **类别不平衡**：阳性率 15.4%，建议结合 `class_weight="balanced"`（模型默认已设）或 `preprocessing.imbalance.SMOTE`，并基于 PR 曲线 / Youden 重选阈值。
- **事后变量**：`postop_*` / `followup_*` / `diff_*` 在结局之后才测得到，作为预测因子会泄漏。`data_utils.get_Xy(df)` 默认剔除；只在敏感性分析里用 `include_leaky=True`。
- **`baseline_ldl_level `** 这一列名末尾有空格——`load_csv()` / `data_utils.load_data()` 都会自动 `strip()`。如果直接 `pd.read_csv` 记得 `df.columns = [c.strip() for c in df.columns]`。
- **患者级数据泄漏**：同一个 `uid` 可能对应多条 lesion 记录。默认随机切分会把同一个患者的不同病变切到两边。需要患者级切分时用 `preprocessing.splits.grouped_stratified_split(..., group_col="uid")`。

---

## 9. 添加新东西

| 想加什么 | 怎么加 |
|---|---|
| 新 ML 模型 | `ml/models/my_model.py` 写一个 `build(preprocessor)`，然后在 `ml/models/__init__.py` 注册到 `MODEL_REGISTRY`。详见 [`ml/models/README.md`](ml/models/README.md) |
| 新评测指标 / 图 | 在 `evaluation/metrics.py` 或 `evaluation/plots.py` 加函数即可，无侵入式 |
| 新预处理变体 | 在 `preprocessing/transformers.py` 加一个 `build_preprocessor(..., scale="...")` 的封装或新预设 |
