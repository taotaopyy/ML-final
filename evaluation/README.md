# `evaluation/` — 模型评测 + SHAP 可解释性

和 `analysis/`、`ml/` 平级的独立模块，**不依赖 `analysis/`**，只依赖 `ml/`
（用它的 Pipeline 结构来抽特征名）。

```
evaluation/
├── metrics.py         # 数值指标 + 阈值搜索 + bootstrap CI
├── plots.py           # ROC / PR / 校准曲线 / 混淆矩阵 / 多模型对比
├── shap_analysis.py   # SHAP TreeExplainer / LinearExplainer / KernelExplainer 自动选择
└── README.md
```

依赖：`shap`、`matplotlib`、`scikit-learn`（已写进 `requirements.txt`）。

---

## 1. 数值指标 `evaluation.metrics`

输入是 `(y_true, y_prob)` 数组，**不需要模型**——所以可以用在 CV 的 out-of-fold
预测、留出测试集预测、或者任何外部数据集上。

```python
from evaluation import metrics

m = metrics.compute_metrics(y_true, y_prob, threshold=0.5)
# {'auc', 'average_precision', 'brier', 'log_loss', 'threshold',
#  'accuracy', 'balanced_accuracy', 'f1', 'precision',
#  'recall_sensitivity', 'specificity', 'ppv', 'npv', 'mcc',
#  'tp','fp','tn','fn', 'youden_j'}

# 找最佳阈值（Youden / F1 / 平衡准确率）
metrics.best_threshold(y_true, y_prob, criterion="youden")
metrics.best_threshold(y_true, y_prob, criterion="f1")

# AUC 的 95% bootstrap CI
metrics.bootstrap_auc_ci(y_true, y_prob, n_boot=1000)
# -> {'auc': 0.71, 'auc_ci_low': 0.68, 'auc_ci_high': 0.74, ...}

# 比较多个模型
oof = {
    "lightgbm": (y_true, lgb_oof),
    "xgboost":  (y_true, xgb_oof),
    "logistic_l2": (y_true, lr_oof),
}
metrics.summarize_models(oof)   # 返回按 AUC 排序的 DataFrame
```

---

## 2. 画图 `evaluation.plots`

所有函数都接受 `ax`，方便拼多面板图；都返回 `ax`。

```python
import matplotlib.pyplot as plt
from evaluation import plots

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
plots.plot_roc(y_true, y_prob, ax=axes[0, 0], label="lightgbm")
plots.plot_pr(y_true, y_prob, ax=axes[0, 1], label="lightgbm")
plots.plot_calibration(y_true, y_prob, ax=axes[1, 0], n_bins=10)
plots.plot_confusion_matrix(y_true, (y_prob >= 0.5).astype(int), ax=axes[1, 1])
plots.save_fig(fig, "results/evaluation/lightgbm_diagnostics.png")

# 多模型对比
plots.plot_models_roc({
    "lightgbm": (y_true, lgb_oof),
    "xgboost":  (y_true, xgb_oof),
    "logistic": (y_true, lr_oof),
})
```

---

## 3. SHAP `evaluation.shap_analysis`

自动识别模型类型并挑合适的 SHAP explainer：

| 模型 | Explainer |
|---|---|
| RandomForest / ExtraTrees / GBT / LightGBM / XGBoost / CatBoost | `TreeExplainer`（最快） |
| LogisticRegression（L1/L2） | `LinearExplainer` |
| SVM / KNN / MLP | `KernelExplainer`（很慢，自动用 100 个背景样本） |

```python
from ml import data_utils
from ml.models import MODEL_REGISTRY
from evaluation import shap_analysis

# 1. 准备模型（要先 fit 过）
df = data_utils.load_data()
X, y, cols = data_utils.get_Xy(df)
pre = data_utils.build_preprocessor(df, cols)
model = MODEL_REGISTRY["lightgbm"](pre)
model.fit(X, y)

# 2. 计算 SHAP 值（默认最多用 1000 行）
shap_out = shap_analysis.compute_shap(model, X, max_samples=1000)
# shap_out = {shap_values, X_transformed, feature_names, explainer_kind, sampled_index}

# 3. 看 top 20 重要变量（按 mean |SHAP|）
print(shap_analysis.top_feature_importance(shap_out, top_k=20))

# 4. 出图（save_path 不传就只返回 matplotlib Figure 不显示）
shap_analysis.plot_summary(shap_out,
                           save_path="results/evaluation/shap_summary.png")
shap_analysis.plot_bar(shap_out,
                       save_path="results/evaluation/shap_bar.png")
shap_analysis.plot_dependence(shap_out, feature="stenosis_percent",
                              save_path="results/evaluation/shap_dep_stenosis.png")
```

返回字典里的 `shap_values` 是形状 `(n_samples, n_features)` 的 numpy 数组，
对应**阳性类**的 SHAP 值，可以直接喂给 `shap.summary_plot` 等官方函数自己画。

---

## 4. 端到端：跑 CV 拿 OOF → 出全套报告

```python
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ml import data_utils
from ml.models import MODEL_REGISTRY
from evaluation import metrics, plots, shap_analysis

# --- 1. 加载 / 选变量 / 选模型 -------------------------------------------
df = data_utils.load_data()
X, y, cols = data_utils.get_Xy(df)
pre = data_utils.build_preprocessor(df, cols)
model = MODEL_REGISTRY["lightgbm"](pre)

# --- 2. 5 折 CV 得到 OOF 概率 --------------------------------------------
cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

# --- 3. 数值评估 ----------------------------------------------------------
m  = metrics.compute_metrics(y.to_numpy(), oof, threshold=0.5)
ci = metrics.bootstrap_auc_ci(y.to_numpy(), oof)
bt = metrics.best_threshold(y.to_numpy(), oof, criterion="youden")
print(f"AUC = {m['auc']:.3f}  (95% CI {ci['auc_ci_low']:.3f}-{ci['auc_ci_high']:.3f})")
print(f"Best Youden threshold = {bt['threshold']:.3f} "
      f"-> Sens={bt['sensitivity']:.3f} Spec={bt['specificity']:.3f}")

# --- 4. 图形评估 ----------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
plots.plot_roc(y, oof,                     ax=axes[0, 0], label="lightgbm")
plots.plot_pr(y, oof,                      ax=axes[0, 1], label="lightgbm")
plots.plot_calibration(y, oof,             ax=axes[1, 0], n_bins=10)
plots.plot_confusion_matrix(y, (oof >= bt["threshold"]).astype(int),
                            ax=axes[1, 1], normalize=True)
plots.save_fig(fig, "results/evaluation/lightgbm_diagnostics.png")

# --- 5. SHAP（在全量上重新 fit 以解释最终模型）---------------------------
model.fit(X, y)
shap_out = shap_analysis.compute_shap(model, X, max_samples=1000)
shap_analysis.plot_summary(shap_out, save_path="results/evaluation/shap_summary.png")
shap_analysis.plot_bar(shap_out,     save_path="results/evaluation/shap_bar.png")
print(shap_analysis.top_feature_importance(shap_out, top_k=15))
```
