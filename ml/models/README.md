# `ml/models/`

每个模型一个独立文件，每个文件只暴露一个函数：

```python
def build(preprocessor) -> sklearn.Pipeline
```

返回的 `Pipeline` 形如 `preprocessor -> classifier`。所有模型共用 [`ml.data_utils.build_preprocessor`](../data_utils.py)（或 [`preprocessing.transformers`](../../preprocessing/README.md) 里的任意预设）构造的预处理器，所以模型之间的差异**只来自分类器本身**。

整体使用方式（命令行 / Python API / CV / 调参 / 保存加载）见上一层的 [`ml/README.md`](../README.md)。本文件只列模型清单和"怎么加一个新模型"。

---

## 模型清单

| 文件 | 分类器 | 备注 |
|---|---|---|
| `logistic.py` | `LogisticRegression` (L2) | 标准 logistic 回归基线，`class_weight="balanced"` |
| `lasso.py` | `LogisticRegression` (L1) | LASSO，自带特征筛选 |
| `random_forest.py` | `RandomForestClassifier` | 500 棵树，`min_samples_leaf=5` |
| `extra_trees.py` | `ExtraTreesClassifier` | 随机阈值树，方差更低 |
| `gradient_boosting.py` | `HistGradientBoostingClassifier` | sklearn 原生直方图 GBT |
| `lightgbm_model.py` | `LGBMClassifier` | LightGBM，叶子优先，通常表格数据最强 |
| `xgboost_model.py` | `XGBClassifier` | XGBoost，`scale_pos_weight` 已按 15% 阳性率设好 |
| `catboost_model.py` | `CatBoostClassifier` | CatBoost，`auto_class_weights="Balanced"`，静默模式 |
| `svm.py` | `SVC`（RBF）+ `probability=True` | 较慢，主要作基准 |
| `knn.py` | `KNeighborsClassifier` | 距离加权，k=25 |
| `mlp.py` | `MLPClassifier` (64, 32) | 小型两层 MLP |

> LightGBM / XGBoost / CatBoost 是**可选依赖**。`ml/models/__init__.py` 里是懒导入，缺哪个包就自动从 `MODEL_REGISTRY` 里跳过该模型，其余 8 个不受影响。

`MODEL_REGISTRY` key 列表：

```
"logistic_l2", "logistic_l1_lasso",
"random_forest", "extra_trees", "gradient_boosting",
"lightgbm", "xgboost", "catboost",
"svm_rbf", "knn", "mlp"
```

查看当前环境实际可用：

```python
from ml.models import MODEL_REGISTRY
print(sorted(MODEL_REGISTRY))
```

---

## 添加一个新模型

1. 在本文件夹里新建 `my_model.py`：

   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import HistGradientBoostingClassifier

   def build(preprocessor):
       clf = HistGradientBoostingClassifier(
           max_iter=500,
           learning_rate=0.05,
           class_weight="balanced",
           random_state=42,
       )
       return Pipeline([("preprocess", preprocessor), ("clf", clf)])
   ```

   注意点：
   - 函数名必须是 `build`，接收 `preprocessor` 并返回 `Pipeline`。
   - Pipeline 里两步必须分别叫 `"preprocess"` 和 `"clf"`，这样调参（`clf__xxx`）和 SHAP 才能自动找到分类器。
   - 推荐处理类别不平衡：`class_weight="balanced"`（sklearn / LightGBM）、`scale_pos_weight=...`（XGBoost）、或 `auto_class_weights="Balanced"`（CatBoost）。

2. 在 `__init__.py` 里注册：

   ```python
   from . import my_model
   MODEL_REGISTRY["my_model"] = my_model.build
   ```

   如果依赖一个可能没装的库，按现有 LightGBM / XGBoost / CatBoost 的样子用 try/except 包起来。

3. 立刻可用：

   ```bash
   python3 -m ml.quickstart --model my_model
   ```

   ```python
   from ml.models import MODEL_REGISTRY
   model = MODEL_REGISTRY["my_model"](preprocessor)
   ```
