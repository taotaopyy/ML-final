# `preprocessing/` — 数据预处理工具箱

> 项目总入口：[`../README.md`](../README.md) · 相关模块：[`ml/`](../ml/README.md) · [`evaluation/`](../evaluation/README.md) · [`analysis/`](../analysis/README.md)

和 `analysis/`、`ml/`、`evaluation/` 平级、互不依赖。`ml/data_utils.py` 里只内置了最小化的预处理（中位数填补 + 标准化 + One-Hot），如果需要更细致的清洗、不同的缩放/编码、KNN 插补、SMOTE 重采样、患者级切分等，用这个文件夹。

```
preprocessing/
├── loaders.py        # CSV 加载 + 列名清洗 + 强制转数值
├── inspect.py        # 缺失 / 基数 / 数值分布 / 离群点报告（返回 DataFrame）
├── cleaning.py       # 删高缺失列/行、去常数列、Winsorize、log 变换、去重
├── transformers.py   # 可配置的 ColumnTransformer 工厂 + 预设
├── imbalance.py      # SMOTE / 随机过采样 / 随机欠采样（imblearn）
├── splits.py         # 分层切分、3-way 切分、患者级切分
└── README.md
```

需要的依赖（都已在 `requirements.txt`）：`pandas`、`numpy`、`scikit-learn`、`imbalanced-learn`。

---

## 1. 加载 + 清列名 `preprocessing.loaders`

```python
from preprocessing import loaders

df = loaders.load_csv()                              # 默认数据文件
df = loaders.load_csv("/path/to/other.csv")          # 任意 CSV
df = loaders.load_csv(clean=False)                   # 不做任何清洗

df = loaders.coerce_numeric(df)                      # 把 object 列尝试转 numeric
```

`clean=True` 会自动 `strip()` 列名 + 删掉整列空的列（原始 CSV 末尾的 `,` 会产生一个空列）。

---

## 2. EDA 报告 `preprocessing.inspect`

每个函数都返回 DataFrame，方便 `to_csv` 或进一步筛选：

```python
from preprocessing import inspect as ins

ins.overview(df)                          # dict：行数/列数/dtype 分布/缺失率/重复行
ins.missing_report(df)                    # 每列缺失数量/缺失率/dtype/唯一值数
ins.cardinality_report(df)                # 看哪些列基本是常数
ins.numeric_summary(df)                   # describe + skew + kurt
ins.outlier_report(df, method="iqr")      # IQR 法离群点统计
ins.outlier_report(df, method="z")        # 鲁棒 z-score
ins.target_balance(y)                     # 类别分布表
```

---

## 3. 清洗操作 `preprocessing.cleaning`

所有函数都是**纯函数**——返回新 DataFrame，不修改输入：

```python
from preprocessing import cleaning

# 删缺失率 > 40% 的列（outcome / id 可以放进 exclude 避免被删）
df_clean, dropped = cleaning.drop_high_missing_columns(df, threshold=0.4,
                                                      exclude=["target_lesion_positive", "uid"])

df_clean, dropped = cleaning.drop_constant_columns(df_clean)
df_clean, n_rows  = cleaning.drop_high_missing_rows(df_clean, threshold=0.5)

# 离群点处理（Winsorize / 截断）
df_clip  = cleaning.clip_outliers_iqr(df_clean, iqr_mult=1.5)
df_winz  = cleaning.winsorize_quantiles(df_clean, lower=0.01, upper=0.99)

# 高偏度列自动 log1p 变换（返回被变换的列名）
df_log, log_cols = cleaning.log_transform_skewed(df_clean, skew_threshold=1.0)

df_dedup, n = cleaning.deduplicate_rows(df_clean)
```

---

## 4. 预处理器 `preprocessing.transformers`

可配置的 ColumnTransformer：

```python
from preprocessing import transformers as tr

pre = tr.build_preprocessor(
    df, cols,
    impute="median",            # "median" / "mean" / "most_frequent" / "knn"
    scale="standard",           # "standard" / "robust" / "minmax" / None
    numeric_transform=None,     # None / "log1p" / "yeo-johnson"
)
```

也有 4 个开箱即用的预设：

```python
tr.standard_preprocessor(df, cols)        # 中位数 + StandardScaler（线性 / SVM / NN 默认）
tr.robust_preprocessor(df, cols)          # 中位数 + RobustScaler（重尾 / 离群点多）
tr.tree_preprocessor(df, cols)            # 中位数 + 不缩放（树模型）
tr.knn_imputed_preprocessor(df, cols)     # KNN 插补 + StandardScaler
```

返回的就是 sklearn `ColumnTransformer`，可以直接喂进 `ml.models` 里任意模型的 `build()`：

```python
from ml.models import MODEL_REGISTRY
pre = tr.robust_preprocessor(df, cols)
model = MODEL_REGISTRY["logistic_l2"](pre)
```

---

## 5. 不平衡处理 `preprocessing.imbalance`

阳性率只有 15.4%。除了模型里的 `class_weight="balanced"`，还可以用 SMOTE 等过采样。**注意 SMOTE 只能在训练时做**，所以用 `imblearn.pipeline.Pipeline`（用法和 sklearn Pipeline 一模一样）：

```python
from preprocessing import imbalance
from preprocessing import transformers as tr
from sklearn.linear_model import LogisticRegression

pre = tr.standard_preprocessor(df, cols)
clf = LogisticRegression(max_iter=2000)

# 三步流水线：预处理 -> SMOTE -> 分类器
pipe = imbalance.wrap_with_resampling(
    pre, clf,
    strategy="smote",                # "smote" / "random_over" / "random_under"
    sampling_strategy="auto",        # 或者 0.5 = 把少数类扩到多数类的 50%
)

# 跟 sklearn 一样用，CV 时 SMOTE 只在每一折的训练集上发生
pipe.fit(X_train, y_train)
y_prob = pipe.predict_proba(X_test)[:, 1]
```

---

## 6. 切分 `preprocessing.splits`

```python
from preprocessing import splits

# 单次分层 train/test
X_tr, X_te, y_tr, y_te = splits.stratified_split(X, y, test_size=0.2)

# 分层 train / val / test
X_tr, X_va, X_te, y_tr, y_va, y_te = splits.stratified_train_val_test(
    X, y, val_size=0.1, test_size=0.2,
)

# 分层 K 折
cv = splits.kfold_splitter(n_splits=5)

# 患者级切分（同一个 uid 的多个 lesion 必须在同一边）
# 注意：需要先把 uid 等分组列保留在 X 里
X_tr, X_te, y_tr, y_te = splits.grouped_stratified_split(
    X_with_uid, y, group_col="uid", test_size=0.2,
)
```

---

## 7. 端到端：清洗 -> 切分 -> SMOTE -> 训练 -> 评测

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from preprocessing import loaders, inspect as ins, cleaning, transformers as tr, imbalance, splits
from ml.models import MODEL_REGISTRY
from evaluation import metrics, plots

# 1. 加载 + EDA
df = loaders.load_csv()
print(ins.overview(df))
print(ins.missing_report(df).head(15))

# 2. 清洗（保护 outcome / id）
df, _ = cleaning.drop_high_missing_columns(df, threshold=0.4,
                                           exclude=["target_lesion_positive", "uid"])
df, _ = cleaning.drop_constant_columns(df)
df    = cleaning.winsorize_quantiles(df, lower=0.01, upper=0.99)

# 3. X / y / cols（可以借用 ml.data_utils.get_Xy 或者自己写）
from ml import data_utils
X, y, cols = data_utils.get_Xy(df)

# 4. 切分
X_tr, X_te, y_tr, y_te = splits.stratified_split(X, y, test_size=0.2)

# 5. 预处理 + SMOTE + 分类器
pre  = tr.standard_preprocessor(df, cols)
clf  = MODEL_REGISTRY["logistic_l2"](pre).named_steps["clf"]
pipe = imbalance.wrap_with_resampling(pre, clf, strategy="smote")

pipe.fit(X_tr, y_tr)
prob = pipe.predict_proba(X_te)[:, 1]
print("test AUC:", roc_auc_score(y_te, prob))
print(metrics.compute_metrics(y_te.to_numpy(), prob))
```
