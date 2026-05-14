# `analysis/` — 统计版逻辑回归

经典的统计学流程，**和 `ml/` 完全独立**：
1. 对每一个候选预测变量做**单变量 logistic 回归**。
2. 把 `p < 0.05` 的变量**全部纳入多变量 logistic 回归**。
3. 报告 OR / 95% CI / p 值、整体 LR 检验、AUC、McFadden 伪 R²。

数据中的事后变量（`postop_*` / `followup_*` / `diff_*`）会被剔除作为主分析；另外提供一个把它们也包进来的敏感性分析。

---

## 文件

```
analysis/
├── logistic_regression.py   # 自包含脚本，无外部依赖（不 import ml/）
└── README.md                # 当前文件
```

---

## 运行

```bash
cd /workspace
python3 analysis/logistic_regression.py
```

执行时间 < 5 秒。所有输出都落在 `results/` 下。

---

## 输出文件解释

| 文件 | 内容 |
|---|---|
| `results/univariate_baseline.csv` | 主分析：每个变量单变量回归的 OR / 95% CI / p，按 p 升序排序 |
| `results/multivariable_baseline.csv` | 主分析：把单变量 p<0.05 的 37 个变量塞进同一个 Logit，OR / 95% CI / p |
| `results/multivariable_baseline_metrics.json` | 主分析多变量模型的整体指标（n / events / 似然 / 伪 R² / AIC / BIC / AUC） |
| `results/univariate_all_variables.csv` | 敏感性：把 `postop_*/followup_*/diff_*` 也加入候选变量 |
| `results/multivariable_all_variables.csv` | 敏感性：对应的多变量模型 |
| `results/multivariable_all_variables_metrics.json` | 敏感性：多变量整体指标 |
| `results/README.md` | 上面六个 CSV/JSON 的列含义与解读 |

### 单变量 / 多变量 CSV 的列含义

| 列 | 含义 |
|---|---|
| `variable` | 变量名（分类变量被 One-Hot 后会带后缀） |
| `n` | 进入这一次回归的样本量 |
| `coef` / `se` | logistic 回归系数 β 及其标准误 |
| `OR` | 比值比 `exp(β)`，每增加 1 个单位预测因子，阳性几率倍数 |
| `OR_CI_low` / `OR_CI_high` | OR 的 95% Wald 置信区间 |
| `p_value` | Wald 检验 p 值 |
| `note` | 拟合异常时的提示（正常为空） |

> OR 的解释和变量尺度有关。比如 `stenosis_percent` 是 0–1 的比例，OR=15.93 表示"从 0% 到 100%"的几率倍数；想要"每增加 10%"的 OR 就开 10 次方（约 1.32）。

### `*_metrics.json` 的字段

```json
{
  "n": 1796,                       // 完整病例数
  "events": 284,                   // 阳性事件数
  "n_predictors": 37,              // 模型里同时拟合的预测因子数
  "llf": -725.60,                  // 模型对数似然
  "ll_null": -784.05,              // 仅截距零模型对数似然
  "pseudo_r2_mcfadden": 0.0746,
  "aic": 1525.19,
  "bic": 1728.45,
  "lr_pvalue": 1.75e-10,           // 模型整体 LR 检验
  "auc": 0.6955                    // 训练集内 ROC-AUC（乐观偏倚）
}
```

---

## 头条结果

### 主分析（剔除事后变量，83 个候选）
- 单变量 p<0.05：37 / 81 个变量
- 多变量：n=1796 完整病例，事件 284，LR p ≈ 2e-10，McFadden R² ≈ 0.075，**AUC ≈ 0.696**
- 调整后仍独立显著：
  - `stenosis_percent` — OR 5.92 (1.55–22.61), p = 0.0093
  - `minimum_luminal_area_mm3` — OR 0.86 (0.76–0.98), p = 0.021
- 单变量 OR 很大的 plaque burden 类变量（`whole_lesion_plaque_burden`、`non_calcified_plaque_burden`、`High-Risk Plaque Flag` 等），在多变量里大都被狭窄程度 / 最小管腔面积"吃掉"——典型的影像学共线性。

### 敏感性分析（含事后变量，135 个候选）
- 单变量 p<0.05：46 / 133
- 多变量：n=1065（事后变量缺失更多，完整病例骤减），AUC ≈ 0.731
- 仅 `stenosis_percent` 独立显著；`followup_wbc / followup_ldl / followup_hba1c` 边缘显著。

---

## 局限性 / 改进方向

- **训练集内 AUC**：未做交叉验证，乐观偏倚明显，尤其在 37–46 个变量同时拟合时。
- **"单变量 p<0.05 → 多变量"是经典但有偏的筛选**：对 plaque burden 这种高度共线的变量族不太友好；推荐用 LASSO（`ml.models["logistic_l1_lasso"]`）或 elastic-net 重新挑变量。
- **完整病例分析**：任一变量缺失就丢整行，事后变量子集尤其明显。若要发表级结果，建议加多重插补。
- **OR 尺度**：plaque burden / volume_ratio 这类 0–1 比例变量的 OR 很大、CI 极宽，正式汇报前应先做尺度变换（如 per 10%）。
- **事后变量**：仅在敏感性分析中出现，不能用作真正的预测模型。

需要直接用 ML 的方式重新建模、做交叉验证、SHAP 解释等，见项目根目录的 [`README.md`](../README.md) 或者 [`ml/README.md`](../ml/README.md) / [`evaluation/README.md`](../evaluation/README.md)。
