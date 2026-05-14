"""
ml/quickstart.py — 端到端最小可运行示例

在项目根目录 /workspace 下运行：

    python3 -m ml.quickstart                 # 默认跑 lightgbm
    python3 -m ml.quickstart random_forest   # 换任意一个模型名
    python3 -m ml.quickstart all             # 11 个模型全部比较

流程：
    1. 加载数据并定义 X / y
    2. 构造共用预处理器
    3. 选择模型
    4. 5 折交叉验证评估（AUC / AP / Sens / Spec）
    5. 在全量数据上重新 fit，把模型保存到 results/ml/<name>.joblib
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ml import data_utils
from ml.models import MODEL_REGISTRY

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = ROOT / "results" / "ml"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42
THRESHOLD = 0.5


def evaluate(name: str, model, X, y, cv) -> dict:
    """5 折 CV 评估，返回一行结果字典。"""
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (oof >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "auc": roc_auc_score(y, oof),
        "average_precision": average_precision_score(y, oof),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "lightgbm"

    # 1. 加载数据 + 定义 X / y
    df = data_utils.load_data()
    X, y, cols = data_utils.get_Xy(df)              # 主分析：剔除事后变量
    print(f"data: {df.shape[0]} rows | predictors: {len(cols)} "
          f"| events: {int(y.sum())}/{len(y)} ({y.mean():.1%})")

    # 2. 共用预处理器
    preprocessor = data_utils.build_preprocessor(df, cols)

    # 3. 选择要跑的模型
    if target == "all":
        names = sorted(MODEL_REGISTRY)
    elif target in MODEL_REGISTRY:
        names = [target]
    else:
        raise SystemExit(
            f"unknown model '{target}'. options: {sorted(MODEL_REGISTRY)} or 'all'"
        )

    # 4. 5 折分层交叉验证
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name in names:
        try:
            pipe = MODEL_REGISTRY[name](preprocessor)
            row = evaluate(name, pipe, X, y, cv)
            print(f"  {name:22s}  AUC={row['auc']:.3f}  AP={row['average_precision']:.3f}  "
                  f"Sens={row['sensitivity']:.3f}  Spec={row['specificity']:.3f}")
            rows.append(row)
        except Exception as exc:
            print(f"  {name:22s}  FAILED: {type(exc).__name__}: {exc}")

    summary = pd.DataFrame(rows).sort_values("auc", ascending=False)
    summary_path = SAVE_DIR / "cv_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved CV summary -> {summary_path}")

    # 5. 把最优（按 AUC 排序的第一个）模型在全量数据上重新 fit，并保存
    best_name = summary.iloc[0]["model"]
    best_model = MODEL_REGISTRY[best_name](preprocessor)
    best_model.fit(X, y)
    model_path = SAVE_DIR / f"{best_name}.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved fitted model -> {model_path}  (best by CV AUC: {best_name})")

    # 加载回来 + 单条预测演示
    loaded = joblib.load(model_path)
    sample = X.iloc[[0]]
    prob = loaded.predict_proba(sample)[0, 1]
    print(f"\nReload check: P(target_lesion_positive=1) for first row = {prob:.3f}")


if __name__ == "__main__":
    main()
