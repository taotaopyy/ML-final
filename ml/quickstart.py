"""
ml/quickstart.py — 端到端最小可运行示例

在项目根目录 /workspace 下运行：

    # 模型选择
    python3 -m ml.quickstart                              # 默认跑 lightgbm
    python3 -m ml.quickstart --model random_forest        # 任意模型名
    python3 -m ml.quickstart --model all                  # 11 个模型全部比较

    # 自定义输入变量（可任选一种）
    python3 -m ml.quickstart --columns age_years,stenosis_percent,baseline_ldl
    python3 -m ml.quickstart --columns-file my_features.txt
    python3 -m ml.quickstart --include-leaky              # 加入 postop_/followup_/diff_

`--columns-file` 是一个每行一个列名的纯文本文件。

流程：
    1. 加载数据并按用户给的列表（或默认）定义 X / y
    2. 构造共用预处理器
    3. 选择模型
    4. 5 折交叉验证评估（AUC / AP / Sens / Spec）
    5. 在全量数据上重新 fit，把模型保存到 results/ml/<name>.joblib
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="lightgbm",
        help="model name in MODEL_REGISTRY, or 'all' to benchmark every model.",
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--columns", default=None,
        help="comma-separated list of column names to use as predictors.",
    )
    grp.add_argument(
        "--columns-file", default=None,
        help="path to a text file with one predictor column name per line "
             "(blank lines and '#' comments are ignored).",
    )
    p.add_argument(
        "--include-leaky", action="store_true",
        help="include postop_/followup_/diff_ variables (only used when "
             "--columns and --columns-file are not given).",
    )
    return p.parse_args()


def load_columns_arg(args: argparse.Namespace) -> list[str] | None:
    if args.columns:
        return [c.strip() for c in args.columns.split(",") if c.strip()]
    if args.columns_file:
        text = Path(args.columns_file).read_text()
        out = []
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line)
        return out
    return None


def main() -> None:
    args = parse_args()
    target = args.model

    # 1. 加载数据 + 定义 X / y
    df = data_utils.load_data()
    user_cols = load_columns_arg(args)
    if user_cols is not None:
        X, y, cols = data_utils.get_Xy(df, columns=user_cols)
        print(f"data: {df.shape[0]} rows | custom predictors: {len(cols)} "
              f"| events: {int(y.sum())}/{len(y)} ({y.mean():.1%})")
    else:
        X, y, cols = data_utils.get_Xy(df, include_leaky=args.include_leaky)
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
