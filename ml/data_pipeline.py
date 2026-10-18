from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "bank-additional-full.csv"

TARGET_COLUMN = "y"

# add duration to leakage
LEAKAGE_COLUMNS = ("duration",)

NUMERIC_COLUMNS = (
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
)

CATEGORICAL_COLUMNS = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
)

# 契约第 3 条：特征列序 —— 去掉目标列与泄漏列后的固定顺序，
# app 端构造 DataFrame 必须严格使用这一列序
FEATURE_COLUMNS = (
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
)

assert set(NUMERIC_COLUMNS) | set(CATEGORICAL_COLUMNS) == set(FEATURE_COLUMNS), (
    "数值列与类别列的并集必须等于特征列序"
)


def load_dataframe(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """读取原始 CSV（分号分隔），并把列名统一为下划线风格。

    例：emp.var.rate → emp_var_rate，与 app/schemas.py 的字段名保持一致。
    """
    frame = pd.read_csv(path, sep=";")
    frame.columns = [str(column).replace(".", "_") for column in frame.columns]

    required = {TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"数据集缺少必需列：{sorted(missing)}")
    return frame


def build_pipeline(estimator) -> Pipeline:
    """把预处理与分类器封装成单个 Pipeline，整体序列化进 model.pkl。"""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            # handle_unknown="ignore"：线上出现没见过的类别时不会崩，编码为全 0
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, list(NUMERIC_COLUMNS)),
            ("cat", categorical, list(CATEGORICAL_COLUMNS)),
        ],
        # HistGradientBoosting 不接受稀疏矩阵，强制稠密输出
        sparse_threshold=0.0,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
