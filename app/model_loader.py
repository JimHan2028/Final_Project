"""模型加载与统一推理出口。

设计要点：
  * 进程内单例，启动时加载一次，请求路径上直接复用；
  * 真 pkl 未就位时回落到占位模型，保证 CI 与本地开发不被训练进度阻塞；
  * 推理输入是「记录字典列表」，真实模型路径下会转成带固定列序的
    DataFrame（契约第 3 条）再交给 Pipeline，调用方无需做任何预处理；
  * 统一屏蔽模型类型差异（predict_proba / decision_function / 无分数）。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 契约第 4 条：模型路径固定，相对项目根定位，不写死绝对路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "ml" / "registry" / "model.pkl"

_model: Any = None


class FakeModel:
    """占位模型：只为跑通链路，不做任何真实预测。

    警告：占位模型无法暴露 pkl 的版本兼容问题，模型回填后必须用真实 pkl
    做一次集成验证（启动后 /health 应显示 model_loaded=true）。
    """

    is_fake = True

    def predict(self, records: Sequence[Mapping[str, Any]]) -> list[str]:
        return ["no"] * len(list(records))


def load_model(path: Path = MODEL_PATH) -> Any:
    """加载模型并写入进程内单例。"""
    global _model

    if not path.exists():
        logger.warning("模型文件不存在（%s），回落到占位模型", path)
        _model = FakeModel()
        return _model

    try:
        import joblib
    except ImportError:
        logger.warning("joblib 未安装（模型段依赖尚未回填），回落到占位模型")
        _model = FakeModel()
        return _model

    _model = joblib.load(path)
    logger.info("模型已加载：%s（%s）", path, type(_model).__name__)
    return _model


def get_model() -> Any:
    """取进程内单例；尚未加载则先加载。"""
    if _model is None:
        load_model()
    return _model


def reset_model() -> None:
    """清空单例缓存（供测试使用）。"""
    global _model
    _model = None


def is_fake() -> bool:
    """当前是否在使用占位模型。"""
    return bool(getattr(get_model(), "is_fake", False))


def _sigmoid(value: float) -> float:
    """把 decision_function 的任意实数压到 (0, 1)。"""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _decision_to_scores(raw: Any) -> list[float]:
    """decision_function 输出可能是 (n,) 或 (n, k)，统一成每样本一个分数。"""
    rows = list(raw)
    if not rows:
        return []
    if hasattr(rows[0], "__iter__"):
        return [_sigmoid(float(max(row))) for row in rows]
    return [_sigmoid(float(value)) for value in rows]


def _records_to_frame(records: Sequence[Mapping[str, Any]]):
    """把记录字典列表转成带固定列序的 DataFrame（契约第 3 条）。

    延迟导入 pandas / ml.data_pipeline：真实模型路径下两者必然可用
    （反序列化 pkl 本身就需要），占位模型路径下则完全不依赖。
    """
    import pandas as pd

    from ml.data_pipeline import FEATURE_COLUMNS

    rows = [
        {column: record[column] for column in FEATURE_COLUMNS} for record in records
    ]
    return pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))


def predict(records: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[float]]:
    """统一推理出口，返回 (labels, scores)。"""
    items = list(records)
    if not items:
        return [], []

    model = get_model()

    # 占位模型路径：不触碰 pandas，保证轻量环境也能跑通 CI
    if getattr(model, "is_fake", False):
        labels = [str(label) for label in model.predict(items)]
        return labels, [0.5] * len(labels)

    frame = _records_to_frame(items)
    labels = [str(label) for label in model.predict(frame)]

    if hasattr(model, "predict_proba"):
        scores = [float(max(row)) for row in model.predict_proba(frame)]
    elif hasattr(model, "decision_function"):
        scores = _decision_to_scores(model.decision_function(frame))
    else:
        scores = [1.0] * len(labels)

    if len(scores) != len(labels):
        scores = [0.5] * len(labels)

    return labels, scores
