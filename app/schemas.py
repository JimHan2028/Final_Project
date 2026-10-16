"""请求 / 响应模型 —— 接口契约的代码化表达。

契约四条（Day1 冻结，任何改动需双方同意后同步）：
  1. 输入结构  {"text": "..."}
  2. 输出结构  {"label": "...", "score": 0.97}
  3. 特征列序  文本分类任务不涉及（全部预处理封装在 pkl 的 Pipeline 内）
  4. 模型路径  ml/registry/model.pkl
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

# 输入文本长度上限：超长输入会拖垮推理并放大内存占用
MAX_TEXT_LENGTH = 5000


class PredictRequest(BaseModel):
    """单条预测请求。"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {"text": "Congratulations! You won a free prize, click here"}
        },
    )

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="待分类的原始文本；不要在此做任何预处理，预处理全部由模型 Pipeline 负责",
    )

    @field_validator("text")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text 不能为纯空白字符")
        return value


class PredictResponse(BaseModel):
    """单条预测结果。"""

    label: str = Field(..., description="预测类别")
    score: float = Field(..., ge=0.0, le=1.0, description="置信度（0~1）")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field(..., description="ok 表示服务可用")
    model_loaded: bool = Field(
        ..., description="True 表示已加载真实 pkl；False 表示当前使用占位模型"
    )
