from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

# 类别字段的长度上限：防御超长垃圾输入；未知类别由模型侧 OneHot 忽略
MAX_CATEGORY_LENGTH = 64


def _category_field(description: str) -> Field:
    return Field(..., max_length=MAX_CATEGORY_LENGTH, description=description)


class PredictRequest(BaseModel):
    """单条客户预测请求：字段与训练数据一一对应（duration 已按泄漏剔除）。"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 56,
                "job": "housemaid",
                "marital": "married",
                "education": "basic.4y",
                "default": "no",
                "housing": "no",
                "loan": "no",
                "contact": "telephone",
                "month": "may",
                "day_of_week": "mon",
                "campaign": 1,
                "pdays": 999,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp_var_rate": 1.1,
                "cons_price_idx": 93.994,
                "cons_conf_idx": -36.4,
                "euribor3m": 4.857,
                "nr_employed": 5191.0,
            }
        },
    )

    age: int = Field(..., ge=17, le=120, description="客户年龄")
    job: str = _category_field("职业（admin./blue-collar/...）")
    marital: str = _category_field("婚姻状况（married/single/divorced）")
    education: str = _category_field("教育程度（university.degree/basic.4y/...）")
    default: str = _category_field("是否有违约记录（no/unknown/yes）")
    housing: str = _category_field("是否有住房贷款（yes/no/unknown）")
    loan: str = _category_field("是否有个人贷款（yes/no/unknown）")
    contact: str = _category_field("联系方式（telephone/cellular）")
    month: str = _category_field("最后一次联系的月份（jan~dec）")
    day_of_week: str = _category_field("最后一次联系的星期（mon~fri）")
    campaign: int = Field(
        ..., ge=1, le=100, description="本次营销活动中对该客户的联系次数"
    )
    pdays: int = Field(
        ...,
        ge=0,
        le=999,
        description="距上次联系的天数；999 表示此前从未联系过",
    )
    previous: int = Field(..., ge=0, le=100, description="本次活动前与该客户联系的次数")
    poutcome: str = _category_field(
        "上一次营销活动的结果（nonexistent/failure/success）"
    )
    emp_var_rate: float = Field(..., description="就业变化率（宏观经济指标）")
    cons_price_idx: float = Field(..., description="消费者价格指数")
    cons_conf_idx: float = Field(..., description="消费者信心指数")
    euribor3m: float = Field(..., description="3 个月欧元同业拆借利率")
    nr_employed: float = Field(..., description="雇员人数（季度指标）")

    @field_validator(
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
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("类别字段不能为纯空白字符")
        return value


class PredictResponse(BaseModel):
    """单条预测结果。"""

    label: str = Field(..., description="预测类别：yes=订阅定期存款，no=不订阅")
    score: float = Field(..., ge=0.0, le=1.0, description="预测类别的置信度（0~1）")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field(..., description="ok 表示服务可用")
    model_loaded: bool = Field(
        ..., description="True 表示已加载真实 pkl；False 表示当前使用占位模型"
    )
