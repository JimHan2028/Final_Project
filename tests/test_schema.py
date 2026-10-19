"""输入校验测试 —— 契约第 1 条的边界。"""

from __future__ import annotations

import pytest

# 删除字段的哨兵：便于在合法样本上做「缺字段」变体
_DELETE = object()

BASE_PAYLOAD = {
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


def _variant(**overrides) -> dict:
    payload = dict(BASE_PAYLOAD)
    for key, value in overrides.items():
        if value is _DELETE:
            payload.pop(key, None)
        else:
            payload[key] = value
    return payload


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="全缺"),
        pytest.param(_variant(age=_DELETE), id="缺age"),
        pytest.param(_variant(job=_DELETE), id="缺job"),
        pytest.param(_variant(poutcome=_DELETE), id="缺poutcome"),
        pytest.param(_variant(emp_var_rate=_DELETE), id="缺宏观指标"),
        pytest.param(_variant(age="abc"), id="age非数字"),
        pytest.param(_variant(age={"min": 18}), id="age是对象"),
        pytest.param(_variant(age=35.5), id="age是小数"),
        pytest.param(_variant(age=15), id="age低于下界"),
        pytest.param(_variant(age=121), id="age高于上界"),
        pytest.param(_variant(campaign=0), id="campaign为0"),
        pytest.param(_variant(campaign=101), id="campaign超上限"),
        pytest.param(_variant(pdays=-1), id="pdays为负"),
        pytest.param(_variant(pdays=1000), id="pdays超上限"),
        pytest.param(_variant(previous=-5), id="previous为负"),
        pytest.param(_variant(job=""), id="空字符串类别"),
        pytest.param(_variant(job="   \n\t  "), id="纯空白类别"),
        pytest.param(_variant(month=None), id="类别为null"),
        pytest.param(_variant(contact=123), id="类别是数字"),
        pytest.param({**BASE_PAYLOAD, "extra": 1}, id="多余字段"),
    ],
)
def test_invalid_payload_rejected(client, payload):
    """非法输入必须被 pydantic 拦在门口，返回 422。"""
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(BASE_PAYLOAD, id="标准样本"),
        pytest.param(_variant(age=17), id="age下界"),
        pytest.param(_variant(age=120), id="age上界"),
        pytest.param(_variant(pdays=0), id="pdays下界"),
        pytest.param(_variant(pdays=999), id="pdays哨兵值"),
        pytest.param(_variant(campaign=1), id="campaign下界"),
        pytest.param(_variant(campaign=100), id="campaign上界"),
        pytest.param(_variant(job="x" * 64), id="类别达到长度上限"),
    ],
)
def test_boundary_payload_accepted(client, payload):
    """恰好落在边界上的输入应通过，验证边界未被 off-by-one 卡住。"""
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_error_body_is_json(client):
    """422 响应体应是结构化 JSON，方便前端解析。"""
    response = client.post("/predict", json={})
    assert response.headers["content-type"].startswith("application/json")
