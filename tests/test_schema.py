"""输入校验测试 —— 契约第 1 条的边界。"""
from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="缺字段"),
        pytest.param({"text": None}, id="null"),
        pytest.param({"text": 123}, id="类型错误-数字"),
        pytest.param({"text": ["a"]}, id="类型错误-数组"),
        pytest.param({"text": ""}, id="空字符串"),
        pytest.param({"text": "   \n\t  "}, id="纯空白"),
        pytest.param({"text": "a" * 5001}, id="超长"),
        pytest.param({"text": "ok", "extra": 1}, id="多余字段"),
    ],
)
def test_invalid_payload_rejected(client, payload):
    """非法输入必须被 pydantic 拦在门口，返回 422。"""
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_max_length_boundary_accepted(client):
    """恰好达到上限应通过，验证边界未被 off-by-one 卡住。"""
    response = client.post("/predict", json={"text": "a" * 5000})
    assert response.status_code == 200


def test_error_body_is_json(client):
    """422 响应体应是结构化 JSON，方便前端解析。"""
    response = client.post("/predict", json={})
    assert response.headers["content-type"].startswith("application/json")
