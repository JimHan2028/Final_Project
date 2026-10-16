"""正常路径测试：/predict 与 /health。"""
from __future__ import annotations


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)


def test_predict_returns_contract_shape(client):
    """输出结构严格等于契约第 2 条：{label, score}，不多不少。"""
    response = client.post(
        "/predict", json={"text": "Congratulations, you won a prize!"}
    )
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"label", "score"}


def test_predict_field_types_and_range(client):
    response = client.post("/predict", json={"text": "hello world"})
    body = response.json()

    assert isinstance(body["label"], str)
    assert body["label"]
    assert isinstance(body["score"], float)
    assert 0.0 <= body["score"] <= 1.0


def test_predict_is_deterministic(client):
    """同一输入连续两次应给出一致结果。"""
    payload = {"text": "free money now"}
    first = client.post("/predict", json=payload).json()
    second = client.post("/predict", json=payload).json()
    assert first == second


def test_openapi_exposes_predict(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]
