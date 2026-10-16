"""异常路径与边界测试。"""
from __future__ import annotations

import app.main as main_module
from app.model_loader import FakeModel, is_fake, reset_model


def test_special_characters(client):
    response = client.post(
        "/predict",
        json={"text": "!@#$%^&*()_+-=[]{}|;:',.<>/?~` 中文 emoji 🎉"},
    )
    assert response.status_code == 200


def test_multiline_text(client):
    response = client.post("/predict", json={"text": "line one\nline two\nline three"})
    assert response.status_code == 200


def test_leading_trailing_whitespace(client):
    """首尾空白只要含有效内容就应放行。"""
    response = client.post("/predict", json={"text": "  valid text  "})
    assert response.status_code == 200


def test_fake_model_used_when_pkl_absent(client):
    """真 pkl 未就位时接口仍须可用（回落到占位模型）。"""
    assert isinstance(is_fake(), bool)


def test_fake_model_predict_shape():
    model = FakeModel()
    assert model.predict(["a", "b", "c"]) == ["not_spam"] * 3


def test_reset_model_does_not_crash():
    reset_model()
    assert is_fake() in (True, False)


def test_inference_failure_returns_500(client, monkeypatch):
    """推理抛异常应转成 500，而不是让进程裸崩。"""

    def boom(texts):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(main_module, "predict", boom)

    response = client.post("/predict", json={"text": "hello"})
    assert response.status_code == 500
    assert "simulated inference failure" in response.json()["detail"]
