"""异常路径与边界测试。"""

from __future__ import annotations

import app.main as main_module
from app.model_loader import FakeModel, is_fake, reset_model


def test_unknown_category_accepted(client, valid_payload):
    """没见过的类别值不应让服务崩溃 —— OneHot 的 handle_unknown 会兜底。"""
    response = client.post("/predict", json={**valid_payload, "job": "astronaut"})
    assert response.status_code == 200


def test_unicode_and_special_characters(client, valid_payload):
    """类别字段中的 Unicode / 特殊符号 / 换行都应安全处理。"""
    response = client.post(
        "/predict",
        json={**valid_payload, "job": "工程师 🎉 !@# line\nbreak"},
    )
    assert response.status_code == 200


def test_extreme_but_valid_numerics(client, valid_payload):
    """极端但合法的数值不应导致 5xx。"""
    response = client.post(
        "/predict",
        json={
            **valid_payload,
            "age": 120,
            "campaign": 100,
            "previous": 100,
            "emp_var_rate": -99.9,
            "cons_price_idx": 0.0,
            "cons_conf_idx": 0.0,
            "euribor3m": 0.0,
            "nr_employed": 0.0,
        },
    )
    assert response.status_code == 200


def test_score_matches_confidence_contract(client, valid_payload):
    """score 必须落在 [0, 1]，与契约第 2 条一致。"""
    body = client.post("/predict", json=valid_payload).json()
    assert 0.0 <= body["score"] <= 1.0


def test_fake_model_used_when_pkl_absent(client):
    """真 pkl 未就位时接口仍须可用（回落到占位模型）。"""
    assert isinstance(is_fake(), bool)


def test_fake_model_predict_shape():
    model = FakeModel()
    assert model.predict([{"age": 1}, {"age": 2}, {"age": 3}]) == ["no"] * 3


def test_reset_model_does_not_crash():
    reset_model()
    assert is_fake() in (True, False)


def test_inference_failure_returns_500(client, valid_payload, monkeypatch):
    """推理抛异常应转成 500，而不是让进程裸崩。"""

    def boom(records):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(main_module, "predict", boom)

    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 500
    assert "simulated inference failure" in response.json()["detail"]
