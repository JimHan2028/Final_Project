"""score 语义回归测试：score 必须是正类（yes）概率，而非预测类的置信度。"""

from __future__ import annotations

import pytest

from app import model_loader


class _StubModel:
    """最小可用的 Pipeline 替身：固定预测 no，yes 概率 0.03。"""

    is_fake = False
    classes_ = ("no", "yes")

    def predict(self, frame):
        return ["no"] * len(frame)

    def predict_proba(self, frame):
        return [[0.97, 0.03] for _ in range(len(frame))]


def test_score_is_yes_probability_not_confidence(monkeypatch, valid_payload):
    monkeypatch.setattr(model_loader, "get_model", lambda: _StubModel())

    labels, scores = model_loader.predict([valid_payload])

    assert labels == ["no"]
    # 旧实现返回 max(row)=0.97（预测类置信度），语义与 label 相悖
    assert scores[0] == pytest.approx(0.03)


def test_score_follows_classes_order(monkeypatch, valid_payload):
    """classes_ 顺序颠倒时，仍须取 yes 那一列。"""

    class _ReversedModel(_StubModel):
        classes_ = ("yes", "no")

        def predict_proba(self, frame):
            return [[0.02, 0.98] for _ in range(len(frame))]

    monkeypatch.setattr(model_loader, "get_model", lambda: _ReversedModel())

    _, scores = model_loader.predict([valid_payload])

    assert scores[0] == pytest.approx(0.02)


def test_score_falls_back_when_classes_unknown(monkeypatch, valid_payload, caplog):
    """没有 classes_ 时退化为最大类概率，并留下告警而非静默返回错误语义。"""

    class _NoClassesModel(_StubModel):
        classes_ = None

    monkeypatch.setattr(model_loader, "get_model", lambda: _NoClassesModel())

    with caplog.at_level("WARNING"):
        _, scores = model_loader.predict([valid_payload])

    assert scores[0] == pytest.approx(0.97)
    assert any("classes_" in record.message for record in caplog.records)


def test_fake_model_path_keeps_neutral_score(monkeypatch, valid_payload):
    """占位模型路径不受本次改动影响，仍返回中性 0.5。"""

    from app.model_loader import FakeModel

    monkeypatch.setattr(model_loader, "get_model", lambda: FakeModel())

    labels, scores = model_loader.predict([valid_payload])

    assert labels == ["no"]
    assert scores == [0.5]
