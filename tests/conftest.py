"""pytest 公共配置与夹具。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    """每个用例一个独立客户端，走完整的 startup / shutdown 流程。"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def valid_payload():
    """一条合法请求体：取自训练集真实样本（duration 已按契约剔除）。"""
    return {
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
