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
