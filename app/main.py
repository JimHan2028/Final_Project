"""FastAPI 服务入口。

启动方式：
    uvicorn app.main:app --reload --port 8000

交互文档：
    http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model_loader import get_model, is_fake, predict
from app.schemas import HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载一次模型，避免首个请求承担加载延迟。"""
    get_model()
    yield


app = FastAPI(
    title="Spam Classifier API",
    description="DevOps / MLOps 期末项目 —— 文本分类推理服务",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """健康检查：进程存活 + 模型是否已就绪。"""
    return HealthResponse(status="ok", model_loaded=not is_fake())


@app.post("/predict", response_model=PredictResponse, tags=["ml"])
def predict_endpoint(payload: PredictRequest) -> PredictResponse:
    """单条文本分类。"""
    try:
        labels, scores = predict([payload.text])
    except Exception as exc:  # 推理异常统一转 500，避免进程裸崩
        raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc

    return PredictResponse(label=labels[0], score=scores[0])
