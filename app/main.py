"""FastAPI 服务入口 —— 银行定期存款营销预测。

启动方式：
    uvicorn app.main:app --reload --port 8000

交互文档：
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles

from app.model_loader import get_model, is_fake, predict
from app.schemas import HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Swagger UI 静态资源自托管：默认走国外 CDN，境内网络加载不出来导致 /docs 空白
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载一次模型，避免首个请求承担加载延迟。"""
    get_model()
    yield


app = FastAPI(
    title="Bank Marketing Prediction API",
    description="DevOps / MLOps 期末项目 —— 银行客户定期存款订阅预测服务",
    version="0.1.0",
    lifespan=lifespan,
    # 关闭默认 /docs（其引用 CDN），用下方自定义路由改为本地资源
    docs_url=None,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/docs", include_in_schema=False)
def swagger_docs():
    """交互文档，资源全部来自本地 /static，无外网依赖。"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Docs",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """健康检查：进程存活 + 模型是否已就绪。"""
    return HealthResponse(status="ok", model_loaded=not is_fake())


@app.post("/predict", response_model=PredictResponse, tags=["ml"])
def predict_endpoint(payload: PredictRequest) -> PredictResponse:
    """单条客户特征预测：是否会订阅定期存款。"""
    record = payload.model_dump()
    try:
        labels, scores = predict([record])
    except Exception as exc:  # 推理异常统一转 500，避免进程裸崩
        raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc

    return PredictResponse(label=labels[0], score=scores[0])
