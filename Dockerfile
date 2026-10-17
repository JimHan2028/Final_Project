# =============================================================================
# 应用镜像 —— 银行定期存款认购预测 API
# 版本唯一事实来源：requirements.txt（禁止在此重复声明版本）
# 构建前提：ml/registry/model.pkl 必须已存在（CI 中先 dvc pull，见 DEPLOYMENT.md）
# =============================================================================
# python 版本与 CI（3.12）保持一致，避免 pkl / 依赖解析差异
FROM python:3.12-slim

ARG APP_UID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 依赖层单独缓存：只改业务代码时不触发重装
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码（含自托管 Swagger 静态资源 app/static）
COPY app ./app
# 模型侧契约：ml/data_pipeline.py 提供 FEATURE_COLUMNS，ml/registry/model.pkl 为真实产物
COPY ml ./ml

# 非 root 运行（容器逃逸面收敛）
RUN useradd --create-home --uid "${APP_UID}" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 健康检查走 /health：不仅看进程存活，还看模型是否真的加载成功
# （若 pkl 缺失会回落 FakeModel，此时 model_loaded=false，容器判定为不健康）
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import json,sys,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)); sys.exit(0 if d.get('model_loaded') else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
