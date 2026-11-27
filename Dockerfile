# syntax=docker/dockerfile:1
# FilingLens API. Multi-stage, slim, non-root. Embeddings run on ONNX (fastembed), so the
# image has no PyTorch; the embedding model is baked in, so containers never download it.
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=10
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
ENV FASTEMBED_CACHE_PATH=/app/models
RUN /app/.venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/models')"

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
COPY --chown=app:app config ./config
COPY --chown=app:app docker/api-entrypoint.sh /usr/local/bin/api-entrypoint
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 FASTEMBED_CACHE_PATH=/app/models \
    DATA_DIR=/app/data HF_HUB_OFFLINE=1
RUN mkdir -p /app/data && chown app:app /app/data
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"
ENTRYPOINT ["api-entrypoint"]
CMD ["uvicorn", "filinglens.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
