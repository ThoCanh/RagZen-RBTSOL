# Multi-stage production Dockerfile for RagZen
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel

FROM python:3.12-slim AS final

WORKDIR /app

RUN useradd -m -u 1000 ragzenuser && \
    mkdir -p /app/.ragzen /app/imports && \
    chown -R ragzenuser:ragzenuser /app

USER ragzenuser

ENV PATH="/home/ragzenuser/.local/bin:${PATH}"
COPY --from=builder /app/dist/*.whl /app/
COPY deployment/docker/ragzen.production.yaml /app/ragzen.yaml
RUN pip install --no-cache-dir /app/*.whl \
    "fastapi>=0.104.0,<1.0.0" \
    "uvicorn[standard]>=0.24.0,<1.0.0" \
    "python-multipart>=0.0.6,<1.0.0" \
    "qdrant-client>=1.7.0,<3.0.0"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health/live').raise_for_status()"

ENTRYPOINT ["ragzen", "--config", "/app/ragzen.yaml", "serve", "--host", "0.0.0.0", "--port", "8000"]
