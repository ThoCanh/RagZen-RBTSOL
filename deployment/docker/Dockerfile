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
    mkdir -p /app/.ragzen && \
    chown -R ragzenuser:ragzenuser /app

USER ragzenuser

ENV PATH="/home/ragzenuser/.local/bin:${PATH}"
ENV RAGZEN_STORAGE_PATH=/app/.ragzen

COPY --from=builder /app/dist/*.whl /app/
RUN pip install --no-cache-dir /app/*.whl "ragzen[server]"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health/live').raise_for_status()"

ENTRYPOINT ["ragzen", "serve", "--host", "0.0.0.0", "--port", "8000"]
