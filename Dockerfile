# syntax=docker/dockerfile:1.4

########################
# Builder Stage
########################
FROM python:3.12-slim AS builder

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8
WORKDIR /app

ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=120

COPY requirements-prod.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt

########################
# Runtime Stage
########################
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

WORKDIR /app

# Runtime deps only
RUN apt-get -o Acquire::ForceIPv4=true update && \
    apt-get -o Acquire::ForceIPv4=true install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r app && useradd -r -g app app \
 && mkdir -p /app && chown app:app /app

# Copy installed python packages
COPY --from=builder /usr/local /usr/local

# Copy application
COPY --chown=app:app . /app

USER app

EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=2
ENV LOG_LEVEL=info

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD exec gunicorn "api.app:app" \
  --bind 0.0.0.0:${PORT} \
  --workers ${WORKERS} \
  --worker-class uvicorn.workers.UvicornWorker \
  --log-level ${LOG_LEVEL} \
  --timeout 60