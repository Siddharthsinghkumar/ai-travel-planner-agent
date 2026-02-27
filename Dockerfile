# syntax=docker/dockerfile:1.4

########################
# Builder Stage
########################
FROM python:3.12-slim AS builder

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --prefix=/install -r requirements.txt

COPY . .

########################
# Final Stage
########################
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

WORKDIR /app

# Create non-root user
RUN groupadd -r app && useradd -r -g app app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy app source
COPY --chown=app:app . .

USER app

EXPOSE 8000

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
