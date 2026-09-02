# syntax=docker/dockerfile:1
# Multi-stage build for small production image
FROM python:3.10-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.10-slim AS runtime
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
# Optional: download model at build time if MODEL_URL is set
ARG MODEL_URL=""
RUN if [ -n "$MODEL_URL" ]; then python -c "import os,urllib.request; url=os.environ.get('MODEL_URL'); print(f'Downloading model from {url}')" 2>/dev/null || true; fi
EXPOSE 8000
ENV PYTHONPATH=/app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
