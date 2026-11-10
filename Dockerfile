# syntax=docker/dockerfile:1

FROM python:3.10-slim

ARG KOKORO_REPO="zaakirio/kokoro-ru"
ARG RUACCENT_REPO="ruaccent/accentuator"
ARG ENABLED_VOICES="sveta"

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    ENABLED_VOICES=${ENABLED_VOICES} \
    LOG_LEVEL=INFO \
    HF_HOME=/root/.cache/huggingface \
    KOKORO_REPO=${KOKORO_REPO} \
    RUACCENT_REPO=${RUACCENT_REPO}

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 10 -r requirements.txt

COPY prepare.py .

# 🔑 Секрет опционален: если передан --secret — попадёт в /run/secrets/hf_token
#    Если не передан — prepare.py сделает анонимные запросы
RUN --mount=type=secret,id=hf_token \
    python prepare.py

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]