# Kokoro-RU TTS (Docker + FastAPI)

Легковесный Docker-сервер с OpenAI-совместимым API (`/v1/audio/speech`) для качественного русского синтеза речи на базе модели `kokoro-ru`. Работает на CPU.

## 🚀 Быстрый старт

1. Соберите образ:
```bash
docker build -t kokoro-ru-tts .
```
2. Пример docker-compose:
```yaml
services:
    kokoro-ru-tts:
        build:
          context: ./kokoro-ru-tts
          secrets:
            - hf_token
        container_name: kokoro-ru-tts
        ports:
            - "8003:8000"
        environment:
            - ENABLED_VOICES=${ENABLED_VOICES:-sveta}
            - LOG_LEVEL=DEBUG
        restart: unless-stopped
        networks:
            - default
        volumes:
            - kokoro-cache:/root/.cache
        healthcheck:
            test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]
            interval: 30s
            timeout: 10s
            retries: 20
            start_period: 90s

volumes:
    kokoro-cache:
        driver: local
        name: kokoro-cache

secrets:
    hf_token:
        environment: HF_TOKEN
```