import os
import sys
import shutil
import logging
from huggingface_hub import hf_hub_download, snapshot_download

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] prepare: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,   # перебиваем возможные другие basicConfig
)
logger = logging.getLogger("prepare")

REPO = os.getenv("KOKORO_REPO", "zaakirio/kokoro-ru")
RUACCENT_REPO = os.getenv("RUACCENT_REPO", "ruaccent/accentuator")

# Опциональный токен из BuildKit secret
token = None
token_file = "/run/secrets/hf_token"
if os.path.exists(token_file):
    with open(token_file) as f:
        token = f.read().strip() or None
    if token:
        logger.info(f"HF_TOKEN: используется ({token[:8]}...)")
    else:
        logger.warning("HF_TOKEN: файл секрета пустой, анонимные запросы")
else:
    logger.info("HF_TOKEN: не задан, анонимные запросы (возможен rate-limit)")

voices_env = os.getenv("ENABLED_VOICES", "sveta")
enabled_voices = [v.strip().lower() for v in voices_env.split(",") if v.strip()]
logger.info(f"Предзагрузка голосов: {', '.join(enabled_voices)}")


def dl_hf(repo: str, filename: str):
    logger.debug(f"hf_hub_download: {repo}/{filename}")
    return hf_hub_download(repo, filename, token=token)


def dl_snapshot(repo: str, **kwargs):
    logger.debug(f"snapshot_download: {repo} {kwargs}")
    return snapshot_download(repo, token=token, **kwargs)


try:
    # 1. ru_g2p.py
    logger.info("Скачивание ru_g2p.py...")
    shutil.copy(dl_hf(REPO, "ru_g2p.py"), "/app/ru_g2p.py")

    # 2. Конфиги
    logger.info("Скачивание конфигов...")
    for f in ("kokoro-config.json", "config.json"):
        shutil.copy(dl_hf(REPO, f), f"/app/{f}")

    # 3. espeak-data
    logger.info("Скачивание espeak-data...")
    dl_snapshot(REPO, allow_patterns="espeak-data/*", local_dir="/app")

    # 4. RUAccent
    logger.info("Скачивание моделей RUAccent...")
    import ruaccent as _ruaccent_pkg

    ruaccent_dir = os.path.dirname(_ruaccent_pkg.__file__)
    logger.info(f"Скачивание моделей ruaccent в {ruaccent_dir}")
    dl_snapshot(RUACCENT_REPO, local_dir=ruaccent_dir)

    # 5. Базовая модель Kokoro
    logger.info("Скачивание базовой модели Kokoro...")
    dl_hf(REPO, "kokoro-ru-v2-base.pth")

    # 6. Все голоса из ENABLED_VOICES
    for voice in enabled_voices:
        logger.info(f"Скачивание голоса: {voice}.pt")
        dl_hf(REPO, f"voices/{voice}.pt")

    logger.info("Подготовка завершена успешно.")
except Exception as e:
    logger.exception(f"Ошибка подготовки: {e}")
    sys.exit(1)
