import os
import io
import re
import time
import logging
import threading
import torch
import soundfile as sf
import numpy as np
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from kokoro import KModel
from huggingface_hub import hf_hub_download
from ru_g2p import RuG2P
from fastapi.responses import Response, JSONResponse

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("kokoro-tts")

if logger.isEnabledFor(logging.DEBUG):
    logger.debug("=" * 60)
    logger.debug("Переменные окружения:")
    logger.debug(f"  LOG_LEVEL={os.getenv('LOG_LEVEL', 'Пусто')}")
    logger.debug(f"  PYTHONUNBUFFERED={os.getenv('PYTHONUNBUFFERED', 'Пусто')}")
    logger.debug(f"  ENABLED_VOICES={os.getenv('ENABLED_VOICES', 'Пусто')}")
    logger.debug(f"  HF_HOME={os.getenv('HF_HOME', 'Пусто')}")
    logger.debug(f"  RUACCENT_CACHE_DIR={os.getenv('RUACCENT_CACHE_DIR', 'Пусто')}")
    logger.debug("=" * 60)

logger.info("=== APP.PY (фоновая загрузка моделей) ===")

REPO = os.getenv("KOKORO_REPO", "zaakirio/kokoro-ru")
MAX_PHONEMES = 500
voices_env = os.getenv("ENABLED_VOICES", "sveta")
enabled_voices = [v.strip().lower() for v in voices_env.split(",") if v.strip()]

# Глобальное состояние загрузки
state = {
    "loading": True,
    "ready": False,
    "error": None,
    "started_at": time.time(),
    "loaded_at": None,
    "stage": "init",
}

# Глобальные переменные, которые заполнит фоновый поток
g2p = None
model_base = None
voice_packs = {}


def load_models():
    """Фоновая загрузка моделей. Не блокирует старт HTTP-сервера."""
    global g2p, model_base, voice_packs
    try:
        state["stage"] = "g2p"
        logger.info("Загрузка G2P...")
        g2p = RuG2P()

        state["stage"] = "base_model"
        logger.info("Загрузка базовой модели...")
        model_base = KModel(
            repo_id=REPO,
            model=hf_hub_download(REPO, "kokoro-ru-v2-base.pth")
        ).eval()

        state["stage"] = "voices"
        logger.info(f"Загрузка голосов: {', '.join(enabled_voices)}...")
        for voice in enabled_voices:
            try:
                path = hf_hub_download(REPO, f"voices/{voice}.pt")
                voice_packs[voice] = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e:
                logger.warning(f"Не удалось загрузить голос '{voice}': {e}")

        if not voice_packs:
            raise RuntimeError("Нет доступных голосов")

        state["ready"] = True
        state["loading"] = False
        state["loaded_at"] = time.time()
        state["stage"] = "ready"
        elapsed = state["loaded_at"] - state["started_at"]
        logger.info(f"Все модели загружены за {elapsed:.1f}с. Сервер готов.")
    except Exception as e:
        state["error"] = str(e)
        state["loading"] = False
        state["stage"] = "error"
        logger.error(f"Ошибка загрузки: {e}")


# Запускаем фоновую загрузку ДО старта uvicorn
_thread = threading.Thread(target=load_models, daemon=True)
_thread.start()

app = FastAPI(title="Kokoro-RU TTS API")


def make_chunks(text: str, max_len: int = MAX_PHONEMES) -> List[str]:
    sentences = re.split(r'(?<=[.!?;])\s+', text)
    chunks = []
    current_text = ""

    def try_add(text_to_add):
        nonlocal current_text
        candidate = (current_text + " " + text_to_add).strip() if current_text else text_to_add
        ipa_candidate, _ = g2p(candidate)
        if len(ipa_candidate) <= max_len:
            current_text = candidate
            return True
        return False

    def commit():
        nonlocal current_text
        if current_text:
            chunks.append(current_text)
            current_text = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if try_add(sentence):
            continue
        commit()
        sub_sentences = re.split(r'(?<=[,])\s+', sentence)
        for sub in sub_sentences:
            sub = sub.strip()
            if not sub:
                continue
            if try_add(sub):
                continue
            commit()
            for word in sub.split(' '):
                if not word:
                    continue
                if try_add(word):
                    continue
                commit()
                current_text = word
    commit()
    return chunks if chunks else [text]


class SpeechRequest(BaseModel):
    input: str
    voice: str = enabled_voices[0] if enabled_voices else "sveta"
    response_format: str = "wav"
    model: Optional[str] = None


@app.get("/health")
async def health():
    """
    200 -> ready (Odysseus может использовать)
    503 -> loading (Odysseus должен подождать)
    """
    if state["error"]:
        return JSONResponse(
            {"status": "error", "error": state["error"], "stage": state["stage"]},
            status_code=500
        )
    if not state["ready"]:
        elapsed = time.time() - state["started_at"]
        return JSONResponse(
            {"status": "loading", "stage": state["stage"], "elapsed_sec": round(elapsed, 1)},
            status_code=503
        )
    return {"status": "ok", "voices": list(voice_packs.keys())}


@app.get("/status")
async def status():
    """Подробный статус для отладки (всегда 200)."""
    elapsed = time.time() - state["started_at"]
    return {
        "ready": state["ready"],
        "loading": state["loading"],
        "stage": state["stage"],
        "error": state["error"],
        "elapsed_sec": round(elapsed, 1),
        "voices_loaded": list(voice_packs.keys()),
    }


@app.post("/v1/audio/speech")
async def synthesize(req: SpeechRequest):
    # Если модели ещё грузятся — возвращаем 503 (Odysseus ретраит)
    if not state["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"Models loading (stage={state['stage']}). Retry in a few seconds."
        )
    if state["error"]:
        raise HTTPException(status_code=500, detail=f"Model load error: {state['error']}")

    if req.voice not in voice_packs:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{req.voice}' not found. Available: {list(voice_packs.keys())}"
        )

    pack = voice_packs[req.voice]
    logger.info(f"Синтез: {len(req.input)} символов, голос={req.voice}")

    try:
        text_chunks = make_chunks(req.input)
        logger.debug(f"Разбито на {len(text_chunks)} чанков")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text splitting error: {str(e)}")

    audio_pieces = []
    for i, chunk in enumerate(text_chunks):
        try:
            ipa, _ = g2p(chunk)
            logger.debug(f"Чанк {i+1}/{len(text_chunks)}: {len(ipa)} фонем")
            if len(ipa) > 510:
                logger.warning(f"Обрезка: {len(ipa)} > 510")
                ipa = ipa[:500]
            with torch.no_grad():
                audio = model_base(ipa, pack[len(ipa) - 1], 1.0, return_output=True).audio
            audio_pieces.append(audio.cpu().numpy())
        except Exception as e:
            logger.error(f"Ошибка в чанке {i+1}: {e}")
            raise HTTPException(status_code=500, detail=f"Synthesis error: {e}")

    if not audio_pieces:
        raise HTTPException(status_code=400, detail="No audio generated")

    full_audio = np.concatenate(audio_pieces)
    logger.info(f"Готово: {len(full_audio)/24000:.2f}с аудио")

    buffer = io.BytesIO()
    sf.write(buffer, full_audio, 24000, format=req.response_format.upper())
    buffer.seek(0)
    return Response(content=buffer.getvalue(), media_type=f"audio/{req.response_format}")
