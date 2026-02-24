from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from kokoro_onnx import Kokoro
from pydantic import BaseModel
from faster_whisper import WhisperModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3:8b")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "af_heart")
DEFAULT_SPEED = float(os.getenv("DEFAULT_SPEED", "1.2"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
DEFAULT_NUM_PREDICT = int(os.getenv("DEFAULT_NUM_PREDICT", "96"))
DEFAULT_NUM_CTX = int(os.getenv("DEFAULT_NUM_CTX", "2048"))

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are OpenClaw, a concise, friendly conversational assistant. Reply in 1-2 short sentences for spoken audio unless asked for detail.",
)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
REPLIES_DIR = ROOT / "iphone_replies"
REPLIES_DIR.mkdir(parents=True, exist_ok=True)

KOKORO_MODEL_PATH = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

app = FastAPI(title="Local Voice iPhone Bridge", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/replies", StaticFiles(directory=str(REPLIES_DIR)), name="replies")

_whisper: WhisperModel | None = None
_tts: Kokoro | None = None
_sessions: dict[str, list[dict[str, str]]] = {}


class TextTurn(BaseModel):
    text: str
    session_id: str | None = None
    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    speed: float = DEFAULT_SPEED


def _get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper


def _get_tts() -> Kokoro:
    global _tts
    if _tts is None:
        _tts = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))
    return _tts


def _session_messages(session_id: str) -> list[dict[str, str]]:
    msgs = _sessions.get(session_id)
    if msgs is None:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        _sessions[session_id] = msgs
    return msgs


def _ollama_chat(messages: list[dict[str, str]], model: str) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.2,
            "num_predict": DEFAULT_NUM_PREDICT,
            "num_ctx": DEFAULT_NUM_CTX,
        },
    }
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=120.0)
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    return THINK_PATTERN.sub("", content).strip()


def _tts_to_wav_url(text: str, voice: str, speed: float) -> str:
    samples, sample_rate = _get_tts().create(text, voice=voice, speed=speed, lang="en-us")
    samples_f32 = np.asarray(samples, dtype=np.float32)
    samples_i16 = np.clip(samples_f32 * 32767.0, -32768, 32767).astype(np.int16)

    fname = f"{uuid.uuid4().hex}.wav"
    path = REPLIES_DIR / fname
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(samples_i16.tobytes())
    _cleanup_replies(max_files=120)
    return f"/replies/{fname}"


def _cleanup_replies(max_files: int = 120) -> None:
    files = sorted(REPLIES_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[max_files:]:
        try:
            old.unlink()
        except OSError:
            pass


def _transcribe_file(local_path: str) -> str:
    segments, _ = _get_whisper().transcribe(local_path, language="en", beam_size=1)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


def _run_turn(text: str, session_id: str, model: str, voice: str, speed: float) -> dict[str, Any]:
    messages = _session_messages(session_id)
    messages.append({"role": "user", "content": text})

    reply = _ollama_chat(messages, model)
    messages.append({"role": "assistant", "content": reply})

    audio_url = _tts_to_wav_url(reply, voice, speed) if reply else None

    return {
        "session_id": session_id,
        "transcript": text,
        "reply_text": reply,
        "audio_url": audio_url,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/session/new")
def new_session() -> dict[str, str]:
    sid = uuid.uuid4().hex
    _session_messages(sid)
    return {"session_id": sid}


@app.post("/chat-text")
def chat_text(turn: TextTurn) -> dict[str, Any]:
    sid = turn.session_id or uuid.uuid4().hex
    return _run_turn(turn.text, sid, turn.model, turn.voice, turn.speed)


@app.post("/talk")
async def talk(
    audio: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    model: str = Form(default=DEFAULT_MODEL),
    voice: str = Form(default=DEFAULT_VOICE),
    speed: float = Form(default=DEFAULT_SPEED),
) -> dict[str, Any]:
    sid = session_id or uuid.uuid4().hex

    suffix = Path(audio.filename or "clip.m4a").suffix or ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await audio.read())

    try:
        transcript = _transcribe_file(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if not transcript:
        return {
            "session_id": sid,
            "transcript": "",
            "reply_text": "",
            "audio_url": None,
        }

    return _run_turn(transcript, sid, model, voice, speed)
