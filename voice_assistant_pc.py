#!/usr/bin/env python3
"""Local Voice Assistant (Windows PC version) — mic → faster-whisper STT → Ollama → Kokoro TTS → speaker.

Features:
  - Streaming TTS: starts speaking first sentence while LLM still generates
  - Voice interrupt: in hands-free mode, speaking into mic cuts off TTS
  - ESC interrupt: press Escape to skip speech in push-to-talk mode
  - Wake word: say "Voice mode" to activate in hands-free mode
  - Conversation memory: auto-summarizes old exchanges to stay within context
  - Audio chimes: subtle tones for recording start/stop feedback
  - Echo cancellation: mic muted briefly after TTS to prevent self-triggering

Requires: faster-whisper, sounddevice, kokoro-onnx, httpx, numpy
"""

import argparse
import json
import math
import msvcrt
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
SILENCE_THRESHOLD = 500  # RMS threshold for VAD
SILENCE_TIMEOUT = 1.5  # seconds of silence before stopping
MIN_SPEECH_DURATION = 0.3  # minimum seconds to consider as speech
INTERRUPT_RMS = 800  # mic RMS to trigger voice interrupt
ECHO_COOLDOWN = 0.5  # seconds to mute mic after TTS finishes
MAX_HISTORY_PAIRS = 20  # max user/assistant exchanges before summarizing

WAKE_WORD = "voice mode"

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
WHISPER_MODEL = "distil-medium.en"

DEFAULT_OUTPUT_DIR = Path.home() / "Documents" / "assistant-output"

FILE_OP_PATTERN = re.compile(
    r"<<FILE_OP>>(.*?)<<END_FILE_OP>>", re.DOTALL
)
THINK_PATTERN = re.compile(
    r"<think>.*?</think>", re.DOTALL
)
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

# Build user-home prefix for file safety check
ALLOWED_PREFIX = str(Path.home()).replace("\\", "/") + "/"

SYSTEM_PROMPT = f"""\
You are a local research assistant for a scientist. You help with:
- Project planning, scheduling, and experiment tracking
- Hypothesis discussion and brainstorming
- Writing and editing notes, logs, summaries, and documents
- Composing files when asked (Markdown, text, etc.)

When the user asks you to write, save, create, or log something to a file,
respond with a <<FILE_OP>> block containing a JSON object:
<<FILE_OP>>{{"action": "write|append", "path": "<full path>", "content": "<file content>"}}<<END_FILE_OP>>

Default save location: {DEFAULT_OUTPUT_DIR}

Keep responses concise for voice conversation. Use natural, conversational language.
Speak in complete sentences suitable for text-to-speech output.
Avoid markdown formatting, bullet points, or code blocks in spoken responses \
unless the user specifically asks for written/formatted content.
/no_think"""

# Voices ordered by quality grade (A -> D), then by type
VOICES = [
    # Grade A / A-
    "af_heart",    # A  — American female (top rated)
    "af_bella",    # A- — American female
    "af_nicole",   # B- — American female
    "bf_emma",     # B- — British female
    # Grade C+
    "af_aoede",    # C+ — American female
    "af_kore",     # C+ — American female
    "af_sarah",    # C+ — American female
    # Grade C
    "af_alloy",    # C  — American female
    "af_nova",     # C  — American female
    "bf_isabella", # C  — British female
    "bm_fable",    # C  — British male
    "bm_george",   # C  — British male
    # Grade C-
    "af_sky",      # C- — American female
    # Grade D+
    "bm_lewis",    # D+ — British male
    # Grade D
    "af_jessica",  # D  — American female
    "af_river",    # D  — American female
    "bf_alice",    # D  — British female
    "bf_lily",     # D  — British female
    "bm_daniel",   # D  — British male
    # Male American (ungraded)
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck",
]

# ---------------------------------------------------------------------------
# Audio chimes — generated programmatically (no external files)
# ---------------------------------------------------------------------------
CHIME_SR = 24000


def _generate_chime(freq: float, duration: float, volume: float = 0.15) -> np.ndarray:
    """Generate a soft sine wave chime."""
    t = np.linspace(0, duration, int(CHIME_SR * duration), dtype=np.float32)
    envelope = np.ones_like(t)
    fade = int(CHIME_SR * 0.02)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    return (np.sin(2 * math.pi * freq * t) * envelope * volume).astype(np.float32)


CHIME_START = _generate_chime(880, 0.08)   # short high A — "I'm listening"
CHIME_STOP = _generate_chime(440, 0.08)    # short low A — "got it"
CHIME_WAKE = _generate_chime(660, 0.12)    # medium E — "wake word heard"


def play_chime(chime: np.ndarray):
    """Play a chime and wait for it to finish."""
    sd.play(chime, samplerate=CHIME_SR)
    sd.wait()


# ---------------------------------------------------------------------------
# Terminal helpers for non-blocking key input (Windows msvcrt)
# ---------------------------------------------------------------------------

def _key_pressed(timeout=0.0):
    """Check for a keypress with optional timeout. Returns the key or None."""
    if timeout <= 0:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            # Handle special keys (arrows, function keys): they come as two chars
            if ch in ('\x00', '\xe0'):
                msvcrt.getwch()  # consume the second byte
                return None
            return ch
        return None

    deadline = time.time() + timeout
    while time.time() < deadline:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ('\x00', '\xe0'):
                msvcrt.getwch()
                return None
            return ch
        time.sleep(0.01)
    return None


# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------

# Timestamp of when TTS last finished playing (for echo cancellation)
_tts_end_time = 0.0


def record_push_to_talk() -> np.ndarray | None:
    print("\n  [Hold SPACE to talk | v:mode  n/p:voice  r:reset  +/-:speed  c:clear  s:save  q:quit]")

    frames = []

    while True:
        ch = _key_pressed(timeout=0.1)
        if ch == "q":
            return "QUIT"
        if ch == "v":
            return "TOGGLE"
        if ch == "n":
            return "NEXT_VOICE"
        if ch == "p":
            return "PREV_VOICE"
        if ch == "r":
            return "RESET_VOICE"
        if ch in ("+", "="):
            return "SPEED_UP"
        if ch in ("-", "_"):
            return "SPEED_DOWN"
        if ch == "c":
            return "CLEAR"
        if ch == "s":
            return "SAVE"
        if ch == " ":
            break

    play_chime(CHIME_START)
    print("\r  Recording... (release SPACE to stop)", end="", flush=True)
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
        blocksize=1024
    )
    stream.start()

    while True:
        data, _ = stream.read(1024)
        frames.append(data.copy())
        ch = _key_pressed(timeout=0.0)
        if ch is not None and ch != " ":
            break
        if ch is None:
            time.sleep(0.02)
            data2, _ = stream.read(1024)
            frames.append(data2.copy())
            ch2 = _key_pressed(timeout=0.05)
            if ch2 is None:
                # Space was released
                break
            elif ch2 != " ":
                break

    stream.stop()
    stream.close()
    play_chime(CHIME_STOP)
    print("\r  Processing...                        ", flush=True)

    if not frames:
        return None

    audio = np.concatenate(frames, axis=0).flatten()
    if len(audio) / SAMPLE_RATE < MIN_SPEECH_DURATION:
        return None
    return audio


def record_vad() -> np.ndarray | None:
    """Record using VAD with wake word support."""
    global _tts_end_time
    print(f'\n  [Say "{WAKE_WORD}" to activate | v:mode  n/p:voice  r:reset  +/-:speed  c:clear  s:save  q:quit]')

    frames = []
    speech_started = False
    silence_start = None

    cooldown_remaining = ECHO_COOLDOWN - (time.time() - _tts_end_time)
    if cooldown_remaining > 0:
        time.sleep(cooldown_remaining)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
        blocksize=1024
    )
    stream.start()

    try:
        while True:
            ch = _key_pressed(timeout=0.0)
            if ch == "q":
                stream.stop(); stream.close()
                return "QUIT"
            if ch == "v":
                stream.stop(); stream.close()
                return "TOGGLE"
            if ch == "n":
                stream.stop(); stream.close()
                return "NEXT_VOICE"
            if ch == "p":
                stream.stop(); stream.close()
                return "PREV_VOICE"
            if ch == "r":
                stream.stop(); stream.close()
                return "RESET_VOICE"
            if ch in ("+", "="):
                stream.stop(); stream.close()
                return "SPEED_UP"
            if ch in ("-", "_"):
                stream.stop(); stream.close()
                return "SPEED_DOWN"
            if ch == "c":
                stream.stop(); stream.close()
                return "CLEAR"
            if ch == "s":
                stream.stop(); stream.close()
                return "SAVE"

            data, _ = stream.read(1024)
            rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))

            if not speech_started:
                if rms > SILENCE_THRESHOLD:
                    speech_started = True
                    silence_start = None
                    frames.append(data.copy())
                    print("\r  Hearing speech...                   ", end="", flush=True)
            else:
                frames.append(data.copy())
                if rms < SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        break
                else:
                    silence_start = None

        stream.stop()
        stream.close()
    except Exception:
        try:
            stream.stop(); stream.close()
        except Exception:
            pass
        return None

    if not frames:
        return None

    audio = np.concatenate(frames, axis=0).flatten()
    if len(audio) / SAMPLE_RATE < MIN_SPEECH_DURATION:
        return None

    # Quick transcribe to check for wake word
    text = transcribe(audio)
    if not text:
        return None

    text_lower = text.lower().strip()

    if not text_lower.startswith(WAKE_WORD):
        print("\r                                      ", end="\r", flush=True)
        return None

    play_chime(CHIME_WAKE)

    after_wake = text[len(WAKE_WORD):].lstrip(" ,.:!?")

    if after_wake.strip():
        print("\r  Processing...                        ", flush=True)
        return after_wake  # Return as string (text, not audio)
    else:
        print("\r  Listening...                         ", end="", flush=True)
        play_chime(CHIME_START)
        return _record_after_wake()


def _record_after_wake() -> np.ndarray | None:
    """Record the actual command after wake word was detected."""
    frames = []
    speech_started = False
    silence_start = None

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
        blocksize=1024
    )
    stream.start()

    wait_start = time.time()
    while True:
        data, _ = stream.read(1024)
        rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))

        if not speech_started:
            if rms > SILENCE_THRESHOLD:
                speech_started = True
                silence_start = None
                frames.append(data.copy())
                print("\r  Recording...                        ", end="", flush=True)
            elif time.time() - wait_start > 5.0:
                stream.stop()
                stream.close()
                print("\r                                      ", end="\r", flush=True)
                return None
        else:
            frames.append(data.copy())
            if rms < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_TIMEOUT:
                    break
            else:
                silence_start = None

    stream.stop()
    stream.close()
    play_chime(CHIME_STOP)
    print("\r  Processing...                        ", flush=True)

    if not frames:
        return None

    audio = np.concatenate(frames, axis=0).flatten()
    if len(audio) / SAMPLE_RATE < MIN_SPEECH_DURATION:
        return None
    return audio


# ---------------------------------------------------------------------------
# Speech-to-Text (faster-whisper for Windows)
# ---------------------------------------------------------------------------

_whisper_model = None

WHISPER_HALLUCINATIONS = {
    "", "you", "thank you", "thanks", "thank you.", "thanks.", "bye",
    "the end", "the end.", "thanks for watching", "thanks for watching.",
    "thank you for watching", "thank you for watching.",
    "please subscribe", "subscribe", "like and subscribe",
    "it seems like you didn't send a message",
    "it seems you didn't send a message",
    "it seems like you may be typing",
    "...", "\u2026",
}


def transcribe(audio: np.ndarray) -> str:
    global _whisper_model

    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    if rms < SILENCE_THRESHOLD * 0.5:
        return ""

    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
        )

    audio_f32 = audio.astype(np.float32) / 32768.0
    segments, _ = _whisper_model.transcribe(
        audio_f32,
        language="en",
        beam_size=1,
        condition_on_previous_text=False,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()

    if text.lower().strip(".!? ") in WHISPER_HALLUCINATIONS:
        return ""
    if len(text) < 3:
        return ""

    return text


# ---------------------------------------------------------------------------
# Conversation memory management
# ---------------------------------------------------------------------------

def trim_conversation(messages: list[dict], model: str) -> list[dict]:
    """Keep conversation within limits by summarizing older exchanges."""
    non_system = [m for m in messages if m["role"] != "system"]
    pairs = len(non_system) // 2

    if pairs <= MAX_HISTORY_PAIRS:
        return messages

    system_msg = messages[0]
    keep_count = MAX_HISTORY_PAIRS * 2
    old_messages = non_system[:-keep_count]
    recent_messages = non_system[-keep_count:]

    old_text = ""
    for msg in old_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        old_text += f"{role}: {msg['content']}\n"

    summary_prompt = [
        {"role": "system", "content": "Summarize this conversation in 2-3 sentences, capturing key facts, decisions, and any important details the user mentioned. Be concise."},
        {"role": "user", "content": old_text}
    ]

    payload = {
        "model": model,
        "messages": summary_prompt,
        "stream": False,
    }

    try:
        resp = httpx.post(OLLAMA_URL, json=payload, timeout=30.0)
        resp.raise_for_status()
        summary = resp.json().get("message", {}).get("content", "")
    except Exception:
        summary = f"[Earlier conversation with {len(old_messages)//2} exchanges was trimmed]"

    print(f"  [Conversation trimmed: summarized {len(old_messages)//2} older exchanges]")

    summary_msg = {
        "role": "system",
        "content": f"Summary of earlier conversation:\n{summary}"
    }

    return [system_msg, summary_msg] + recent_messages


# ---------------------------------------------------------------------------
# LLM Chat via Ollama — streaming with sentence callback
# ---------------------------------------------------------------------------

def chat_streaming(messages: list[dict], model: str, sentence_queue: queue.Queue):
    """Stream LLM response, pushing complete sentences to sentence_queue."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    full_response = []
    sentence_buffer = ""
    in_think_block = False
    in_file_op = False

    try:
        with httpx.stream("POST", OLLAMA_URL, json=payload, timeout=120.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_response.append(token)

                    if "<think>" in token:
                        in_think_block = True
                    if not in_think_block:
                        print(token, end="", flush=True)
                    if "</think>" in token:
                        in_think_block = False
                        continue

                    if "<<FILE_OP>>" in token:
                        in_file_op = True
                    if in_file_op:
                        if "<<END_FILE_OP>>" in token:
                            in_file_op = False
                        continue

                    if not in_think_block and not in_file_op:
                        sentence_buffer += token

                        parts = SENTENCE_END.split(sentence_buffer)
                        if len(parts) > 1:
                            for sentence in parts[:-1]:
                                sentence = sentence.strip()
                                if sentence:
                                    sentence_queue.put(sentence)
                            sentence_buffer = parts[-1]

                if chunk.get("done"):
                    break
    except httpx.ConnectError:
        sentence_queue.put(None)
        return "[Error: Cannot connect to Ollama. Is it running?]"
    except httpx.HTTPStatusError as e:
        sentence_queue.put(None)
        return f"[Error: Ollama returned {e.response.status_code}]"

    print()

    remaining = sentence_buffer.strip()
    if remaining:
        sentence_queue.put(remaining)

    sentence_queue.put(None)

    response = "".join(full_response)
    response = THINK_PATTERN.sub("", response).strip()
    return response


# ---------------------------------------------------------------------------
# File Operations
# ---------------------------------------------------------------------------

def handle_file_ops(response: str) -> str:
    matches = FILE_OP_PATTERN.findall(response)
    if not matches:
        return response

    for raw_json in matches:
        try:
            op = json.loads(raw_json.strip())
        except json.JSONDecodeError:
            print("  [Warning: Could not parse file operation JSON]")
            continue

        action = op.get("action", "write")
        path_str = os.path.expanduser(op.get("path", ""))
        content = op.get("content", "")
        path = Path(path_str)

        # Safety: normalize to forward slashes for comparison
        normalized = str(path).replace("\\", "/")
        if not normalized.startswith(ALLOWED_PREFIX):
            print(f"  [Blocked: path {path} is outside allowed prefix]")
            continue

        try:
            if action == "create_dir":
                path.mkdir(parents=True, exist_ok=True)
                print(f"  [Created directory: {path}]")
            elif action == "append":
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a") as f:
                    f.write(content)
                print(f"  [Appended to: {path}]")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    f.write(content)
                print(f"  [Wrote file: {path}]")
        except OSError as e:
            print(f"  [File operation error: {e}]")

    return FILE_OP_PATTERN.sub("", response).strip()


# ---------------------------------------------------------------------------
# Text-to-Speech (Kokoro ONNX) — with voice interrupt + echo cancellation
# ---------------------------------------------------------------------------

# PC version: look for models in script directory or user home
_script_dir = Path(__file__).parent
KOKORO_MODEL_PATH = _script_dir / "models" / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = _script_dir / "models" / "voices-v1.0.bin"

_tts_model = None


def _clean_for_tts(text: str) -> str:
    """Strip markdown and other formatting that sounds bad when spoken."""
    text = re.sub(r"\*{1,3}", "", text)
    text = re.sub(r"_{1,3}", "", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"`{1,3}", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _play_with_interrupt(samples, sample_rate, vad_mode: bool) -> bool:
    """Play audio, monitoring for interrupt. Returns True if interrupted."""
    global _tts_end_time
    sd.play(samples, samplerate=sample_rate)

    if vad_mode:
        try:
            mic_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
                blocksize=1024
            )
            mic_stream.start()
            while sd.get_stream().active:
                mic_data, _ = mic_stream.read(1024)
                mic_rms = np.sqrt(np.mean(mic_data.astype(np.float32) ** 2))
                if mic_rms > INTERRUPT_RMS:
                    sd.stop()
                    mic_stream.stop()
                    mic_stream.close()
                    _tts_end_time = time.time()
                    print("  [Interrupted]")
                    return True
            mic_stream.stop()
            mic_stream.close()
        except Exception:
            sd.wait()
    else:
        # Monitor keyboard for ESC
        while sd.get_stream().active:
            ch = _key_pressed(timeout=0.1)
            if ch == "\x1b":
                sd.stop()
                print("  [Skipped]")
                _tts_end_time = time.time()
                return True

    _tts_end_time = time.time()
    return False


def speak_streaming(sentence_queue: queue.Queue, voice: str, speed: float, vad_mode: bool):
    """Consume sentences from queue, synthesize and play with interrupt support."""
    global _tts_model

    if _tts_model is None:
        from kokoro_onnx import Kokoro
        _tts_model = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))

    while True:
        try:
            sentence = sentence_queue.get(timeout=30.0)
        except queue.Empty:
            break

        if sentence is None:
            break

        sentence = _clean_for_tts(sentence)
        if not sentence or sentence.startswith("[Error"):
            continue

        try:
            samples, sample_rate = _tts_model.create(
                sentence, voice=voice, speed=speed, lang="en-us"
            )
            interrupted = _play_with_interrupt(samples, sample_rate, vad_mode)
            if interrupted:
                while True:
                    try:
                        remaining = sentence_queue.get_nowait()
                        if remaining is None:
                            break
                    except queue.Empty:
                        break
                return
        except Exception as e:
            print(f"  [TTS error: {e}]")


# ---------------------------------------------------------------------------
# Conversation log saving
# ---------------------------------------------------------------------------

def save_conversation(messages: list[dict]):
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = DEFAULT_OUTPUT_DIR / f"conversation_{ts}.md"

    lines = [f"# Voice Assistant Conversation \u2014 {time.strftime('%Y-%m-%d %H:%M')}\n\n"]
    for msg in messages:
        if msg["role"] == "system":
            continue
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"**{role}:** {msg['content']}\n\n")

    path.write_text("".join(lines))
    print(f"  [Conversation saved to {path}]")


# ---------------------------------------------------------------------------
# VAD without wake word
# ---------------------------------------------------------------------------

def _record_vad_no_wake() -> np.ndarray | None:
    """VAD recording without wake word."""
    global _tts_end_time
    print("\n  [Listening... speak to begin | v:mode  n/p:voice  r:reset  +/-:speed  c:clear  s:save  q:quit]")

    frames = []
    speech_started = False
    silence_start = None

    cooldown_remaining = ECHO_COOLDOWN - (time.time() - _tts_end_time)
    if cooldown_remaining > 0:
        time.sleep(cooldown_remaining)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
        blocksize=1024
    )
    stream.start()

    try:
        while True:
            ch = _key_pressed(timeout=0.0)
            if ch == "q":
                stream.stop(); stream.close()
                return "QUIT"
            if ch == "v":
                stream.stop(); stream.close()
                return "TOGGLE"
            if ch == "n":
                stream.stop(); stream.close()
                return "NEXT_VOICE"
            if ch == "p":
                stream.stop(); stream.close()
                return "PREV_VOICE"
            if ch == "r":
                stream.stop(); stream.close()
                return "RESET_VOICE"
            if ch in ("+", "="):
                stream.stop(); stream.close()
                return "SPEED_UP"
            if ch in ("-", "_"):
                stream.stop(); stream.close()
                return "SPEED_DOWN"
            if ch == "c":
                stream.stop(); stream.close()
                return "CLEAR"
            if ch == "s":
                stream.stop(); stream.close()
                return "SAVE"

            data, _ = stream.read(1024)
            rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))

            if not speech_started:
                if rms > SILENCE_THRESHOLD:
                    speech_started = True
                    silence_start = None
                    frames.append(data.copy())
                    play_chime(CHIME_START)
                    print("\r  Recording...                        ", end="", flush=True)
            else:
                frames.append(data.copy())
                if rms < SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        break
                else:
                    silence_start = None

        stream.stop()
        stream.close()
        play_chime(CHIME_STOP)
        print("\r  Processing...                        ", flush=True)

    except Exception:
        try:
            stream.stop(); stream.close()
        except Exception:
            pass

    if not frames:
        return None

    audio = np.concatenate(frames, axis=0).flatten()
    if len(audio) / SAMPLE_RATE < MIN_SPEECH_DURATION:
        return None
    return audio


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local Voice Assistant (PC)")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama model name")
    parser.add_argument("--voice", default="af_heart", help="Kokoro TTS voice")
    parser.add_argument("--speed", type=float, default=1.0, help="TTS speed (0.5-2.0)")
    parser.add_argument("--vad", action="store_true", help="Start in always-listening mode")
    parser.add_argument("--no-wake", action="store_true", help="Disable wake word in VAD mode")
    args = parser.parse_args()

    voice = args.voice
    voice_idx = VOICES.index(voice) if voice in VOICES else 0
    speed = max(0.5, min(2.0, args.speed))
    use_wake_word = not args.no_wake

    print("=" * 60)
    print("  Local Voice Assistant (PC)")
    print(f"  Model: {args.model}  |  Voice: {voice}  |  Speed: {speed:.1f}x")
    if args.vad:
        wake_status = f'Wake word: "{WAKE_WORD}"' if use_wake_word else "Wake word: disabled"
        print(f"  {wake_status}")
    print("=" * 60)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    vad_mode = args.vad

    # Preload models at startup
    print("\n  Loading models...")
    global _tts_model, _whisper_model

    print("    Whisper STT (faster-whisper)...", end=" ", flush=True)
    from faster_whisper import WhisperModel
    _whisper_model = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
    )
    # Warm up with silent audio
    _whisper_model.transcribe(
        np.zeros(SAMPLE_RATE, dtype=np.float32),
        language="en",
        beam_size=1,
        condition_on_previous_text=False,
    )
    print("done")

    print("    Kokoro TTS...", end=" ", flush=True)
    from kokoro_onnx import Kokoro
    _tts_model = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))
    print("done")

    print(f"\n  Mode: {'Always-listening (VAD)' if vad_mode else 'Push-to-talk'}")
    print("  Ready.")
    play_chime(CHIME_WAKE)

    while True:
        try:
            if vad_mode:
                if use_wake_word:
                    result = record_vad()
                else:
                    result = _record_vad_no_wake()
            else:
                result = record_push_to_talk()

            if result is None:
                continue
            if isinstance(result, str):
                if result == "QUIT":
                    print("\n  Goodbye!")
                    break
                if result == "TOGGLE":
                    vad_mode = not vad_mode
                    mode_name = "Always-listening (VAD)" if vad_mode else "Push-to-talk"
                    print(f"\n  Switched to: {mode_name}")
                    continue
                if result == "NEXT_VOICE":
                    voice_idx = (voice_idx + 1) % len(VOICES)
                    voice = VOICES[voice_idx]
                    print(f"\n  Voice: {voice}")
                    continue
                if result == "PREV_VOICE":
                    voice_idx = (voice_idx - 1) % len(VOICES)
                    voice = VOICES[voice_idx]
                    print(f"\n  Voice: {voice}")
                    continue
                if result == "RESET_VOICE":
                    voice_idx = 0
                    voice = VOICES[0]
                    print(f"\n  Voice reset: {voice}")
                    continue
                if result == "SPEED_UP":
                    speed = min(2.0, speed + 0.1)
                    print(f"\n  Speed: {speed:.1f}x")
                    continue
                if result == "SPEED_DOWN":
                    speed = max(0.5, speed - 0.1)
                    print(f"\n  Speed: {speed:.1f}x")
                    continue
                if result == "CLEAR":
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                    print("\n  Conversation history cleared.")
                    continue
                if result == "SAVE":
                    save_conversation(messages)
                    continue

                # Pre-transcribed text from wake word
                text = result

            else:
                text = transcribe(result)

            if not text:
                print("  [No speech detected]")
                continue

            print(f"\n  You: {text}")

            messages = trim_conversation(messages, args.model)

            messages.append({"role": "user", "content": text})
            print("\n  Assistant: ", end="", flush=True)

            sq = queue.Queue()
            tts_thread = threading.Thread(
                target=speak_streaming,
                args=(sq, voice, speed, vad_mode),
                daemon=True
            )
            tts_thread.start()

            response = chat_streaming(messages, args.model, sq)
            messages.append({"role": "assistant", "content": response})

            tts_thread.join(timeout=60.0)

            handle_file_ops(response)

        except KeyboardInterrupt:
            print("\n\n  Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n  [Error: {e}]")
            continue


if __name__ == "__main__":
    main()
