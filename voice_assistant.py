#!/usr/bin/env python3
"""Local Voice Assistant — mic → MLX-Whisper STT → Ollama → Kokoro TTS → speaker."""

import argparse
import json
import os
import re
import select
import sys
import termios
import time
import tty
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

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"

DEFAULT_OUTPUT_DIR = Path.home() / "Documents" / "assistant-output"

FILE_OP_PATTERN = re.compile(
    r"<<FILE_OP>>(.*?)<<END_FILE_OP>>", re.DOTALL
)

SYSTEM_PROMPT = """\
You are a local research assistant for a scientist. You help with:
- Project planning, scheduling, and experiment tracking
- Hypothesis discussion and brainstorming
- Writing and editing notes, logs, summaries, and documents
- Composing files when asked (Markdown, text, etc.)

When the user asks you to write, save, create, or log something to a file,
respond with a <<FILE_OP>> block containing a JSON object:
<<FILE_OP>>{"action": "write|append", "path": "<full path>", "content": "<file content>"}<<END_FILE_OP>>

Default save location: /Users/bs1026/Documents/assistant-output/
Google Drive: /Users/bs1026/Library/CloudStorage/GoogleDrive-brenneths@gmail.com/
OneDrive: /Users/bs1026/Library/CloudStorage/OneDrive-PrincetonUniversity/

Keep responses concise for voice conversation. Use natural, conversational language.
Speak in complete sentences suitable for text-to-speech output.
Avoid markdown formatting, bullet points, or code blocks in spoken responses \
unless the user specifically asks for written/formatted content."""

# Voices ordered by quality grade (A → D), then by type
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
# Terminal helpers for non-blocking key input
# ---------------------------------------------------------------------------

def _set_raw_mode(fd):
    old = termios.tcgetattr(fd)
    tty.setraw(fd)
    return old


def _restore_mode(fd, old):
    termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _key_pressed(timeout=0.0):
    fd = sys.stdin.fileno()
    rlist, _, _ = select.select([fd], [], [], timeout)
    if rlist:
        return sys.stdin.read(1)
    return None


# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------

def record_push_to_talk() -> np.ndarray | None:
    print("\n  [Hold SPACE to talk | v:mode  n/p:voice  +/-:speed  c:clear  s:save  q:quit]")

    fd = sys.stdin.fileno()
    old_settings = _set_raw_mode(fd)
    frames = []

    try:
        while True:
            ch = _key_pressed(timeout=0.1)
            if ch == "q":
                _restore_mode(fd, old_settings)
                return "QUIT"
            if ch == "v":
                _restore_mode(fd, old_settings)
                return "TOGGLE"
            if ch == "n":
                _restore_mode(fd, old_settings)
                return "NEXT_VOICE"
            if ch == "p":
                _restore_mode(fd, old_settings)
                return "PREV_VOICE"
            if ch == "r":
                _restore_mode(fd, old_settings)
                return "RESET_VOICE"
            if ch in ("+", "="):
                _restore_mode(fd, old_settings)
                return "SPEED_UP"
            if ch in ("-", "_"):
                _restore_mode(fd, old_settings)
                return "SPEED_DOWN"
            if ch == "c":
                _restore_mode(fd, old_settings)
                return "CLEAR"
            if ch == "s":
                _restore_mode(fd, old_settings)
                return "SAVE"
            if ch == " ":
                break

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
            if ch != " " and ch is not None:
                break
            if ch is None:
                time.sleep(0.02)
                data2, _ = stream.read(1024)
                frames.append(data2.copy())
                ch2 = _key_pressed(timeout=0.05)
                if ch2 is None:
                    break
                elif ch2 != " ":
                    break

        stream.stop()
        stream.close()
        print("\r  Processing...                        ", flush=True)

    finally:
        _restore_mode(fd, old_settings)

    if not frames:
        return None

    audio = np.concatenate(frames, axis=0).flatten()
    if len(audio) / SAMPLE_RATE < MIN_SPEECH_DURATION:
        return None
    return audio


def record_vad() -> np.ndarray | None:
    print("\n  [Listening... speak to begin | ESC:interrupt  v:mode  n/p:voice  +/-:speed  c:clear  s:save  q:quit]")

    fd = sys.stdin.fileno()
    old_settings = _set_raw_mode(fd)
    frames = []
    speech_started = False
    silence_start = None

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
            blocksize=1024
        )
        stream.start()

        while True:
            ch = _key_pressed(timeout=0.0)
            if ch == "q":
                stream.stop(); stream.close()
                _restore_mode(fd, old_settings)
                return "QUIT"
            if ch == "v":
                stream.stop(); stream.close()
                _restore_mode(fd, old_settings)
                return "TOGGLE"
            if ch == "n":
                stream.stop(); stream.close()
                _restore_mode(fd, old_settings)
                return "NEXT_VOICE"
            if ch == "p":
                stream.stop(); stream.close()
                _restore_mode(fd, old_settings)
                return "PREV_VOICE"
            if ch in ("+", "="):
                stream.stop(); stream.close()
                _restore_mode(fd, old_settings)
                return "SPEED_UP"
            if ch in ("-", "_"):
                stream.stop(); stream.close()
                _restore_mode(fd, old_settings)
                return "SPEED_DOWN"
            if ch == "c":
                stream.stop(); stream.close()
                _restore_mode(fd, old_settings)
                return "CLEAR"
            if ch == "s":
                stream.stop(); stream.close()
                _restore_mode(fd, old_settings)
                return "SAVE"

            data, _ = stream.read(1024)
            rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))

            if not speech_started:
                if rms > SILENCE_THRESHOLD:
                    speech_started = True
                    silence_start = None
                    frames.append(data.copy())
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
        print("\r  Processing...                        ", flush=True)

    finally:
        _restore_mode(fd, old_settings)

    if not frames:
        return None

    audio = np.concatenate(frames, axis=0).flatten()
    if len(audio) / SAMPLE_RATE < MIN_SPEECH_DURATION:
        return None
    return audio


# ---------------------------------------------------------------------------
# Speech-to-Text
# ---------------------------------------------------------------------------

_whisper_loaded = False

# Whisper hallucinates these during silence or noise — ignore them
WHISPER_HALLUCINATIONS = {
    "", "you", "thank you", "thanks", "thank you.", "thanks.", "bye",
    "the end", "the end.", "thanks for watching", "thanks for watching.",
    "thank you for watching", "thank you for watching.",
    "please subscribe", "subscribe", "like and subscribe",
    "it seems like you didn't send a message",
    "it seems you didn't send a message",
    "it seems like you may be typing",
    "...", "…",
}

def transcribe(audio: np.ndarray) -> str:
    global _whisper_loaded
    import mlx_whisper

    _whisper_loaded = True

    # Check if audio is mostly silence
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    if rms < SILENCE_THRESHOLD * 0.5:
        return ""

    audio_f32 = audio.astype(np.float32) / 32768.0
    result = mlx_whisper.transcribe(
        audio_f32,
        path_or_hf_repo=WHISPER_MODEL,
        language="en",
    )
    text = result.get("text", "").strip()

    # Filter hallucinations
    if text.lower().strip(".!? ") in WHISPER_HALLUCINATIONS:
        return ""
    if len(text) < 3:
        return ""

    return text


# ---------------------------------------------------------------------------
# LLM Chat via Ollama
# ---------------------------------------------------------------------------

def chat(messages: list[dict], model: str) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    full_response = []
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
                    print(token, end="", flush=True)
                if chunk.get("done"):
                    break
    except httpx.ConnectError:
        return "[Error: Cannot connect to Ollama. Is it running?]"
    except httpx.HTTPStatusError as e:
        return f"[Error: Ollama returned {e.response.status_code}]"

    print()
    return "".join(full_response)


# ---------------------------------------------------------------------------
# File Operations
# ---------------------------------------------------------------------------

ALLOWED_PREFIX = "/Users/bs1026/"

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

        if not str(path).startswith(ALLOWED_PREFIX):
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
# Text-to-Speech (Kokoro ONNX)
# ---------------------------------------------------------------------------

KOKORO_MODEL_PATH = Path.home() / "ai-env" / "models" / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = Path.home() / "ai-env" / "models" / "voices-v1.0.bin"

_tts_model = None

def _clean_for_tts(text: str) -> str:
    """Strip markdown and other formatting that sounds bad when spoken."""
    text = re.sub(r"\*{1,3}", "", text)       # bold/italic asterisks
    text = re.sub(r"_{1,3}", "", text)         # bold/italic underscores
    text = re.sub(r"#{1,6}\s*", "", text)      # headings
    text = re.sub(r"`{1,3}", "", text)         # code ticks
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links → text only
    text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)  # bullet points
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)  # numbered lists
    text = re.sub(r"\n{2,}", ". ", text)       # collapse double newlines
    text = re.sub(r"\s{2,}", " ", text)        # collapse whitespace
    return text.strip()


def speak(text: str, voice: str = "af_heart", speed: float = 1.0):
    global _tts_model

    text = _clean_for_tts(text)
    if not text or text.startswith("[Error"):
        return

    if _tts_model is None:
        from kokoro_onnx import Kokoro
        _tts_model = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))

    try:
        samples, sample_rate = _tts_model.create(text, voice=voice, speed=speed, lang="en-us")
        sd.play(samples, samplerate=sample_rate)
        fd = sys.stdin.fileno()
        old_settings = _set_raw_mode(fd)
        try:
            while sd.get_stream().active:
                ch = _key_pressed(timeout=0.1)
                if ch == "\x1b":  # Escape key
                    sd.stop()
                    print("  [Skipped]")
                    break
        finally:
            _restore_mode(fd, old_settings)
    except Exception as e:
        print(f"  [TTS error: {e}]")


# ---------------------------------------------------------------------------
# Conversation log saving
# ---------------------------------------------------------------------------

def save_conversation(messages: list[dict]):
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = DEFAULT_OUTPUT_DIR / f"conversation_{ts}.md"

    lines = [f"# Voice Assistant Conversation — {time.strftime('%Y-%m-%d %H:%M')}\n\n"]
    for msg in messages:
        if msg["role"] == "system":
            continue
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"**{role}:** {msg['content']}\n\n")

    path.write_text("".join(lines))
    print(f"  [Conversation saved to {path}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local Voice Assistant")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model name")
    parser.add_argument("--voice", default="af_heart", help="Kokoro TTS voice")
    parser.add_argument("--speed", type=float, default=1.0, help="TTS speed (0.5-2.0)")
    parser.add_argument("--vad", action="store_true", help="Start in always-listening mode")
    args = parser.parse_args()

    voice = args.voice
    voice_idx = VOICES.index(voice) if voice in VOICES else 0
    speed = max(0.5, min(2.0, args.speed))

    print("=" * 60)
    print("  Local Voice Assistant")
    print(f"  Model: {args.model}  |  Voice: {voice}  |  Speed: {speed:.1f}x")
    print("=" * 60)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    vad_mode = args.vad

    # Preload models at startup so there's no delay mid-conversation
    print("\n  Loading models...")
    global _tts_model, _whisper_loaded
    import mlx_whisper
    print("    Whisper STT...", end=" ", flush=True)
    # Warm up whisper by transcribing a tiny silent clip
    mlx_whisper.transcribe(
        np.zeros(SAMPLE_RATE, dtype=np.float32),
        path_or_hf_repo=WHISPER_MODEL,
        language="en",
    )
    _whisper_loaded = True
    print("done")

    print("    Kokoro TTS...", end=" ", flush=True)
    from kokoro_onnx import Kokoro
    _tts_model = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))
    print("done")

    print(f"\n  Mode: {'Always-listening (VAD)' if vad_mode else 'Push-to-talk'}")
    print("  Ready.")

    while True:
        try:
            if vad_mode:
                result = record_vad()
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

            text = transcribe(result)
            if not text:
                print("  [No speech detected]")
                continue

            print(f"\n  You: {text}")

            messages.append({"role": "user", "content": text})
            print("\n  Assistant: ", end="", flush=True)
            response = chat(messages, args.model)
            messages.append({"role": "assistant", "content": response})

            clean_response = handle_file_ops(response)
            speak(clean_response, voice=voice, speed=speed)

        except KeyboardInterrupt:
            print("\n\n  Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n  [Error: {e}]")
            continue


if __name__ == "__main__":
    main()
