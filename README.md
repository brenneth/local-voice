# local-voice

Fully local voice assistant. No cloud APIs, 100% private.

**Pipeline:** Microphone → Whisper (STT) → Ollama (LLM) → Kokoro (TTS) → Speaker

Two versions:
- **Mac** (`voice_assistant.py`) — MLX-Whisper on Apple Silicon
- **PC** (`voice_assistant_pc.py`) — faster-whisper on Windows

## Mac Setup

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- [Ollama](https://ollama.com) with `qwen3:14b` or `qwen3:32b`
- [Homebrew](https://brew.sh)

### Install

```bash
# System dependency
brew install portaudio

# Create venv and install Python packages
python3 -m venv ~/ai-env
source ~/ai-env/bin/activate
pip install mlx-whisper sounddevice kokoro-onnx httpx numpy

# Pull Ollama models
ollama pull qwen3:14b
ollama pull qwen3:32b  # optional, for "powerful" mode

# Download Kokoro TTS model files
mkdir -p ~/ai-env/models
cd ~/ai-env/models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

### Usage

```bash
./start-assistant.sh                        # default (qwen3:14b, push-to-talk)
./start-assistant.sh --model qwen3:32b      # more powerful model
./start-assistant.sh --vad                  # hands-free mode with wake word
./start-assistant.sh --vad --no-wake        # hands-free without wake word
./start-assistant.sh --voice af_bella       # different voice
./start-assistant.sh --speed 1.2            # faster speech
```

## PC (Windows) Setup

### Requirements

- Windows 10/11
- Python 3.10+
- [Ollama for Windows](https://ollama.com) with `qwen3:8b`
- ~16GB RAM recommended

### Install

Option A — run `setup-pc.bat` (does everything automatically).

Option B — manual:

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements-pc.txt

mkdir models
curl -L -o models\kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models\voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

ollama pull qwen3:8b
```

### Usage

```cmd
start-assistant-pc.bat                      # default (qwen3:8b, push-to-talk)
start-assistant-pc.bat --vad                # hands-free mode
start-assistant-pc.bat --model qwen3:14b    # if you have more RAM
start-assistant-pc.bat --voice af_bella     # different voice
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Space` (hold) | Record in push-to-talk mode |
| `v` | Toggle push-to-talk / hands-free (VAD) |
| `n` / `p` | Next / previous voice |
| `r` | Reset voice to af_heart |
| `+` / `-` | Speed up / slow down TTS |
| `ESC` | Skip current speech |
| `c` | Clear conversation history |
| `s` | Save conversation to file |
| `q` | Quit |

## Features

- **Streaming TTS** — starts speaking the first sentence while the LLM is still generating
- **Voice interrupt** — in hands-free mode, speaking into the mic cuts off TTS playback
- **Wake word** — say "Voice mode" to activate in hands-free mode (configurable)
- **Conversation memory** — auto-summarizes older exchanges to stay within context limits
- **Audio chimes** — subtle tones for recording start/stop feedback
- **Echo cancellation** — mic muted briefly after TTS to prevent self-triggering
- **File operations** — ask the assistant to save notes, logs, or documents to disk

## Designing for Low-Latency "Thinking Partner" Use

If your goal is to think out loud with minimal friction (for example on a MacBook with AirPods), use this setup strategy:

1. **Prefer wired input for the mic path when possible**
   - Bluetooth earbuds usually switch to a lower-quality bidirectional profile when the microphone is active.
   - Best quality+latency combo: USB mic (input) + earbuds/headphones (output).

2. **Use push-to-talk for fastest turn quality**
   - Push-to-talk avoids false triggers and lets you keep VAD thresholds conservative.
   - If you want hands-free flow, enable `--vad` and tune silence settings in `voice_assistant.py`.

3. **Keep local models small for interactive speed**
   - `qwen3:14b` is a good default for Mac.
   - Use larger models only when you need deeper reasoning and can tolerate extra delay.

4. **Layer lightweight filters before transcription**
   - Prioritize simple, low-latency cleanup (noise suppression, gain normalization, optional high-pass filter) over heavy DSP.
   - Aggressive filtering can hurt recognition accuracy more than it helps.

5. **Use a hybrid local/cloud routing policy for research-heavy turns**
   - Keep normal brainstorming local for privacy and speed.
   - Route only "research" or "web-dependent" queries to a cloud model/API (for example Gemini), then feed the result back into local conversation memory.
   - This preserves responsiveness while still giving you high-quality external knowledge when needed.

6. **Treat the assistant like a two-lane system**
   - **Fast lane (default):** local STT + local LLM + local TTS for rapid idea iteration.
   - **Deep lane (on demand):** optional cloud call for synthesis, fact-checking, or citations.

This architecture usually gives the best balance of latency, privacy, and answer quality for "help me think through problems" workflows.

## If You Want to Clone Something More Complete

If you want an off-the-shelf project instead of building your own pipeline, these are strong options:

1. **OpenWebUI + Ollama + Whisper/Faster-Whisper**
   - Repo: https://github.com/open-webui/open-webui
   - Best when you want a polished UI, tool use, and easy model switching.
   - Good "thinking partner" option if voice can be routed through browser/desktop integrations.

2. **Home Assistant Assist (local voice stack)**
   - Repo: https://github.com/home-assistant/core
   - Best when you want robust wake-word + STT/TTS orchestration and automation hooks.
   - More infrastructure-heavy, but mature for always-on voice agents.

3. **OpenVoiceOS**
   - Repo: https://github.com/OpenVoiceOS/OpenVoiceOS
   - Best when you want a dedicated open-source voice assistant framework with plugins.
   - Strong for extensibility, but more setup complexity than this repo.

### Quick recommendation

For your specific goal (low-latency voice + problem-solving on a Mac), keep this repo as the core and add optional cloud research routing. If you want to switch to a more full-featured platform, start with **OpenWebUI** first because it is usually the fastest path to a polished daily workflow.

## Voices

28 voices available, sorted by quality. Top voices:

| Voice | Grade | Description |
|-------|-------|-------------|
| `af_heart` | A | American female (default) |
| `af_bella` | A- | American female |
| `af_nicole` | B- | American female |
| `bf_emma` | B- | British female |
