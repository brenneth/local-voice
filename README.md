# local-voice

Fully local voice assistant running on Apple Silicon. No cloud APIs, 100% private.

**Pipeline:** Microphone → MLX-Whisper (STT) → Ollama (LLM) → Kokoro (TTS) → Speaker

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- [Ollama](https://ollama.com) with `qwen2.5:7b` or `qwen2.5:32b`
- [Homebrew](https://brew.sh)

## Install

```bash
# System dependency
brew install portaudio

# Create venv and install Python packages
python3 -m venv ~/ai-env
source ~/ai-env/bin/activate
pip install mlx-whisper sounddevice kokoro-onnx httpx numpy

# Pull an Ollama model
ollama pull qwen2.5:7b

# Download Kokoro TTS model files
mkdir -p ~/ai-env/models
cd ~/ai-env/models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

## Usage

```bash
# Quick start
./start-assistant.sh

# Options
python voice_assistant.py --model qwen2.5:32b   # use larger model
python voice_assistant.py --vad                   # hands-free mode
python voice_assistant.py --voice af_bella        # different voice
python voice_assistant.py --speed 1.2             # faster speech
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

## Voices

28 voices available, sorted by quality. Top voices:

| Voice | Grade | Description |
|-------|-------|-------------|
| `af_heart` | A | American female (default) |
| `af_bella` | A- | American female |
| `af_nicole` | B- | American female |
| `bf_emma` | B- | British female |

## File Operations

Ask the assistant to save files and it will write them to `~/Documents/assistant-output/` by default. It can also write to Google Drive or OneDrive cloud storage paths.
