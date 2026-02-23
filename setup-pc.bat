@echo off
REM One-time setup for Local Voice Assistant on Windows PC
REM Run this from the local-voice directory

echo === Local Voice Assistant - PC Setup ===
echo.

REM Create virtual environment
echo Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing Python packages...
pip install -r requirements-pc.txt

REM Create models directory
echo Creating models directory...
mkdir models 2>nul

REM Download Kokoro TTS models
echo Downloading Kokoro TTS model files...
echo (This may take a few minutes — ~340MB total)
curl -L -o models\kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models\voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

REM Pull Ollama model
echo.
echo Pulling Ollama model (qwen3:8b)...
echo Make sure Ollama is installed and running!
ollama pull qwen3:8b

echo.
echo === Setup complete! ===
echo.
echo To start the assistant:
echo   start-assistant-pc.bat
echo.
echo Or with hands-free mode:
echo   start-assistant-pc.bat --vad
echo.
pause
