@echo off
REM Local Voice Assistant — Windows launcher
REM Usage: start-assistant-pc.bat [--model qwen3:8b] [--vad] [--voice af_heart]

call venv\Scripts\activate.bat
python voice_assistant_pc.py %*
pause
