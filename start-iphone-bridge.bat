@echo off
cd /d %~dp0
call venv\Scripts\activate.bat
python -m uvicorn iphone_bridge_server:app --host 0.0.0.0 --port 8765
