# iPhone OpenClaw Companion

This adds a quick iPhone companion app (Expo) plus a local bridge API.

## What is included

- `iphone_bridge_server.py`: local FastAPI bridge
- `iphone-openclaw/`: Expo iOS app (mic capture + auto-talk loop)
- `start-iphone-bridge.bat`: run bridge server on port `8765`
- `start-iphone-app.bat`: run Expo app server

## Quick start

1. Run `start-iphone-bridge.bat`
2. Run `start-iphone-app.bat`
3. Open the app in Expo Go on iPhone
4. Set server URL in app to your reachable PC IP (`http://<ip>:8765`)
5. Tap `Auto Talk`

## Tuning

Bridge defaults to faster STT model for lower latency:

- `WHISPER_MODEL=small.en`

You can override before launching bridge, for example:

```bat
set WHISPER_MODEL=large-v3-turbo
start-iphone-bridge.bat
```
