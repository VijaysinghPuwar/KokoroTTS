@echo off
setlocal
if not exist .venv (
  echo Missing .venv. Run install_deps.bat first.
  pause & exit /b 1
)
call .venv\Scripts\activate
if exist kokoro-v1.0.onnx (
  python kokoro_tts_gui.py
) else (
  echo Missing model: kokoro-v1.0.onnx
  pause & exit /b 1
)
