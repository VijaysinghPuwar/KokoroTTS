# Kokoro TTS — Tkinter Desktop (Windows) · v1.1

A lightweight Tkinter GUI for **Kokoro TTS (ONNX)** with 4 curated English voices  
(2 female, 2 male), speed control, Play/Stop, and **Save as WAV**.

> ⚠️ Model files are **not** in this repo. Download them (see below) and place them
> next to `kokoro_tts_gui.py`.

---

## What’s new in v1.1

- Cleaner **one-click** scripts: `install_deps.bat` (venv + deps) and `run_app.bat`.
- Simpler README with explicit download/placement steps for model files.
- Minor UX polish and safer defaults (speed 0.5×–2.0×).

---

## Download the model files

Download these two files from the official Kokoro ONNX release page and put them in
the **same folder** as `kokoro_tts_gui.py`:

- `kokoro-v1.0.onnx`
- `voices-v1.0.bin`

> If you only find `kokoro-v1.0.fp16.onnx`, download it and **rename** to
> `kokoro-v1.0.onnx`.

---

## Quick start (Windows)

1. **Install Python 3.12 or 3.13** (check “Add Python to PATH”).
2. Clone or download this repo.
3. Place **`kokoro-v1.0.onnx`** and **`voices-v1.0.bin`** beside `kokoro_tts_gui.py`.
4. Double-click **`install_deps.bat`** (creates `.venv` and installs deps).
5. Double-click **`run_app.bat`** to launch the GUI.

If PowerShell blocks scripts:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
