# Kokoro TTS — Tkinter Desktop (Windows)

A lightweight Tkinter GUI for **Kokoro TTS (ONNX)** with 4 curated English voices  
(2 female, 2 male), speed control, Play/Stop, and **Save as WAV**.

> This repo keeps the code small. **Model files are not checked in.**  
> You’ll download them from the official releases and place them next to the script.

---

## What you need to download

Download these two files from the official Kokoro ONNX releases page:

- **`kokoro-v1.0.onnx`**  
- **`voices-v1.0.bin`**

Put both files in the **same folder** as `kokoro_tts_gui.py`.

> Tip: On the releases page, expand **Assets** and download the exact filenames above.  
> If you get a file named `kokoro-v1.0.fp16.onnx`, **rename it** to `kokoro-v1.0.onnx`.

---

## Quick start (Windows)

1. **Install Python 3.12 or 3.13** (check “Add Python to PATH” during setup).
2. **Download the two files** listed above and place them next to `kokoro_tts_gui.py`.
3. Double-click **`install_deps.bat`**  
   – Creates a `.venv` and installs the required packages.
4. Double-click **`run_app.bat`** to launch the GUI.

If PowerShell blocks scripts, open PowerShell and run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
