# KokoroTTS (Hybrid GUI)

**KokoroTTS** is a fast, privacy‑first desktop app that turns text into natural‑sounding speech using the **Kokoro (ONNX)** model. This version runs on **macOS (Apple Silicon: M1–M4, CoreML/CPU)** and **Windows 10/11 (CUDA/DirectML/CPU)** with smart, automatic provider selection.

---

## ✨ Highlights

* **Cross‑platform**: one app for macOS + Windows
* **Smart acceleration** (AUTO):

  * macOS → **CoreML → CPU**
  * Windows → **CUDA → DirectML → CPU**
* **Voice validation**: scan `voices-v1.0.bin` and show only working voices
* **Chunked synthesis**: long text handled safely and quickly
* **Warm‑up**: trims first‑run latency
* **Diagnostics panel**: shows ORT versions, providers (available/active), GPU/Metal info, paths
* **Reliable playback & export**: instant preview + streamed **WAV** writes

---

## 🧩 What you can do

* Convert any text to speech in a click
* Preview (Play/Stop) and export **WAV** files
* Switch curated English voices (expandable)
* Adjust speed **0.5×–2.0×**
* Work 100% **offline**—your text never leaves your device

---

## 📦 Requirements

* **Python** 3.9–3.12 (3.10–3.12 recommended)
* Files next to the script:

  * `kokoro-v1.0.onnx` *or* `kokoro-v1.0.fp16.onnx`
  * `voices-v1.0.bin`
* Internet **not required** after you have the files
* Tip: **FP16** model is often faster with CoreML on Apple Silicon

---

## 🧰 Install — macOS (Apple Silicon: M1–M4)

1. Open **Terminal**, create & activate a venv

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
2. Install dependencies

   ```bash
   python -m pip install onnxruntime soundfile sounddevice numpy
   ```
3. Place files in the project folder

   ```
   kokoro_tts_hybrid_cross.py
   kokoro-v1.0.onnx   # or kokoro-v1.0.fp16.onnx
   voices-v1.0.bin
   ```
4. Verify providers

   ```bash
   python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
   ```

   Expect: `['CoreMLExecutionProvider', 'CPUExecutionProvider']` (CPU fallback is fine).
5. Run

   ```bash
   python kokoro_tts_hybrid_cross.py
   ```

---

## 🧰 Install — Windows 10/11

1. Open **PowerShell**, create & activate a venv

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   py -m pip install --upgrade pip
   ```
2. Install dependencies (CPU baseline)

   ```powershell
   py -m pip install onnxruntime soundfile sounddevice numpy
   ```

   If `sounddevice` fails: install PortAudio (e.g., `winget install PortAudio.PortAudio`) and retry.
3. Place files next to the script

   ```
   kokoro_tts_hybrid_cross.py
   kokoro-v1.0.onnx
   voices-v1.0.bin
   ```
4. (Optional) GPU acceleration

   * **DirectML** (NVIDIA/AMD/Intel):

     ```powershell
     py -m pip uninstall -y onnxruntime
     py -m pip install onnxruntime-directml==1.20.1
     ```
   * **CUDA** (NVIDIA only; drivers/CUDA runtime required):

     ```powershell
     py -m pip uninstall -y onnxruntime onnxruntime-directml
     py -m pip install onnxruntime-gpu==1.20.1
     ```
5. Verify providers

   ```powershell
   py -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
   ```

   Look for `CUDAExecutionProvider` or `DmlExecutionProvider` (CPU always present).
6. Run

   ```powershell
   py .\kokoro_tts_hybrid_cross.py
   ```

---

## ▶️ Using the App

1. **Paste or type text**
2. Choose a **Voice** (validated from `voices-v1.0.bin`)
3. Adjust **Speed** (0.5×–2.0×)
4. Pick **Compute**:

   * **AUTO** (recommended)
   * **CoreML** (macOS Apple Silicon)
   * **CUDA** (Windows + NVIDIA)
   * **DirectML** (Windows + most GPUs)
   * **CPU** (portable fallback)
5. **▶ Speak** to preview, **■ Stop** to stop
6. **💾 Save WAV…** to export (streamed writes)
7. **Diagnostics…** to view ORT versions, providers, GPU/Metal, and paths

> If a requested accelerator isn’t engaged, the app warns and continues on CPU.

---

## 🔊 Audio Backend

* Prefers **sounddevice** on both OSes (low latency)
* Falls back to **winsound** on Windows if needed
* **soundfile** handles reliable, chunked WAV writes

---

## ⚡ Tips for Speed & Stability

* Apple Silicon: try **FP16** + **CoreML**
* Keep **AUTO** unless debugging a provider
* Very long text is handled automatically via chunking

---

## 🧪 Sanity Checks

* **Audio beep**

  ```bash
  python -c "import numpy as np, sounddevice as sd, math; sr=48000; t=np.linspace(0,0.4,int(sr*0.4),False); sd.play((0.1*np.sin(2*math.pi*440*t)).astype('float32'), sr); sd.wait(); print('beep ok')"
  ```
* **Providers**

  ```bash
  python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
  ```

---

## 🛠️ Troubleshooting

**macOS — no CoreML**

* `python -m pip install -U onnxruntime`
* Confirm arm64: `python -c "import platform; print(platform.system(), platform.machine())"` (should be `Darwin arm64`)

**Windows — no CUDA/DirectML**

* Install the correct ORT build (see GPU options)
* Re‑verify providers; ensure NVIDIA drivers/CUDA runtime for CUDA builds

**No audio / wrong device**

```python
import sounddevice as sd
print(sd.query_devices(), '\nDefault:', sd.default.device)
# Set output index
sd.default.device = (None, 3)
```

**PATH warnings in PowerShell**

* Harmless for imports; only affects launching console scripts globally

---

## ⌨️ Shortcuts

* **Speak**: macOS `⌘ + Enter` · Windows `Ctrl + Enter`
* **Save WAV**: macOS `⌘ + S` · Windows `Ctrl + S`
* **Stop**: `Esc`

---

## 🧭 Roadmap

* Batch mode (text list → multiple WAVs)
* SSML‑style helpers (pauses/emphasis)
* Voice sample previews
* Default device selection in UI
* Session autosave

---

## 📝 Changelog (v3.0 vs older Windows‑only)

* Cross‑platform GUI (macOS + Windows) with adaptive providers
* CoreML/CUDA/DirectML support + assertions & banner
* Voice validation pipeline
* Chunked progress (synthesis & saving) + warm‑up
* Diagnostics panel (providers, GPU/Metal, versions, paths)
* Unified playback (sounddevice preferred; winsound fallback)

---

## 🙌 Credits

* **Model**: Kokoro (ONNX)
* **Runtime**: ONNX Runtime (CoreML / CUDA / DirectML / CPU)
* **Audio**: sounddevice + soundfile (winsound fallback on Windows)
* **GUI**: Tkinter

**Type → Listen → Export.** Run locally, run fast—on an **M4 Pro** Mac or a Windows box with **CUDA/DirectML**.
