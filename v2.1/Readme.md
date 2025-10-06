# Kokoro TTS Hybrid — v2.1 (Windows • CPU/CUDA/DirectML)

A polished Windows GUI for the **Kokoro** ONNX text‑to‑speech model with selectable **CPU**, **CUDA**, or **DirectML** backends. Version **2.1** focuses on reliability, diagnostics, and real‑world UX: provider assertion, live voice validation, GPU‑friendly chunking, warm‑up, shortcuts, and logging.

> Main script: `kokoro_tts_hybrid.py`

---

## 🚀 What’s new in **v2.1** (vs. v2.0)

**Provider control & correctness**

* **Primary‑provider assertion**: if you select **CUDA** or **DirectML**, the app verifies the ONNX Runtime session actually starts with that provider and warns if it didn’t (e.g., fallback to CPU).
* **Provider pruning**: the **Compute** dropdown now hides unavailable options based on `ort.get_available_providers()`.
* **Smarter re‑init**: model is only re‑initialized when the compute choice changes (faster repeat runs).

**Voice handling**

* **Live voice validation**: scans `voices‑v1.0.bin` at startup and populates the voice list with only the IDs that actually synthesize.
* **Rescan Voices** button to refresh after changing voice packs.
* Expanded candidate set (e.g., Nova/Aria/Luna/Sky/Willow, Orion/Atlas/River/Rowan/Leo + optional UK Isla/Arthur). Only valid ones are shown.

**Diagnostics & observability**

* New **Diagnostics…** panel (toolbar) with:

  * onnxruntime version, kokoro‑onnx version
  * available vs. active providers
  * CUDA GPU model (via `nvidia‑smi`, if present)
  * resolved model/voices file paths
* **Startup/error logging** to `startuplog.txt` with stack traces for easier support.

**Latency & throughput**

* **Warm‑up inference** on init to reduce first‑utterance delay.
* **Bigger text chunks** (~480 chars) for fewer model calls and better GPU utilization (still sentence‑aware).

**UI/UX polish**

* New **char/word counter**, right‑click context menu (Cut/Copy/Paste/Select All), scroll bar on the editor.
* Keyboard shortcuts: **Ctrl+Enter** Speak, **Ctrl+S** Save, **Esc** Stop.
* Subtle theme improvements; window layout/spacing refined.
* Provider banner shows the **active** providers string after synth.

> TL;DR: 2.1 makes hardware selection trustworthy, voices self‑verifying, troubleshooting obvious, and the whole flow snappier.

---

## ✨ Features

* CPU/CUDA/DirectML execution with **AUTO** fallback.
* Live **voice validation** + quick Rescan.
* Sentence‑aware **chunked synthesis** with progress and **chunked WAV saving**.
* Async, multi‑run‑safe playback (temp‑WAV + safe cleanup/Stop).
* Clear status text: Synthesizing / Playing / Saving / Done.
* One‑click **Diagnostics…** with environment details.

---

## 🧰 Requirements

* **Windows 10/11**, **Python 3.9+**
* **Choose ONE** ONNX Runtime build:

  * CPU: `pip install onnxruntime`
  * CUDA: `pip install onnxruntime-gpu` (NVIDIA drivers/CUDA required)
  * DirectML: `pip install onnxruntime-directml`
* Plus: `numpy`, `soundfile`, `kokoro_onnx`
* Optional (for auto‑download): `huggingface_hub`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy soundfile kokoro_onnx
# pick ONE of these:
pip install onnxruntime            # CPU
# pip install onnxruntime-gpu      # CUDA
# pip install onnxruntime-directml # DirectML
# optional for AUTO_DOWNLOAD=True
# pip install huggingface_hub
```

Place these beside the script (if not using auto‑download):

* `kokoro-v1.0.onnx` (or `kokoro-v1.0.fp16.onnx`)
* `voices-v1.0.bin`

---

## ▶️ Quick Start

```bash
python kokoro_tts_hybrid.py
```

1. Paste text → 2) Pick **Voice**, **Speed**, **Compute** → 3) **▶ Speak** or **💾 Save WAV…**.
   Watch the status bar and **Active providers** banner. Use **Diagnostics…** if something looks off.

---

## 🔎 Diagnostics

**Toolbar → Diagnostics…** shows:

* `onnxruntime` / `kokoro_onnx` versions
* available vs active providers
* CUDA GPU model (if `nvidia-smi` available)
* model/voice file paths

Use this when filing issues or verifying that CUDA/DML is actually used.

---

## 🗣️ Voices

The app validates candidate voice IDs at startup. Only those that successfully synthesize are listed.

* Click **🔄 Rescan Voices** after changing voice packs.
* Default language: `en-us`; adjust at call sites if needed.

---

## ⚙️ Configuration

* `AUTO_DOWNLOAD = False` by default. Set `True` to pull assets from:

  * Model: `onnx-community/Kokoro-82M-v1.0-ONNX`
  * Voices (fallback order): same repo → `hexgrad/Kokoro-82M`
* If assets are missing and auto‑download is off, you’ll get a clear error with instructions.

---

## 🧯 Troubleshooting

* **Selected CUDA/DML but banner shows CPU** → Click **Diagnostics…**. Ensure the correct ORT wheel is installed and drivers are present. The app warns if GPU wasn’t engaged.
* **No voices listed** → Check `voices-v1.0.bin` path or use **Rescan Voices**.
* **Playback issues** → Confirm `soundfile`/libsndfile is installed; verify temp directory write access (AV tools can lock files).
* **Long first utterance** → Warm‑up reduces this, but very large texts still run chunk‑by‑chunk.

---

## 🗓️ Versioning & Changelog

* **2.1** (this release)

  * Provider assertion + pruning; smarter re‑init
  * Live voice validation + Rescan button
  * Diagnostics panel; startup/error logging
  * Warm‑up; larger, GPU‑friendly chunks; UI/shortcut polish
* **2.0**

  * Compute selector (AUTO/CPU/CUDA/DirectML), provider banner
  * Optional auto‑download; chunked synth & save; safer playback; progress UI
* **1.x**

  * Initial GUI & basic synthesis

> Note: The in‑code banner may read “v3” for internal iteration; repository version is **2.1**.


Kokoro ONNX model & voice assets by their respective authors. Thanks to ONNX Runtime (CPU/CUDA/DirectML), NumPy, SoundFile,
