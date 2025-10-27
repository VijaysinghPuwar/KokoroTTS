# Kokoro TTS Hybrid — v5.0 README

A cross-platform GUI for **Kokoro TTS** with first-class accelerator support (CoreML / CUDA / DirectML / CPU), presets, history, loudness & speed controls—now with **online voice discovery**.

> Platforms: **macOS** (Apple Silicon CoreML or CPU) and **Windows** (CUDA / DirectML / CPU)

---

## What’s new in v5.0 (since v4)

**Big features**

* **Online Voice Catalog + Auto-Validation**

  * Pulls full voice IDs from official sources (`VOICES.md` / ONNX model card) and **validates** them against your local `voices-v1.0.bin` by doing a tiny synthesis probe.
  * Falls back to a small offline set if you’re disconnected, so the UI never comes up empty.
* **Provider Auto-Resolution**

  * “AUTO” now smart-selects the **best available**: CoreML (mac arm64) → CUDA (NVIDIA) → DirectML (Windows) → CPU.
  * Clear banner of **active providers** and improved diagnostics text.

**UX & reliability**

* **Smarter progress** (chunked synth + save) with stable, non-jumpy percentages.
* **First-run warm-up** to reduce the “first audio is slow” penalty.
* **Safer playback**: unified stop handling for `sounddevice` and `winsound`, with temp-file cleanup retries on Windows.
* **History quality**: clearer “Play/Save” actions, compact text previews, easy double-click to reload.
* **Presets**: same flow, but now also remember the selected compute backend.

**Dev & maintenance**

* **Config discovery**: tolerant model/voice file detection + optional **auto-download** via `huggingface_hub`.
* **Better errors**: clearer messages when the requested accelerator isn’t actually engaged (e.g., CUDA requested but CPU used).
* **Code cleanup**: eliminated duplicate/legacy worker blocks that caused `NameError: VOICE_CANDIDATES` in edge paths, tightened typing, smaller helpers, and clearer separation of responsibilities.

---

## Quick Start

### 1) Install deps

```bash
pip install onnxruntime soundfile numpy
# Optional playback backend (preferred):
pip install sounddevice
# Optional auto-download support:
pip install huggingface_hub requests
```

### 2) Place assets next to the script

```
kokoro-v1.0.onnx          # or kokoro-v1.0.fp16.onnx (CoreML may like FP16)
voices-v1.0.bin
kokoro_tts_hybrid_cross.py
```

> Or set `AUTO_DOWNLOAD = True` to fetch from Hugging Face automatically.

### 3) Run

```bash
python kokoro_tts_hybrid_cross.py
```

---

## Using accelerators

The **Compute** dropdown exposes:

* **AUTO** (recommended) → CoreML → CUDA → DirectML → CPU
* **CPU**, **CoreML** (macOS/arm64), **CUDA** (NVIDIA), **DirectML** (Windows)

> The **Diagnostics…** button shows `onnxruntime` version, available & active providers, model/voices paths, and GPU info (CUDA/Metal when available).

---

## Voice catalog (new in v5)

* On launch (or **🔄 Rescan voices**) v5:

  1. fetches the official voice lists online,
  2. **probes** each ID locally via a tiny synthesis to confirm it actually exists in your `voices-v1.0.bin`,
  3. populates the combobox with friendly labels like `Bella — American English (F)`.

> Offline? You still get a small curated set so the app remains usable.

---

## Differences vs v4 (migration notes)

* **Static `VOICE_CANDIDATES`** (v4) → **Online discovery + validation** (v5).
* **Manual accelerator pick** (v4) → **AUTO hierarchy + assert/notify if not engaged** (v5).
* **Occasional first-utterance lag** (v4) → **Warm-up call** (v5).
* **Progress jumps** (v4) → **Chunk-aware progress for synth & save** (v5).
* **Potential duplicate worker / `VOICE_CANDIDATES` NameError** (v4) → **Removed dead code path & unified workers** (v5).

No breaking config changes: your `kokoro_presets.json` continues to work. Presets now also remember the compute mode.

---

## Troubleshooting

* **“Requested accelerator not engaged” warning**

  * Check your drivers (CUDA), hardware support, or run with **CPU** to compare.
* **No audio device**

  * Install `sounddevice` (preferred) or rely on `winsound` (Windows only).
* **No voices listed**

  * Ensure `voices-v1.0.bin` is present; if offline, the fallback list will still appear.
* **Slow first line**

  * Expected once per session; v5 pre-warms to minimize this.

---

## File layout

```
kokoro_tts_hybrid_cross.py
kokoro_presets.json                # created on first save/update of presets
kokoro-v1.0.onnx | kokoro-v1.0.fp16.onnx
voices-v1.0.bin  | voices/voices-v1.0.bin
startuplog.txt                     # basic startup logs
```

---

## License & credits

* Uses **onnxruntime**, **soundfile**, and optionally **sounddevice**.
* Kokoro model & voices per their respective licenses.
* Online voice discovery reads official voice listings for convenience.

---

## Changelog

### v5.0

* Online voice fetch + local validation
* AUTO accelerator resolution & provider assertion
* Warm-up synthesis to reduce first-call latency
* Chunk-aware progress for synthesis and saving
* More robust playback/cleanup on Windows
* Diagnostics improvements (GPU/Metal/CUDA hints)
* Removed duplicate worker block causing `NameError` in rare paths
* General refactors & UX polish

### v4.0

* Redesigned layout, History & Presets
* Language & loudness controls
* Smarter status updates
* Initial accelerator selection and diagnostics

---

Happy narrating!
