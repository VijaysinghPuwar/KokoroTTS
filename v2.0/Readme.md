# Kokoro TTS GUI — v2.0 (Windows)

A lightweight Tkinter GUI for the **Kokoro** ONNX TTS model. This document focuses on **what’s new in v2.0** compared to **v1.0**. Future changes will roll into **v3.0**.

---

## What’s New in v2.0 (vs v1.0)

* **Compute Backend Selector**

  * Choose **AUTO / CPU / CUDA / DirectML** at runtime.
  * AUTO prefers CUDA → DirectML → CPU based on availability.

* **Chunked Synthesis**

  * Long text is split into sentence-aware chunks, synthesized sequentially, and concatenated.
  * Smoother handling of large inputs and reduced memory spikes.

* **Deterministic Progress UI**

  * Real-time percent progress while **Synthesizing…** and **Saving WAV…**
  * Keeps the app responsive for long passages.

* **Provider Banner**

  * Displays the active ONNX Runtime providers (e.g., `CUDAExecutionProvider, CPUExecutionProvider`).

* **Windows-Native Playback**

  * Switched from `simpleaudio` (v1.0) to **`winsound`** for async playback with stop/purge support.

* **Flexible Asset Discovery**

  * Model: accepts `kokoro-v1.0.onnx` **or** `kokoro-v1.0.fp16.onnx`.
  * Voices: supports `voices-v1.0.bin` **or** `voices/voices-v1.0.bin`.

* **Optional Auto‑Download (off by default)**

  * Can pull model/voices via `huggingface_hub` when enabled.

---

## Quick Comparison

| Capability          | v1.0            | v2.0                                    |
| ------------------- | --------------- | --------------------------------------- |
| Compute backends    | CPU only        | AUTO / CPU / CUDA / DirectML            |
| Text handling       | Single pass     | Chunked synthesis + concat              |
| Progress            | Basic/none      | Determinate % for synth & save          |
| Playback            | `simpleaudio`   | Windows `winsound` (async, stop, purge) |
| Provider visibility | –               | Shows active providers                  |
| Model discovery     | Single file     | ONNX **or** FP16 variant                |
| Voices path         | Single location | Root **or** `voices/` subfolder         |

---

## Minimal Setup

```bash
# Pick ONE runtime backend:
pip install onnxruntime      # CPU
# pip install onnxruntime-gpu   # NVIDIA CUDA
# pip install onnxruntime-directml  # DirectML (AMD/Intel/NVIDIA)

# Common deps
pip install numpy soundfile kokoro-onnx
```

**Files expected:**

* Model: `kokoro-v1.0.onnx` **or** `kokoro-v1.0.fp16.onnx`
* Voices: `voices-v1.0.bin` (same folder) **or** `voices/voices-v1.0.bin`

Run:

```bash
python kokoro_tts_gui_gpu.py
```

---

## Notes

* CUDA requires a compatible NVIDIA GPU + drivers; DirectML works across most modern GPUs.
* For extremely long inputs, chunking ensures the UI stays responsive and the progress bar remains accurate.

---

## Next Version

Any additional changes after this release will be published as **v3.0** along with an updated comparison.
