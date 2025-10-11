# KokoroTTS v3.0 — What’s New 

This document lists **only the changes and improvements** introduced in **v3.0** compared to **v2.1** of `kokoro_tts_hybrid.py`.

---

## 🚀 Highlights

* **Provider assertion & banner**: detects the **actual** ONNX Runtime provider in use and warns if your requested accelerator (CUDA/DirectML) isn’t engaged.
* **Voice validation**: scans `voices-v1.0.bin` on startup and only shows voices that truly work.
* **Diagnostics panel**: one click shows ORT versions, available/active providers, GPU info, and model/voice paths.
* **Chunked progress & saving**: visible progress during synthesis and **streamed** WAV writes (no giant buffers).
* **Warm-up pass**: first-run latency reduced via a short initial inference.
* **Polished UI/UX**: cleaner status, progress, context menu, counters, and platform-appropriate theme.

---

## 🧠 Providers & Acceleration

* **New**: `resolve_providers()` refined with **AUTO** preference (CUDA → DirectML → CPU).
* **New**: `_assert_primary_provider()` raises a friendly warning if the primary provider isn’t what you selected (e.g., chose CUDA but running on CPU).
* **New**: Provider banner shows **active providers** after synthesis.
* **New**: `_session_providers_string()` for robust provider introspection across ORT session variants.

---

## 🔊 Voice Handling

* **New**: `validate_voices_async()` pre-checks every candidate ID from `VOICE_CANDIDATES` against your `voices-v1.0.bin`, filtering out non-working voices.
* **New**: “Rescan Voices” button to re-validate after changing models/packs.
* **Improved**: Voice dropdown restores last selection when possible.

---

## ⚡ Performance & Stability

* **New**: **Warm-up** inference (“Hi” with `af_bella`) to reduce first synthesis latency.
* **New**: **Chunked synthesis** with percent progress callbacks for long text.
* **New**: **Chunked WAV writer** (`save_wav_chunked`) to prevent high memory usage on big outputs.
* **Improved**: Background worker threads for synthesis and saving keep the UI responsive.
* **Improved**: Safer temporary file lifecycle on Windows playback (handles file locks).

---

## 🖥️ UI / UX

* **New**: Clear **status line + progress bar** with text (Synthesizing… / Saving… / Playing…).
* **New**: **Character/word counter** auto-updates on edit.
* **New**: **Context menu** (Cut/Copy/Paste/Select All).
* **New**: **Provider banner** (“Active providers: …”) under the progress row.
* **Improved**: Platform-appropriate fonts/themes (Vista on Windows); polished spacing and layout.
* **Improved**: Keyboard shortcuts

  * Speak: `Ctrl+Enter`
  * Save: `Ctrl+S`
  * Stop: `Esc`

---

## 🧪 Diagnostics & Logging

* **New**: **Diagnostics dialog** shows:

  * onnxruntime version and kokoro-onnx version
  * **available** vs **active** providers
  * **CUDA GPU name** via `nvidia-smi` (if present)
  * model/voice file paths
* **New**: **Startup logging** to `startuplog.txt` with timestamps and tracebacks on failure.

---

## 🧰 Configuration & Dev Experience

* **New**: Optional **auto-download** of model/voices from Hugging Face when `AUTO_DOWNLOAD=True`.
* **Improved**: Robust asset discovery (`_find_first`), works with multiple expected paths.
* **Improved**: Error surfaces via message boxes; UI state (buttons/progress) is restored on exceptions.
* **Improved**: Cleaner separation of concerns (providers, voice validation, playback, chunking, diagnostics).

---

## 🔄 Backwards Compatibility & Migration

* v3.0 remains compatible with your existing **Kokoro ONNX** model and **voices** files.
* `VOICE_CANDIDATES` still accepts your custom label/ID pairs; v3.0 just filters to **working** ones.
* Existing Windows workflows continue; v3.0 adds guardrails (assertions, diagnostics) and better long-text handling.

---

## ✍️ Summary

v3.0 focuses on **confidence and control**: you can now **see and trust** which accelerator is actually running, validate voices up front, watch **real progress** on long jobs, and diagnose issues instantly—all while enjoying a smoother, faster UI.
