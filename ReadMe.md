# KokoroTTS-Windows

**KokoroTTS-Windows** is a lightweight desktop application that turns text into natural-sounding speech using the **Kokoro (ONNX)** text-to-speech engine.  
It provides a simple Tkinter interface designed for fast previews, clear exports, and dependable offline use.

## What it does
- Converts any typed or pasted text into speech within a click.
- Lets you preview audio instantly (Play/Stop) and export clean **WAV** files.
- Offers four curated English voices (2 female, 2 male) for quick voice switching.
- Supports adjustable speaking rate to match narration styles or accessibility needs.
- Runs locally on Windows, keeping content private on your device.

## Why it’s useful
- **Creators:** generate narration for videos, shorts, and tutorials without cloud round-trips.
- **Students & Professionals:** listen to notes, drafts, and reports for faster review.
- **Accessibility:** produce clear audio versions of documents and announcements.
- **Developers & Researchers:** a minimal GUI to audition Kokoro voice quality and cadence.

## Design goals
- **Fast start, low friction:** a single window with essential controls only.
- **Reliable output:** consistent WAV exports suitable for editing pipelines.
- **Local first:** process on your machine to reduce latency and preserve privacy.
- **Small footprint:** no heavyweight frameworks; pure Python + ONNX runtime.

## Core features at a glance
- 🎤 4 built-in English voice profiles (2F, 2M)  
- 🕒 Adjustable speed (0.5×–2.0×)  
- ▶️ Instant preview & stop control  
- 💾 High-quality WAV export  
- 🔒 Offline, local processing

## Roadmap (high-level)
- Voice preview samples per option
- Batch text → multi-file export
- SSML-style punctuation helpers
- Optional auto-save of last session text

---

**KokoroTTS-Windows** is built for clarity and speed—type, listen, export.  
Perfect for creators, learners, and anyone who wants high-quality TTS without the clutter.
