# KokoroTTS v4.0 — What’s New

This release focuses on modernising the desktop experience for Kokoro TTS. Version 4 introduces a redesigned workspace, richer playback controls, and workflow helpers that make it easier to iterate on long-form narration projects.

---

## 🚀 Highlights

* **Split layout dashboard** – text entry, controls, and insights now live side-by-side so nothing is hidden behind dialogs.
* **Session history** – every speak/save run is captured with timestamp, voice, language, speed, loudness, and a preview of the text.
* **Reusable presets** – store favourite voice/language/speed combinations and sync them to `kokoro_presets.json` for future sessions.
* **Language picker + loudness trim** – send any text with an explicit locale and apply ±6 dB gain before playback or export.
* **Smarter status** – progress readouts include your chosen voice, language and speed, and the character counter estimates runtime.

---

## 🎛 Interface upgrades

| Area | Improvement |
| --- | --- |
| **Input panel** | Toolbar buttons for quick text import (`.txt`, `.md`) and one-click clearing. |
| **Statistics** | Live character/word count with a speaking time estimate that respects the speed slider. |
| **Context menu** | Cut/Copy/Paste/Select-All still available with right-click/ctrl-click. |
| **Provider banner** | Always shows the currently active ONNX Runtime providers after each synthesis. |

---

## 🗂 Workflow tools

* **History tab**
  * Tree view of recent renders (plays and saves) with double-click to reload text.
  * Buttons to apply captured settings or re-run synthesis instantly.
  * Keeps up to 80 entries while avoiding duplicates from long sessions.
* **Presets tab**
  * Save, update, apply and delete named profiles.
  * Presets capture voice, language, speed, loudness gain and compute preference.
  * Stored in `kokoro_presets.json` beside the app for easy sharing/version control.

---

## 🔈 Audio controls

* **Language dropdown** exposes a curated list of locale codes (`en-us`, `fr-fr`, `ja-jp`, …) and sends the chosen value directly to `kokoro.create`.
* **Loudness slider** lets you attenuate or boost output between –6 dB and +6 dB before playback/saving with safe clipping.
* **History entries** record the exact loudness and compute provider choice used for every render.

---

## 🧠 Diagnostics & shortcuts

* Diagnostics window now reports the location of `kokoro_presets.json` along with provider information.
* Keyboard shortcuts expanded:
  * Speak: `Ctrl/Cmd + Enter`
  * Save: `Ctrl/Cmd + S`
  * Import text: `Ctrl/Cmd + O`
  * Stop: `Esc`

---

## ⚙️ Under the hood

* Text synthesis, playback and saving still stream on background threads for responsive UI.
* All progress notifications share a unified helper so voice/lang/speed info is consistent.
* WAV saving uses the existing chunked writer but now includes volume-adjusted buffers.

---

## ✅ Upgrade notes

v4.0 remains compatible with the v1.0 Kokoro ONNX model and voice packs. Drop the new `kokoro_tts.py` alongside your existing assets, optionally copy over your old presets, and enjoy the upgraded tooling.
