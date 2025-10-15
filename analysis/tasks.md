# Proposed Maintenance Tasks

## Fix a Typo
- **Location:** `v2.1/Readme.md`
- **Issue:** The quick-start instructions mix numbering styles (`1.` → `2)` → `3)`), which reads like a typographical error in the step list.
- **Proposed Task:** Normalize the numbering (e.g., use `1.`, `2.`, `3.`) so the steps render consistently in Markdown.
- **Why it matters:** Consistent numbering improves readability and avoids confusing users skimming the setup steps.

## Fix a Bug
- **Location:** `v3.0/kokoro_tts.py`
- **Issue:** `validate_voices_async` promises to invoke its callback on the Tk thread, but when `app.after` is unavailable (such as when the module is embedded and no global `app` exists) it calls `on_done` directly from the worker thread. That can mutate Tk widgets off the main thread and also suppresses the error dialog path.
- **Proposed Task:** Refactor the callback dispatch so it always schedules back onto the UI thread (e.g., accept a `tk.Widget` for `after` or use `self.after`), falling back safely when Tk is unavailable.
- **Why it matters:** Prevents thread-safety issues and ensures error reporting still runs even when the global `app` variable is not defined.

## Fix a Documentation Discrepancy
- **Location:** Root `ReadMe.md`
- **Issue:** The installation instructions still reference `kokoro_tts_hybrid_cross.py`, but the actual launch script for v3.0 lives at `v3.0/kokoro_tts.py`.
- **Proposed Task:** Update the README paths/commands to point at the current entry point (and mention the directory structure if needed).
- **Why it matters:** Keeps setup guidance accurate for newcomers so they launch the correct script.

## Improve a Test
- **Location:** `v3.0/kokoro_tts.py`
- **Issue:** The `chunk_text` helper that powers chunked synthesis lacks automated tests, even though its sentence-splitting logic and length guard are critical for long-form synthesis.
- **Proposed Task:** Introduce unit tests (e.g., with `pytest`) that cover punctuation splitting, `max_len` boundaries, and edge cases like whitespace-only input to guard against regressions.
- **Why it matters:** Ensures future changes to text-chunking keep honoring the expected chunk sizes and don’t break synthesis progress reporting.
