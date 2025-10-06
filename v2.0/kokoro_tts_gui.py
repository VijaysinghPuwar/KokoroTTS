# kokoro_tts_gui_gpu.py — Windows GUI for Kokoro TTS with CPU/CUDA/DirectML selection
import os, sys, re, time, threading, tempfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import soundfile as sf
import winsound
import onnxruntime as ort

from kokoro_onnx import Kokoro  # type: ignore

# ========== Config ==========
AUTO_DOWNLOAD = False                 # Set True to auto-download with huggingface_hub
BASE_DIR = Path(__file__).resolve().parent
MODEL_NAMES = ["kokoro-v1.0.onnx", "kokoro-v1.0.fp16.onnx"]
VOICE_NAMES = ["voices-v1.0.bin", str(Path("voices") / "voices-v1.0.bin")]

VOICES = [
    ("Female — Bella (US)", "af_bella"),
    ("Female — Heart (US)", "af_heart"),
    ("Male — Michael (US)", "am_michael"),
    ("Male — Fenrir (US)", "am_fenrir"),
]

# ========== Locate assets ==========
def _find_first(paths):
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    return None

MODEL_PATH  = _find_first([BASE_DIR / n for n in MODEL_NAMES])
VOICES_PATH = _find_first([BASE_DIR / n for n in VOICE_NAMES])

# ========== Optional auto-download ==========
def _maybe_download():
    global MODEL_PATH, VOICES_PATH
    if not AUTO_DOWNLOAD:
        return
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return
    if MODEL_PATH is None:
        try:
            local = hf_hub_download("onnx-community/Kokoro-82M-v1.0-ONNX", "kokoro-v1.0.onnx", local_dir=str(BASE_DIR))
            dst = BASE_DIR / "kokoro-v1.0.onnx"
            if Path(local) != dst:
                Path(local).replace(dst)
            MODEL_PATH = dst
        except Exception as e:
            print(f"[WARN] ONNX download failed: {e}")
    if VOICES_PATH is None:
        for repo, fname in [
            ("onnx-community/Kokoro-82M-v1.0-ONNX", "voices-v1.0.bin"),
            ("hexgrad/Kokoro-82M", "voices/voices-v1.0.bin"),
        ]:
            try:
                local = hf_hub_download(repo_id=repo, filename=fname, local_dir=str(BASE_DIR))
                dst = BASE_DIR / "voices-v1.0.bin"
                if Path(local) != dst:
                    Path(local).replace(dst)
                VOICES_PATH = dst
                break
            except Exception:
                continue

# ========== Provider selection helpers ==========
def available_providers():
    # e.g., ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
    return set(ort.get_available_providers())

def resolve_providers(choice: str):
    """
    Map UI choice -> providers list (ordered by preference).
    'AUTO' = CUDA if present, else DML, else CPU.
    """
    avail = available_providers()
    if choice == "CPU":
        return ["CPUExecutionProvider"]
    if choice == "CUDA":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if choice == "DirectML":
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    # AUTO:
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "DmlExecutionProvider" in avail:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def providers_string(sess_like) -> str:
    for attr in ("session", "_session", "sess", "_sess"):
        s = getattr(sess_like, attr, None)
        if s is not None and hasattr(s, "get_providers"):
            return ", ".join(s.get_providers())
    # Fallback: report availability (not necessarily the one used)
    return "available: " + ", ".join(sorted(available_providers()))

# ========== Model init ==========
_kokoro = None
_active_providers = None   # for UI display

def init_kokoro(providers_choice: str) -> Kokoro:
    global _kokoro, _active_providers
    if _kokoro is not None:
        return _kokoro
    _maybe_download()
    if MODEL_PATH is None or VOICES_PATH is None:
        raise FileNotFoundError(
            "Missing model files.\n"
            "Place 'kokoro-v1.0.onnx' (or 'kokoro-v1.0.fp16.onnx') and 'voices-v1.0.bin' next to this script,\n"
            "or enable AUTO_DOWNLOAD and install 'huggingface_hub'."
        )

    provs = resolve_providers(providers_choice)
    # Try to pass providers to Kokoro if wrapper supports it
    try:
        _kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH), providers=provs)  # type: ignore[arg-type]
    except TypeError:
        # Wrapper doesn't support 'providers'; fall back to default (ORT will choose)
        _kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
    # Read back the actual providers used (or availability)
    try:
        used = providers_string(_kokoro)
    except Exception:
        used = "unknown"
    _active_providers = used
    print("[Kokoro providers]", used)
    return _kokoro

# ========== Playback (winsound, multi-run safe) ==========
class WinSoundPlayer:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._current_tmp = None

    def stop(self):
        self._stop_evt.set()
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
        self._cleanup_tmp()

    def play(self, samples: np.ndarray, sr: int, on_done=None):
        def worker():
            try:
                with self._lock:
                    self._stop_evt.clear()
                    self._cleanup_tmp()

                    fd, tmp = tempfile.mkstemp(prefix="kokoro_", suffix=".wav")
                    os.close(fd)
                    self._current_tmp = tmp
                    sf.write(tmp, samples, sr)

                    winsound.PlaySound(tmp, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)

                # Poll to allow quick stop
                dur = float(len(samples)) / float(sr) if sr else 0.0
                step = 0.1
                waited = 0.0
                while waited < dur and not self._stop_evt.wait(step):
                    waited += step
            except Exception as e:
                messagebox.showerror("Playback error", str(e))
            finally:
                try:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                except Exception:
                    pass
                self._cleanup_tmp()
                if on_done:
                    on_done()
        threading.Thread(target=worker, daemon=True).start()

    def _cleanup_tmp(self):
        if not self._current_tmp:
            return
        p = Path(self._current_tmp)
        for _ in range(6):
            try:
                if p.exists():
                    p.unlink()
                self._current_tmp = None
                return
            except PermissionError:
                time.sleep(0.12)

_player = WinSoundPlayer()

# ========== Text chunking + chunked synthesis (for % by text) ==========
def chunk_text(text: str, max_len: int = 220):
    parts = re.split(r'([.!?]+[\s\n]+)', text)
    sentences = []
    for i in range(0, len(parts), 2):
        a = parts[i]; b = parts[i+1] if i+1 < len(parts) else ""
        s = (a or "") + (b or "")
        if s.strip(): sentences.append(s.strip())
    if not sentences:
        sentences = [text]
    chunks, buf = [], ""
    for s in sentences:
        if not buf: buf = s
        elif len(buf) + 1 + len(s) <= max_len:
            buf += " " + s
        else:
            chunks.append(buf); buf = s
    if buf: chunks.append(buf)
    return chunks

def synthesize_chunked(kokoro, text, voice_id, speed, lang="en-us", on_progress=None):
    chunks = chunk_text(text)
    total_chars = sum(len(c) for c in chunks) or 1
    sr = None
    outs = []
    done = 0
    for ch in chunks:
        s, sr = kokoro.create(ch, voice=voice_id, speed=speed, lang=lang)
        outs.append(s)
        done += len(ch)
        pct = int((done / total_chars) * 100)
        if on_progress:
            on_progress(min(99, max(1, pct)))
    samples = np.concatenate(outs) if outs else np.zeros((0,), dtype=np.float32)
    if on_progress:
        on_progress(100)
    return samples, sr

# ========== Chunked save ==========
def save_wav_chunked(samples: np.ndarray, sr: int, fpath: str, progress_cb=None, chunk_frames: int = 48000):
    total_frames = int(len(samples))
    Path(fpath).parent.mkdir(parents=True, exist_ok=True)
    channels = (samples.shape[1] if samples.ndim > 1 else 1)
    with sf.SoundFile(fpath, mode='w', samplerate=sr, channels=channels) as f:
        pos = 0
        while pos < total_frames:
            end = min(pos + chunk_frames, total_frames)
            f.write(samples[pos:end])
            pos = end
            if progress_cb:
                pct = int((pos / total_frames) * 100) if total_frames else 100
                progress_cb(pct)

# ========== Tk GUI ==========
class KokoroApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kokoro TTS — Windows (CPU/CUDA/DirectML)")
        self.geometry("820x620")
        self.minsize(700, 540)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        ttk.Label(self, text="Kokoro Text-to-Speech", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, sticky="w", padx=12, pady=(12, 6)
        )

        self.text_box = tk.Text(self, wrap="word", font=("Segoe UI", 11))
        self.text_box.grid(row=1, column=0, sticky="nsew", padx=12)
        self.text_box.insert("1.0", "Type or paste text here and click Speak…")
        self.text_box.focus_set(); self.text_box.mark_set("insert", "end")

        controls = ttk.Frame(self); controls.grid(row=2, column=0, sticky="ew", padx=12, pady=12)
        controls.columnconfigure(1, weight=1); controls.columnconfigure(5, weight=1)

        ttk.Label(controls, text="Voice:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.voice_combo = ttk.Combobox(controls, state="readonly",
                                        values=[f"{label}  ({vid})" for label, vid in VOICES])
        self.voice_combo.current(0); self.voice_combo.grid(row=0, column=1, sticky="ew")

        ttk.Label(controls, text="Speed:").grid(row=0, column=2, sticky="e", padx=(16, 8))
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Scale(controls, from_=0.5, to=2.0, variable=self.speed_var).grid(row=0, column=3, sticky="ew")

        ttk.Label(controls, text="Compute:").grid(row=0, column=4, sticky="e", padx=(16, 8))
        self.compute_combo = ttk.Combobox(controls, state="readonly",
                                          values=["AUTO", "CPU", "CUDA", "DirectML"])
        self.compute_combo.current(0); self.compute_combo.grid(row=0, column=5, sticky="ew")

        # Buttons
        btns = ttk.Frame(self); btns.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 10))
        btns.columnconfigure(0, weight=1)
        self.speak_btn = ttk.Button(btns, text="▶ Speak", command=self.on_speak); self.speak_btn.grid(row=0, column=0, sticky="w")
        ttk.Button(btns, text="■ Stop", command=self.on_stop).grid(row=0, column=1, padx=(8, 0))
        self.save_btn  = ttk.Button(btns, text="💾 Save WAV…", command=self.on_save); self.save_btn.grid(row=0, column=2, padx=(8, 0))

        # Status + Progress
        status_row = ttk.Frame(self); status_row.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 10))
        status_row.columnconfigure(0, weight=1)
        self.status = tk.StringVar(value="Ready")
        ttk.Label(status_row, textvariable=self.status, anchor="w").grid(row=0, column=0, sticky="w")

        self.pbar = ttk.Progressbar(status_row, orient="horizontal", mode="determinate", maximum=100)
        self.pbar.grid(row=0, column=1, sticky="e", padx=(12, 0)); self._pbar_visible = True; self._hide_pbar()

        # Provider banner (read-only)
        self.provider_lbl = ttk.Label(self, text="", anchor="w", foreground="#3a3a3a")
        self.provider_lbl.grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 10))

    # ---- Progress helpers
    def _show_pbar(self):
        if not self._pbar_visible:
            self.pbar.grid(); self._pbar_visible = True
    def _hide_pbar(self):
        if self._pbar_visible:
            self.pbar.stop(); self.pbar.grid_remove(); self._pbar_visible = False
    def pbar_text_progress(self, pct: int, prefix: str = "Synthesizing…"):
        self.pbar.stop(); self.pbar.configure(mode="determinate", maximum=100)
        self._show_pbar()
        pct_i = max(0, min(100, int(pct)))
        self.pbar["value"] = pct_i
        self.status.set(f"{prefix} {pct_i}%")
    def pbar_done(self, msg: str = "Done"):
        self.status.set(msg); self._hide_pbar()

    # ---- Helpers
    def _voice_id(self):
        idx = self.voice_combo.current()
        return VOICES[idx][1] if idx >= 0 else VOICES[0][1]

    def on_stop(self):
        _player.stop()
        self.status.set("Stopped")

    # ---- Actions
    def on_speak(self):
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Empty text", "Please enter some text."); return
        _player.stop()

        voice_id = self._voice_id()
        speed = max(0.5, min(2.0, round(float(self.speed_var.get()), 1)))
        compute_choice = self.compute_combo.get()

        self.speak_btn.config(state="disabled")
        self.pbar_text_progress(1, f"Synthesizing… (voice: {voice_id}, speed: {speed:.1f}x)")

        def worker():
            global _kokoro, _active_providers
            try:
                # Re-init model if provider choice changed from previous boot
                _kokoro = None
                kokoro = init_kokoro(compute_choice)

                def on_prog(p):
                    self.after(0, self.pbar_text_progress, p, "Synthesizing…")

                samples, sr = synthesize_chunked(kokoro, text, voice_id, speed, "en-us", on_progress=on_prog)

                # Show provider banner
                banner = _active_providers or "(providers unknown)"
                self.after(0, self.provider_lbl.config, {"text": f"Active providers: {banner}"})

                self.after(0, self.pbar_text_progress, 100, "Synthesizing…")
                self.after(0, self.status.set, "Playing…")
                _player.play(samples, sr, on_done=lambda: self.after(0, self._on_playback_finished))

            except Exception as e:
                self.after(0, self.pbar_done, "Error")
                self.after(0, lambda: messagebox.showerror("Error", f"Speech generation failed:\n{e}"))
                self.after(0, lambda: self.speak_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_playback_finished(self):
        self.pbar_done("Playback finished")
        self.speak_btn.config(state="normal")

    def on_save(self):
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Empty text", "Please enter some text."); return

        voice_id = self._voice_id()
        speed = max(0.5, min(2.0, round(float(self.speed_var.get()), 1)))
        compute_choice = self.compute_combo.get()

        fpath = filedialog.asksaveasfilename(defaultextension=".wav",
                                             filetypes=[("WAV audio", "*.wav")],
                                             title="Save spoken audio as...")
        if not fpath: return

        self.speak_btn.config(state="disabled"); self.save_btn.config(state="disabled")
        self.pbar_text_progress(1, f"Synthesizing… (voice: {voice_id}, speed: {speed:.1f}x)")

        def worker():
            global _kokoro, _active_providers
            try:
                _kokoro = None
                kokoro = init_kokoro(compute_choice)

                def on_prog(p):
                    self.after(0, self.pbar_text_progress, p, "Synthesizing…")

                samples, sr = synthesize_chunked(kokoro, text, voice_id, speed, "en-us", on_progress=on_prog)

                banner = _active_providers or "(providers unknown)"
                self.after(0, self.provider_lbl.config, {"text": f"Active providers: {banner}"})

                def set_pct(p_save):
                    self.after(0, self.pbar_text_progress, p_save, "Saving WAV…")

                save_wav_chunked(samples, sr, fpath, progress_cb=set_pct, chunk_frames=48000)

                self.after(0, self.pbar_done, f"Saved: {fpath}")
                self.after(0, lambda: messagebox.showinfo("Saved", f"Audio saved to:\n{fpath}"))
            except Exception as e:
                self.after(0, self.pbar_done, "Error")
                self.after(0, lambda: messagebox.showerror("Error", f"Save failed:\n{e}"))
            finally:
                self.after(0, lambda: self.speak_btn.config(state="normal"))
                self.after(0, lambda: self.save_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

# ========== Main ==========
if __name__ == "__main__":
    try:
        app = KokoroApp()
        app.mainloop()
    except Exception as exc:
        messagebox.showerror("Startup error", str(exc))
        sys.exit(1)
