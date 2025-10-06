# kokoro_tts_hybrid.py — Windows GUI for Kokoro TTS with CPU/CUDA/DirectML selection
# v3 — provider assertion, voice validation, diagnostics, chunked progress, warm-up, polished UI

import os, sys, re, time, threading, tempfile, traceback, datetime, subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import soundfile as sf
import winsound
import onnxruntime as ort

from kokoro_onnx import Kokoro  # type: ignore

# ============================ Config ============================
AUTO_DOWNLOAD = False  # Set True to auto-download with huggingface_hub (requires pip install huggingface_hub)
BASE_DIR = Path(__file__).resolve().parent
MODEL_NAMES = ["kokoro-v1.0.onnx", "kokoro-v1.0.fp16.onnx"]
VOICE_NAMES = ["voices-v1.0.bin", str(Path("voices") / "voices-v1.0.bin")]

# Candidate voices (labels are yours; ids must match what's inside voices-v1.0.bin)
# We'll validate these on startup and only show those that work.
VOICE_CANDIDATES = [
    ("Female — Bella (US)",   "af_bella"),
    ("Female — Heart (US)",   "af_heart"),
    ("Male — Michael (US)",   "am_michael"),
    ("Male — Fenrir (US)",    "am_fenrir"),
    # Extra female
    ("Female — Nova (US)",    "af_nova"),
    ("Female — Aria (US)",    "af_aria"),
    ("Female — Luna (US)",    "af_luna"),
    ("Female — Sky (US)",     "af_sky"),
    ("Female — Willow (US)",  "af_willow"),
    # Extra male
    ("Male — Orion (US)",     "am_orion"),
    ("Male — Atlas (US)",     "am_atlas"),
    ("Male — River (US)",     "am_river"),
    ("Male — Rowan (US)",     "am_rowan"),
    ("Male — Leo (US)",       "am_leo"),
    # Optional UK (only if your voice pack provides them)
    ("Female — Isla (UK)",    "bf_isla"),
    ("Male — Arthur (UK)",    "bm_arthur"),
]

AVAILABLE_VOICES: list[tuple[str, str]] = []  # (label, voice_id) after validation

# ============================ Locate assets ============================
def _find_first(paths):
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    return None

MODEL_PATH  = _find_first([BASE_DIR / n for n in MODEL_NAMES])
VOICES_PATH = _find_first([BASE_DIR / n for n in VOICE_NAMES])

# ============================ Optional auto-download ============================
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

# ============================ Provider helpers ============================
def available_providers() -> set[str]:
    # e.g. {'CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider', 'AzureExecutionProvider'}
    return set(ort.get_available_providers())

def resolve_providers(choice: str) -> list[str]:
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
    # AUTO
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "DmlExecutionProvider" in avail:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def _session_providers_string(sess_like) -> str:
    for attr in ("session", "_session", "sess", "_sess"):
        s = getattr(sess_like, attr, None)
        if s is not None and hasattr(s, "get_providers"):
            return ", ".join(s.get_providers())
    return "unknown"

def _assert_primary_provider(kokoro_obj, wanted_primary: str):
    """Raise if the first (primary) provider is not what we asked for (when available)."""
    for attr in ("session", "_session", "sess", "_sess"):
        s = getattr(kokoro_obj, attr, None)
        if s is not None and hasattr(s, "get_providers"):
            used = s.get_providers()
            if wanted_primary and wanted_primary in {"CUDAExecutionProvider", "DmlExecutionProvider"}:
                if used and used[0] != wanted_primary:
                    raise RuntimeError(f"Requested {wanted_primary} but active providers: {used}")
            return used
    return None

# ============================ Model init ============================
_kokoro: Kokoro | None = None
_active_providers_str: str | None = None  # for UI

def init_kokoro(providers_choice: str) -> Kokoro:
    """
    Create Kokoro with requested providers; assert primary if possible.
    """
    global _kokoro, _active_providers_str
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
    primary = provs[0] if provs else "CPUExecutionProvider"

    # Try passing providers into Kokoro if wrapper supports it
    try:
        _kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH), providers=provs)  # type: ignore[arg-type]
    except TypeError:
        # Wrapper might not expose providers; fall back to default (ORT decides)
        _kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))

    # Assert/record used providers
    try:
        used = _session_providers_string(_kokoro)
        _active_providers_str = used
        # If we can see providers, enforce primary when the GPU provider is requested
        try:
            _assert_primary_provider(_kokoro, primary)
        except RuntimeError as e:
            # Surface a friendly message (still allow CPU to run)
            messagebox.showwarning("GPU not engaged", str(e))
    except Exception:
        _active_providers_str = "unknown"

    # Warm-up to cut first-latency cost
    try:
        _kokoro.create("Hi", voice="af_bella", speed=1.0, lang="en-us")
    except Exception:
        pass

    print("[Kokoro providers]", _active_providers_str)
    return _kokoro

# ============================ Voice validation ============================
def validate_voices_async(compute_choice: str, on_done):
    """
    Validate which voice IDs actually exist in voices-v1.0.bin.
    Calls on_done(valid_list) back on the Tk thread.
    """
    def worker():
        valid: list[tuple[str, str]] = []
        exc_first = None
        try:
            k = init_kokoro(compute_choice)
        except Exception as e:
            exc_first = e
            k = None

        if k is not None:
            for label, vid in VOICE_CANDIDATES:
                try:
                    k.create("Hi", voice=vid, speed=1.0, lang="en-us")
                    valid.append((label, vid))
                except Exception:
                    continue

        def finish():
            if exc_first:
                messagebox.showerror("Voice scan failed", str(exc_first))
                on_done([])
            else:
                on_done(valid)

        # Bounce to Tk after thread
        try:
            app.after(0, finish)  # type: ignore[name-defined]
        except Exception:
            on_done(valid)

    threading.Thread(target=worker, daemon=True).start()

# ============================ Playback (winsound) ============================
class WinSoundPlayer:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._current_tmp: str | None = None

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

                # Poll for completion / allow Stop
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

# ============================ Text chunking / synth ============================
def chunk_text(text: str, max_len: int = 480):
    """
    Split into sentence-like chunks up to ~max_len for fewer model calls (GPU-friendly).
    """
    parts = re.split(r'([.!?]+[\s\n]+)', text)
    sentences = []
    for i in range(0, len(parts), 2):
        a = parts[i]
        b = parts[i + 1] if i + 1 < len(parts) else ""
        s = (a or "") + (b or "")
        if s.strip():
            sentences.append(s.strip())
    if not sentences:
        sentences = [text]

    chunks, buf = [], ""
    for s in sentences:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_len:
            buf += " " + s
        else:
            chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
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

# ============================ Chunked save ============================
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

# ============================ Diagnostics helpers ============================
def _get_gpu_name_cuda() -> str | None:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=1)
        lines = out.decode(errors="ignore").strip().splitlines()
        return lines[0] if lines else None
    except Exception:
        return None

def get_diagnostics_text() -> str:
    lines = []
    lines.append(f"onnxruntime: {ort.__version__}")
    try:
        import kokoro_onnx as kx  # type: ignore
        lines.append(f"kokoro-onnx: {getattr(kx, '__version__', 'unknown')}")
    except Exception:
        lines.append("kokoro-onnx: unknown")

    lines.append(f"available providers: {', '.join(sorted(available_providers()))}")
    lines.append(f"active providers: {_active_providers_str or 'unknown'}")
    cuda_name = _get_gpu_name_cuda()
    if cuda_name:
        lines.append(f"CUDA GPU: {cuda_name}")
    lines.append(f"model: {MODEL_PATH or '(missing)'}")
    lines.append(f"voices: {VOICES_PATH or '(missing)'}")
    return "\n".join(lines)

# ============================ GUI ============================
class KokoroApp(tk.Tk):
    PADX = 12
    PADY = 10

    def __init__(self):
        super().__init__()
        self.title("Kokoro TTS Hybrid — Windows (CPU/CUDA/DirectML)")
        self.geometry("880x660")
        self.minsize(760, 580)

        self._last_compute_choice = None
        self._last_voice_vid = None

        # base grid
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Header
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=self.PADX, pady=(self.PADY, 6))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Kokoro Text-to-Speech", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Diagnostics…", command=self._show_diag).grid(row=0, column=1, sticky="e")

        # Text
        text_wrap = ttk.Frame(self); text_wrap.grid(row=1, column=0, sticky="nsew", padx=self.PADX)
        text_wrap.columnconfigure(0, weight=1); text_wrap.rowconfigure(0, weight=1)

        self.text_box = tk.Text(text_wrap, wrap="word", font=("Segoe UI", 11), undo=True)
        self.text_box.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(text_wrap, orient="vertical", command=self.text_box.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        self.text_box.configure(yscrollcommand=vscroll.set)

        self.text_box.insert("1.0", "Type or paste text here and click Speak…")
        self.text_box.focus_set(); self.text_box.mark_set("insert", "end")

        # Context menu
        self._ctx = tk.Menu(self, tearoff=0)
        for label, ev in [("Cut", "<<Cut>>"), ("Copy", "<<Copy>>"), ("Paste", "<<Paste>>"), (None, None), ("Select All", "<<SelectAll>>")]:
            if label is None: self._ctx.add_separator()
            else: self._ctx.add_command(label=label, command=lambda e=ev: self.text_box.event_generate(e))
        self.text_box.bind("<Button-3>", lambda e: self._ctx.tk_popup(e.x_root, e.y_root))

        # Meta (counter)
        meta = ttk.Frame(self); meta.grid(row=2, column=0, sticky="ew", padx=self.PADX, pady=(6, 0))
        meta.columnconfigure(0, weight=1)
        self.chars_var = tk.StringVar(value="0 chars • 0 words")
        ttk.Label(meta, textvariable=self.chars_var, foreground="#666").grid(row=0, column=0, sticky="w")
        self.text_box.bind("<<Modified>>", self._on_text_modified)
        self._update_char_counter()

        # Controls
        controls = ttk.Frame(self); controls.grid(row=3, column=0, sticky="ew", padx=self.PADX, pady=self.PADY)
        for c in (1, 3, 5, 7): controls.columnconfigure(c, weight=1)

        ttk.Label(controls, text="Voice:").grid(row=0, column=0, sticky="w", padx=(0,8))
        self.voice_combo = ttk.Combobox(controls, state="readonly", values=["(scanning…)"])
        self.voice_combo.current(0); self.voice_combo.grid(row=0, column=1, sticky="ew")

        ttk.Label(controls, text="Speed:").grid(row=0, column=2, sticky="e", padx=(16,8))
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_scale = ttk.Scale(controls, from_=0.5, to=2.0, variable=self.speed_var)
        self.speed_scale.grid(row=0, column=3, sticky="ew")

        ttk.Label(controls, text="Compute:").grid(row=0, column=4, sticky="e", padx=(16,8))
        self.compute_combo = ttk.Combobox(controls, state="readonly", values=["AUTO", "CPU", "CUDA", "DirectML"])
        self.compute_combo.current(0); self.compute_combo.grid(row=0, column=5, sticky="ew")
        self._prune_unavailable_providers()

        self.rescan_btn = ttk.Button(controls, text="🔄 Rescan Voices", command=self._rescan_voices)
        self.rescan_btn.grid(row=0, column=6, padx=(12,0), sticky="e")

        # Buttons
        btns = ttk.Frame(self); btns.grid(row=4, column=0, sticky="ew", padx=self.PADX, pady=(0, self.PADY))
        btns.columnconfigure(0, weight=1)
        self.speak_btn = ttk.Button(btns, text="▶ Speak", command=self.on_speak); self.speak_btn.grid(row=0, column=0, sticky="w")
        ttk.Button(btns, text="■ Stop", command=self.on_stop).grid(row=0, column=1, padx=(8,0))
        self.save_btn = ttk.Button(btns, text="💾 Save WAV…", command=self.on_save); self.save_btn.grid(row=0, column=2, padx=(8,0))

        # Status + Progress
        status_row = ttk.Frame(self); status_row.grid(row=5, column=0, sticky="ew", padx=self.PADX, pady=(0, self.PADY))
        status_row.columnconfigure(0, weight=1); status_row.columnconfigure(1, weight=1)
        self.status = tk.StringVar(value="Ready")
        ttk.Label(status_row, textvariable=self.status, anchor="w").grid(row=0, column=0, sticky="w")
        self.pbar = ttk.Progressbar(status_row, orient="horizontal", mode="determinate", maximum=100)
        self.pbar.grid(row=0, column=1, sticky="ew", padx=(12,0)); self._pbar_visible = True; self._hide_pbar()

        # Provider banner
        self.provider_lbl = ttk.Label(self, text="", anchor="w", foreground="#3a3a3a")
        self.provider_lbl.grid(row=6, column=0, sticky="ew", padx=self.PADX, pady=(0, self.PADY//2))

        # Keybindings & theme
        self.bind_all("<Control-Return>", lambda e: self.on_speak())
        self.bind_all("<Control-s>", lambda e: self.on_save())
        self.bind_all("<Escape>", lambda e: self.on_stop())
        try:
            style = ttk.Style(self)
            if "vista" in style.theme_names(): style.theme_use("vista")
        except Exception:
            pass

        # Initial voice scan (after window shown)
        self.after(200, self._rescan_voices)

    # -------- Progress helpers --------
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

    # -------- Helpers --------
    def _voice_id(self):
        src = AVAILABLE_VOICES if AVAILABLE_VOICES else VOICE_CANDIDATES[:4]
        idx = self.voice_combo.current()
        return src[idx][1] if 0 <= idx < len(src) else src[0][1]

    def _prune_unavailable_providers(self):
        try:
            avail = available_providers()
        except Exception:
            avail = {"CPUExecutionProvider"}
        keep = ["AUTO", "CPU"]
        if "CUDAExecutionProvider" in avail: keep.append("CUDA")
        if "DmlExecutionProvider" in avail: keep.append("DirectML")
        cur = self.compute_combo.get()
        self.compute_combo.configure(values=keep)
        if cur in keep: self.compute_combo.set(cur)
        else: self.compute_combo.current(0)

    def _on_text_modified(self, _evt=None):
        if self.text_box.edit_modified():
            self.text_box.edit_modified(False)
            self._update_char_counter()

    def _update_char_counter(self):
        txt = self.text_box.get("1.0", "end-1c")
        n = len(txt); words = len(txt.split()) if n else 0
        self.chars_var.set(f"{n} chars • {words} words")

    def set_provider_banner(self, text: str):
        self.provider_lbl.config(text=f"Active providers: {text}")

    def _populate_voices(self, voices_list: list[tuple[str,str]]):
        if not voices_list:
            data = [f"{l}  ({v})" for (l, v) in VOICE_CANDIDATES[:4]]
            self.voice_combo.configure(values=data); self.voice_combo.current(0)
            return
        data = [f"{l}  ({v})" for (l, v) in voices_list]
        self.voice_combo.configure(values=data)
        # restore last selection if possible
        if self._last_voice_vid:
            try:
                pos = [vid for _, vid in voices_list].index(self._last_voice_vid)
                self.voice_combo.current(pos)
            except ValueError:
                self.voice_combo.current(0)
        else:
            self.voice_combo.current(0)

    def _rescan_voices(self):
        self.rescan_btn.config(state="disabled")
        self.status.set("Scanning voices…")
        compute_choice = self.compute_combo.get()

        def on_done(valid_list):
            global AVAILABLE_VOICES
            AVAILABLE_VOICES = valid_list
            self._populate_voices(AVAILABLE_VOICES)
            n = len(AVAILABLE_VOICES) if AVAILABLE_VOICES else len(VOICE_CANDIDATES[:4])
            self.status.set(f"Ready — {n} voice(s) available")
            self.rescan_btn.config(state="normal")

        validate_voices_async(compute_choice, on_done)

    def _show_diag(self):
        diag = tk.Toplevel(self); diag.title("Diagnostics"); diag.geometry("560x380")
        diag.resizable(True, True)
        txt = tk.Text(diag, wrap="word", font=("Consolas", 10))
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", get_diagnostics_text())
        txt.configure(state="disabled")
        ttk.Button(diag, text="Close", command=diag.destroy).pack(pady=6)

    # -------- Actions --------
    def on_stop(self):
        _player.stop()
        self.status.set("Stopped")

    def on_speak(self):
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Empty text", "Please enter some text.")
            return

        _player.stop()
        voice_id = self._voice_id(); self._last_voice_vid = voice_id
        speed = max(0.5, min(2.0, round(float(self.speed_var.get()), 1)))
        compute_choice = self.compute_combo.get()
        need_reinit = (self._last_compute_choice != compute_choice)

        self.speak_btn.config(state="disabled")
        self.pbar_text_progress(1, f"Synthesizing… (voice: {voice_id}, speed: {speed:.1f}x)")

        def worker():
            global _kokoro, _active_providers_str
            try:
                if need_reinit:
                    _kokoro = None
                    self._last_compute_choice = compute_choice

                kokoro = init_kokoro(compute_choice)

                def on_prog(p):
                    self.after(0, self.pbar_text_progress, p, "Synthesizing…")

                samples, sr = synthesize_chunked(kokoro, text, voice_id, speed, "en-us", on_progress=on_prog)

                banner = _active_providers_str or "(providers unknown)"
                self.after(0, self.set_provider_banner, banner)

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
            messagebox.showinfo("Empty text", "Please enter some text.")
            return

        voice_id = self._voice_id(); self._last_voice_vid = voice_id
        speed = max(0.5, min(2.0, round(float(self.speed_var.get()), 1)))
        compute_choice = self.compute_combo.get()
        need_reinit = (self._last_compute_choice != compute_choice)

        fpath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
            title="Save spoken audio as..."
        )
        if not fpath:
            return

        self.speak_btn.config(state="disabled"); self.save_btn.config(state="disabled")
        self.pbar_text_progress(1, f"Synthesizing… (voice: {voice_id}, speed: {speed:.1f}x)")

        def worker():
            global _kokoro, _active_providers_str
            try:
                if need_reinit:
                    _kokoro = None
                    self._last_compute_choice = compute_choice

                kokoro = init_kokoro(compute_choice)

                def on_prog(p):
                    self.after(0, self.pbar_text_progress, p, "Synthesizing…")

                samples, sr = synthesize_chunked(kokoro, text, voice_id, speed, "en-us", on_progress=on_prog)

                banner = _active_providers_str or "(providers unknown)"
                self.after(0, self.set_provider_banner, banner)

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

# ============================ Main ============================
if __name__ == "__main__":
    try:
        # Robust startup logging
        with open(BASE_DIR / "startuplog.txt", "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.datetime.now()}] launching\n")
        app = KokoroApp()
        app.mainloop()
    except Exception as exc:
        tb = traceback.format_exc()
        try:
            messagebox.showerror("Startup error", f"{exc}\n\n{tb}")
        except Exception:
            pass
        with open(BASE_DIR / "startuplog.txt", "a", encoding="utf-8") as f:
            f.write(tb + "\n")
        print("Startup error:", exc)
        sys.exit(1)
