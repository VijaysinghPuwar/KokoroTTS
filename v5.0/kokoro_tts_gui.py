# kokoro_tts_hybrid_cross.py — macOS & Windows GUI for Kokoro TTS
# v5.0 — faster voice scan (with cache), safer providers, fixed VOICE_CANDIDATES bug,
#        tighter threading, improved progress/status, minor perf cleanup.
# Platforms: macOS (Apple Silicon/CoreML, CPU) and Windows (CUDA/DirectML/CPU)

from __future__ import annotations

import json
import os
import sys
import re
import time
import threading
import tempfile
import traceback
import datetime
import subprocess
import platform
import textwrap
from pathlib import Path
from typing import Iterable, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np
import soundfile as sf
import onnxruntime as ort

# --- Playback backend (Windows/macOS) ---
try:
    import sounddevice as sd          # Preferred (works on macOS and Windows)
    PLAYBACK = "sd"
except Exception:
    try:
        import winsound               # Windows fallback
        PLAYBACK = "winsound"
    except Exception:
        sd = None
        winsound = None
        PLAYBACK = "none"

from kokoro_onnx import Kokoro  # type: ignore

# ============================ Config ============================
BASE_DIR = Path(__file__).resolve().parent
AUTO_DOWNLOAD = False  # True: auto download with huggingface_hub if missing
MODEL_NAMES = ["kokoro-v1.0.onnx", "kokoro-v1.0.fp16.onnx"]  # FP16 may help CoreML
VOICE_NAMES = ["voices-v1.0.bin", str(Path("voices") / "voices-v1.0.bin")]
PRESETS_FILE = BASE_DIR / "kokoro_presets.json"
VOICE_CACHE = BASE_DIR / ".voicecache.json"  # cache of validated voice IDs

ONLINE_VOICE_SOURCES = [
    "https://huggingface.co/hexgrad/Kokoro-82M/raw/main/VOICES.md",
    "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/raw/main/README.md",
]

# When offline or fetch fails, still show a few usable voices
FALLBACK_VOICES = [
    ("Bella — American English (F)", "af_bella"),
    ("Heart — American English (F)", "af_heart"),
    ("Michael — American English (M)", "am_michael"),
    ("Fenrir — American English (M)", "am_fenrir"),
]

LANGUAGE_OPTIONS: list[tuple[str, str]] = [
    ("English (US)", "en-us"),
    ("English (UK)", "en-gb"),
    ("English (Australia)", "en-au"),
    ("English (India)", "en-in"),
    ("Spanish", "es-es"),
    ("French", "fr-fr"),
    ("German", "de-de"),
    ("Italian", "it-it"),
    ("Portuguese (Brazil)", "pt-br"),
    ("Japanese", "ja-jp"),
    ("Korean", "ko-kr"),
    ("Chinese (Simplified)", "zh-cn"),
]

# ============================ Small utilities ============================
def _find_first(paths):
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    return None

MODEL_PATH  = _find_first([BASE_DIR / n for n in MODEL_NAMES])
VOICES_PATH = _find_first([BASE_DIR / n for n in VOICE_NAMES])

def _http_get(url: str, timeout: float = 6.0) -> str | None:
    """Tiny HTTP GET with graceful fallbacks (requests -> urllib). Returns text or None."""
    try:
        import requests  # type: ignore
        r = requests.get(url, timeout=timeout)
        if r.ok and r.text:
            return r.text
    except Exception:
        pass
    try:
        from urllib.request import urlopen
        with urlopen(url, timeout=timeout) as resp:  # type: ignore[attr-defined]
            data = resp.read()
            return data.decode("utf-8", errors="ignore")
    except Exception:
        return None

def _title_case_id(vid: str) -> str:
    base = vid.split("_", 1)[-1]
    return base.replace("-", " ").replace("_", " ").strip().title()

def _region_gender_from_prefix(vid: str) -> tuple[str, str] | None:
    if len(vid) < 2 or "_" not in vid:
        return None
    pref = vid.split("_", 1)[0]
    if len(pref) != 2:
        return None
    lang_key = pref[0]
    sex_key = pref[1]
    region_map = {
        "a": "American English",
        "b": "British English",
        "j": "Japanese",
        "z": "Mandarin Chinese",
        "e": "Spanish",
        "f": "French",
        "h": "Hindi",
        "i": "Italian",
        "p": "Brazilian Portuguese",
    }
    gender_map = {"f": "F", "m": "M"}
    return (region_map.get(lang_key, "Unknown"), gender_map.get(sex_key, "?"))

def _pretty_label_for_voice(vid: str) -> str:
    name = _title_case_id(vid)
    rg = _region_gender_from_prefix(vid) or ("Unknown", "?")
    return f"{name} — {rg[0]} ({rg[1]})"

def _parse_voice_ids_from_voices_md(md_text: str) -> list[str]:
    if not md_text:
        return []
    ids = set()
    for m in re.finditer(r"`([a-z]{1,2}[fm]_[a-z0-9\-]+)`", md_text):
        ids.add(m.group(1).strip())
    for m in re.finditer(r"\b([abjzefhip][fm]_[a-z0-9\-]+)\b", md_text):
        ids.add(m.group(1).strip())
    return sorted(ids)

def _parse_voice_ids_from_model_card(md_text: str) -> list[str]:
    if not md_text:
        return []
    ids = set()
    for m in re.finditer(r"\(`([a-z]{1,2}[fm]_[a-z0-9\-]+)`\)", md_text):
        ids.add(m.group(1).strip())
    return sorted(ids)

def fetch_online_voice_ids() -> list[str]:
    """Try each source in ONLINE_VOICE_SOURCES until one yields voice ids."""
    for url in ONLINE_VOICE_SOURCES:
        txt = _http_get(url)
        if not txt:
            continue
        if url.endswith("VOICES.md"):
            vids = _parse_voice_ids_from_voices_md(txt)
        else:
            vids = _parse_voice_ids_from_model_card(txt)
        if vids:
            return vids
    return []

def format_voices_for_ui(vids: Iterable[str]) -> list[Tuple[str, str]]:
    out: list[Tuple[str, str]] = [(_pretty_label_for_voice(vid), vid) for vid in vids]
    out.sort(key=lambda x: (x[0].split(" — ")[-1], x[0]))
    return out

def _read_voice_cache() -> list[str]:
    try:
        data = json.loads(VOICE_CACHE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("valid_voice_ids"), list):
            return [str(v) for v in data["valid_voice_ids"]]
    except Exception:
        pass
    return []

def _write_voice_cache(valid_ids: list[str]) -> None:
    try:
        VOICE_CACHE.write_text(json.dumps({"valid_voice_ids": valid_ids}, indent=2), encoding="utf-8")
    except Exception:
        pass

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
    try:
        return set(ort.get_available_providers())
    except Exception:
        return {"CPUExecutionProvider"}

def _is_apple_silicon() -> bool:
    try:
        return platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")
    except Exception:
        return False

def _is_windows() -> bool:
    return platform.system() == "Windows"

def resolve_providers(choice: str) -> list[str]:
    """
    Map UI choice -> providers list (ordered by preference).
    AUTO: CoreML (mac arm64) > CUDA (Windows/NVIDIA) > DirectML (Windows) > CPU.
    """
    avail = available_providers()
    if choice == "CPU":
        return ["CPUExecutionProvider"]
    if choice == "CoreML":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if choice == "CUDA":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if choice == "DirectML":
        return ["DmlExecutionProvider", "CPUExecutionProvider"]

    # AUTO
    if _is_apple_silicon() and "CoreMLExecutionProvider" in avail:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "DmlExecutionProvider" in avail:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def _session_providers_string(sess_like) -> str:
    for attr in ("session", "_session", "sess", "_sess"):
        s = getattr(sess_like, attr, None)
        if s is not None and hasattr(s, "get_providers"):
            try:
                return ", ".join(s.get_providers())
            except Exception:
                return "unknown"
    return "unknown"

def _assert_primary_provider(kokoro_obj, wanted_primary: str):
    """Raise if the first (primary) provider is not what we asked for (when available)."""
    for attr in ("session", "_session", "sess", "_sess"):
        s = getattr(kokoro_obj, attr, None)
        if s is not None and hasattr(s, "get_providers"):
            used = s.get_providers()
            if wanted_primary and wanted_primary in {
                "CoreMLExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"
            }:
                if used and used[0] != wanted_primary:
                    raise RuntimeError(f"Requested {wanted_primary} but active providers: {used}")
            return used
    return None

# ============================ Engine singleton ============================
_kokoro: Kokoro | None = None
_active_providers_str: str | None = None  # for UI
_engine_lock = threading.Lock()

def init_kokoro(providers_choice: str) -> Kokoro:
    """
    Create Kokoro with requested providers; assert primary if possible.
    Thread-safe & cached.
    """
    global _kokoro, _active_providers_str
    with _engine_lock:
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

        try:
            _kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH), providers=provs)  # type: ignore[arg-type]
        except TypeError:
            _kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))

        # Assert/record used providers
        try:
            used = _session_providers_string(_kokoro)
            _active_providers_str = used
            try:
                _assert_primary_provider(_kokoro, primary)
            except RuntimeError as e:
                # Still allow CPU to run; notify once
                try:
                    messagebox.showwarning("Requested accelerator not engaged", str(e))
                except Exception:
                    pass
        except Exception:
            _active_providers_str = "unknown"

        # Warm-up (non-fatal if it fails)
        try:
            _kokoro.create("Hi", voice="af_bella", speed=1.0, lang="en-us")
        except Exception:
            pass

        print("[Kokoro providers]", _active_providers_str)
        return _kokoro

# ============================ Voice validation (async) ============================
AVAILABLE_VOICES: list[tuple[str, str]] = []  # (label, voice_id) after validation

def validate_voices_async(compute_choice: str, on_done):
    """
    1) Fetch the *full* list of Kokoro voices online (VOICES.md/README) or use cache/fallback.
    2) Validate which ones are truly usable with the user's local voices pack
       by attempting a tiny synthesis call per-id.
    3) Cache valid IDs to speed up future launches.
    Calls on_done(valid_list) back on the Tk thread.
    """
    def worker():
        valid: list[tuple[str, str]] = []
        exc_first: Exception | None = None

        # Step A: gather candidates
        online_ids: list[str] = []
        try:
            online_ids = fetch_online_voice_ids()
        except Exception as e:
            exc_first = exc_first or e

        if not online_ids:
            # Use cache if any
            cached = _read_voice_cache()
            if cached:
                online_ids = cached

        if not online_ids:
            # Fallback to a tiny known-good subset
            online_ids = [vid for _lbl, vid in FALLBACK_VOICES]

        # Step B: init engine
        try:
            k = init_kokoro(compute_choice)
        except Exception as e:
            k = None
            exc_first = exc_first or e

        # Step C: probe each voice quickly
        valid_ids: list[str] = []
        if k is not None:
            for vid in online_ids:
                try:
                    k.create("Hi", voice=vid, speed=1.0, lang="en-us")
                    valid_ids.append(vid)
                except Exception:
                    continue

        # Step D: if nothing validated, fallback UI still shouldn’t be empty
        if not valid_ids:
            valid_ui = FALLBACK_VOICES
        else:
            _write_voice_cache(valid_ids)
            valid_ui = format_voices_for_ui(valid_ids)

        def finish():
            if exc_first and not valid_ui:
                messagebox.showerror("Voice scan failed", str(exc_first))
                on_done([])
            else:
                on_done(valid_ui)

        try:
            app.after(0, finish)  # type: ignore[name-defined]
        except Exception:
            finish()

    threading.Thread(target=worker, daemon=True).start()

# ============================ Playback (sounddevice or winsound) ============================
class SDPlayer:
    """Playback using sounddevice (preferred)."""
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    def stop(self):
        self._stop_evt.set()
        try:
            sd.stop(ignore_errors=True)
        except Exception:
            pass

    def play(self, samples: np.ndarray, sr: int, on_done=None):
        def worker():
            try:
                with self._lock:
                    self._stop_evt.clear()
                    x = samples.astype(np.float32, copy=False)
                    sd.play(x, sr, blocking=False)
                dur = float(len(samples)) / float(sr) if sr else 0.0
                step = 0.1; waited = 0.0
                while waited < dur and not self._stop_evt.wait(step):
                    waited += step
            except Exception as e:
                messagebox.showerror("Playback error", str(e))
            finally:
                try:
                    sd.stop(ignore_errors=True)
                except Exception:
                    pass
                if on_done: on_done()
        t = threading.Thread(target=worker, daemon=True); self._thread = t; t.start()

class WinSoundPlayer:
    """Playback using winsound (Windows fallback, no extra deps)."""
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
                dur = float(len(samples)) / float(sr) if sr else 0.0
                step = 0.1; waited = 0.0
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
                if on_done: on_done()
        threading.Thread(target=worker, daemon=True).start()

    def _cleanup_tmp(self):
        if not self._current_tmp: return
        p = Path(self._current_tmp)
        for _ in range(6):
            try:
                if p.exists(): p.unlink()
                self._current_tmp = None; return
            except PermissionError:
                time.sleep(0.12)

# choose backend
if PLAYBACK == "sd":
    _player = SDPlayer()
elif PLAYBACK == "winsound":
    _player = WinSoundPlayer()
else:
    class NoPlayer:
        def stop(self): pass
        def play(self, *a, **kw): messagebox.showerror("Playback", "No audio backend available.")
    _player = NoPlayer()

# ============================ Text chunking / synth ============================
def chunk_text(text: str, max_len: int = 480):
    """
    Split into sentence-like chunks up to ~max_len for fewer model calls.
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

# ============================ Audio helpers ============================
def apply_volume_boost(samples: np.ndarray, boost_db: float) -> np.ndarray:
    if samples.size == 0 or abs(boost_db) < 1e-3:
        return samples
    factor = float(10 ** (boost_db / 20.0))
    boosted = samples.astype(np.float32, copy=False) * factor
    np.clip(boosted, -1.0, 1.0, out=boosted)
    return boosted

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
def _get_metal_info() -> str | None:
    if platform.system() != "Darwin": return None
    try:
        out = subprocess.check_output(["/usr/sbin/system_profiler", "SPDisplaysDataType"], timeout=2)
        text = out.decode(errors="ignore")
        m = re.search(r"Chipset Model:\s*(.+)", text)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return None

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
    if _is_apple_silicon():
        lines.append("CPU: Apple Silicon (arm64)")
        metal = _get_metal_info()
        if metal:
            lines.append(f"Metal GPU: {metal}")
    if _is_windows():
        cuda_name = _get_gpu_name_cuda()
        if cuda_name:
            lines.append(f"CUDA GPU: {cuda_name}")
    lines.append(f"model: {MODEL_PATH or '(missing)'}")
    lines.append(f"voices: {VOICES_PATH or '(missing)'}")
    lines.append(f"presets file: {PRESETS_FILE}")
    return "\n".join(lines)

def _humanize_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    mins = int(seconds // 60)
    secs = int(round(seconds % 60))
    if mins:
        return f"{mins}:{secs:02d} min"
    return f"{secs}s"

# ============================ GUI ============================
class KokoroApp(tk.Tk):
    PADX = 14
    PADY = 12
    HISTORY_LIMIT = 80

    def __init__(self):
        super().__init__()
        title_suffix = "macOS (CPU/CoreML)" if platform.system() == "Darwin" else "Windows (CPU/CUDA/DirectML)"
        self.title(f"Kokoro TTS Hybrid — {title_suffix}")
        self.geometry("1040x720")
        self.minsize(900, 640)

        self._last_compute_choice: str | None = None
        self._last_voice_vid: str | None = None
        self._history_id = 0
        self.history_data: list[dict[str, object]] = []
        self.presets: list[dict[str, object]] = []

        self.speed_var = tk.DoubleVar(value=1.0)
        self.volume_var = tk.DoubleVar(value=0.0)
        self.chars_var = tk.StringVar(value="0 chars • 0 words")
        self.preset_var = tk.StringVar()

        # base grid
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        main = ttk.Frame(self)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(1, weight=1)

        # Header
        header = ttk.Frame(main)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=self.PADX, pady=(self.PADY, 6))
        header.columnconfigure(0, weight=1)
        friendly_font = ("SF Pro", 16, "bold") if platform.system() == "Darwin" else ("Segoe UI", 16, "bold")
        ttk.Label(header, text="Kokoro Text-to-Speech", font=friendly_font).grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Diagnostics…", command=self._show_diag).grid(row=0, column=1, sticky="e")

        # Text area + tools
        text_frame = ttk.Frame(main)
        text_frame.grid(row=1, column=0, sticky="nsew", padx=(self.PADX, self.PADX // 2), pady=(0, self.PADY))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(1, weight=1)

        text_header = ttk.Frame(text_frame)
        text_header.grid(row=0, column=0, sticky="ew")
        text_header.columnconfigure(0, weight=1)
        ttk.Label(text_header, text="Input text", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        text_tools = ttk.Frame(text_header)
        text_tools.grid(row=0, column=1, sticky="e")
        ttk.Button(text_tools, text="📂 Open…", command=self.on_import_text).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(text_tools, text="🧹 Clear", command=self.on_clear_text).grid(row=0, column=1)

        text_font = ("SF Pro Text", 12) if platform.system() == "Darwin" else ("Segoe UI", 11)
        self.text_box = tk.Text(text_frame, wrap="word", font=text_font, undo=True)
        self.text_box.grid(row=1, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_box.yview)
        vscroll.grid(row=1, column=1, sticky="ns")
        self.text_box.configure(yscrollcommand=vscroll.set)
        self.text_box.focus_set()

        # Context menu
        self._ctx = tk.Menu(self, tearoff=0)
        for label, ev in [("Cut", "<<Cut>>"), ("Copy", "<<Copy>>"), ("Paste", "<<Paste>>"), (None, None), ("Select All", "<<SelectAll>>")]:
            if label is None:
                self._ctx.add_separator()
            else:
                self._ctx.add_command(label=label, command=lambda e=ev: self.text_box.event_generate(e))
        self.text_box.bind("<Button-3>", lambda e: self._ctx.tk_popup(e.x_root, e.y_root))
        self.text_box.bind("<Control-Button-1>", lambda e: self._ctx.tk_popup(e.x_root, e.y_root))
        self.text_box.bind("<Button-2>", lambda e: self._ctx.tk_popup(e.x_root, e.y_root))

        ttk.Label(text_frame, textvariable=self.chars_var, foreground="#666").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        self.text_box.bind("<<Modified>>", self._on_text_modified)

        # Side controls
        side = ttk.Frame(main)
        side.grid(row=1, column=1, sticky="nsew", padx=(self.PADX // 2, self.PADX), pady=(0, self.PADY))
        side.columnconfigure(0, weight=1)
        side.rowconfigure(4, weight=1)

        voice_group = ttk.Labelframe(side, text="Voice & Language")
        voice_group.grid(row=0, column=0, sticky="ew", pady=(0, self.PADY // 2))
        for col in (1,):
            voice_group.columnconfigure(col, weight=1)

        ttk.Label(voice_group, text="Voice:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(4, 2))
        self.voice_combo = ttk.Combobox(voice_group, state="readonly", values=["(scanning…)"])
        self.voice_combo.current(0)
        self.voice_combo.grid(row=0, column=1, sticky="ew", pady=(4, 2))

        lang_values = [f"{label} — {code}" for (label, code) in LANGUAGE_OPTIONS]
        ttk.Label(voice_group, text="Language:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(2, 6))
        self.lang_combo = ttk.Combobox(voice_group, state="readonly", values=lang_values)
        if lang_values:
            self.lang_combo.current(0)
        self.lang_combo.grid(row=1, column=1, sticky="ew", pady=(2, 6))

        perf_group = ttk.Labelframe(side, text="Delivery & Compute")
        perf_group.grid(row=1, column=0, sticky="ew", pady=(0, self.PADY // 2))
        perf_group.columnconfigure(1, weight=1)

        ttk.Label(perf_group, text="Speed:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(4, 2))
        self.speed_scale = ttk.Scale(perf_group, from_=0.5, to=2.0, variable=self.speed_var, command=self._on_speed_changed)
        self.speed_scale.grid(row=0, column=1, sticky="ew", pady=(4, 2))
        self.speed_value_lbl = ttk.Label(perf_group, text="1.0×")
        self.speed_value_lbl.grid(row=0, column=2, padx=(8, 0))

        ttk.Label(perf_group, text="Loudness:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(2, 2))
        self.volume_scale = ttk.Scale(perf_group, from_=-6.0, to=6.0, variable=self.volume_var, command=self._on_volume_changed)
        self.volume_scale.grid(row=1, column=1, sticky="ew", pady=(2, 2))
        self.volume_value_lbl = ttk.Label(perf_group, text="0.0 dB")
        self.volume_value_lbl.grid(row=1, column=2, padx=(8, 0))

        ttk.Label(perf_group, text="Compute:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(6, 4))
        self.compute_combo = ttk.Combobox(perf_group, state="readonly", values=["AUTO", "CPU"])
        self.compute_combo.current(0)
        self.compute_combo.grid(row=2, column=1, sticky="ew", pady=(6, 4))
        self._prune_unavailable_providers()
        self.rescan_btn = ttk.Button(perf_group, text="🔄 Rescan voices", command=self._rescan_voices)
        self.rescan_btn.grid(row=2, column=2, padx=(8, 0), pady=(6, 4))

        actions = ttk.Frame(side)
        actions.grid(row=2, column=0, sticky="ew", pady=(0, self.PADY // 2))
        actions.columnconfigure(0, weight=1)

        self.speak_btn = ttk.Button(actions, text="▶ Speak", command=self.on_speak)
        self.speak_btn.grid(row=0, column=0, sticky="ew")
        ttk.Button(actions, text="■ Stop", command=self.on_stop).grid(row=0, column=1, padx=(8, 0))
        self.save_btn = ttk.Button(actions, text="💾 Save WAV…", command=self.on_save)
        self.save_btn.grid(row=0, column=2, padx=(8, 0))

        status_row = ttk.Frame(side)
        status_row.grid(row=3, column=0, sticky="ew", pady=(0, self.PADY // 2))
        status_row.columnconfigure(0, weight=1)
        status_row.columnconfigure(1, weight=1)
        self.status = tk.StringVar(value="Ready")
        ttk.Label(status_row, textvariable=self.status, anchor="w").grid(row=0, column=0, sticky="w")
        self.pbar = ttk.Progressbar(status_row, orient="horizontal", mode="determinate", maximum=100)
        self.pbar.grid(row=0, column=1, sticky="ew", padx=(12, 0))
        self._pbar_visible = True
        self._hide_pbar()

        self.provider_lbl = ttk.Label(side, text="Active providers: —", anchor="w", foreground="#3a3a3a")
        self.provider_lbl.grid(row=4, column=0, sticky="ew", pady=(0, self.PADY // 2))

        self.notebook = ttk.Notebook(side)
        self.notebook.grid(row=5, column=0, sticky="nsew")
        side.rowconfigure(5, weight=1)

        self._build_history_tab()
        self._build_presets_tab()

        # Bindings & theme
        self._bind_shortcuts()
        try:
            style = ttk.Style(self)
            if platform.system() == "Darwin" and "aqua" in style.theme_names():
                style.theme_use("aqua")
            elif "vista" in style.theme_names():
                style.theme_use("vista")
        except Exception:
            pass

        self._update_speed_label()
        self._update_volume_label()
        self._update_char_counter()
        self.lang_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_char_counter())

        self._load_presets()
        self.after(200, self._rescan_voices)

    # -------- UI builders --------
    def _build_history_tab(self):
        history_tab = ttk.Frame(self.notebook)
        history_tab.columnconfigure(0, weight=1)
        history_tab.rowconfigure(0, weight=1)
        self.notebook.add(history_tab, text="History")

        columns = ("time", "voice", "speed", "lang", "gain", "action", "preview")
        self.history_tree = ttk.Treeview(history_tab, columns=columns, show="headings", selectmode="browse")
        headings = {
            "time": "Time",
            "voice": "Voice",
            "speed": "Speed",
            "lang": "Lang",
            "gain": "Gain",
            "action": "Action",
            "preview": "Preview",
        }
        widths = {"time": 80, "voice": 140, "speed": 70, "lang": 80, "gain": 70, "action": 110, "preview": 260}
        for col in columns:
            self.history_tree.heading(col, text=headings[col])
            self.history_tree.column(col, width=widths[col], anchor="w")

        yscroll = ttk.Scrollbar(history_tab, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=yscroll.set)
        self.history_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        self.history_tree.bind("<Double-1>", self._on_history_double_click)

        history_btns = ttk.Frame(history_tab)
        history_btns.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        history_btns.columnconfigure(0, weight=1)
        history_btns.columnconfigure(1, weight=1)
        history_btns.columnconfigure(2, weight=1)

        ttk.Button(history_btns, text="Load text", command=self._on_history_load_text).grid(row=0, column=0, sticky="ew")
        ttk.Button(history_btns, text="Apply settings", command=self._on_history_apply_settings).grid(row=0, column=1, sticky="ew", padx=(6, 6))
        ttk.Button(history_btns, text="Speak again", command=self._on_history_speak).grid(row=0, column=2, sticky="ew")

        ttk.Label(history_btns, text="Double-click an entry to load its text.", foreground="#666").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )

    def _build_presets_tab(self):
        presets_tab = ttk.Frame(self.notebook)
        presets_tab.columnconfigure(0, weight=1)
        self.notebook.add(presets_tab, text="Presets")

        ttk.Label(
            presets_tab,
            text="Presets remember voice, language, speed, loudness and compute choices.",
            wraplength=320,
            foreground="#444",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.preset_combo = ttk.Combobox(presets_tab, textvariable=self.preset_var, state="readonly", values=[])
        self.preset_combo.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        preset_btns = ttk.Frame(presets_tab)
        preset_btns.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        preset_btns.columnconfigure(0, weight=1)
        preset_btns.columnconfigure(1, weight=1)
        preset_btns.columnconfigure(2, weight=1)

        ttk.Button(preset_btns, text="Apply", command=self._apply_preset).grid(row=0, column=0, sticky="ew")
        ttk.Button(preset_btns, text="Save new…", command=self._save_preset_as_new).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(preset_btns, text="Update", command=self._update_selected_preset).grid(row=0, column=2, sticky="ew")

        ttk.Button(presets_tab, text="Delete", command=self._delete_selected_preset).grid(row=3, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(presets_tab, text=f"Stored at: {PRESETS_FILE}", foreground="#666", wraplength=320).grid(row=4, column=0, sticky="w")

    # -------- Helpers --------
    def _bind_shortcuts(self):
        if platform.system() == "Darwin":
            self.bind_all("<Command-Return>", lambda e: self.on_speak())
            self.bind_all("<Command-s>", lambda e: self.on_save())
            self.bind_all("<Command-o>", lambda e: self.on_import_text())
        else:
            self.bind_all("<Control-Return>", lambda e: self.on_speak())
            self.bind_all("<Control-s>", lambda e: self.on_save())
            self.bind_all("<Control-o>", lambda e: self.on_import_text())
        self.bind_all("<Escape>", lambda e: self.on_stop())

    def _on_speed_changed(self, _value=None):
        self._update_speed_label()
        self._update_char_counter()

    def _on_volume_changed(self, _value=None):
        self._update_volume_label()

    def _update_speed_label(self):
        try:
            speed = float(self.speed_var.get())
        except Exception:
            speed = 1.0
        self.speed_value_lbl.configure(text=f"{speed:.1f}×")

    def _update_volume_label(self):
        try:
            gain = float(self.volume_var.get())
        except Exception:
            gain = 0.0
        self.volume_value_lbl.configure(text=f"{gain:+.1f} dB")

    def _load_presets(self):
        if PRESETS_FILE.exists():
            try:
                data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
                presets = data.get("presets", [])
                if isinstance(presets, list):
                    self.presets = [p for p in presets if isinstance(p, dict) and "name" in p]
            except Exception as exc:
                messagebox.showwarning("Preset load failed", f"Could not read presets:\n{exc}")
                self.presets = []
        else:
            self.presets = []
        self._refresh_preset_combo()

    def _save_presets(self):
        try:
            PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {"presets": self.presets}
            PRESETS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            messagebox.showwarning("Preset save failed", f"Could not save presets:\n{exc}")

    def _refresh_preset_combo(self):
        names = [str(p.get("name", "")) for p in self.presets]
        self.preset_combo.configure(values=names)
        current = self.preset_var.get()
        if current and current in names:
            self.preset_combo.set(current)
        elif names:
            self.preset_var.set(names[0])
            self.preset_combo.current(0)
        else:
            self.preset_var.set("")
            self.preset_combo.set("")

    def _current_settings_as_preset(self, name: str) -> dict[str, object]:
        return {
            "name": name,
            "voice": self._voice_id(),
            "language": self._language_code(),
            "speed": float(self.speed_var.get()),
            "volume_db": float(self.volume_var.get()),
            "compute": self.compute_combo.get(),
        }

    def _selected_preset(self):
        name = self.preset_var.get().strip()
        for preset in self.presets:
            if str(preset.get("name", "")) == name:
                return preset
        return None

    def _apply_preset(self):
        preset = self._selected_preset()
        if not preset:
            messagebox.showinfo("Presets", "Select a preset to apply.")
            return
        self._set_voice_by_id(str(preset.get("voice", "")))
        self._set_language_by_code(str(preset.get("language", "")))
        if "speed" in preset: self.speed_var.set(float(preset.get("speed", 1.0)))
        if "volume_db" in preset: self.volume_var.set(float(preset.get("volume_db", 0.0)))
        compute = preset.get("compute")
        if compute: self.compute_combo.set(str(compute))
        self._update_speed_label(); self._update_volume_label(); self._update_char_counter()
        self.status.set(f"Applied preset: {preset.get('name')}")

    def _save_preset_as_new(self):
        name = simpledialog.askstring("Preset name", "Save current settings as:", parent=self)
        if not name: return
        name = name.strip()
        if not name: return
        existing = next((p for p in self.presets if p.get("name") == name), None)
        preset = self._current_settings_as_preset(name)
        if existing:
            if not messagebox.askyesno("Overwrite preset", f"Preset '{name}' exists. Overwrite it?"):
                return
            existing.clear(); existing.update(preset)
        else:
            self.presets.append(preset)
        self.preset_var.set(name)
        self._refresh_preset_combo()
        self._save_presets()
        self.status.set(f"Saved preset: {name}")

    def _update_selected_preset(self):
        preset = self._selected_preset()
        if not preset:
            messagebox.showinfo("Presets", "Select a preset to update.")
            return
        preset.clear(); preset.update(self._current_settings_as_preset(self.preset_var.get()))
        self._save_presets()
        self.status.set(f"Updated preset: {self.preset_var.get()}")

    def _delete_selected_preset(self):
        preset = self._selected_preset()
        if not preset:
            messagebox.showinfo("Presets", "Select a preset to delete.")
            return
        name = str(preset.get("name"))
        if not messagebox.askyesno("Delete preset", f"Remove preset '{name}'?"):
            return
        self.presets = [p for p in self.presets if p is not preset]
        self._refresh_preset_combo()
        self._save_presets()
        self.status.set(f"Deleted preset: {name}")

    def _history_selection(self):
        sel = self.history_tree.selection()
        if not sel:
            return None
        iid = sel[0]
        for entry in self.history_data:
            if entry.get("iid") == iid:
                return entry
        return None

    def _voice_label_for_id(self, voice_id: str) -> str:
        source = AVAILABLE_VOICES if AVAILABLE_VOICES else FALLBACK_VOICES
        for label, vid in source:
            if vid == voice_id:
                return f"{label} ({vid})"
        return voice_id

    def _add_history_entry(
        self,
        action: str,
        text: str,
        voice: str,
        speed: float,
        lang: str,
        boost_db: float,
        compute: str,
        filepath: str | None = None,
    ):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        preview = textwrap.shorten(" ".join(text.split()), width=120, placeholder="…") if text else "(empty)"
        iid = f"h{self._history_id}"; self._history_id += 1
        entry = {
            "iid": iid, "time": timestamp, "text": text, "voice": voice, "language": lang,
            "speed": speed, "volume_db": boost_db, "action": action, "filepath": filepath, "compute": compute,
        }
        self.history_data.insert(0, entry)
        action_desc = f"{action} ({Path(filepath).name})" if filepath else action
        self.history_tree.insert(
            "", 0, iid=iid,
            values=(timestamp, self._voice_label_for_id(voice), f"{speed:.1f}×", lang, f"{boost_db:+.1f} dB", action_desc, preview),
        )
        if len(self.history_data) > self.HISTORY_LIMIT:
            old = self.history_data.pop()
            try: self.history_tree.delete(old.get("iid"))
            except Exception: pass

    def _set_text(self, text: str):
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", text)
        self.text_box.edit_modified(False)
        self._update_char_counter()

    def _set_voice_by_id(self, voice_id: str) -> bool:
        source = AVAILABLE_VOICES if AVAILABLE_VOICES else FALLBACK_VOICES
        data = [f"{label}  ({vid})" for (label, vid) in source]
        self.voice_combo.configure(values=data)
        for idx, (_, vid) in enumerate(source):
            if vid == voice_id:
                self.voice_combo.current(idx)
                return True
        return False

    def _set_language_by_code(self, lang_code: str) -> bool:
        for idx, (_, code) in enumerate(LANGUAGE_OPTIONS):
            if code == lang_code:
                self.lang_combo.current(idx)
                return True
        return False

    def _on_history_load_text(self):
        entry = self._history_selection()
        if not entry:
            messagebox.showinfo("History", "Select a history entry first.")
            return
        self._set_text(str(entry.get("text", "")))
        self.status.set("Loaded text from history")

    def _apply_history_settings(self, entry):
        self._set_voice_by_id(str(entry.get("voice", "")))
        self._set_language_by_code(str(entry.get("language", "")))
        if "speed" in entry: self.speed_var.set(float(entry.get("speed", 1.0)))
        if "volume_db" in entry: self.volume_var.set(float(entry.get("volume_db", 0.0)))
        if entry.get("compute"): self.compute_combo.set(str(entry.get("compute")))
        self._update_speed_label(); self._update_volume_label(); self._update_char_counter()

    def _on_history_apply_settings(self):
        entry = self._history_selection()
        if not entry:
            messagebox.showinfo("History", "Select a history entry first.")
            return
        self._apply_history_settings(entry)
        self.status.set("Restored voice and options from history")

    def _on_history_speak(self):
        entry = self._history_selection()
        if not entry:
            messagebox.showinfo("History", "Select a history entry first.")
            return
        self._set_text(str(entry.get("text", "")))
        self._apply_history_settings(entry)
        self.on_speak()

    def _on_history_double_click(self, _event):
        self._on_history_load_text()

    def on_import_text(self):
        path = filedialog.askopenfilename(
            title="Open text file…",
            filetypes=[("Text", "*.txt"), ("Markdown", "*.md"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            content = Path(path).read_text(encoding="utf-8")
        except Exception:
            content = Path(path).read_text(encoding="utf-8", errors="ignore")
        self._set_text(content)
        self.status.set(f"Loaded text from {Path(path).name}")

    def on_clear_text(self):
        self._set_text("")
        self.status.set("Cleared text input")

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
        src = AVAILABLE_VOICES if AVAILABLE_VOICES else FALLBACK_VOICES
        if not src:
            raise RuntimeError("No voices available")
        idx = self.voice_combo.current()
        return src[idx][1] if 0 <= idx < len(src) else src[0][1]

    def _language_code(self) -> str:
        idx = self.lang_combo.current()
        if 0 <= idx < len(LANGUAGE_OPTIONS):
            return LANGUAGE_OPTIONS[idx][1]
        return LANGUAGE_OPTIONS[0][1] if LANGUAGE_OPTIONS else "en-us"

    def _volume_boost_db(self) -> float:
        try:
            return float(self.volume_var.get())
        except Exception:
            return 0.0

    def _prune_unavailable_providers(self):
        try:
            avail = available_providers()
        except Exception:
            avail = {"CPUExecutionProvider"}
        keep = ["AUTO", "CPU"]
        if "CoreMLExecutionProvider" in avail and _is_apple_silicon(): keep.append("CoreML")
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
        n = len(txt)
        words = len(txt.split()) if txt.strip() else 0
        try:
            speed = float(self.speed_var.get()) or 1.0
        except Exception:
            speed = 1.0
        speed = max(0.2, min(4.0, speed))
        est_secs = 0.0
        if words:
            base_wpm = 165.0
            est_secs = (words / base_wpm) * 60.0
            est_secs /= speed
        parts = [f"{n} chars", f"{words} words"]
        if est_secs:
            parts.append(f"≈{_humanize_duration(est_secs)}")
        self.chars_var.set(" • ".join(parts))

    def set_provider_banner(self, text: str):
        self.provider_lbl.config(text=f"Active providers: {text}")

    def _populate_voices(self, voices_list: list[tuple[str,str]]):
        source = voices_list if voices_list else FALLBACK_VOICES
        if not source:
            self.voice_combo.configure(values=["(no voices)"])
            self.voice_combo.set("(no voices)")
            return
        data = [f"{label}  ({vid})" for (label, vid) in source]
        self.voice_combo.configure(values=data)
        if self._last_voice_vid:
            for idx, (_, vid) in enumerate(source):
                if vid == self._last_voice_vid:
                    self.voice_combo.current(idx)
                    break
            else:
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
            n = len(AVAILABLE_VOICES) if AVAILABLE_VOICES else len(FALLBACK_VOICES)
            self.status.set(f"Ready — {n} voice(s) available")
            self.rescan_btn.config(state="normal")

        validate_voices_async(compute_choice, on_done)

    def _show_diag(self):
        diag = tk.Toplevel(self); diag.title("Diagnostics"); diag.geometry("560x380")
        diag.resizable(True, True)
        mono = ("Menlo", 11) if platform.system()=="Darwin" else ("Consolas", 10)
        txt = tk.Text(diag, wrap="word", font=mono)
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
        speed = max(0.5, min(2.0, float(self.speed_var.get())))
        lang = self._language_code()
        boost_db = self._volume_boost_db()
        compute_choice = self.compute_combo.get()
        need_reinit = (self._last_compute_choice != compute_choice)

        self.speak_btn.config(state="disabled"); self.save_btn.config(state="disabled")
        summary = f"Synthesizing… ({voice_id}, {lang}, {speed:.1f}×)"
        self.pbar_text_progress(1, summary)

        def worker():
            global _kokoro, _active_providers_str
            try:
                if need_reinit:
                    with _engine_lock:
                        _kokoro = None
                    self._last_compute_choice = compute_choice

                kokoro = init_kokoro(compute_choice)

                def on_prog(p):
                    self.after(0, self.pbar_text_progress, p, summary)

                samples, sr = synthesize_chunked(kokoro, text, voice_id, speed, lang, on_progress=on_prog)
                samples = apply_volume_boost(samples, boost_db)

                banner = _active_providers_str or "(providers unknown)"
                self.after(0, self.set_provider_banner, banner)

                self.after(0, self.pbar_text_progress, 100, summary)
                self.after(0, self.status.set, "Playing…")
                self.after(0, lambda: self._add_history_entry("Play", text, voice_id, speed, lang, boost_db, compute_choice))
                _player.play(samples, sr, on_done=lambda: self.after(0, self._on_playback_finished))
            except Exception as e:
                self.after(0, self.pbar_done, "Error")
                self.after(0, lambda: messagebox.showerror("Error", f"Speech generation failed:\n{e}"))
                self.after(0, lambda: self.speak_btn.config(state="normal"))
                self.after(0, lambda: self.save_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_playback_finished(self):
        self.pbar_done("Playback finished")
        self.speak_btn.config(state="normal"); self.save_btn.config(state="normal")

    def on_save(self):
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Empty text", "Please enter some text.")
            return

        voice_id = self._voice_id(); self._last_voice_vid = voice_id
        speed = max(0.5, min(2.0, float(self.speed_var.get())))
        lang = self._language_code()
        boost_db = self._volume_boost_db()
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
        summary = f"Synthesizing… ({voice_id}, {lang}, {speed:.1f}×)"
        self.pbar_text_progress(1, summary)

        def worker():
            global _kokoro, _active_providers_str
            try:
                if need_reinit:
                    with _engine_lock:
                        _kokoro = None
                    self._last_compute_choice = compute_choice

                kokoro = init_kokoro(compute_choice)

                def on_prog(p):
                    self.after(0, self.pbar_text_progress, p, summary)

                samples, sr = synthesize_chunked(kokoro, text, voice_id, speed, lang, on_progress=on_prog)
                samples = apply_volume_boost(samples, boost_db)

                banner = _active_providers_str or "(providers unknown)"
                self.after(0, self.set_provider_banner, banner)

                def set_pct(p_save):
                    self.after(0, self.pbar_text_progress, p_save, "Saving WAV…")

                save_wav_chunked(samples, sr, fpath, progress_cb=set_pct, chunk_frames=48000)

                self.after(0, self.pbar_done, f"Saved: {fpath}")
                self.after(0, lambda: messagebox.showinfo("Saved", f"Audio saved to:\n{fpath}"))
                self.after(0, lambda: self._add_history_entry("Save", text, voice_id, speed, lang, boost_db, compute_choice, fpath))
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
        with open(BASE_DIR / "startuplog.txt", "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.datetime.now()}] launching ({platform.system()})\n")
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
