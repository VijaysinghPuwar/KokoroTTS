import os
import sys
import threading
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Audio + TTS deps
import numpy as np
import soundfile as sf
import simpleaudio as sa

# Kokoro ONNX
# API: Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin"); .create(text, voice="...", speed=1.0, lang="en-us")
# Reference usage shown in community docs/articles. 
from kokoro_onnx import Kokoro  # type: ignore

# Optional: auto-download model/voices on first run (can be disabled by setting AUTO_DOWNLOAD=False)
AUTO_DOWNLOAD = True

# Where to store the downloaded weights (script directory by default)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "kokoro-v1.0.onnx"
VOICES_PATH = BASE_DIR / "voices-v1.0.bin"

# Voices (2 female, 2 male) from Kokoro VOICES.md
VOICES = [
    ("Female — Bella (US)", "af_bella"),
    ("Female — Heart (US)", "af_heart"),
    ("Male — Michael (US)", "am_michael"),
    ("Male — Fenrir (US)", "am_fenrir"),
]

# Globals for player state
_current_play_obj = None
_current_temp_wav = None
_kokoro = None


def download_weights_if_needed():
    """
    Downloads kokoro-v1.0.onnx and voices-v1.0.bin if they don't exist,
    using huggingface_hub. We grab the ONNX model from the ONNX Community repo
    and the voices binary from the kokoro-onnx release mirror.
    """
    if not AUTO_DOWNLOAD:
        return

    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return  # huggingface_hub not installed; user will place files manually.

    # ONNX model (community hub hosts ONNX export)
    if not MODEL_PATH.exists():
        try:
            local = hf_hub_download(
                repo_id="onnx-community/Kokoro-82M-v1.0-ONNX",
                filename="kokoro-v1.0.onnx",
                local_dir=str(BASE_DIR)
            )
            Path(local).rename(MODEL_PATH)
        except Exception as e:
            print(f"[WARN] Could not auto-download ONNX: {e}")

    # Voices binary (several mirrors exist; try community first)
    if not VOICES_PATH.exists():
        candidates = [
            # Common hosting locations over 2025—try multiple names/paths if needed.
            ("onnx-community/Kokoro-82M-v1.0-ONNX", "voices-v1.0.bin"),
            ("hexgrad/Kokoro-82M", "voices/voices-v1.0.bin"),  # fallback (path may change)
        ]
        for repo, filename in candidates:
            try:
                local = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(BASE_DIR))
                Path(local).rename(VOICES_PATH)
                break
            except Exception:
                continue


def init_kokoro():
    global _kokoro
    if _kokoro is not None:
        return _kokoro

    # attempt auto-download if missing
    download_weights_if_needed()

    if not MODEL_PATH.exists() or not VOICES_PATH.exists():
        raise FileNotFoundError(
            "Missing model files.\n"
            "Please place 'kokoro-v1.0.onnx' and 'voices-v1.0.bin' next to this script, "
            "or enable AUTO_DOWNLOAD and install huggingface_hub."
        )
    _kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
    return _kokoro


def stop_audio():
    global _current_play_obj
    if _current_play_obj and _current_play_obj.is_playing():
        _current_play_obj.stop()


def speak_async(text, voice_id, speed, on_done=None):
    """
    Generate audio with Kokoro and play it, in a worker thread to keep the UI responsive.
    """
    def worker():
        global _current_play_obj, _current_temp_wav
        try:
            kokoro = init_kokoro()

            # Kokoro accepts lang codes; en-us is default for these voices.
            # Speed in kokoro-onnx is generally ~0.5 to 2.0; round to one decimal to avoid edge behavior.
            speed = float(np.clip(round(speed, 1), 0.5, 2.0))

            samples, sr = kokoro.create(
                text,
                voice=voice_id,
                speed=speed,
                lang="en-us",
            )

            # Write to a temp WAV and play using simpleaudio
            # Use 24kHz (Kokoro default sample rate)
            if _current_temp_wav and Path(_current_temp_wav).exists():
                try:
                    Path(_current_temp_wav).unlink()
                except Exception:
                    pass

            fd, tmp = tempfile.mkstemp(prefix="kokoro_", suffix=".wav")
            os.close(fd)
            _current_temp_wav = tmp
            sf.write(tmp, samples, sr)

            wave_obj = sa.WaveObject.from_wave_file(tmp)
            _current_play_obj = wave_obj.play()
            _current_play_obj.wait_done()

        except Exception as e:
            messagebox.showerror("Error", f"Speech generation failed:\n{e}")
        finally:
            if on_done:
                on_done()

    t = threading.Thread(target=worker, daemon=True)
    t.start()


def save_wav(text, voice_id, speed):
    """
    Generate audio and prompt user to save as WAV.
    """
    try:
        kokoro = init_kokoro()
        speed = float(np.clip(round(speed, 1), 0.5, 2.0))
        samples, sr = kokoro.create(text, voice=voice_id, speed=speed, lang="en-us")

        fpath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
            title="Save spoken audio as..."
        )
        if fpath:
            sf.write(fpath, samples, sr)
            messagebox.showinfo("Saved", f"Audio saved to:\n{fpath}")
    except Exception as e:
        messagebox.showerror("Error", f"Save failed:\n{e}")


class KokoroApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kokoro TTS — Tkinter")
        self.geometry("720x540")
        self.minsize(640, 480)

        # Main layout
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Label(self, text="Kokoro Text-to-Speech", font=("Segoe UI", 16, "bold"))
        header.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        # Text input
        self.text_box = tk.Text(self, wrap="word", font=("Segoe UI", 11))
        self.text_box.grid(row=1, column=0, sticky="nsew", padx=12)
        self.text_box.insert(
            "1.0",
            "Type or paste text here and click Speak…"
        )

        # Controls frame
        controls = ttk.Frame(self)
        controls.grid(row=2, column=0, sticky="ew", padx=12, pady=12)
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Voice:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.voice_var = tk.StringVar(value=VOICES[0][1])
        self.voice_combo = ttk.Combobox(
            controls,
            state="readonly",
            values=[f"{label}  ({vid})" for label, vid in VOICES]
        )
        self.voice_combo.current(0)
        self.voice_combo.grid(row=0, column=1, sticky="ew")

        # map back to id
        def get_selected_voice_id():
            idx = self.voice_combo.current()
            return VOICES[idx][1] if idx >= 0 else VOICES[0][1]
        self.get_selected_voice_id = get_selected_voice_id

        # Speed
        ttk.Label(controls, text="Speed:").grid(row=0, column=2, sticky="e", padx=(16, 8))
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_scale = ttk.Scale(controls, from_=0.5, to=2.0, variable=self.speed_var)
        self.speed_scale.grid(row=0, column=3, sticky="ew")
        controls.columnconfigure(3, weight=1)

        # Buttons
        btns = ttk.Frame(self)
        btns.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=0)
        btns.columnconfigure(2, weight=0)

        self.speak_btn = ttk.Button(btns, text="▶ Speak", command=self.on_speak)
        self.speak_btn.grid(row=0, column=0, sticky="w")

        self.stop_btn = ttk.Button(btns, text="■ Stop", command=stop_audio)
        self.stop_btn.grid(row=0, column=1, padx=(8, 0))

        self.save_btn = ttk.Button(btns, text="💾 Save WAV…", command=self.on_save)
        self.save_btn.grid(row=0, column=2, padx=(8, 0))

        # Status bar
        self.status = tk.StringVar(value="Ready")
        statusbar = ttk.Label(self, textvariable=self.status, anchor="w")
        statusbar.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 8))

    def on_speak(self):
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Empty text", "Please enter some text.")
            return

        voice_id = self.get_selected_voice_id()
        speed = self.speed_var.get()

        self.status.set(f"Generating speech… (voice: {voice_id}, speed: {speed:.1f}x)")
        self.speak_btn.config(state="disabled")

        def done():
            self.status.set("Done")
            self.speak_btn.config(state="normal")

        speak_async(text, voice_id, speed, on_done=done)

    def on_save(self):
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Empty text", "Please enter some text.")
            return
        voice_id = self.get_selected_voice_id()
        speed = self.speed_var.get()
        save_wav(text, voice_id, speed)


if __name__ == "__main__":
    try:
        app = KokoroApp()
        app.mainloop()
    except Exception as exc:
        messagebox.showerror("Startup error", str(exc))
        sys.exit(1)
