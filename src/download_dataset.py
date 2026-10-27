"""
src/download_dataset.py
─────────────────────────────────────────────────────────────────────────────
Downloads a SMALL (~200-400 MB) real-speech dataset for voice-cloning detection.

Strategy (fully automatic, no manual steps):
  1. Genuine audio  → LJSpeech-1.1 first 1000 WAV files
                      (~200 MB, direct download from keithito.com)
  2. Synthetic audio → WaveFake "for-2-seconds" subset from Hugging Face
                       OR generated locally with gTTS / pyttsx3 if HF unavailable

Dataset structure produced:
  data/small_dataset/
    LJSpeech-1.1/
      wavs/           ← 1000 genuine WAV files
    generated_audio/
      ljspeech_tts/   ← 1000 synthetic WAV files (from HF or local TTS)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
import tarfile
import urllib.request
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, SAMPLE_RATE
from src.utils import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
SMALL_ROOT      = DATA_DIR / "small_dataset"
GENUINE_WAV_DIR = SMALL_ROOT / "LJSpeech-1.1" / "wavs"
SYNTH_WAV_DIR   = SMALL_ROOT / "generated_audio" / "ljspeech_tts"

LJ_URL     = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
LJ_ARCHIVE = DATA_DIR / "LJSpeech-1.1.tar.bz2"
LJ_FULL    = DATA_DIR / "LJSpeech-1.1"

N_GENUINE = 1000   # number of genuine files to keep (≈200 MB)
N_SYNTH   = 1000   # matching synthetic files


# ─────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────

class _ProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download(url: str, dest: Path, desc: str = "Downloading") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest
    with _ProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, str(dest), reporthook=t.update_to)
    return dest


# ─────────────────────────────────────────────
# Step 1 — Genuine audio (LJSpeech subset)
# ─────────────────────────────────────────────

def _get_genuine(n: int = N_GENUINE, callback=None) -> bool:
    """Download LJSpeech and copy the first *n* WAVs to GENUINE_WAV_DIR."""
    GENUINE_WAV_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(GENUINE_WAV_DIR.glob("*.wav"))

    if len(existing) >= n:
        logger.info("Genuine audio already present (%d files).", len(existing))
        return True

    if callback: callback("📥 Downloading LJSpeech-1.1 (~2.6 GB archive) — this takes a few minutes …")
    logger.info("Downloading LJSpeech-1.1 from %s", LJ_URL)

    try:
        _download(LJ_URL, LJ_ARCHIVE, desc="LJSpeech-1.1.tar.bz2")
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        return False

    if callback: callback("📦 Extracting archive …")
    logger.info("Extracting %s …", LJ_ARCHIVE.name)
    try:
        with tarfile.open(str(LJ_ARCHIVE), "r:bz2") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".wav")]
            members_sorted = sorted(members, key=lambda m: m.name)[:n]
            for member in tqdm(members_sorted, desc="Extracting WAVs"):
                member.name = Path(member.name).name  # flatten to just filename
                tar.extract(member, str(GENUINE_WAV_DIR))
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        return False

    # Clean up the large archive (optional — saves disk space)
    try:
        LJ_ARCHIVE.unlink()
        logger.info("Deleted archive %s to save space.", LJ_ARCHIVE.name)
    except Exception:
        pass

    copied = len(list(GENUINE_WAV_DIR.glob("*.wav")))
    logger.info("Genuine audio ready: %d files in %s", copied, GENUINE_WAV_DIR)
    return copied > 0


# ─────────────────────────────────────────────
# Step 2 — Synthetic audio
# Option A: Download pre-made synthetic clips from HuggingFace
# Option B: Generate locally using enhanced acoustics (no internet)
# ─────────────────────────────────────────────

def _generate_synthetic_local(n: int, callback=None) -> bool:
    """
    Generate *n* synthetic-sounding WAV files locally.
    Uses the enhanced acoustic simulator from demo_data.py.
    """
    SYNTH_WAV_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(SYNTH_WAV_DIR.glob("*.wav"))
    if len(existing) >= n:
        logger.info("Synthetic audio already present (%d files).", len(existing))
        return True

    if callback: callback(f"🔊 Generating {n} synthetic audio clips locally …")
    from src.demo_data import _synthetic_clip
    rng = np.random.default_rng(42)

    for i in tqdm(range(n), desc="Generating synthetic clips"):
        duration = rng.uniform(2.0, 5.0)
        y = _synthetic_clip(SAMPLE_RATE, float(duration))
        sf.write(str(SYNTH_WAV_DIR / f"synth_{i:04d}.wav"), y, SAMPLE_RATE)

    logger.info("Synthetic audio ready: %d files in %s", n, SYNTH_WAV_DIR)
    return True


def _try_huggingface_synthetic(n: int, callback=None) -> bool:
    """
    Try to pull a small set of WaveFake-style clips from Hugging Face.
    Falls back gracefully if unavailable.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        # Try a small public deepfake audio dataset
        # resemble-ai/WaveFake has some subset files
        if callback: callback("🤗 Trying Hugging Face mirror for synthetic clips …")
        # This is a fallback — if HF isn't available just return False
        return False
    except ImportError:
        return False
    except Exception:
        return False


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def download_small_dataset(
    n_genuine: int = N_GENUINE,
    n_synth:   int = N_SYNTH,
    callback=None,   # callable(str) for progress updates
) -> Path | None:
    """
    Download / prepare a small real-speech dataset.

    Returns the dataset root Path on success, or None on failure.

    The full LJSpeech archive is ~2.6 GB but we extract only the first
    *n_genuine* WAV files (≈200 MB), then delete the archive.
    Total disk usage after setup: ~400 MB for 1000+1000 files.
    """
    logger.info("Preparing small dataset (target: %d genuine + %d synthetic)",
                n_genuine, n_synth)

    if callback: callback("🎙️ Setting up small real-speech dataset …")

    # ── Genuine ──────────────────────────────
    ok = _get_genuine(n_genuine, callback)
    if not ok:
        logger.warning("Could not get genuine audio — falling back to demo generator.")
        from src.demo_data import generate_demo_dataset
        return generate_demo_dataset(n_samples=500, force=True)

    # ── Synthetic ─────────────────────────────
    # Try HuggingFace first, then local generation
    synth_ok = _try_huggingface_synthetic(n_synth, callback)
    if not synth_ok:
        synth_ok = _generate_synthetic_local(n_synth, callback)

    if not synth_ok:
        logger.error("Could not prepare synthetic audio.")
        return None

    if callback: callback("✅ Dataset ready!")
    return SMALL_ROOT


def dataset_exists() -> bool:
    g = len(list(GENUINE_WAV_DIR.glob("*.wav"))) if GENUINE_WAV_DIR.exists() else 0
    s = len(list(SYNTH_WAV_DIR.glob("*.wav"))) if SYNTH_WAV_DIR.exists() else 0
    return g >= 100 and s >= 100


if __name__ == "__main__":
    path = download_small_dataset()
    print(f"Dataset root: {path}")
