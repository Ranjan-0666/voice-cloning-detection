"""
src/audio_features.py
Extracts a fixed-length numerical feature vector from a single audio file.

Feature vector layout (31 dimensions):
  [0:13]  MFCC means   (N_MFCC = 13)
  [13:26] MFCC stds    (N_MFCC = 13)
  [26]    spectral centroid mean
  [27]    spectral bandwidth mean
  [28]    spectral rolloff mean
  [29]    zero-crossing rate mean
  [30]    RMS energy mean
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SAMPLE_RATE, MAX_DURATION, MIN_DURATION,
    N_MFCC, HOP_LENGTH, N_FFT, N_FEATURES, FEATURE_NAMES,
)
from src.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def load_audio(
    path: Path,
    sr: int = SAMPLE_RATE,
    max_duration: float = MAX_DURATION,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Load audio from *path*.

    Returns (waveform, sample_rate) or (None, 0) on failure.
    Handles: mono/stereo, resampling, truncation, silence, corrupted files.
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required. Install it with: pip install librosa")

    path = Path(path)
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None, 0

    try:
        y, file_sr = librosa.load(
            str(path),
            sr=sr,
            mono=True,          # fold stereo → mono
            duration=max_duration,
        )
    except Exception as exc:
        logger.warning("Cannot load %s — %s", path.name, exc)
        return None, 0

    # Skip near-silent clips
    if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
        logger.warning("Silent or empty audio: %s", path.name)
        return None, 0

    duration = len(y) / sr
    if duration < MIN_DURATION:
        logger.warning("Too short (%.2fs): %s", duration, path.name)
        return None, 0

    return y, sr


def extract_features(
    path_or_waveform,
    sr: int = SAMPLE_RATE,
) -> Optional[np.ndarray]:
    """
    Extract the canonical 31-dimensional feature vector.

    *path_or_waveform* can be a file path (str/Path) or a pre-loaded
    NumPy waveform array (in which case *sr* must match the loaded rate).

    Returns a 1-D float32 array of length N_FEATURES, or None on failure.
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required.")

    # ── Load waveform ────────────────────────
    if isinstance(path_or_waveform, (str, Path)):
        y, sr = load_audio(path_or_waveform, sr=sr)
        if y is None:
            return None
    else:
        y = np.asarray(path_or_waveform, dtype=np.float32)
        if y.ndim > 1:
            y = librosa.to_mono(y)

    try:
        # ── MFCCs ────────────────────────────
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=N_MFCC,
            n_fft=N_FFT, hop_length=HOP_LENGTH,
        )
        mfcc_mean = mfcc.mean(axis=1)   # shape (N_MFCC,)
        mfcc_std  = mfcc.std(axis=1)    # shape (N_MFCC,)

        # ── Spectral centroid ─────────────────
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        )
        centroid_mean = float(centroid.mean())

        # ── Spectral bandwidth ────────────────
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        )
        bandwidth_mean = float(bandwidth.mean())

        # ── Spectral rolloff ──────────────────
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        )
        rolloff_mean = float(rolloff.mean())

        # ── Zero-crossing rate ────────────────
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
        zcr_mean = float(zcr.mean())

        # ── RMS energy ───────────────────────
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        rms_mean = float(rms.mean())

        # ── Assemble feature vector ───────────
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [centroid_mean, bandwidth_mean, rolloff_mean, zcr_mean, rms_mean],
        ]).astype(np.float32)

        assert len(features) == N_FEATURES, (
            f"Feature length mismatch: got {len(features)}, expected {N_FEATURES}"
        )
        return features

    except Exception as exc:
        logger.error("Feature extraction failed: %s", exc)
        return None


def get_audio_info(path) -> dict:
    """Return basic metadata about an audio file (for display in the UI)."""
    path = Path(path)
    info = {
        "filename"  : path.name,
        "format"    : path.suffix.upper().lstrip("."),
        "duration_s": None,
        "sample_rate": None,
        "channels"  : None,
        "valid"     : False,
        "error"     : None,
    }
    try:
        sf_info = sf.info(str(path))
        info["duration_s"]  = round(sf_info.duration, 3)
        info["sample_rate"] = sf_info.samplerate
        info["channels"]    = sf_info.channels
        info["valid"]       = True
    except Exception as exc:
        info["error"] = str(exc)
    return info
