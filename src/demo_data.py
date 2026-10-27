"""
src/demo_data.py
Generates a small synthetic demo dataset so the app can train and demonstrate
all features WITHOUT the full WaveFake audio download (~72 GB).

Generates:
  - 200 "genuine" sine-wave clips with natural voice characteristics
  - 200 "synthetic" clips with GAN-like spectral artefacts

These are NOT real voice recordings — they are acoustic simulations used only
for demonstration and testing. Performance metrics from demo data do not reflect
real-world voice-cloning detection accuracy.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_DIR, SAMPLE_RATE, WAVEFAKE_ROOT,
    GENUINE_SUBDIR, SYNTH_SUBDIR, RANDOM_STATE,
)
from src.utils import get_logger

logger = get_logger(__name__)

DEMO_ROOT = DATA_DIR / "demo_wavefake"
N_SAMPLES  = 200   # per class
DURATION   = 2.0   # seconds

rng = np.random.default_rng(RANDOM_STATE)

# ─────────────────────────────────────────────
# Waveform generators
# ─────────────────────────────────────────────

def _genuine_clip(sr: int, duration: float) -> np.ndarray:
    """
    Simulate a genuine speech clip:
    - Multiple harmonics (fundamental + overtones)
    - Slight pitch modulation (prosody)
    - Band-limited noise (breath/room)
    - Natural RMS variation
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    f0 = rng.uniform(85, 255)                  # human fundamental pitch range
    harmonics = [1.0, 0.55, 0.30, 0.18, 0.10, 0.06]

    # Slow pitch vibrato (< 5 Hz)
    vibrato_rate = rng.uniform(4.0, 6.0)
    vibrato_depth = rng.uniform(0.01, 0.04)
    pitch_mod = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)

    y = np.zeros(n)
    for k, amp in enumerate(harmonics, start=1):
        y += amp * np.sin(2 * np.pi * f0 * k * pitch_mod * t)

    # Amplitude envelope (speech-like bursts)
    n_bursts = int(duration * rng.uniform(2.5, 5.0))
    envelope = np.zeros(n)
    for _ in range(n_bursts):
        center = rng.integers(0, n)
        width  = rng.integers(n // (n_bursts * 2), n // n_bursts)
        start  = max(0, center - width // 2)
        end    = min(n, center + width // 2)
        envelope[start:end] += 1.0
    envelope = np.convolve(envelope, np.ones(sr // 20) / (sr // 20), mode="same")
    envelope = np.clip(envelope, 0, 1)

    y = y * envelope
    y += rng.normal(0, 0.003, n)              # breath noise

    # Normalise
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * rng.uniform(0.55, 0.75)

    return y.astype(np.float32)


def _synthetic_clip(sr: int, duration: float) -> np.ndarray:
    """
    Simulate a GAN-synthesised clip:
    - More uniform harmonics (less natural decay)
    - Reduced vibrato
    - Spectrally flat residual (vocoder artefact)
    - Slightly elevated high-frequency energy
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    f0 = rng.uniform(100, 240)
    harmonics = [1.0, 0.70, 0.58, 0.50, 0.44, 0.38, 0.30]  # flatter decay

    # Reduced vibrato (vocoders often under-model prosody)
    vibrato_depth = rng.uniform(0.002, 0.008)
    vibrato_rate  = rng.uniform(3.0, 5.0)
    pitch_mod = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)

    y = np.zeros(n)
    for k, amp in enumerate(harmonics, start=1):
        y += amp * np.sin(2 * np.pi * f0 * k * pitch_mod * t)

    # Flatter envelope (less natural silence)
    envelope = 0.7 + 0.3 * np.abs(np.sin(np.pi * t / duration))
    y = y * envelope

    # Spectrally flat noise (vocoder quantisation artefact)
    y += rng.uniform(-0.015, 0.015, n)

    # High-frequency "ringing" artefact common in GAN vocoders
    hf_freq = rng.uniform(8000, 10000)
    hf_amp  = rng.uniform(0.02, 0.06)
    y += hf_amp * np.sin(2 * np.pi * hf_freq * t)

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * rng.uniform(0.55, 0.75)

    return y.astype(np.float32)


# ─────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────

def generate_demo_dataset(n_samples: int = N_SAMPLES, force: bool = False) -> Path:
    """
    Write demo WAV files under DEMO_ROOT and return the root path.

    Returns DEMO_ROOT so callers can point WAVEFAKE_ROOT at it.
    """
    genuine_dir = DEMO_ROOT / GENUINE_SUBDIR
    synth_dir   = DEMO_ROOT / SYNTH_SUBDIR / "demo_vocoder"

    if (genuine_dir.exists() and synth_dir.exists()
            and len(list(genuine_dir.glob("*.wav"))) >= n_samples
            and not force):
        logger.info("Demo dataset already exists at %s", DEMO_ROOT)
        return DEMO_ROOT

    genuine_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %d genuine demo clips …", n_samples)
    for i in range(n_samples):
        y = _genuine_clip(SAMPLE_RATE, DURATION)
        sf.write(str(genuine_dir / f"DEMO_GENUINE_{i:04d}.wav"), y, SAMPLE_RATE)

    logger.info("Generating %d synthetic demo clips …", n_samples)
    for i in range(n_samples):
        y = _synthetic_clip(SAMPLE_RATE, DURATION)
        sf.write(str(synth_dir / f"DEMO_SYNTH_{i:04d}.wav"), y, SAMPLE_RATE)

    logger.info("Demo dataset ready at %s", DEMO_ROOT)
    return DEMO_ROOT


if __name__ == "__main__":
    path = generate_demo_dataset()
    print(f"Demo dataset generated at: {path}")
