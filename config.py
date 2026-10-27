"""
config.py — Central configuration for Voice Cloning Detection project.
All tuneable parameters live here so every module imports a single source of truth.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Project root (one level up from this file)
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

# ─────────────────────────────────────────────
# Dataset paths  (WaveFake structure)
# ─────────────────────────────────────────────
DATA_DIR        = BASE_DIR / "data"
PROCESSED_DIR   = DATA_DIR / "processed"
MANIFEST_PATH   = PROCESSED_DIR / "dataset_manifest.csv"

# WaveFake places genuine LJSpeech wav files under:
#   <dataset_root>/LJSpeech-1.1/wavs/
# and synthetic files under per-vocoder sub-directories:
#   <dataset_root>/generated_audio/<vocoder_name>/
#
# Auto-detect the dataset root from common extraction locations.
def _find_wavefake_root(base: Path) -> Path:
    """
    Try common locations where any compatible dataset has been placed.
    Priority:
      1. WAVEFAKE_ROOT env var (manual override)
      2. small_dataset/  (auto-downloaded LJSpeech subset — recommended)
      3. WaveFake/       (full 72 GB Zenodo download)
      4. WaveFake-main/  (GitHub code repo — no audio)
      5. demo_wavefake/  (generated demo data — instant, no download)
    """
    if os.environ.get("WAVEFAKE_ROOT"):
        return Path(os.environ["WAVEFAKE_ROOT"])
    candidates = [
        base / "small_dataset",          # ← 400 MB auto-download (recommended)
        base / "WaveFake",               # ← full 72 GB dataset
        base / "WaveFake-main" / "WaveFake-main",
        base / "WaveFake-main",
        base / "demo_wavefake",          # ← instant demo (no real audio)
    ]
    for c in candidates:
        if (c / "LJSpeech-1.1").exists() or (c / "generated_audio").exists():
            return c
    return base / "WaveFake"   # fallback — will trigger setup instructions

WAVEFAKE_ROOT = _find_wavefake_root(DATA_DIR)

# Mode flags
DEMO_MODE        = "demo_wavefake" in str(WAVEFAKE_ROOT)
SMALL_DATASET    = "small_dataset" in str(WAVEFAKE_ROOT)

# Sub-paths inside WAVEFAKE_ROOT
GENUINE_SUBDIR  = "LJSpeech-1.1/wavs"
SYNTH_SUBDIR    = "generated_audio"

# ─────────────────────────────────────────────
# Audio processing
# ─────────────────────────────────────────────
SAMPLE_RATE     = 22050        # Hz — resample all audio to this rate
MAX_DURATION    = 10.0         # seconds — truncate longer clips
MIN_DURATION    = 0.3          # seconds — skip clips that are too short
N_MFCC          = 13           # number of MFCC coefficients
HOP_LENGTH      = 512
N_FFT           = 2048
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}

# ─────────────────────────────────────────────
# Feature vector layout
# (mean + std for each MFCC → 2*N_MFCC, plus 5 spectral/energy features)
# ─────────────────────────────────────────────
FEATURE_NAMES = (
    [f"mfcc_mean_{i}" for i in range(N_MFCC)]
    + [f"mfcc_std_{i}"  for i in range(N_MFCC)]
    + [
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "spectral_rolloff_mean",
        "zcr_mean",
        "rms_mean",
    ]
)
N_FEATURES = len(FEATURE_NAMES)   # 2*13 + 5 = 31

# ─────────────────────────────────────────────
# Label encoding
# ─────────────────────────────────────────────
LABEL_GENUINE  = 0
LABEL_SPOOF    = 1
LABEL_MAP      = {0: "Genuine", 1: "Synthetic (Suspected Spoof)"}

# ─────────────────────────────────────────────
# Dataset sampling (set to None to use all files)
# ─────────────────────────────────────────────
MAX_GENUINE_SAMPLES  = 2000
MAX_SYNTH_SAMPLES    = 2000

# ─────────────────────────────────────────────
# Train / test split
# ─────────────────────────────────────────────
TEST_SIZE       = 0.20
RANDOM_STATE    = 42

# ─────────────────────────────────────────────
# Random Forest hyper-parameters
# ─────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators"  : 200,
    "max_depth"     : None,
    "min_samples_split": 5,
    "min_samples_leaf" : 2,
    "max_features"  : "sqrt",
    "class_weight"  : "balanced",
    "random_state"  : RANDOM_STATE,
    "n_jobs"        : -1,
}

# ─────────────────────────────────────────────
# Model persistence
# ─────────────────────────────────────────────
MODELS_DIR      = BASE_DIR / "models"
MODEL_PATH      = MODELS_DIR / "rf_voice_cloning.joblib"
SCALER_PATH     = MODELS_DIR / "feature_scaler.joblib"
FEATURE_CFG     = MODELS_DIR / "feature_config.json"

# ─────────────────────────────────────────────
# Reports / outputs
# ─────────────────────────────────────────────
REPORTS_DIR     = BASE_DIR / "reports"
CONFUSION_MATRIX_PATH   = REPORTS_DIR / "confusion_matrix.png"
FEATURE_IMP_PATH        = REPORTS_DIR / "feature_importance.png"
ROC_CURVE_PATH          = REPORTS_DIR / "roc_curve.png"
METRICS_PATH            = REPORTS_DIR / "metrics.json"

# ─────────────────────────────────────────────
# Cybersecurity / inference
# ─────────────────────────────────────────────
SPOOF_THRESHOLD = 0.50   # probability above which audio is flagged as synthetic
HISTORY_PATH    = PROCESSED_DIR / "prediction_history.csv"
RETAIN_AUDIO    = False  # do NOT store uploaded audio by default

# ─────────────────────────────────────────────
# Ensure required directories exist at import time
# ─────────────────────────────────────────────
for _d in [DATA_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
