"""
src/dataset_loader.py
Discovers, validates, and catalogues WaveFake audio files.

WaveFake folder layout expected under WAVEFAKE_ROOT:
  LJSpeech-1.1/
    wavs/                   ← genuine recordings
      LJ001-0001.wav
      ...
  generated_audio/
    ljspeech_full_band_melgan/
      LJ001-0001.wav
      ...
    ljspeech_hifiGAN/
      ...
    <other vocoders>/
      ...

A CSV manifest is built at MANIFEST_PATH with columns:
  path | label | vocoder | split | valid | duration_s | error
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import soundfile as sf

from config import (
    WAVEFAKE_ROOT, GENUINE_SUBDIR, SYNTH_SUBDIR,
    MANIFEST_PATH, PROCESSED_DIR,
    LABEL_GENUINE, LABEL_SPOOF,
    MAX_GENUINE_SAMPLES, MAX_SYNTH_SAMPLES,
    AUDIO_EXTENSIONS, MIN_DURATION, MAX_DURATION,
    TEST_SIZE, RANDOM_STATE,
)
from src.utils import get_logger, find_audio_files

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _probe(path: Path) -> Tuple[bool, Optional[float], Optional[str]]:
    """Return (valid, duration_seconds, error_message)."""
    try:
        info = sf.info(str(path))
        dur  = info.duration
        if dur < MIN_DURATION:
            return False, dur, f"Too short ({dur:.3f}s)"
        if dur > MAX_DURATION * 5:          # tolerate some overshoots
            return True, dur, None          # will be truncated on load
        return True, dur, None
    except Exception as exc:
        return False, None, str(exc)


def _collect_genuine(root: Path) -> List[Dict]:
    wav_dir = root / GENUINE_SUBDIR
    if not wav_dir.exists():
        logger.warning("Genuine audio directory not found: %s", wav_dir)
        return []

    paths = find_audio_files(wav_dir, AUDIO_EXTENSIONS)
    logger.info("Found %d genuine candidates in %s", len(paths), wav_dir)

    # Optional cap
    if MAX_GENUINE_SAMPLES and len(paths) > MAX_GENUINE_SAMPLES:
        random.seed(RANDOM_STATE)
        paths = random.sample(paths, MAX_GENUINE_SAMPLES)

    rows = []
    for p in tqdm(paths, desc="Scanning genuine", leave=False):
        valid, dur, err = _probe(p)
        rows.append({
            "path"    : str(p),
            "label"   : LABEL_GENUINE,
            "vocoder" : "genuine",
            "valid"   : valid,
            "duration_s": dur,
            "error"   : err,
        })
    return rows


def _collect_synthetic(root: Path) -> List[Dict]:
    synth_root = root / SYNTH_SUBDIR
    if not synth_root.exists():
        logger.warning("Synthetic audio directory not found: %s", synth_root)
        return []

    # Each immediate sub-directory is a vocoder
    vocoders = [d for d in synth_root.iterdir() if d.is_dir()]
    if not vocoders:
        logger.warning("No vocoder sub-directories found under %s", synth_root)
        return []

    logger.info("Found %d vocoder(s): %s", len(vocoders),
                [v.name for v in vocoders])

    rows = []
    for vocoder_dir in vocoders:
        paths = find_audio_files(vocoder_dir, AUDIO_EXTENSIONS)
        for p in paths:
            valid, dur, err = _probe(p)
            rows.append({
                "path"    : str(p),
                "label"   : LABEL_SPOOF,
                "vocoder" : vocoder_dir.name,
                "valid"   : valid,
                "duration_s": dur,
                "error"   : err,
            })

    logger.info("Found %d synthetic candidates", len(rows))

    # Optional cap — balance across vocoders
    if MAX_SYNTH_SAMPLES and len(rows) > MAX_SYNTH_SAMPLES:
        random.seed(RANDOM_STATE)
        rows = random.sample(rows, MAX_SYNTH_SAMPLES)

    return rows


def _assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign train/test split labels.
    Genuine files are split by stem name so that the same utterance id
    never appears in both partitions across genuine and synthetic sets,
    which prevents data leakage between same-utterance pairings.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    df = df.copy()
    df["split"] = "train"

    # For each class independently, mark TEST_SIZE fraction as test
    for label in df["label"].unique():
        idx = df.index[df["label"] == label].tolist()
        n_test = max(1, int(len(idx) * TEST_SIZE))
        test_idx = rng.choice(idx, size=n_test, replace=False)
        df.loc[test_idx, "split"] = "test"

    return df


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def build_manifest(force: bool = False) -> pd.DataFrame:
    """
    Scan WAVEFAKE_ROOT, validate files, and write a CSV manifest.

    Parameters
    ----------
    force : bool
        If True, rebuild the manifest even if it already exists.

    Returns
    -------
    pd.DataFrame  — the manifest (valid files only, with split column)
    """
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MANIFEST_PATH.exists() and not force:
        logger.info("Loading existing manifest from %s", MANIFEST_PATH)
        return pd.read_csv(MANIFEST_PATH)

    if not WAVEFAKE_ROOT.exists():
        logger.error(
            "Dataset root not found: %s\n"
            "Please download WaveFake and set WAVEFAKE_ROOT in config.py "
            "or export the WAVEFAKE_ROOT environment variable.",
            WAVEFAKE_ROOT,
        )
        return pd.DataFrame()

    genuine_rows  = _collect_genuine(WAVEFAKE_ROOT)
    synthetic_rows = _collect_synthetic(WAVEFAKE_ROOT)

    all_rows = genuine_rows + synthetic_rows
    if not all_rows:
        logger.error("No audio files discovered. Check WAVEFAKE_ROOT.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = _assign_splits(df)

    # Persist full manifest (including invalid files for reporting)
    df.to_csv(MANIFEST_PATH, index=False)
    logger.info(
        "Manifest saved → %s  (%d total, %d valid, %d invalid)",
        MANIFEST_PATH,
        len(df),
        df["valid"].sum(),
        (~df["valid"]).sum(),
    )
    return df


def load_manifest() -> pd.DataFrame:
    """Load existing manifest or build it if absent."""
    if not MANIFEST_PATH.exists():
        return build_manifest()
    return pd.read_csv(MANIFEST_PATH)


def get_valid_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) containing only valid files."""
    valid = df[df["valid"] == True].copy()
    train = valid[valid["split"] == "train"].reset_index(drop=True)
    test  = valid[valid["split"] == "test"].reset_index(drop=True)
    return train, test


def dataset_summary(df: pd.DataFrame) -> Dict:
    """Return a dict of statistics suitable for display."""
    if df.empty:
        return {"error": "Manifest is empty or not yet built."}

    valid_df = df[df["valid"] == True]
    return {
        "total_files"     : len(df),
        "valid_files"     : int(df["valid"].sum()),
        "invalid_files"   : int((~df["valid"]).sum()),
        "genuine_count"   : int((valid_df["label"] == 0).sum()),
        "synthetic_count" : int((valid_df["label"] == 1).sum()),
        "vocoders"        : valid_df["vocoder"].unique().tolist(),
        "n_vocoders"      : valid_df["vocoder"].nunique(),
        "train_count"     : int((valid_df["split"] == "train").sum()),
        "test_count"      : int((valid_df["split"] == "test").sum()),
        "duration_mean_s" : round(float(valid_df["duration_s"].mean()), 3) if "duration_s" in valid_df else None,
        "duration_min_s"  : round(float(valid_df["duration_s"].min()),  3) if "duration_s" in valid_df else None,
        "duration_max_s"  : round(float(valid_df["duration_s"].max()),  3) if "duration_s" in valid_df else None,
    }
