"""
src/predict.py
Inference pipeline: loads trained model, extracts features, returns prediction.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    LABEL_MAP, SPOOF_THRESHOLD,
    HISTORY_PATH, RETAIN_AUDIO, FEATURE_CFG,
)
from src.audio_features import extract_features, get_audio_info
from src.train_model import load_model, model_is_trained
from src.utils import get_logger, load_json

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Core prediction
# ─────────────────────────────────────────────

def predict(
    audio_path,
    threshold: float = SPOOF_THRESHOLD,
) -> Dict:
    """
    Run the full inference pipeline for a single audio file.

    Parameters
    ----------
    audio_path : str | Path
        Path to the audio file to analyse.
    threshold : float
        Probability cut-off above which a file is classified as synthetic.

    Returns
    -------
    dict with keys:
        success, label_id, label_name, spoof_probability, genuine_probability,
        is_suspicious, warning, audio_info, features, error
    """
    result = {
        "success"            : False,
        "label_id"           : None,
        "label_name"         : None,
        "spoof_probability"  : None,
        "genuine_probability": None,
        "is_suspicious"      : False,
        "warning"            : None,
        "audio_info"         : {},
        "features"           : {},
        "error"              : None,
    }

    # ── 0. Model check ────────────────────────
    if not model_is_trained():
        result["error"] = (
            "Model has not been trained yet. "
            "Go to the 'Model Training' page and click 'Train Model'."
        )
        return result

    # ── 1. Validate audio ─────────────────────
    audio_path = Path(audio_path)
    info = get_audio_info(audio_path)
    result["audio_info"] = info

    if not info["valid"]:
        result["error"] = f"Invalid audio file: {info.get('error', 'unknown error')}"
        return result

    # ── 2. Extract features ───────────────────
    feat_vec = extract_features(audio_path)
    if feat_vec is None:
        result["error"] = "Feature extraction failed — audio may be corrupted or too short."
        return result

    # ── 3. Load model ─────────────────────────
    clf, scaler = load_model()
    if clf is None:
        result["error"] = "Could not load trained model from disk."
        return result

    # ── 4. Scale & predict ────────────────────
    X = feat_vec.reshape(1, -1)
    X_scaled = scaler.transform(X)

    proba        = clf.predict_proba(X_scaled)[0]   # [P(genuine), P(spoof)]
    spoof_prob   = float(proba[1])
    genuine_prob = float(proba[0])

    # Apply configurable threshold
    label_id   = 1 if spoof_prob >= threshold else 0
    label_name = LABEL_MAP[label_id]

    # ── 5. Cybersecurity assessment ───────────
    is_suspicious = (label_id == 1)
    warning = None
    if is_suspicious:
        warning = (
            "⚠️  SUSPECTED SYNTHETIC SPEECH DETECTED\n\n"
            f"This recording has a {spoof_prob*100:.1f}% probability of being "
            "AI-generated or cloned voice audio.\n\n"
            "Recommendation: Do NOT rely solely on this system for high-stakes "
            "decisions. Perform independent identity verification before authorising "
            "sensitive actions (financial transactions, access control, etc.)."
        )

    # ── 6. Build features dict for display ────
    try:
        cfg        = load_json(FEATURE_CFG)
        feat_names = cfg.get("feature_names", [])
        feat_dict  = {n: round(float(v), 5) for n, v in zip(feat_names, feat_vec)}
    except Exception:
        feat_dict = {}

    result.update({
        "success"            : True,
        "label_id"           : label_id,
        "label_name"         : label_name,
        "spoof_probability"  : round(spoof_prob,   4),
        "genuine_probability": round(genuine_prob, 4),
        "is_suspicious"      : is_suspicious,
        "warning"            : warning,
        "features"           : feat_dict,
    })

    # ── 7. Optional prediction history ────────
    _log_prediction(audio_path, result, threshold)

    return result


# ─────────────────────────────────────────────
# History logging
# ─────────────────────────────────────────────

def _log_prediction(path: Path, result: Dict, threshold: float) -> None:
    """Append prediction to a local CSV history (metadata only, no audio)."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not HISTORY_PATH.exists()
    try:
        with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow([
                    "timestamp", "filename", "label", "spoof_prob",
                    "genuine_prob", "is_suspicious", "threshold",
                ])
            writer.writerow([
                datetime.now().isoformat(),
                path.name if not RETAIN_AUDIO else str(path),
                result.get("label_name", ""),
                result.get("spoof_probability", ""),
                result.get("genuine_probability", ""),
                result.get("is_suspicious", ""),
                threshold,
            ])
    except Exception as exc:
        logger.warning("Could not write prediction history: %s", exc)


def load_history() -> Optional[object]:
    """Return prediction history as a DataFrame, or None."""
    if not HISTORY_PATH.exists():
        return None
    try:
        import pandas as pd
        return pd.read_csv(HISTORY_PATH)
    except Exception:
        return None
