"""
src/train_model.py
Trains a Random Forest Classifier on extracted audio features.

Steps:
  1. Load / build dataset manifest.
  2. Extract features for every valid training file.
  3. Fit a StandardScaler (training data only — no leakage).
  4. Train sklearn RandomForestClassifier.
  5. Save model, scaler, and feature config via Joblib.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODEL_PATH, SCALER_PATH, FEATURE_CFG,
    RF_PARAMS, FEATURE_NAMES, N_FEATURES,
    LABEL_MAP, RANDOM_STATE, MODELS_DIR,
)
from src.audio_features import extract_features
from src.dataset_loader import load_manifest, get_valid_splits
from src.utils import get_logger, Timer, save_json

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Feature matrix builder
# ─────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    desc: str = "Extracting",
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Extract features for every row in *df*.

    Returns
    -------
    X : ndarray shape (n_valid, N_FEATURES)
    y : ndarray shape (n_valid,)
    skipped : list of paths that failed extraction
    """
    X_rows, y_rows, skipped = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc, leave=False):
        feat = extract_features(row["path"])
        if feat is None:
            skipped.append(row["path"])
            continue
        if len(feat) != N_FEATURES:
            skipped.append(row["path"])
            continue
        X_rows.append(feat)
        y_rows.append(int(row["label"]))

    if not X_rows:
        return np.empty((0, N_FEATURES)), np.empty(0), skipped

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32), skipped


# ─────────────────────────────────────────────
# Training entry-point
# ─────────────────────────────────────────────

def train(
    rf_params: Optional[Dict] = None,
    progress_callback=None,   # callable(message: str) — for Streamlit
) -> Dict:
    """
    Full training pipeline.

    Parameters
    ----------
    rf_params : dict | None
        Override RF_PARAMS from config (merged, not replaced).
    progress_callback : callable | None
        Called with status strings so a UI can display progress.

    Returns
    -------
    dict with keys: success, message, metrics (accuracy, …), timing
    """
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    # ── 1. Load manifest ──────────────────────
    log("Loading dataset manifest …")
    manifest = load_manifest()
    if manifest.empty:
        return {"success": False, "message": "Dataset manifest is empty. Run dataset setup first."}

    train_df, test_df = get_valid_splits(manifest)
    log(f"Train: {len(train_df)} files | Test: {len(test_df)} files")

    if len(train_df) < 10:
        return {"success": False, "message": f"Too few training samples ({len(train_df)}). Check dataset."}

    # ── 2. Extract features ───────────────────
    log("Extracting training features …")
    with Timer() as t_feat:
        X_train, y_train, skip_train = build_feature_matrix(train_df, "Train features")
    log(f"Training features done in {t_feat} — skipped {len(skip_train)} files")

    log("Extracting test features …")
    X_test, y_test, skip_test = build_feature_matrix(test_df, "Test features")
    log(f"Test features done — skipped {len(skip_test)} files")

    if len(X_train) < 5:
        return {"success": False, "message": "Feature extraction produced too few samples."}

    # ── 3. Scale features (fit on TRAIN only) ─
    log("Fitting StandardScaler on training data …")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test) if len(X_test) > 0 else np.empty((0, N_FEATURES))

    # ── 4. Train Random Forest ────────────────
    params = {**RF_PARAMS, **(rf_params or {})}
    log(f"Training RandomForestClassifier with params: {params}")

    clf = RandomForestClassifier(**params)
    with Timer() as t_train:
        clf.fit(X_train_scaled, y_train)
    log(f"Training complete in {t_train}")

    # ── 5. Evaluate on held-out test set ──────
    metrics = {}
    if len(X_test_scaled) > 0:
        log("Evaluating on test set …")
        with Timer() as t_eval:
            from src.evaluate_model import evaluate
            metrics = evaluate(clf, scaler, X_test_scaled, y_test)
        log(f"Evaluation done in {t_eval}")
    else:
        log("WARNING: No test samples available — skipping evaluation.")

    # ── 6. Persist model artifacts ────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf,    MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    feature_config = {
        "feature_names" : FEATURE_NAMES,
        "n_features"    : N_FEATURES,
        "label_map"     : LABEL_MAP,
        "rf_params"     : params,
        "train_samples" : int(len(X_train)),
        "test_samples"  : int(len(X_test)),
        "skipped_train" : len(skip_train),
        "skipped_test"  : len(skip_test),
        "training_time_s": float(t_train.elapsed),
    }
    save_json(feature_config, FEATURE_CFG)
    log(f"Model saved → {MODEL_PATH}")
    log(f"Scaler saved → {SCALER_PATH}")

    return {
        "success"       : True,
        "message"       : "Training complete.",
        "metrics"       : metrics,
        "feature_config": feature_config,
        "training_time_s": float(t_train.elapsed),
    }


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_model():
    """Return (clf, scaler) or (None, None) if not yet trained."""
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    try:
        clf    = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return clf, scaler
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return None, None


def model_is_trained() -> bool:
    return MODEL_PATH.exists() and SCALER_PATH.exists()


if __name__ == "__main__":
    result = train()
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2))
    if result.get("metrics"):
        print("Metrics:", result["metrics"])

