"""
tests/test_features.py
Unit tests for audio feature extraction and label mapping.
Run with: pytest tests/ -v
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Ensure project root is on path ───────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import N_FEATURES, FEATURE_NAMES, LABEL_GENUINE, LABEL_SPOOF
from src.audio_features import extract_features, load_audio, get_audio_info


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_wav(path: Path, duration: float = 1.0, sr: int = 22050, corrupt: bool = False):
    """Write a synthetic sine-wave WAV to *path*."""
    import soundfile as sf
    if corrupt:
        path.write_bytes(b"NOT_VALID_AUDIO")
        return
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(str(path), y, sr)


# ─────────────────────────────────────────────
# Feature extraction tests
# ─────────────────────────────────────────────

class TestFeatureExtraction:

    def test_feature_vector_length(self, tmp_path):
        """Extracted feature vector must have exactly N_FEATURES elements."""
        wav = tmp_path / "test.wav"
        _make_wav(wav)
        feat = extract_features(wav)
        assert feat is not None, "extract_features returned None for valid audio"
        assert len(feat) == N_FEATURES, (
            f"Expected {N_FEATURES} features, got {len(feat)}"
        )

    def test_feature_names_count(self):
        """FEATURE_NAMES list length must match N_FEATURES."""
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_feature_vector_dtype(self, tmp_path):
        """Feature vector should be float32."""
        wav = tmp_path / "test.wav"
        _make_wav(wav)
        feat = extract_features(wav)
        assert feat is not None
        assert feat.dtype == np.float32

    def test_feature_no_nan(self, tmp_path):
        """Feature vector must not contain NaN or Inf values."""
        wav = tmp_path / "test.wav"
        _make_wav(wav)
        feat = extract_features(wav)
        assert feat is not None
        assert not np.any(np.isnan(feat)), "NaN in feature vector"
        assert not np.any(np.isinf(feat)), "Inf in feature vector"

    def test_different_durations_same_shape(self, tmp_path):
        """Feature vector shape should be consistent regardless of audio length."""
        shapes = set()
        for dur in [0.5, 1.0, 3.0, 8.0]:
            wav = tmp_path / f"test_{dur}.wav"
            _make_wav(wav, duration=dur)
            feat = extract_features(wav)
            if feat is not None:
                shapes.add(len(feat))
        assert len(shapes) == 1, f"Inconsistent feature shapes: {shapes}"

    def test_stereo_handled(self, tmp_path):
        """Stereo audio should be converted to mono without error."""
        import soundfile as sf
        sr = 22050
        t  = np.linspace(0, 1.0, sr, endpoint=False)
        stereo = np.stack([0.5 * np.sin(2 * np.pi * 440 * t),
                           0.5 * np.sin(2 * np.pi * 880 * t)], axis=1).astype(np.float32)
        wav = tmp_path / "stereo.wav"
        sf.write(str(wav), stereo, sr)
        feat = extract_features(wav)
        assert feat is not None
        assert len(feat) == N_FEATURES


# ─────────────────────────────────────────────
# Invalid audio handling
# ─────────────────────────────────────────────

class TestInvalidAudioHandling:

    def test_missing_file_returns_none(self, tmp_path):
        """extract_features on a missing path must return None gracefully."""
        feat = extract_features(tmp_path / "nonexistent.wav")
        assert feat is None

    def test_corrupt_file_returns_none(self, tmp_path):
        """extract_features on a corrupt file must return None gracefully."""
        bad = tmp_path / "corrupt.wav"
        _make_wav(bad, corrupt=True)
        feat = extract_features(bad)
        assert feat is None

    def test_silent_audio_returns_none(self, tmp_path):
        """Near-silent audio should be rejected."""
        import soundfile as sf
        wav = tmp_path / "silence.wav"
        sf.write(str(wav), np.zeros(22050, dtype=np.float32), 22050)
        feat = extract_features(wav)
        # Silent → load_audio returns None → extract_features returns None
        assert feat is None

    def test_get_audio_info_valid(self, tmp_path):
        """get_audio_info should return valid=True for good files."""
        wav = tmp_path / "good.wav"
        _make_wav(wav)
        info = get_audio_info(wav)
        assert info["valid"] is True
        assert info["duration_s"] > 0

    def test_get_audio_info_corrupt(self, tmp_path):
        """get_audio_info should return valid=False for corrupt files."""
        bad = tmp_path / "bad.wav"
        _make_wav(bad, corrupt=True)
        info = get_audio_info(bad)
        assert info["valid"] is False
        assert info["error"] is not None


# ─────────────────────────────────────────────
# Label mapping tests
# ─────────────────────────────────────────────

class TestLabelMapping:

    def test_genuine_label_is_zero(self):
        assert LABEL_GENUINE == 0

    def test_spoof_label_is_one(self):
        assert LABEL_SPOOF == 1

    def test_label_map_keys(self):
        from config import LABEL_MAP
        assert 0 in LABEL_MAP
        assert 1 in LABEL_MAP

    def test_label_map_strings(self):
        from config import LABEL_MAP
        assert "Genuine" in LABEL_MAP[0]
        assert "Synthetic" in LABEL_MAP[1] or "Spoof" in LABEL_MAP[1]


# ─────────────────────────────────────────────
# Model persistence tests (requires trained model)
# ─────────────────────────────────────────────

class TestModelPersistence:

    def test_feature_config_schema(self):
        """If feature_config.json exists, validate its schema."""
        from config import FEATURE_CFG
        if not FEATURE_CFG.exists():
            pytest.skip("Model not yet trained — skipping persistence test")

        with open(FEATURE_CFG) as f:
            cfg = json.load(f)

        assert "feature_names" in cfg
        assert "n_features" in cfg
        assert cfg["n_features"] == N_FEATURES
        assert len(cfg["feature_names"]) == N_FEATURES

    def test_model_files_exist(self):
        """After training, model and scaler files must be present."""
        from config import MODEL_PATH, SCALER_PATH
        if not MODEL_PATH.exists():
            pytest.skip("Model not yet trained — skipping file-existence test")
        assert MODEL_PATH.exists(), "Model file missing"
        assert SCALER_PATH.exists(), "Scaler file missing"

    def test_model_loadable(self):
        """Saved model must be loadable with joblib."""
        from config import MODEL_PATH, SCALER_PATH
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            pytest.skip("Model not yet trained")
        import joblib
        clf    = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        assert clf    is not None
        assert scaler is not None


# ─────────────────────────────────────────────
# Prediction pipeline test
# ─────────────────────────────────────────────

class TestPrediction:

    def test_prediction_without_model(self, tmp_path):
        """predict() must return a clear error when model is not trained."""
        from config import MODEL_PATH, SCALER_PATH
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            pytest.skip("Model IS trained — this test only applies pre-training")

        wav = tmp_path / "audio.wav"
        _make_wav(wav)
        from src.predict import predict
        result = predict(wav)
        assert result["success"] is False
        assert result["error"] is not None

    def test_prediction_with_model(self, tmp_path):
        """If model is trained, predict() must return success=True for valid audio."""
        from config import MODEL_PATH, SCALER_PATH
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            pytest.skip("Model not yet trained")

        wav = tmp_path / "audio.wav"
        _make_wav(wav)
        from src.predict import predict
        result = predict(wav)
        assert result["success"] is True
        assert result["label_id"] in (0, 1)
        assert 0.0 <= result["spoof_probability"] <= 1.0
        assert 0.0 <= result["genuine_probability"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
