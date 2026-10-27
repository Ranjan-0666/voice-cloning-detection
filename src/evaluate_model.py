"""
src/evaluate_model.py
Computes and persists all evaluation metrics for the trained Random Forest.

Called automatically from train_model.py but can also be run standalone.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CONFUSION_MATRIX_PATH, FEATURE_IMP_PATH, ROC_CURVE_PATH,
    METRICS_PATH, FEATURE_NAMES, LABEL_MAP, REPORTS_DIR,
)
from src.utils import get_logger, save_json, Timer

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Core evaluation function
# ─────────────────────────────────────────────

def evaluate(
    clf: RandomForestClassifier,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_plots: bool = True,
) -> Dict:
    """
    Evaluate *clf* on the (already-scaled) *X_test*.

    Returns a metrics dict and optionally saves plots to REPORTS_DIR.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with Timer() as t_infer:
        y_pred       = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]  # P(spoof)

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    cm    = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=[LABEL_MAP[0], LABEL_MAP[1]],
        output_dict=True,
    )

    # ROC-AUC (binary)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    except ValueError:
        roc_auc = None
        fpr, tpr = None, None

    metrics = {
        "accuracy"        : round(float(acc),  4),
        "precision"       : round(float(prec), 4),
        "recall"          : round(float(rec),  4),
        "f1_score"        : round(float(f1),   4),
        "roc_auc"         : round(float(roc_auc), 4) if roc_auc is not None else None,
        "n_test_samples"  : int(len(y_test)),
        "inference_time_s": round(float(t_infer.elapsed), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    save_json(metrics, METRICS_PATH)
    logger.info(
        "Metrics — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %s",
        acc, prec, rec, f1, roc_auc,
    )

    if save_plots:
        _plot_confusion_matrix(cm)
        _plot_feature_importance(clf)
        if fpr is not None:
            _plot_roc_curve(fpr, tpr, roc_auc)

    return metrics


# ─────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────

def _plot_confusion_matrix(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Genuine", "Synthetic"],
        yticklabels=["Genuine", "Synthetic"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=120)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", CONFUSION_MATRIX_PATH)


def _plot_feature_importance(clf: RandomForestClassifier, top_n: int = 20) -> None:
    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    names       = [FEATURE_NAMES[i] for i in indices]
    values      = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("viridis", len(indices))
    ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title(f"Top-{top_n} Feature Importances (Random Forest)")
    plt.tight_layout()
    fig.savefig(FEATURE_IMP_PATH, dpi=120)
    plt.close(fig)
    logger.info("Feature importance plot saved → %s", FEATURE_IMP_PATH)


def _plot_roc_curve(fpr, tpr, auc_score) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Voice Cloning Detection")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(ROC_CURVE_PATH, dpi=120)
    plt.close(fig)
    logger.info("ROC curve saved → %s", ROC_CURVE_PATH)


# ─────────────────────────────────────────────
# Standalone run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from src.train_model import load_model
    from src.dataset_loader import load_manifest, get_valid_splits
    from src.train_model import build_feature_matrix

    clf, scaler = load_model()
    if clf is None:
        print("Model not trained yet. Run: python -m src.train_model")
        sys.exit(1)

    manifest = load_manifest()
    _, test_df = get_valid_splits(manifest)
    X_test, y_test, _ = build_feature_matrix(test_df, "Test features")

    if len(X_test) == 0:
        print("No test samples available.")
        sys.exit(1)

    X_test_scaled = scaler.transform(X_test)
    metrics = evaluate(clf, scaler, X_test_scaled, y_test)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("classification_report",)}, indent=2))
