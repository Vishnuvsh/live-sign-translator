"""
evaluate_model.py
=================
Load the saved gesture model and evaluate it on the processed test dataset.

Displays:
  - Accuracy, Precision, Recall, F1 Score (per class + macro average)
  - Confusion matrix (terminal + matplotlib popup)

Usage:
    python training/evaluate_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gesture_config import (
    FEATURE_VECTOR_SIZE,
    GESTURE_NAMES,
    GESTURES,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_DIR,
)
from src.utils import get_logger, validate_file_exists

logger = get_logger("evaluate_model")


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_model_and_classes():
    """Load the trained model + class labels from the .pkl file."""
    validate_file_exists(MODEL_PATH, "Gesture model")
    data = joblib.load(MODEL_PATH)
    if isinstance(data, dict):
        return data["model"], data.get("classes", GESTURE_NAMES)
    return data, GESTURE_NAMES


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load data for evaluation.
    Prefers the processed combined CSV; falls back to raw CSVs if unavailable.
    """
    if PROCESSED_DATA_PATH.is_file():
        logger.info(f"Loading processed dataset: {PROCESSED_DATA_PATH}")
        data = np.loadtxt(PROCESSED_DATA_PATH, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        y = data[:, 0].astype(int)
        X = data[:, 1:].astype(np.float32)
        return X, y

    # Fallback: re-read raw CSVs
    logger.info("Processed dataset not found — reading raw CSVs.")
    X_list, y_list = [], []
    for g in GESTURES:
        safe  = g["name"].replace(" ", "_").lower()
        path  = RAW_DATA_DIR / f"{g['id']:02d}_{safe}.csv"
        if not path.is_file():
            continue
        rows = np.loadtxt(path, delimiter=",")
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        y_list.append(rows[:, 0].astype(int))
        X_list.append(rows[:, 1:])

    if not X_list:
        raise FileNotFoundError(
            "No data found. Run collect_data.py first, then train_model.py."
        )
    return np.vstack(X_list).astype(np.float32), np.concatenate(y_list).astype(int)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X: np.ndarray, y: np.ndarray, class_names: list[str]) -> None:
    """Compute and print all evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_pred = model.predict(X)
    unique_ids = sorted(set(y.tolist()))

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="macro", zero_division=0)
    rec  = recall_score   (y, y_pred, average="macro", zero_division=0)
    f1   = f1_score       (y, y_pred, average="macro", zero_division=0)

    print("\n" + "="*60)
    print(" Gesture Model Evaluation Report")
    print("="*60)
    print(f"  Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision (macro) : {prec:.4f}")
    print(f"  Recall (macro)    : {rec:.4f}")
    print(f"  F1 Score (macro)  : {f1:.4f}")
    print()
    print("  Per-class metrics:")
    print("-"*60)
    print(classification_report(y, y_pred, target_names=class_names, zero_division=0))

    # --- Text confusion matrix ---
    cm = confusion_matrix(y, y_pred, labels=unique_ids)
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print("         " + "  ".join(f"{n[:5]:>5}" for n in class_names))
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:5d}" for v in row)
        name = class_names[i] if i < len(class_names) else str(unique_ids[i])
        print(f"  {name[:7]:<7} {row_str}")
    print("="*60 + "\n")

    # --- Matplotlib visual confusion matrix (optional) ---
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
        ax.set_title("Gesture Classifier — Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(MODEL_PATH.parent / "confusion_matrix.png", dpi=150)
        print("  Confusion matrix image saved to: models/confusion_matrix.png")
        plt.show()
    except ImportError:
        print("  (Install matplotlib to see visual confusion matrix)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "="*60)
    print(" Sign Language Model Evaluation")
    print("="*60)

    print("  Loading model …")
    model, class_names = load_model_and_classes()
    print(f"  Model type : {type(model).__name__}")
    print(f"  Classes    : {class_names}\n")

    print("  Loading evaluation data …")
    X, y = load_test_data()
    print(f"  Samples    : {len(X)}\n")

    evaluate(model, X, y, class_names)


if __name__ == "__main__":
    main()
