"""
train_model.py
==============
Training pipeline for the gesture classifier.

Pipeline:
    Raw CSV files (data/raw/)
        ↓ Load & combine
    Combined dataset
        ↓ Validate
    Train / Test split (80/20, stratified)
        ↓
    Random Forest training
        ↓ 
    Cross-validation (5-fold)
        ↓
    Evaluate on held-out test set
        ↓
    Print metrics
        ↓
    Save model + class labels to models/gesture_model.pkl

Usage:
    python training/train_model.py
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
    NUM_GESTURES,
    RAW_DATA_DIR,
    PROCESSED_DATA_PATH,
)
from src.utils import ensure_dir, get_logger

logger = get_logger("train_model")


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_raw_csvs() -> tuple[np.ndarray, np.ndarray]:
    """
    Load all per-gesture CSV files from data/raw/ and combine into X, y arrays.

    Each CSV row format:
        gesture_id, feat0, feat1, ..., feat62  (1 + 63 = 64 columns)

    Returns
    -------
    X : np.ndarray, shape (N, 63)
    y : np.ndarray, shape (N,), dtype int  — gesture IDs
    """
    X_list, y_list = [], []
    found_any = False

    for g in GESTURES:
        safe_name = g["name"].replace(" ", "_").lower()
        csv_path  = RAW_DATA_DIR / f"{g['id']:02d}_{safe_name}.csv"

        if not csv_path.is_file():
            logger.warning(f"CSV not found — skipping gesture '{g['name']}': {csv_path}")
            continue

        rows = np.loadtxt(csv_path, delimiter=",")
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)

        if rows.shape[1] != FEATURE_VECTOR_SIZE + 1:
            logger.error(
                f"CSV {csv_path.name} has {rows.shape[1]} columns, "
                f"expected {FEATURE_VECTOR_SIZE + 1}. Skipping."
            )
            continue

        y_list.append(rows[:, 0].astype(int))
        X_list.append(rows[:, 1:])
        found_any = True
        logger.info(f"Loaded {len(rows):4d} samples for '{g['name']}'")

    if not found_any:
        raise FileNotFoundError(
            f"No training CSV files found in {RAW_DATA_DIR}.\n"
            "Please run: python training/collect_data.py"
        )

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(int)
    return X, y


def save_processed_dataset(X: np.ndarray, y: np.ndarray) -> None:
    """Save the combined dataset as a single CSV (optional, for inspection)."""
    ensure_dir(PROCESSED_DATA_PATH.parent)
    combined = np.column_stack([y, X])
    np.savetxt(PROCESSED_DATA_PATH, combined, delimiter=",", fmt="%.6f")
    logger.info(f"Processed dataset saved: {PROCESSED_DATA_PATH}")


# ---------------------------------------------------------------------------
# 2. Validation
# ---------------------------------------------------------------------------

def validate_dataset(X: np.ndarray, y: np.ndarray) -> None:
    """Check dataset integrity and print a summary."""
    assert X.shape[1] == FEATURE_VECTOR_SIZE, (
        f"Feature dimension mismatch: got {X.shape[1]}, expected {FEATURE_VECTOR_SIZE}"
    )
    assert len(X) == len(y), "X and y length mismatch."
    assert not np.isnan(X).any(), "Dataset contains NaN values."
    assert not np.isinf(X).any(), "Dataset contains Inf values."

    print("\n" + "="*60)
    print(" Dataset Summary")
    print("="*60)
    print(f"  Total samples : {len(X)}")
    print(f"  Feature dims  : {X.shape[1]}")
    print(f"  Classes found : {sorted(set(y.tolist()))}")
    print()
    for g in GESTURES:
        count = int((y == g["id"]).sum())
        bar   = "#" * (count // 10)
        print(f"  [{g['id']:2d}] {g['name']:<12} {count:4d} samples  {bar}")
    print("="*60 + "\n")

    unique_classes = set(y.tolist())
    if len(unique_classes) < 2:
        raise ValueError(
            "At least 2 gesture classes are required for training. "
            f"Found: {unique_classes}"
        )


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def train(X: np.ndarray, y: np.ndarray):
    """
    Train a Random Forest classifier and return it along with test data.

    Why Random Forest?
    - Handles small datasets well without overfitting (bagging reduces variance)
    - No feature scaling required (tree-based)
    - Provides calibrated class probabilities via predict_proba
    - Fast inference (< 1 ms per prediction)
    - Naturally handles multi-class problems
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    # --- Train / Test split (stratified to preserve class ratios) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples\n")

    # --- Instantiate model ---
    model = RandomForestClassifier(
        n_estimators=200,       # 200 trees — good balance of accuracy vs speed
        max_depth=None,         # Grow full trees — RF pruning via bagging
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",    # sqrt(63) ≈ 8 features per split — standard setting
        class_weight="balanced", # Handle any class imbalance
        random_state=42,
        n_jobs=-1,              # Use all CPU cores
    )

    # --- 5-fold cross-validation on training set ---
    print("  Running 5-fold cross-validation …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()

    # --- Final training on full training set ---
    print("  Training final model …")
    model.fit(X_train, y_train)
    print("  Training complete.\n")

    # --- Test set evaluation ---
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    # Build class names list ordered by their numeric IDs
    unique_ids   = sorted(set(y.tolist()))
    class_names  = [GESTURE_NAMES[i] if i < len(GESTURE_NAMES) else str(i) for i in unique_ids]

    print("="*60)
    print(" Test Set Results")
    print("="*60)
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)\n")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=unique_ids)
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    header = "         " + "  ".join(f"{n[:4]:>4}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:4d}" for v in row)
        print(f"  {class_names[i][:8]:<8} {row_str}")
    print("="*60 + "\n")

    return model, X_test, y_test, unique_ids, class_names


# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------

def save_model(model, class_names: list[str]) -> None:
    """Save the trained model and metadata to models/gesture_model.pkl."""
    ensure_dir(MODEL_PATH.parent)
    payload = {
        "model":   model,
        "classes": class_names,
    }
    joblib.dump(payload, MODEL_PATH, compress=3)
    logger.info(f"Model saved: {MODEL_PATH}")
    print(f"  [OK] Model saved to: {MODEL_PATH}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "="*60)
    print(" Sign Language Gesture Model Training")
    print("="*60 + "\n")

    # Load
    print("  Loading training data …")
    X, y = load_raw_csvs()
    save_processed_dataset(X, y)
    validate_dataset(X, y)

    # Train
    model, X_test, y_test, unique_ids, class_names = train(X, y)

    # Save
    save_model(model, class_names)

    print("Training pipeline complete!")
    print("Run evaluate_model.py for detailed evaluation.\n")


if __name__ == "__main__":
    main()
