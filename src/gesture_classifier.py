"""
gesture_classifier.py
=====================
Loads the trained Random Forest model and performs gesture prediction.

Responsibilities:
- Load the model from disk (with clear error if missing)
- Predict the gesture class and confidence from a feature vector
- Apply the confidence threshold (returns None if below threshold)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from src.gesture_config import (
    CONFIDENCE_THRESHOLD,
    FEATURE_VECTOR_SIZE,
    GESTURE_NAMES,
    MODEL_PATH,
)
from src.utils import get_logger, validate_file_exists

logger = get_logger(__name__)


@dataclass
class Prediction:
    """
    Result of a single gesture classification.

    Attributes:
        gesture_name : Display name of the predicted gesture (e.g. 'Hello')
        confidence   : Classifier's probability for this prediction (0–1)
        accepted     : True if confidence ≥ CONFIDENCE_THRESHOLD
    """
    gesture_name: str
    confidence: float
    accepted: bool


class GestureClassifier:
    """
    Wraps the trained scikit-learn RandomForestClassifier.

    Usage:
        clf = GestureClassifier()
        clf.load()
        pred = clf.predict(feature_vector)
        if pred and pred.accepted:
            print(pred.gesture_name)
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self._model_path = model_path
        self._threshold = confidence_threshold
        self._model = None
        self._classes: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the trained model from disk.

        Raises:
            FileNotFoundError: if the .pkl file does not exist.
            ValueError: if the loaded object is not a valid classifier.
        """
        validate_file_exists(self._model_path, label="Gesture model")

        data = joblib.load(self._model_path)

        # Support saving either a raw model or a dict with metadata
        if isinstance(data, dict):
            self._model   = data["model"]
            self._classes = data.get("classes", GESTURE_NAMES)
        else:
            self._model   = data
            self._classes = GESTURE_NAMES

        logger.info(
            f"Model loaded from {self._model_path} | "
            f"classes={self._classes}"
        )

    def predict(self, feature_vector: np.ndarray) -> Optional[Prediction]:
        """
        Predict the gesture from a 63-dimensional feature vector.

        Parameters
        ----------
        feature_vector : np.ndarray, shape (63,)

        Returns
        -------
        Prediction dataclass, or None if the model is not loaded or the
        feature vector is invalid.
        """
        if self._model is None:
            logger.warning("Model not loaded. Call load() first.")
            return None

        if feature_vector is None or feature_vector.shape != (FEATURE_VECTOR_SIZE,):
            return None

        try:
            # Reshape to (1, 63) for scikit-learn
            X = feature_vector.reshape(1, -1)

            # Get class probabilities
            proba = self._model.predict_proba(X)[0]  # shape: (num_classes,)
            class_index = int(np.argmax(proba))
            confidence  = float(proba[class_index])

            # Map index to gesture name via stored classes list
            if class_index < len(self._classes):
                gesture_name = self._classes[class_index]
            else:
                logger.error(f"Predicted class index {class_index} out of range.")
                return None

            accepted = confidence >= self._threshold

            return Prediction(
                gesture_name=gesture_name,
                confidence=confidence,
                accepted=accepted,
            )

        except Exception as exc:
            logger.error(f"Prediction failed: {exc}")
            return None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def confidence_threshold(self) -> float:
        return self._threshold
