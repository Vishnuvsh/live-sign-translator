"""
test_gesture_classifier.py
==========================
Unit tests for the GestureClassifier (model-free path).
Tests confidence filtering and the Prediction dataclass without
requiring an actual trained model on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gesture_classifier import GestureClassifier, Prediction
from src.gesture_config     import CONFIDENCE_THRESHOLD, FEATURE_VECTOR_SIZE, GESTURE_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_model(predicted_class: int, confidence: float, num_classes: int = 10):
    """Return a mock sklearn model that always predicts the given class and confidence."""
    model = MagicMock()
    proba = np.zeros(num_classes, dtype=np.float64)
    proba[predicted_class] = confidence
    # Distribute remaining probability
    remaining = (1.0 - confidence) / max(num_classes - 1, 1)
    for i in range(num_classes):
        if i != predicted_class:
            proba[i] = remaining
    model.predict_proba.return_value = np.array([proba])
    return model


def make_feature_vector(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, FEATURE_VECTOR_SIZE).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPredictionDataclass:

    def test_accepted_when_above_threshold(self):
        pred = Prediction(
            gesture_name="Hello",
            confidence=0.95,
            accepted=True,
        )
        assert pred.accepted is True

    def test_rejected_when_below_threshold(self):
        pred = Prediction(
            gesture_name="Hello",
            confidence=0.50,
            accepted=False,
        )
        assert pred.accepted is False


class TestGestureClassifier:

    def _make_classifier(self, predicted_class=0, confidence=0.90):
        clf = GestureClassifier(confidence_threshold=CONFIDENCE_THRESHOLD)
        clf._model   = make_mock_model(predicted_class, confidence)
        clf._classes = GESTURE_NAMES
        return clf

    def test_predict_returns_prediction(self):
        clf = self._make_classifier(predicted_class=0, confidence=0.92)
        fv  = make_feature_vector()
        pred = clf.predict(fv)
        assert pred is not None
        assert isinstance(pred, Prediction)

    def test_predict_correct_gesture_name(self):
        clf = self._make_classifier(predicted_class=0, confidence=0.92)
        fv  = make_feature_vector()
        pred = clf.predict(fv)
        assert pred.gesture_name == GESTURE_NAMES[0]  # "Hello"

    def test_prediction_accepted_above_threshold(self):
        clf = self._make_classifier(predicted_class=2, confidence=0.95)
        fv  = make_feature_vector()
        pred = clf.predict(fv)
        assert pred.accepted is True

    def test_prediction_rejected_below_threshold(self):
        clf = self._make_classifier(predicted_class=2, confidence=0.40)
        fv  = make_feature_vector()
        pred = clf.predict(fv)
        assert pred.accepted is False

    def test_predict_none_when_model_not_loaded(self):
        clf = GestureClassifier()
        # _model is None (not loaded)
        fv  = make_feature_vector()
        pred = clf.predict(fv)
        assert pred is None

    def test_predict_none_on_wrong_feature_size(self):
        clf = self._make_classifier()
        wrong_fv = np.zeros(10, dtype=np.float32)  # wrong shape
        pred = clf.predict(wrong_fv)
        assert pred is None

    def test_predict_none_on_none_input(self):
        clf = self._make_classifier()
        pred = clf.predict(None)
        assert pred is None

    def test_is_loaded_false_before_load(self):
        clf = GestureClassifier()
        assert clf.is_loaded is False

    def test_is_loaded_true_after_mock_load(self):
        clf = self._make_classifier()
        assert clf.is_loaded is True

    def test_confidence_threshold_property(self):
        clf = GestureClassifier(confidence_threshold=0.75)
        assert clf.confidence_threshold == 0.75

    def test_load_raises_on_missing_file(self):
        clf = GestureClassifier(model_path=Path("/nonexistent/model.pkl"))
        with pytest.raises(FileNotFoundError):
            clf.load()
