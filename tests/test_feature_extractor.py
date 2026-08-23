"""
test_feature_extractor.py
=========================
Unit tests for feature extraction and normalization logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_extractor import FeatureExtractor
from src.gesture_config    import FEATURE_VECTOR_SIZE, NUM_LANDMARKS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeLandmark:
    """Minimal landmark object mimicking mediapipe NormalizedLandmark."""
    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z


def make_fake_landmarks(offset_x=0.0, offset_y=0.0, scale=1.0) -> list[_FakeLandmark]:
    """Generate 21 fake landmarks with deterministic positions."""
    rng = np.random.default_rng(seed=42)
    coords = rng.uniform(0.1, 0.9, size=(NUM_LANDMARKS, 3)) * scale
    coords[:, 0] += offset_x
    coords[:, 1] += offset_y
    return [_FakeLandmark(*c) for c in coords]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeatureExtractor:

    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_output_shape(self):
        """Feature vector must be exactly FEATURE_VECTOR_SIZE = 63."""
        lms = make_fake_landmarks()
        fv  = self.extractor.extract(lms)
        assert fv is not None
        assert fv.shape == (FEATURE_VECTOR_SIZE,), (
            f"Expected shape ({FEATURE_VECTOR_SIZE},), got {fv.shape}"
        )

    def test_output_dtype(self):
        """Feature vector must be float32."""
        lms = make_fake_landmarks()
        fv  = self.extractor.extract(lms)
        assert fv.dtype == np.float32

    def test_values_in_range(self):
        """All feature values must be in [-1, 1] after normalization."""
        lms = make_fake_landmarks()
        fv  = self.extractor.extract(lms)
        assert fv is not None
        assert np.all(fv >= -1.0 - 1e-5), f"Min value {fv.min()} below -1"
        assert np.all(fv <= 1.0 + 1e-5), f"Max value {fv.max()} above 1"

    def test_position_invariance(self):
        """
        Shifting all landmarks by the same offset must produce the same
        feature vector (translation invariance via wrist subtraction).
        """
        lms_a = make_fake_landmarks(offset_x=0.0, offset_y=0.0)
        lms_b = make_fake_landmarks(offset_x=0.3, offset_y=0.2)  # shifted

        fv_a = self.extractor.extract(lms_a)
        fv_b = self.extractor.extract(lms_b)

        assert fv_a is not None and fv_b is not None
        np.testing.assert_allclose(
            fv_a, fv_b, atol=1e-5,
            err_msg="Feature vectors differ after positional shift — "
                    "translation invariance broken."
        )

    def test_wrist_is_zero(self):
        """After extraction, wrist (landmark 0) coordinates should be ~0."""
        lms = make_fake_landmarks()
        fv  = self.extractor.extract(lms)
        assert fv is not None
        # First 3 values correspond to wrist (x, y, z) after translation
        wrist_x, wrist_y, wrist_z = fv[0], fv[1], fv[2]
        assert abs(wrist_x) < 1e-5, f"Wrist x not zero: {wrist_x}"
        assert abs(wrist_y) < 1e-5, f"Wrist y not zero: {wrist_y}"
        assert abs(wrist_z) < 1e-5, f"Wrist z not zero: {wrist_z}"

    def test_none_landmarks_returns_none(self):
        """Passing None should return None, not raise an exception."""
        fv = self.extractor.extract(None)
        assert fv is None

    def test_wrong_landmark_count_returns_none(self):
        """Passing fewer than 21 landmarks should return None."""
        lms = make_fake_landmarks()[:10]  # only 10 landmarks
        fv  = self.extractor.extract(lms)
        assert fv is None

    def test_extract_from_raw(self):
        """extract_from_raw() should match extract() given the same data."""
        lms = make_fake_landmarks()
        raw = [(lm.x, lm.y, lm.z) for lm in lms]

        fv_from_lm  = self.extractor.extract(lms)
        fv_from_raw = self.extractor.extract_from_raw(raw)

        assert fv_from_lm  is not None
        assert fv_from_raw is not None
        np.testing.assert_allclose(fv_from_lm, fv_from_raw, atol=1e-6)
