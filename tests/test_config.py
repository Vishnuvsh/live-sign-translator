"""
test_config.py
==============
Tests that the configuration module is self-consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gesture_config import (
    CONFIDENCE_THRESHOLD,
    FEATURE_VECTOR_SIZE,
    GESTURE_ID_MAP,
    GESTURE_NAMES,
    GESTURE_SPOKEN,
    GESTURES,
    NUM_GESTURES,
    NUM_LANDMARKS,
    COORDS_PER_LANDMARK,
    STABILITY_MIN_AGREE,
    STABILITY_WINDOW,
    WORD_COOLDOWN_SECONDS,
)


class TestConfig:

    def test_feature_vector_size_correct(self):
        """FEATURE_VECTOR_SIZE must equal NUM_LANDMARKS × COORDS_PER_LANDMARK."""
        assert FEATURE_VECTOR_SIZE == NUM_LANDMARKS * COORDS_PER_LANDMARK == 63

    def test_num_gestures_matches_list(self):
        assert NUM_GESTURES == len(GESTURES)
        assert NUM_GESTURES == len(GESTURE_NAMES)

    def test_gesture_ids_are_unique(self):
        ids = [g["id"] for g in GESTURES]
        assert len(ids) == len(set(ids)), "Gesture IDs are not unique."

    def test_gesture_names_are_unique(self):
        names = [g["name"] for g in GESTURES]
        assert len(names) == len(set(names)), "Gesture names are not unique."

    def test_spoken_words_present_for_all_gestures(self):
        for g in GESTURES:
            assert "spoken" in g, f"Gesture {g['name']} missing 'spoken' key."
            assert g["spoken"], f"Gesture {g['name']} has empty 'spoken' value."

    def test_derived_maps_consistent(self):
        """GESTURE_ID_MAP and GESTURE_SPOKEN must be consistent with GESTURES."""
        for g in GESTURES:
            assert GESTURE_ID_MAP[g["name"]] == g["id"]
            assert GESTURE_SPOKEN[g["id"]] == g["spoken"]

    def test_confidence_threshold_in_range(self):
        assert 0.0 < CONFIDENCE_THRESHOLD <= 1.0

    def test_stability_window_larger_than_min_agree(self):
        assert STABILITY_WINDOW >= STABILITY_MIN_AGREE, (
            "STABILITY_WINDOW must be >= STABILITY_MIN_AGREE."
        )

    def test_cooldown_positive(self):
        assert WORD_COOLDOWN_SECONDS > 0

    def test_ten_gestures_defined(self):
        """Prototype must define exactly 10 gestures."""
        assert NUM_GESTURES == 10, (
            f"Expected 10 gestures, found {NUM_GESTURES}."
        )

    def test_gesture_names_non_empty(self):
        for name in GESTURE_NAMES:
            assert name and isinstance(name, str)
