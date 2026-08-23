"""
test_prediction_stabilizer.py
==============================
Unit tests for the PredictionStabilizer temporal smoother.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prediction_stabilizer import PredictionStabilizer


class TestPredictionStabilizer:

    def setup_method(self):
        # Small window for fast testing
        self.stab = PredictionStabilizer(window_size=5, min_agree=4)

    def _fill(self, predictions: list) -> list:
        """Push a sequence of predictions and collect results."""
        results = []
        for p in predictions:
            results.append(self.stab.update(p))
        return results

    # ----- Confirmation -----

    def test_confirms_stable_gesture(self):
        results = self._fill(["Hello"] * 5)
        # Once window is full (5 frames), last result should be confirmed
        assert results[-1] == "Hello"

    def test_does_not_confirm_inconsistent(self):
        results = self._fill(["Hello", "Yes", "Hello", "No", "Yes"])
        # No single label has 4/5 agreement
        assert results[-1] is None

    def test_does_not_confirm_before_window_full(self):
        results = self._fill(["Hello"] * 3)  # only 3 out of 5 frames
        assert all(r is None for r in results)

    def test_confirms_after_window_fills(self):
        results = self._fill(["Hello"] * 5)
        assert results[4] == "Hello"

    # ----- None / No gesture -----

    def test_none_predictions_dont_confirm(self):
        results = self._fill([None] * 5)
        assert all(r is None for r in results)

    def test_mixed_none_and_gesture_blocked(self):
        results = self._fill(["Hello", None, "Hello", None, "Hello"])
        # 3 / 5 — below min_agree=4
        assert results[-1] is None

    # ----- Reset -----

    def test_reset_clears_window(self):
        self._fill(["Hello"] * 5)
        self.stab.reset()
        assert self.stab.current_window == []
        # After reset, we need to refill the window
        results = self._fill(["Hello"] * 3)
        assert all(r is None for r in results)

    # ----- Properties -----

    def test_most_common(self):
        self._fill(["Hello", "Hello", "Yes", "Hello"])
        assert self.stab.most_common == "Hello"

    def test_most_common_empty(self):
        assert self.stab.most_common is None

    def test_current_window_copy(self):
        self._fill(["Hello", "Yes"])
        window = self.stab.current_window
        assert isinstance(window, list)
        assert len(window) == 2

    # ----- Edge: single class with min agreement -----

    def test_just_below_threshold_not_confirmed(self):
        # 3 out of 5 — min_agree is 4 → should not confirm
        results = self._fill(["Hello", "Hello", "Hello", "Yes", "Yes"])
        assert results[-1] is None

    def test_exactly_at_threshold_confirmed(self):
        # 4 out of 5 — exactly meets min_agree=4
        results = self._fill(["Hello", "Hello", "Hello", "Hello", "Yes"])
        assert results[-1] == "Hello"
