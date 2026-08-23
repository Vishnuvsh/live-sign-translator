"""
prediction_stabilizer.py
========================
Temporal smoothing for gesture predictions.

Problem
-------
The classifier runs on every frame (~30 fps). A single misclassified frame
or brief occlusion would cause flickering or incorrect confirmations.

Solution
--------
Maintain a rolling window of the last N predictions. Only confirm a gesture
when the same label appears at least MIN_AGREE times within that window.
This is a majority-vote style filter applied over time.

Example (STABILITY_WINDOW=10, STABILITY_MIN_AGREE=8):

  Window: [Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Yes]
  Count of "Hello" = 9 → 9 >= 8 → CONFIRMED: Hello

  Window: [Hello, Yes, Hello, None, Yes, Hello, None, Yes, Hello, Yes]
  No single label reaches 8 → NOT CONFIRMED
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Optional

from src.gesture_config import STABILITY_MIN_AGREE, STABILITY_WINDOW
from src.utils import get_logger

logger = get_logger(__name__)

# Sentinel value used when no gesture was detected in a frame
_NO_GESTURE = "__NONE__"


class PredictionStabilizer:
    """
    Maintains a rolling window of gesture predictions and emits a confirmed
    gesture only when the same prediction dominates the window.

    Usage:
        stab = PredictionStabilizer()
        confirmed = stab.update("Hello")   # returns "Hello" if confirmed, else None
        confirmed = stab.update(None)      # None for frames with no detection
    """

    def __init__(
        self,
        window_size: int  = STABILITY_WINDOW,
        min_agree: int    = STABILITY_MIN_AGREE,
    ) -> None:
        self._window:   deque[str]  = deque(maxlen=window_size)
        self._min_agree: int        = min_agree
        self._window_size: int      = window_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, prediction: Optional[str]) -> Optional[str]:
        """
        Push the latest prediction into the window and check for stability.

        Parameters
        ----------
        prediction : str | None
            The gesture name from the latest frame, or None / 'Unknown Gesture'
            if no valid detection.

        Returns
        -------
        str | None
            The confirmed gesture name if stable, otherwise None.
        """
        label = prediction if prediction else _NO_GESTURE
        self._window.append(label)

        if len(self._window) < self._window_size:
            # Not enough history yet — wait for the window to fill
            return None

        return self._get_stable_prediction()

    def reset(self) -> None:
        """Clear the prediction window (e.g. after a word is confirmed)."""
        self._window.clear()

    @property
    def current_window(self) -> list[str]:
        """Returns a copy of the current window contents (for debugging / testing)."""
        return list(self._window)

    @property
    def most_common(self) -> Optional[str]:
        """Returns the most common non-None label in the window without confirming."""
        if not self._window:
            return None
        counts = Counter(lbl for lbl in self._window if lbl != _NO_GESTURE)
        if not counts:
            return None
        top, _ = counts.most_common(1)[0]
        return top

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_stable_prediction(self) -> Optional[str]:
        """
        Count occurrences of each label in the window.
        Return the label if it meets the minimum agreement threshold,
        excluding the NO_GESTURE sentinel.
        """
        counts = Counter(lbl for lbl in self._window if lbl != _NO_GESTURE)
        if not counts:
            return None

        top_label, top_count = counts.most_common(1)[0]

        if top_count >= self._min_agree:
            logger.debug(f"Gesture CONFIRMED: {top_label} ({top_count}/{self._window_size})")
            return top_label

        return None
