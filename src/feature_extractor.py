"""
feature_extractor.py
====================
Converts raw MediaPipe hand landmarks into a normalised, position-independent
feature vector suitable for ML classification.

Normalisation Strategy
----------------------
Raw MediaPipe landmarks are in normalised image coordinates (x, y ∈ [0, 1], z
is a relative depth value).  If we used these directly, the feature vector
would change depending on where in the frame the hand is and how large it
appears — making the classifier less robust.

We apply the following steps:

1. **Translate to wrist origin**
   Subtract wrist coordinates (landmark 0) from all landmark coordinates.
   This makes the feature vector hand-position independent.

2. **Scale normalisation**
   Divide by the maximum absolute coordinate value across the entire hand
   (after translation). This maps all values into the range [-1, 1] and
   makes recognition less sensitive to hand distance from the camera.

3. **Flatten**
   Concatenate all 21 × (x, y, z) values into a 1-D NumPy array of length 63.

The resulting 63-dimensional vector is stable, compact, and directly usable
as input features for the Random Forest classifier.
"""

from __future__ import annotations

import numpy as np

from src.gesture_config import COORDS_PER_LANDMARK, FEATURE_VECTOR_SIZE, NUM_LANDMARKS
from src.utils import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """
    Converts a list of 21 MediaPipe NormalizedLandmark objects into a
    normalised 63-dimensional NumPy feature vector.
    """

    def extract(self, landmarks) -> np.ndarray | None:
        """
        Parameters
        ----------
        landmarks : list of mediapipe NormalizedLandmark
            Must contain exactly NUM_LANDMARKS (21) entries.

        Returns
        -------
        np.ndarray of shape (FEATURE_VECTOR_SIZE,) = (63,), dtype float32
        None if landmarks are invalid or extraction fails.
        """
        if landmarks is None:
            return None

        if len(landmarks) != NUM_LANDMARKS:
            logger.warning(
                f"Expected {NUM_LANDMARKS} landmarks, got {len(landmarks)}. Skipping."
            )
            return None

        try:
            # Step 1: Extract raw (x, y, z) into a (21, 3) array
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks],
                dtype=np.float32,
            )  # shape: (21, 3)

            # Step 2: Translate — subtract wrist (landmark 0) coordinates
            # After this, the wrist is at origin (0, 0, 0)
            wrist = coords[0].copy()
            coords -= wrist  # shape: (21, 3)

            # Step 3: Scale — divide by max absolute value across all coords
            # This normalises hand size so distance-from-camera doesn't matter
            max_abs = np.max(np.abs(coords))
            if max_abs > 1e-6:
                coords /= max_abs
            else:
                # Edge case: all landmarks collapsed to a single point
                logger.debug("All landmarks at same position — returning zero vector.")
                return np.zeros(FEATURE_VECTOR_SIZE, dtype=np.float32)

            # Step 4: Flatten (21, 3) → (63,)
            feature_vector = coords.flatten()

            assert feature_vector.shape == (FEATURE_VECTOR_SIZE,), (
                f"Feature vector shape mismatch: {feature_vector.shape}"
            )

            return feature_vector

        except Exception as exc:
            logger.error(f"Feature extraction failed: {exc}")
            return None

    def extract_from_raw(self, raw_coords: list[tuple[float, float, float]]) -> np.ndarray | None:
        """
        Alternative entry point: accepts a list of (x, y, z) tuples directly.
        Useful for reprocessing saved data during training.
        """
        if len(raw_coords) != NUM_LANDMARKS:
            return None

        class _FakeLandmark:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        fake_landmarks = [_FakeLandmark(*c) for c in raw_coords]
        return self.extract(fake_landmarks)
