"""
hand_detector.py
================
MediaPipe Hands wrapper.

Responsibilities:
- Initialise MediaPipe Hands with configured parameters
- Process a frame and return raw landmark data
- Draw landmark annotations onto the frame
- Compute the bounding box of the detected hand
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from src.gesture_config import (
    DETECTION_CONFIDENCE,
    MAX_NUM_HANDS,
    TRACKING_CONFIDENCE,
)
from src.utils import get_logger

logger = get_logger(__name__)

# MediaPipe drawing utilities (module-level singletons — cheap to create once)
_mp_hands    = mp.solutions.hands
_mp_drawing  = mp.solutions.drawing_utils
_mp_styles   = mp.solutions.drawing_styles


@dataclass
class HandLandmarks:
    """
    Structured result from a single hand detection.

    Attributes:
        landmarks: List of 21 NormalizedLandmark objects (x, y, z ∈ [0, 1])
        handedness: 'Left' or 'Right'
        bbox: Pixel bounding box as (x_min, y_min, x_max, y_max)
    """
    landmarks: list   # List[mediapipe.framework.formats.landmark_pb2.NormalizedLandmark]
    handedness: str
    bbox: tuple[int, int, int, int]  # x_min, y_min, x_max, y_max


class HandDetector:
    """
    Wraps MediaPipe Hands for single-hand landmark detection.

    Example:
        detector = HandDetector()
        result = detector.detect(bgr_frame)
        if result:
            annotated_frame = detector.draw(bgr_frame, result)
    """

    def __init__(
        self,
        max_num_hands: int            = MAX_NUM_HANDS,
        detection_confidence: float   = DETECTION_CONFIDENCE,
        tracking_confidence: float    = TRACKING_CONFIDENCE,
    ) -> None:
        self._hands = _mp_hands.Hands(
            static_image_mode=False,         # Video mode: tracking after initial detection
            max_num_hands=max_num_hands,
            model_complexity=0,              # 0 is faster, 1 is more accurate (reduces lag)
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        logger.info(
            f"HandDetector initialised — max_hands={max_num_hands}, "
            f"det={detection_confidence}, track={tracking_confidence}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, bgr_frame: np.ndarray) -> Optional[HandLandmarks]:
        """
        Run MediaPipe hand detection on a BGR frame.

        Returns the first detected hand's HandLandmarks, or None if no hand
        is found. Processes in RGB internally (MediaPipe requirement).
        """
        if bgr_frame is None:
            return None

        h, w = bgr_frame.shape[:2]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        # Mark as not writeable to pass by reference (performance optimisation)
        rgb_frame.flags.writeable = False
        results = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None

        # Use only the first detected hand
        hand_lm = results.multi_hand_landmarks[0]

        # Determine handedness label
        handedness = "Unknown"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        # Compute pixel bounding box from normalised landmark coordinates
        xs = [lm.x * w for lm in hand_lm.landmark]
        ys = [lm.y * h for lm in hand_lm.landmark]
        padding = 20
        x_min = max(0, int(min(xs)) - padding)
        y_min = max(0, int(min(ys)) - padding)
        x_max = min(w, int(max(xs)) + padding)
        y_max = min(h, int(max(ys)) + padding)

        return HandLandmarks(
            landmarks=hand_lm.landmark,
            handedness=handedness,
            bbox=(x_min, y_min, x_max, y_max),
        )

    def draw(self, bgr_frame: np.ndarray, hand: HandLandmarks) -> np.ndarray:
        """
        Draw landmark skeleton and bounding box on a copy of the frame.
        Returns the annotated frame (does not modify in place).
        """
        frame = bgr_frame.copy()

        # Build a NormalizedLandmarkList proto for the drawing utility
        proto = _build_landmark_list_proto(hand.landmarks)

        _mp_drawing.draw_landmarks(
            frame,
            proto,
            _mp_hands.HAND_CONNECTIONS,
            _mp_styles.get_default_hand_landmarks_style(),
            _mp_styles.get_default_hand_connections_style(),
        )

        # Draw bounding box
        x_min, y_min, x_max, y_max = hand.bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 120), 2)

        # Label handedness
        cv2.putText(
            frame,
            hand.handedness,
            (x_min, y_min - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 120),
            1,
            cv2.LINE_AA,
        )
        return frame

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()
        logger.info("HandDetector closed.")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _build_landmark_list_proto(landmarks):
    """
    Re-wrap a list of NormalizedLandmark objects into the proto format
    expected by mp_drawing.draw_landmarks().
    """
    landmark_list = _mp_hands.HandLandmark  # access the enum (not needed directly)
    proto = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks:
        new_lm = proto.landmark.add()
        new_lm.x = lm.x
        new_lm.y = lm.y
        new_lm.z = lm.z
    return proto
