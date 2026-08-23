"""
camera.py
=========
Webcam abstraction layer.

Responsibilities:
- Open and release the webcam safely
- Read frames with proper error handling
- Expose camera metadata (width, height, fps)
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

from src.gesture_config import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH
from src.utils import get_logger

logger = get_logger(__name__)


class Camera:
    """
    Thread-safe webcam wrapper.

    Usage:
        cam = Camera()
        cam.start()
        frame = cam.read()   # returns None if no frame available yet
        cam.stop()
    """

    def __init__(
        self,
        index: int = CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the webcam and begin reading frames in a background thread."""
        if self._running:
            logger.warning("Camera already running.")
            return

        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self._index}. "
                "Check that the webcam is connected and not in use by another application."
            )

        # Apply requested resolution and FPS
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)

        # Log actual values the driver accepted
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened — requested {self._width}×{self._height}, "
                    f"actual {actual_w}×{actual_h}")

        self._running = True
        self._error = None
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraThread")
        self._thread.start()
        logger.info("Camera capture thread started.")

    def stop(self) -> None:
        """Stop the capture thread and release the webcam."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._frame = None
        logger.info("Camera stopped and released.")

    def read(self) -> Optional[np.ndarray]:
        """
        Return the latest frame (BGR), or None if not available.
        Safe to call from any thread.
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    @property
    def width(self) -> int:
        if self._cap:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return self._width

    @property
    def height(self) -> int:
        if self._cap:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self._height

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Background thread: continuously grab frames from the webcam."""
        consecutive_failures = 0
        max_failures = 10

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._error = "Camera disconnected unexpectedly."
                logger.error(self._error)
                self._running = False
                break

            ret, frame = self._cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                logger.warning(f"Frame read failed ({consecutive_failures}/{max_failures})")
                if consecutive_failures >= max_failures:
                    self._error = "Too many consecutive frame read failures. Camera may be disconnected."
                    logger.error(self._error)
                    self._running = False
                    break
                continue

            consecutive_failures = 0
            # Flip horizontally for a mirror-view (more natural for the user)
            frame = cv2.flip(frame, 1)

            with self._lock:
                self._frame = frame
