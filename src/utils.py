"""
utils.py
========
Shared utility helpers for the Sign Language Translator.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Image / frame helpers
# ---------------------------------------------------------------------------

def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR frame to RGB (required by MediaPipe)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert RGB frame back to OpenCV BGR."""
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a frame to the given dimensions."""
    return cv2.resize(frame, (width, height))


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_scale: float = 0.8,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 120, 255),
    thickness: int = 2,
    padding: int = 6,
) -> np.ndarray:
    """Draw text with a filled rectangular background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x - padding, y - h - padding),
        (x + w + padding, y + baseline + padding),
        bg_color,
        -1,
    )
    # Draw text on top
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return frame


def draw_bounding_box(
    frame: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    color: tuple[int, int, int] = (0, 255, 120),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a bounding box around a detected hand."""
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
    return frame


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it does not exist. Returns path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_file_exists(path: Path, label: str = "File") -> None:
    """Raise FileNotFoundError with a helpful message if path does not exist."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found: {path}\n"
            "Please ensure you have run the required setup steps."
        )


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------

class FPSCounter:
    """Rolling-average FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._times: list[float] = []
        self._window = window

    def tick(self) -> float:
        """Record a frame tick and return current FPS."""
        now = time.time()
        self._times.append(now)
        # Keep only the last `window` timestamps
        if len(self._times) > self._window:
            self._times = self._times[-self._window:]
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0
