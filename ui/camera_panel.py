"""
camera_panel.py
===============
CustomTkinter widget that displays the live webcam feed with overlays.

Responsibilities:
- Display the camera frame (converted to Tkinter-compatible format)
- Overlay gesture name, confidence, and FPS
- Refresh itself using after() loop (non-blocking)
"""

from __future__ import annotations

from typing import Optional, Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import customtkinter as ctk
except ImportError:
    raise ImportError("Please install customtkinter: pip install customtkinter")

from src.gesture_config import CAMERA_WIDTH, CAMERA_HEIGHT, UI_FONT_FAMILY


class CameraPanel(ctk.CTkFrame):
    """
    A CTkFrame that renders live camera frames with landmark overlays.

    The parent is responsible for calling update_frame() with pre-annotated
    frames so that drawing logic stays in the pipeline, not in the UI.
    """

    PANEL_W = 640
    PANEL_H = 480

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self._photo_image: Optional[ImageTk.PhotoImage] = None

        # Canvas for displaying the camera frame
        self._canvas = ctk.CTkCanvas(
            self,
            width=self.PANEL_W,
            height=self.PANEL_H,
            bg="#0d0d0d",
            highlightthickness=0,
        )
        self._canvas.pack(fill="both", expand=True)

        # Placeholder text shown before camera starts
        self._placeholder_id = self._canvas.create_text(
            self.PANEL_W // 2,
            self.PANEL_H // 2,
            text="Camera not started\n\nClick  'Start Camera'  to begin",
            fill="#555555",
            font=(UI_FONT_FAMILY, 16),
            justify="center",
        )
        self._image_id: Optional[int] = None

        # Status labels overlaid below the feed
        self._gesture_var    = ctk.StringVar(value="—")
        self._confidence_var = ctk.StringVar(value="—")
        self._status_var     = ctk.StringVar(value="WAITING")
        self._fps_var        = ctk.StringVar(value="0 FPS")

        self._build_overlay_labels()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_frame(
        self,
        bgr_frame: np.ndarray,
        gesture_name: str = "—",
        confidence: float = 0.0,
        status: str = "WAITING",
        fps: float = 0.0,
    ) -> None:
        """
        Update the displayed frame and overlay info.
        Must be called from the main thread (or via root.after).
        """
        # Convert BGR → RGB → PIL → PhotoImage
        rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb)
        pil  = pil.resize((self.PANEL_W, self.PANEL_H), Image.LANCZOS)
        self._photo_image = ImageTk.PhotoImage(image=pil)

        # Remove placeholder text on first real frame
        if self._placeholder_id is not None:
            self._canvas.delete(self._placeholder_id)
            self._placeholder_id = None

        # Draw (or update) the image on canvas
        if self._image_id is None:
            self._image_id = self._canvas.create_image(0, 0, anchor="nw", image=self._photo_image)
        else:
            self._canvas.itemconfig(self._image_id, image=self._photo_image)

        # Update overlay labels
        self._gesture_var.set(gesture_name.upper())
        self._confidence_var.set(f"{confidence * 100:.0f}%")
        self._status_var.set(status)
        self._fps_var.set(f"{fps:.0f} FPS")

        # Status colour
        status_colour = {
            "CONFIRMED":  "#00e676",
            "DETECTING":  "#ffab40",
            "NO HAND":    "#ef5350",
            "WAITING":    "#757575",
            "LOW CONF":   "#ff7043",
        }.get(status, "#ffffff")
        self._status_label.configure(text_color=status_colour)

    def show_placeholder(self) -> None:
        """Revert to the 'camera not started' placeholder."""
        if self._image_id is not None:
            self._canvas.delete(self._image_id)
            self._image_id = None
        self._photo_image = None
        self._placeholder_id = self._canvas.create_text(
            self.PANEL_W // 2,
            self.PANEL_H // 2,
            text="Camera not started\n\nClick  'Start Camera'  to begin",
            fill="#555555",
            font=(UI_FONT_FAMILY, 16),
            justify="center",
        )
        self._gesture_var.set("—")
        self._confidence_var.set("—")
        self._status_var.set("WAITING")
        self._fps_var.set("0 FPS")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_overlay_labels(self) -> None:
        """Build the gesture / confidence / status / fps labels below the canvas."""
        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        info_frame.pack(fill="x", padx=8, pady=(4, 6))

        # Gesture label (large)
        ctk.CTkLabel(
            info_frame, text="GESTURE", font=(UI_FONT_FAMILY, 10), text_color="#888888"
        ).grid(row=0, column=0, sticky="w", padx=(0, 4))

        ctk.CTkLabel(
            info_frame,
            textvariable=self._gesture_var,
            font=(UI_FONT_FAMILY, 22, "bold"),
            text_color="#ffffff",
        ).grid(row=1, column=0, sticky="w")

        # Confidence
        ctk.CTkLabel(
            info_frame, text="CONFIDENCE", font=(UI_FONT_FAMILY, 10), text_color="#888888"
        ).grid(row=0, column=1, padx=20, sticky="w")
        ctk.CTkLabel(
            info_frame,
            textvariable=self._confidence_var,
            font=(UI_FONT_FAMILY, 22, "bold"),
            text_color="#64b5f6",
        ).grid(row=1, column=1, padx=20, sticky="w")

        # Status
        ctk.CTkLabel(
            info_frame, text="STATUS", font=(UI_FONT_FAMILY, 10), text_color="#888888"
        ).grid(row=0, column=2, padx=20, sticky="w")
        self._status_label = ctk.CTkLabel(
            info_frame,
            textvariable=self._status_var,
            font=(UI_FONT_FAMILY, 22, "bold"),
            text_color="#00e676",
        )
        self._status_label.grid(row=1, column=2, padx=20, sticky="w")

        # FPS (small, right-aligned)
        ctk.CTkLabel(
            info_frame,
            textvariable=self._fps_var,
            font=(UI_FONT_FAMILY, 11),
            text_color="#555555",
        ).grid(row=0, column=3, rowspan=2, padx=(40, 0), sticky="e")

        info_frame.columnconfigure(3, weight=1)
