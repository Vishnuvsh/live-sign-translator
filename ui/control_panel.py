"""
control_panel.py
================
Right-side control panel widget.

Contains:
- Sentence display area
- Control buttons (Start/Stop Camera, Speak, Clear, Undo, Exit)
- System status section (Camera, Model, Hand)
- Recent word history
"""

from __future__ import annotations

from typing import Callable, Optional

try:
    import customtkinter as ctk
except ImportError:
    raise ImportError("Please install customtkinter: pip install customtkinter")

from src.gesture_config import UI_FONT_FAMILY


class ControlPanel(ctk.CTkFrame):
    """
    The right-hand side panel of the application.

    All button callbacks are injected by MainWindow to keep UI logic
    decoupled from application logic.
    """

    def __init__(
        self,
        parent,
        on_start_camera:    Callable,
        on_stop_camera:     Callable,
        on_speak_sentence:  Callable,
        on_clear_sentence:  Callable,
        on_clear_last_word: Callable,
        on_exit:            Callable,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)

        self._on_start         = on_start_camera
        self._on_stop          = on_stop_camera
        self._on_speak         = on_speak_sentence
        self._on_clear         = on_clear_sentence
        self._on_clear_last    = on_clear_last_word
        self._on_exit          = on_exit

        # String vars
        self._sentence_var     = ctk.StringVar(value="")
        self._cam_status_var   = ctk.StringVar(value="⬤  Disconnected")
        self._model_status_var = ctk.StringVar(value="⬤  Not Loaded")
        self._hand_status_var  = ctk.StringVar(value="⬤  Not Detected")

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API — called by MainWindow to update state
    # ------------------------------------------------------------------

    def set_sentence(self, sentence: str) -> None:
        self._sentence_var.set(sentence if sentence else "")
        self._sentence_display.configure(state="normal")
        self._sentence_display.delete("1.0", "end")
        self._sentence_display.insert("1.0", sentence)
        self._sentence_display.configure(state="disabled")

    def set_camera_status(self, connected: bool) -> None:
        if connected:
            self._cam_status_var.set("⬤  Connected")
            self._cam_status_label.configure(text_color="#00e676")
            self._start_btn.configure(state="disabled")
            self._stop_btn.configure(state="normal")
        else:
            self._cam_status_var.set("⬤  Disconnected")
            self._cam_status_label.configure(text_color="#ef5350")
            self._start_btn.configure(state="normal")
            self._stop_btn.configure(state="disabled")

    def set_model_status(self, loaded: bool) -> None:
        if loaded:
            self._model_status_var.set("⬤  Loaded")
            self._model_status_label.configure(text_color="#00e676")
        else:
            self._model_status_var.set("⬤  Not Loaded")
            self._model_status_label.configure(text_color="#ef5350")

    def set_hand_status(self, detected: bool) -> None:
        if detected:
            self._hand_status_var.set("⬤  Detected")
            self._hand_status_label.configure(text_color="#00e676")
        else:
            self._hand_status_var.set("⬤  Not Detected")
            self._hand_status_label.configure(text_color="#757575")

    def show_error(self, message: str) -> None:
        """Flash an error message in the status area."""
        self._error_label.configure(text=f"⚠ {message}", text_color="#ff5252")
        self.after(5000, lambda: self._error_label.configure(text=""))

    def show_info(self, message: str) -> None:
        self._error_label.configure(text=f"ℹ {message}", text_color="#64b5f6")
        self.after(3000, lambda: self._error_label.configure(text=""))

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.configure(fg_color="#1a1a2e", corner_radius=12)

        # ----- Title -----
        ctk.CTkLabel(
            self,
            text="AI Sign Language Translator",
            font=(UI_FONT_FAMILY, 16, "bold"),
            text_color="#e0e0e0",
        ).pack(pady=(18, 4))

        ctk.CTkLabel(
            self,
            text="Accessibility · Offline · Real-Time",
            font=(UI_FONT_FAMILY, 10),
            text_color="#555577",
        ).pack(pady=(0, 14))

        self._add_divider()

        # ----- Sentence area -----
        ctk.CTkLabel(
            self,
            text="RECOGNISED SENTENCE",
            font=(UI_FONT_FAMILY, 10, "bold"),
            text_color="#888899",
        ).pack(anchor="w", padx=16, pady=(10, 4))

        sentence_frame = ctk.CTkFrame(self, fg_color="#0d0d1a", corner_radius=8)
        sentence_frame.pack(fill="x", padx=16, pady=(0, 8))

        self._sentence_display = ctk.CTkTextbox(
            sentence_frame,
            height=90,
            font=(UI_FONT_FAMILY, 18, "bold"),
            text_color="#ffffff",
            fg_color="#0d0d1a",
            wrap="word",
            state="disabled",
        )
        self._sentence_display.pack(fill="both", expand=True, padx=8, pady=8)

        # ----- Word action buttons -----
        word_btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        word_btn_frame.pack(fill="x", padx=16, pady=(0, 8))

        ctk.CTkButton(
            word_btn_frame,
            text="🔊  Speak Sentence",
            command=self._on_speak,
            fg_color="#1565c0",
            hover_color="#1976d2",
            font=(UI_FONT_FAMILY, 13, "bold"),
            height=38,
            corner_radius=8,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=2)

        ctk.CTkButton(
            word_btn_frame,
            text="⌫  Undo Word",
            command=self._on_clear_last,
            fg_color="#37474f",
            hover_color="#455a64",
            font=(UI_FONT_FAMILY, 13),
            height=38,
            corner_radius=8,
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=2)

        ctk.CTkButton(
            word_btn_frame,
            text="🗑  Clear All",
            command=self._on_clear,
            fg_color="#b71c1c",
            hover_color="#c62828",
            font=(UI_FONT_FAMILY, 13),
            height=38,
            corner_radius=8,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)

        word_btn_frame.columnconfigure(0, weight=1)
        word_btn_frame.columnconfigure(1, weight=1)

        self._add_divider()

        # ----- Camera controls -----
        ctk.CTkLabel(
            self,
            text="CAMERA CONTROL",
            font=(UI_FONT_FAMILY, 10, "bold"),
            text_color="#888899",
        ).pack(anchor="w", padx=16, pady=(10, 4))

        cam_btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        cam_btn_frame.pack(fill="x", padx=16, pady=(0, 8))

        self._start_btn = ctk.CTkButton(
            cam_btn_frame,
            text="▶  Start Camera",
            command=self._on_start,
            fg_color="#1b5e20",
            hover_color="#2e7d32",
            font=(UI_FONT_FAMILY, 13, "bold"),
            height=38,
            corner_radius=8,
        )
        self._start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self._stop_btn = ctk.CTkButton(
            cam_btn_frame,
            text="■  Stop Camera",
            command=self._on_stop,
            fg_color="#4a1942",
            hover_color="#6a1b9a",
            font=(UI_FONT_FAMILY, 13),
            height=38,
            corner_radius=8,
            state="disabled",
        )
        self._stop_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        cam_btn_frame.columnconfigure(0, weight=1)
        cam_btn_frame.columnconfigure(1, weight=1)

        self._add_divider()

        # ----- System status -----
        ctk.CTkLabel(
            self,
            text="SYSTEM STATUS",
            font=(UI_FONT_FAMILY, 10, "bold"),
            text_color="#888899",
        ).pack(anchor="w", padx=16, pady=(10, 6))

        status_frame = ctk.CTkFrame(self, fg_color="#111122", corner_radius=8)
        status_frame.pack(fill="x", padx=16, pady=(0, 8))

        rows = [
            ("Camera",  self._cam_status_var,   "#ef5350"),
            ("Model",   self._model_status_var,  "#ef5350"),
            ("Hand",    self._hand_status_var,   "#757575"),
        ]
        self._cam_status_label   = None
        self._model_status_label = None
        self._hand_status_label  = None

        for i, (label_text, var, color) in enumerate(rows):
            ctk.CTkLabel(
                status_frame,
                text=f"{label_text}:",
                font=(UI_FONT_FAMILY, 11),
                text_color="#888888",
            ).grid(row=i, column=0, sticky="w", padx=(12, 4), pady=4)

            lbl = ctk.CTkLabel(
                status_frame,
                textvariable=var,
                font=(UI_FONT_FAMILY, 11, "bold"),
                text_color=color,
            )
            lbl.grid(row=i, column=1, sticky="w", pady=4)

            if label_text == "Camera":
                self._cam_status_label = lbl
            elif label_text == "Model":
                self._model_status_label = lbl
            elif label_text == "Hand":
                self._hand_status_label = lbl

        status_frame.columnconfigure(1, weight=1)

        # Error / info message label
        self._error_label = ctk.CTkLabel(
            self,
            text="",
            font=(UI_FONT_FAMILY, 11),
            text_color="#ff5252",
            wraplength=280,
        )
        self._error_label.pack(padx=16, pady=(0, 6))

        # Spacer
        ctk.CTkLabel(self, text="").pack(expand=True)

        self._add_divider()

        # ----- Exit -----
        ctk.CTkButton(
            self,
            text="✕  Exit Application",
            command=self._on_exit,
            fg_color="#1c1c1c",
            hover_color="#2c2c2c",
            text_color="#888888",
            font=(UI_FONT_FAMILY, 12),
            height=36,
            corner_radius=8,
        ).pack(fill="x", padx=16, pady=(0, 16))

    def _add_divider(self) -> None:
        ctk.CTkFrame(self, height=1, fg_color="#2a2a3e").pack(fill="x", padx=16, pady=2)
