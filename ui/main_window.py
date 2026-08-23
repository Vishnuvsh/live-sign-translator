"""
main_window.py
==============
Root Tkinter / CustomTkinter window.

Wires together:
  Camera → HandDetector → FeatureExtractor → GestureClassifier
  → PredictionStabilizer → SentenceBuilder → SpeechEngine
  → CameraPanel / ControlPanel (UI)

The camera update loop runs via Tkinter's after() scheduler,
keeping everything on the main thread while remaining non-blocking.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

try:
    import customtkinter as ctk
except ImportError:
    raise ImportError("Install customtkinter:  pip install customtkinter")

from src.camera               import Camera
from src.feature_extractor    import FeatureExtractor
from src.gesture_classifier   import GestureClassifier
from src.gesture_config       import (
    CONFIDENCE_THRESHOLD,
    MODEL_PATH,
    UI_COLOR_SCHEME,
    UI_FONT_FAMILY,
    UI_THEME,
    UI_UPDATE_MS,
    UI_WINDOW_TITLE,
)
from src.hand_detector        import HandDetector
from src.prediction_stabilizer import PredictionStabilizer
from src.sentence_builder     import SentenceBuilder
from src.speech_engine        import SpeechEngine
from src.utils                import FPSCounter, get_logger
from ui.camera_panel          import CameraPanel
from ui.control_panel         import ControlPanel

logger = get_logger(__name__)


class MainWindow:
    """
    Application root window.

    Lifecycle:
        win = MainWindow()
        win.run()      # blocks until window is closed
    """

    def __init__(self) -> None:
        # --- CustomTkinter global settings ---
        ctk.set_appearance_mode(UI_THEME)
        ctk.set_default_color_theme(UI_COLOR_SCHEME)

        self._root = ctk.CTk()
        self._root.title(UI_WINDOW_TITLE)
        self._root.geometry("1050x620")
        self._root.minsize(900, 580)
        self._root.protocol("WM_DELETE_WINDOW", self._on_exit)

        # --- Core pipeline objects ---
        self._camera      = Camera()
        self._detector    = HandDetector()
        self._extractor   = FeatureExtractor()
        self._classifier  = GestureClassifier()
        self._stabilizer  = PredictionStabilizer()
        self._speech      = SpeechEngine()
        self._fps_counter = FPSCounter()

        self._sentence_builder = SentenceBuilder(
            on_word_added=self._on_word_added
        )

        # --- State ---
        self._camera_running    = False
        self._model_loaded      = False
        self._last_confirmed: Optional[str] = None
        self._update_job: Optional[str] = None   # after() job ID

        # --- Load model ---
        self._load_model()

        # --- Start TTS engine ---
        self._speech.start()

        # --- Build UI ---
        self._build_layout()

        logger.info("MainWindow initialised.")

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Create the two-column layout: camera left, controls right."""
        self._root.configure(fg_color="#0d0d1a")

        # Header
        header = ctk.CTkFrame(self._root, fg_color="#16213e", height=50, corner_radius=0)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="✋  Real-Time Sign Language Translator",
            font=(UI_FONT_FAMILY, 15, "bold"),
            text_color="#e8eaf6",
        ).pack(side="left", padx=20, pady=10)

        ctk.CTkLabel(
            header,
            text="Accessibility Prototype · Offline · 10 Gestures",
            font=(UI_FONT_FAMILY, 10),
            text_color="#4a4a6a",
        ).pack(side="right", padx=20)

        # Main body
        body = ctk.CTkFrame(self._root, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=12, pady=8)

        # Camera panel (left, 2/3 width)
        self._cam_panel = CameraPanel(
            body,
            fg_color="#111122",
            corner_radius=12,
        )
        self._cam_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # Control panel (right, fixed width)
        self._ctrl_panel = ControlPanel(
            body,
            on_start_camera    = self._start_camera,
            on_stop_camera     = self._stop_camera,
            on_speak_sentence  = self._speak_sentence,
            on_clear_sentence  = self._clear_sentence,
            on_clear_last_word = self._clear_last_word,
            on_exit            = self._on_exit,
            width=310,
        )
        self._ctrl_panel.pack(side="right", fill="y")
        self._ctrl_panel.pack_propagate(False)

        # Initial status
        self._ctrl_panel.set_camera_status(False)
        self._ctrl_panel.set_model_status(self._model_loaded)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Attempt to load the gesture model. Show a clear message if missing."""
        try:
            self._classifier.load()
            self._model_loaded = True
            logger.info("Gesture model loaded successfully.")
        except FileNotFoundError:
            self._model_loaded = False
            logger.warning(
                f"Model not found at {MODEL_PATH}. "
                "Run training/collect_data.py then training/train_model.py first."
            )
        except Exception as exc:
            self._model_loaded = False
            logger.error(f"Model load error: {exc}")

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------

    def _start_camera(self) -> None:
        if self._camera_running:
            return
        try:
            self._camera.start()
            self._camera_running = True
            self._ctrl_panel.set_camera_status(True)

            if not self._model_loaded:
                self._ctrl_panel.show_error(
                    "Model not loaded — gestures won't be recognised.\n"
                    "Run the training pipeline first."
                )

            # Kick off the UI update loop
            self._schedule_update()
            logger.info("Camera started.")
        except RuntimeError as exc:
            self._ctrl_panel.show_error(str(exc))
            logger.error(f"Camera start failed: {exc}")

    def _stop_camera(self) -> None:
        if not self._camera_running:
            return

        # Cancel the scheduled after() job
        if self._update_job:
            self._root.after_cancel(self._update_job)
            self._update_job = None

        self._camera.stop()
        self._camera_running = False
        self._stabilizer.reset()
        self._cam_panel.show_placeholder()
        self._ctrl_panel.set_camera_status(False)
        self._ctrl_panel.set_hand_status(False)
        logger.info("Camera stopped.")

    # ------------------------------------------------------------------
    # Main recognition loop
    # ------------------------------------------------------------------

    def _schedule_update(self) -> None:
        """Schedule the next frame update via after()."""
        self._update_job = self._root.after(UI_UPDATE_MS, self._process_frame)

    def _process_frame(self) -> None:
        """
        Called every UI_UPDATE_MS ms while the camera is running.

        Pipeline:
          Frame → Detect hand → Extract features → Classify → Stabilize
          → (if confirmed) Add to sentence / Speak
          → Update UI
        """
        if not self._camera_running:
            return

        frame = self._camera.read()
        if frame is None:
            self._schedule_update()
            return

        fps = self._fps_counter.tick()

        # --- Hand detection ---
        hand = self._detector.detect(frame)
        hand_detected = hand is not None
        self._ctrl_panel.set_hand_status(hand_detected)

        gesture_name = "—"
        confidence   = 0.0
        status       = "NO HAND"

        if hand:
            frame = self._detector.draw(frame, hand)

            # --- Feature extraction ---
            fv = self._extractor.extract(hand.landmarks)

            if fv is not None and self._model_loaded:
                # --- Classification ---
                prediction = self._classifier.predict(fv)

                if prediction:
                    gesture_name = prediction.gesture_name
                    confidence   = prediction.confidence

                    if prediction.accepted:
                        status = "DETECTING"

                        # --- Temporal stabilization ---
                        confirmed = self._stabilizer.update(prediction.gesture_name)
                        if confirmed:
                            status = "CONFIRMED"
                            self._on_gesture_confirmed(confirmed)
                            self._stabilizer.reset()
                    else:
                        gesture_name = "Unknown"
                        status       = "LOW CONF"
                        self._stabilizer.update(None)
                else:
                    self._stabilizer.update(None)
                    status = "DETECTING"
            else:
                self._stabilizer.update(None)
                status = "DETECTING" if not self._model_loaded else "DETECTING"
        else:
            self._stabilizer.update(None)

        # --- Update UI ---
        self._cam_panel.update_frame(frame, gesture_name, confidence, status, fps)

        # Schedule next frame
        self._schedule_update()

    # ------------------------------------------------------------------
    # Gesture → Sentence → Speech
    # ------------------------------------------------------------------

    def _on_gesture_confirmed(self, gesture_name: str) -> None:
        """Handle a newly confirmed (stabilized) gesture."""
        added = self._sentence_builder.add_gesture(gesture_name)
        if added:
            logger.info(f"Gesture confirmed and added: {gesture_name}")
            # Speak only the newly added word (not the full sentence)
            from src.gesture_config import GESTURES
            spoken = next(
                (g["spoken"] for g in GESTURES if g["name"] == gesture_name),
                gesture_name,
            )
            self._speech.speak(spoken)
            self._ctrl_panel.set_sentence(self._sentence_builder.sentence)

    def _on_word_added(self, word: str) -> None:
        """Callback fired by SentenceBuilder when a word is accepted."""
        self._ctrl_panel.set_sentence(self._sentence_builder.sentence)

    def _speak_sentence(self) -> None:
        sentence = self._sentence_builder.sentence
        if sentence:
            self._speech.speak_sentence(sentence)
        else:
            self._ctrl_panel.show_info("Sentence is empty.")

    def _clear_sentence(self) -> None:
        self._sentence_builder.clear()
        self._ctrl_panel.set_sentence("")

    def _clear_last_word(self) -> None:
        removed = self._sentence_builder.remove_last_word()
        if removed:
            self._ctrl_panel.set_sentence(self._sentence_builder.sentence)
        else:
            self._ctrl_panel.show_info("Nothing to undo.")

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def _on_exit(self) -> None:
        """Gracefully shut down all resources before closing."""
        logger.info("Shutting down …")
        self._stop_camera()
        self._speech.stop()
        self._detector.close()
        self._root.destroy()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the Tkinter main loop."""
        self._root.mainloop()
