"""
speech_engine.py
================
Offline Text-to-Speech using pyttsx3.

Design
------
pyttsx3 is NOT thread-safe when called from multiple threads simultaneously.
We therefore run all TTS operations inside a single dedicated daemon thread
that processes a queue of speech requests. The UI thread simply enqueues
requests and the speech thread processes them asynchronously.

This ensures the webcam feed and UI remain fully responsive even during
long speech synthesis operations.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

from src.utils import get_logger

logger = get_logger(__name__)

# Sentinel object used to signal the speech thread to stop
_STOP_SENTINEL = object()


class SpeechEngine:
    """
    Thread-safe offline TTS engine backed by pyttsx3.

    Usage:
        engine = SpeechEngine()
        engine.start()
        engine.speak("Hello")
        engine.speak_sentence("Hello Thank You")
        engine.stop()
    """

    def __init__(self, rate: int = 150, volume: float = 1.0) -> None:
        self._rate   = rate    # Words per minute
        self._volume = volume  # 0.0 – 1.0
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._error: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background TTS thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._speech_loop,
            daemon=True,
            name="SpeechThread",
        )
        self._thread.start()
        logger.info("SpeechEngine started.")

    def stop(self) -> None:
        """Gracefully stop the TTS thread."""
        if not self._running:
            return
        self._running = False
        self._queue.put(_STOP_SENTINEL)
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("SpeechEngine stopped.")

    def speak(self, text: str) -> None:
        """
        Queue a single word or phrase for speech.
        Non-blocking — returns immediately.
        """
        if not text or not text.strip():
            return
        self._queue.put(("speak", text.strip()))
        logger.debug(f"Queued speech: '{text}'")

    def speak_sentence(self, sentence: str) -> None:
        """Queue the full sentence for speech. Non-blocking."""
        self.speak(sentence)

    def clear_queue(self) -> None:
        """Discard any pending speech requests."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        logger.debug("Speech queue cleared.")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _speech_loop(self) -> None:
        """
        Background thread entry point.
        Uses SAPI.SpVoice directly on Windows to avoid pyttsx3 thread hanging bugs.
        """
        import sys
        
        # Use Windows SAPI directly to avoid pyttsx3 loop freezes
        if sys.platform == "win32":
            try:
                import pythoncom
                pythoncom.CoInitialize()
                import win32com.client
                engine = win32com.client.Dispatch("SAPI.SpVoice")
                engine.Rate = -2  # Slower for clarity
                
                # Select female voice (Zira) if available
                voices = engine.GetVoices()
                for voice in voices:
                    if "Zira" in voice.GetDescription():
                        engine.Voice = voice
                        break
                        
                logger.info("SAPI.SpVoice engine initialised.")
            except Exception as exc:
                self._error = f"SAPI init failed: {exc}"
                logger.error(self._error)
                self._running = False
                return

            while self._running:
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if item is _STOP_SENTINEL:
                    break

                action, text = item
                if action == "speak":
                    try:
                        # 1 = SVSFlagsAsync (not blocking totally, but we want it to block the queue, so 0)
                        engine.Speak(text, 0)
                        logger.debug(f"Spoke: '{text}'")
                    except Exception as exc:
                        logger.error(f"TTS speak error: {exc}")
                        
            pythoncom.CoUninitialize()
            logger.info("SpeechEngine thread exiting.")
            return

        # Fallback for non-Windows (if ever needed)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 130)
            logger.info("pyttsx3 engine initialised.")
        except Exception as exc:
            self._error = f"TTS init failed: {exc}"
            logger.error(self._error)
            self._running = False
            return

        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is _STOP_SENTINEL:
                break

            action, text = item
            if action == "speak":
                try:
                    engine.say(text)
                    engine.runAndWait()
                    logger.debug(f"Spoke: '{text}'")
                except Exception as exc:
                    logger.error(f"TTS speak error: {exc}")

        try:
            engine.stop()
        except Exception:
            pass
        logger.info("SpeechEngine thread exiting.")
