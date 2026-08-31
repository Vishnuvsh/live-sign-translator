"""
sentence_builder.py
===================
Builds the recognised sentence word by word with duplicate prevention and
cooldown logic.

Rules
-----
1. A confirmed gesture adds its spoken word to the sentence.
2. The same word cannot be added again until WORD_COOLDOWN_SECONDS have passed.
   This prevents "Hello Hello Hello …" when the user holds the sign.
3. After the cooldown, the user CAN intentionally repeat the same word.
4. The sentence has a maximum word limit to avoid unbounded growth.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from src.gesture_config import GESTURE_SPOKEN, WORD_COOLDOWN_SECONDS
from src.utils import get_logger

logger = get_logger(__name__)

MAX_SENTENCE_WORDS = 50  # Safety cap — prevents memory issues


class SentenceBuilder:
    """
    Accumulates confirmed gesture words into a readable sentence.

    Usage:
        sb = SentenceBuilder(on_word_added=lambda w: print(f"Added: {w}"))
        sb.add_gesture("Hello")
        sb.add_gesture("Hello")   # ignored — within cooldown
        print(sb.sentence)         # "Hello"
    """

    def __init__(
        self,
        cooldown_seconds: float = WORD_COOLDOWN_SECONDS,
        on_word_added: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._cooldown       = cooldown_seconds
        self._on_word_added  = on_word_added   # callback fired when a word is accepted
        self._words: list[str] = []
        self._last_word: Optional[str] = None
        self._last_add_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_gesture(self, gesture_name: str) -> bool:
        """
        Attempt to add the spoken word corresponding to a gesture name.

        Parameters
        ----------
        gesture_name : str
            The confirmed gesture name (e.g. 'Hello').

        Returns
        -------
        bool : True if the word was added, False if it was blocked by cooldown.
        """
        spoken_word = self._resolve_spoken(gesture_name)
        if spoken_word is None:
            logger.warning(f"Unknown gesture name: '{gesture_name}' — skipping.")
            return False

        now = time.monotonic()

        # Cooldown & Duplicate check: block exact same consecutive word completely
        # (This prevents spam like "What What What" if the user holds their hand steady)
        if spoken_word == self._last_word:
            logger.debug(f"Word '{spoken_word}' blocked (consecutive duplicate)")
            return False

        # Safety cap
        if len(self._words) >= MAX_SENTENCE_WORDS:
            logger.warning("Sentence word limit reached — ignoring new word.")
            return False

        # Accept the word
        self._words.append(spoken_word)
        self._last_word     = spoken_word
        self._last_add_time = now
        logger.info(f"Word added: '{spoken_word}' | Sentence: '{self.sentence}'")

        if self._on_word_added:
            self._on_word_added(spoken_word)

        return True

    def remove_last_word(self) -> Optional[str]:
        """Remove and return the last word in the sentence (undo)."""
        if self._words:
            removed = self._words.pop()
            logger.info(f"Last word removed: '{removed}'")
            return removed
        return None

    def clear(self) -> None:
        """Clear the entire sentence and reset state."""
        self._words.clear()
        self._last_word     = None
        self._last_add_time = 0.0
        logger.info("Sentence cleared.")

    @property
    def sentence(self) -> str:
        """Return the accumulated sentence as a space-joined string."""
        return " ".join(self._words)

    @property
    def words(self) -> list[str]:
        """Return a copy of the word list."""
        return self._words.copy()

    @property
    def last_word(self) -> Optional[str]:
        return self._last_word

    @property
    def is_empty(self) -> bool:
        return len(self._words) == 0

    @property
    def cooldown_remaining(self) -> float:
        """Seconds remaining in the cooldown for the last word (0 if expired)."""
        elapsed = time.monotonic() - self._last_add_time
        remaining = self._cooldown - elapsed
        return max(0.0, remaining)

    @property
    def time_since_last_add(self) -> float:
        """Seconds since the last word was added."""
        if self.is_empty:
            return 0.0
        return time.monotonic() - self._last_add_time

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_spoken(gesture_name: str) -> Optional[str]:
        """
        Look up the spoken word for a gesture name.
        First tries GESTURE_SPOKEN dict (keyed by int id), then falls back
        to a name-based reverse lookup.
        """
        # GESTURE_SPOKEN is {id: spoken_word}, so build a name→spoken map
        from src.gesture_config import GESTURES
        name_to_spoken = {g["name"]: g["spoken"] for g in GESTURES}
        return name_to_spoken.get(gesture_name)
