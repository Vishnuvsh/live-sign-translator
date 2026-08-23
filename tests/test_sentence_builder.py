"""
test_sentence_builder.py
========================
Unit tests for the SentenceBuilder — word accumulation, cooldown, and undo.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sentence_builder import SentenceBuilder


class TestSentenceBuilder:

    def setup_method(self):
        # Use a very short cooldown for fast testing
        self.sb = SentenceBuilder(cooldown_seconds=0.05)

    # ----- Basic addition -----

    def test_add_single_word(self):
        added = self.sb.add_gesture("Hello")
        assert added is True
        assert self.sb.sentence == "Hello"

    def test_add_multiple_words(self):
        self.sb.add_gesture("Hello")
        time.sleep(0.06)
        self.sb.add_gesture("Thank You")
        time.sleep(0.06)
        self.sb.add_gesture("Help")
        assert self.sb.sentence == "Hello Thank You Help"

    def test_sentence_space_joined(self):
        self.sb.add_gesture("Hello")
        time.sleep(0.06)
        self.sb.add_gesture("Yes")
        assert " " in self.sb.sentence

    # ----- Cooldown / deduplication -----

    def test_same_word_within_cooldown_blocked(self):
        self.sb.add_gesture("Hello")
        added_again = self.sb.add_gesture("Hello")  # immediately — within cooldown
        assert added_again is False
        assert self.sb.sentence == "Hello"

    def test_same_word_after_cooldown_allowed(self):
        self.sb.add_gesture("Hello")
        time.sleep(0.10)  # wait longer than cooldown (0.05 s)
        added = self.sb.add_gesture("Hello")
        assert added is True
        assert self.sb.sentence == "Hello Hello"

    def test_different_words_no_cooldown_restriction(self):
        self.sb.add_gesture("Hello")
        added = self.sb.add_gesture("Yes")  # different word — no cooldown
        assert added is True

    # ----- Unknown gesture -----

    def test_unknown_gesture_returns_false(self):
        added = self.sb.add_gesture("InvalidGestureName")
        assert added is False
        assert self.sb.sentence == ""

    # ----- Undo -----

    def test_remove_last_word(self):
        self.sb.add_gesture("Hello")
        time.sleep(0.06)
        self.sb.add_gesture("Yes")
        removed = self.sb.remove_last_word()
        assert removed == "Yes"
        assert self.sb.sentence == "Hello"

    def test_remove_from_empty_returns_none(self):
        removed = self.sb.remove_last_word()
        assert removed is None

    # ----- Clear -----

    def test_clear_empties_sentence(self):
        self.sb.add_gesture("Hello")
        self.sb.clear()
        assert self.sb.sentence == ""
        assert self.sb.is_empty is True

    # ----- Properties -----

    def test_words_property(self):
        self.sb.add_gesture("Hello")
        time.sleep(0.06)
        self.sb.add_gesture("Stop")
        assert self.sb.words == ["Hello", "Stop"]

    def test_last_word_property(self):
        self.sb.add_gesture("Hello")
        assert self.sb.last_word == "Hello"

    def test_is_empty_initial(self):
        assert self.sb.is_empty is True

    # ----- Callback -----

    def test_on_word_added_callback_fired(self):
        received = []
        sb = SentenceBuilder(
            cooldown_seconds=0.05,
            on_word_added=lambda w: received.append(w),
        )
        sb.add_gesture("Hello")
        assert received == ["Hello"]
