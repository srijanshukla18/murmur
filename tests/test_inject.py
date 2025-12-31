"""Tests for src/murmur/inject.py"""

import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Mock Quartz before importing inject
quartz_mock = MagicMock()
sys.modules['Quartz'] = quartz_mock
quartz_mock.CGEventCreateKeyboardEvent = MagicMock(return_value=MagicMock())
quartz_mock.CGEventKeyboardSetUnicodeString = MagicMock()
quartz_mock.CGEventPost = MagicMock()
quartz_mock.CGEventSourceCreate = MagicMock(return_value=MagicMock())
quartz_mock.kCGEventSourceStateHIDSystemState = 1
quartz_mock.kCGHIDEventTap = 0

from murmur.inject import StreamingInjector


class TestStreamingInjectorInit:
    """Test StreamingInjector initialization."""

    def test_init_default_values(self):
        injector = StreamingInjector()
        assert injector._max_updates_per_sec == 4
        assert injector._max_backspace_chars == 30
        assert injector._keystroke_delay == 0.002
        assert injector._backspace_delay == 0.001

    def test_init_custom_max_updates(self):
        injector = StreamingInjector(max_updates_per_sec=10)
        assert injector._max_updates_per_sec == 10

    def test_init_max_updates_minimum_one(self):
        injector = StreamingInjector(max_updates_per_sec=0)
        assert injector._max_updates_per_sec == 1

    def test_init_max_updates_negative_becomes_one(self):
        injector = StreamingInjector(max_updates_per_sec=-5)
        assert injector._max_updates_per_sec == 1

    def test_init_custom_max_backspace(self):
        injector = StreamingInjector(max_backspace_chars=50)
        assert injector._max_backspace_chars == 50

    def test_init_max_backspace_negative_becomes_zero(self):
        injector = StreamingInjector(max_backspace_chars=-10)
        assert injector._max_backspace_chars == 0

    def test_init_custom_keystroke_delay(self):
        injector = StreamingInjector(keystroke_delay_seconds=0.005)
        assert injector._keystroke_delay == 0.005

    def test_init_custom_backspace_delay(self):
        injector = StreamingInjector(backspace_delay_seconds=0.003)
        assert injector._backspace_delay == 0.003

    def test_init_typed_text_empty(self):
        injector = StreamingInjector()
        assert injector._typed_text == ""

    def test_init_last_update_time_zero(self):
        injector = StreamingInjector()
        assert injector._last_update_time == 0.0

    def test_init_creates_event_source(self):
        quartz_mock.CGEventSourceCreate.reset_mock()
        injector = StreamingInjector()
        quartz_mock.CGEventSourceCreate.assert_called_once()


class TestStreamingInjectorReset:
    """Test StreamingInjector reset method."""

    def test_reset_clears_typed_text(self):
        injector = StreamingInjector()
        injector._typed_text = "some text"
        injector.reset()
        assert injector._typed_text == ""

    def test_reset_clears_last_update_time(self):
        injector = StreamingInjector()
        injector._last_update_time = 123.456
        injector.reset()
        assert injector._last_update_time == 0.0


class TestStreamingInjectorTypedText:
    """Test typed_text property."""

    def test_typed_text_returns_current_text(self):
        injector = StreamingInjector()
        injector._typed_text = "hello world"
        assert injector.typed_text == "hello world"

    def test_typed_text_thread_safe(self):
        injector = StreamingInjector()
        injector._typed_text = "test"

        results = []

        def read_text():
            for _ in range(100):
                results.append(injector.typed_text)

        threads = [threading.Thread(target=read_text) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == "test" for r in results)


class TestStreamingInjectorUpdate:
    """Test StreamingInjector update method."""

    def test_update_empty_text_returns_false(self):
        injector = StreamingInjector()
        result = injector.update("")
        assert result is False

    def test_update_none_text_returns_false(self):
        injector = StreamingInjector()
        result = injector.update(None)
        assert result is False

    def test_update_same_text_returns_false(self):
        injector = StreamingInjector()
        injector._typed_text = "hello"
        injector._last_update_time = 0  # ensure not throttled
        result = injector.update("hello")
        assert result is False

    def test_update_throttled_returns_false(self):
        injector = StreamingInjector(max_updates_per_sec=1)
        injector._typed_text = ""
        injector._last_update_time = time.time()  # just updated
        result = injector.update("hello")
        assert result is False

    def test_update_force_bypasses_throttle(self):
        injector = StreamingInjector(max_updates_per_sec=1)
        injector._typed_text = ""
        injector._last_update_time = time.time()  # just updated
        result = injector.update("hello", force=True)
        assert result is True

    def test_update_new_text_returns_true(self):
        injector = StreamingInjector()
        injector._typed_text = ""
        injector._last_update_time = 0
        result = injector.update("hello")
        assert result is True

    def test_update_updates_typed_text(self):
        injector = StreamingInjector()
        injector._typed_text = ""
        injector._last_update_time = 0
        injector.update("hello")
        assert injector._typed_text == "hello"

    def test_update_updates_last_update_time(self):
        injector = StreamingInjector()
        injector._typed_text = ""
        injector._last_update_time = 0
        before = time.time()
        injector.update("hello")
        assert injector._last_update_time >= before

    def test_update_appends_text(self):
        injector = StreamingInjector()
        injector._typed_text = "hello"
        injector._last_update_time = 0
        injector.update("hello world")
        assert injector._typed_text == "hello world"

    def test_update_with_deletion(self):
        injector = StreamingInjector()
        injector._typed_text = "hello world"
        injector._last_update_time = 0
        injector.update("hello")
        assert injector._typed_text == "hello"


class TestStreamingInjectorDiffLogic:
    """Test the diff computation in update method."""

    def test_diff_appends_suffix_only(self):
        injector = StreamingInjector()
        injector._typed_text = "hello"
        injector._last_update_time = 0

        quartz_mock.CGEventPost.reset_mock()
        injector.update("hello world")

        calls = quartz_mock.CGEventPost.call_args_list
        assert len(calls) > 0

    def test_diff_with_common_prefix(self):
        injector = StreamingInjector()
        injector._typed_text = "hello world"
        injector._last_update_time = 0

        injector.update("hello there")
        assert injector._typed_text == "hello there"

    def test_diff_complete_replacement(self):
        injector = StreamingInjector()
        injector._typed_text = "abc"
        injector._last_update_time = 0
        injector._max_backspace_chars = 100

        injector.update("xyz")
        assert injector._typed_text == "xyz"


class TestStreamingInjectorBackspaces:
    """Test _send_backspaces method."""

    def test_send_backspaces_creates_key_events(self):
        injector = StreamingInjector()
        quartz_mock.CGEventPost.reset_mock()

        injector._send_backspaces(3)

        assert quartz_mock.CGEventPost.call_count == 6  # 2 per backspace (down+up)

    def test_send_backspaces_uses_correct_keycode(self):
        injector = StreamingInjector()
        quartz_mock.CGEventCreateKeyboardEvent.reset_mock()

        injector._send_backspaces(1)

        calls = quartz_mock.CGEventCreateKeyboardEvent.call_args_list
        assert any(call[0][1] == 51 for call in calls)  # keycode 51 = backspace


class TestStreamingInjectorTypeText:
    """Test _type_text method."""

    def test_type_text_creates_key_events_per_char(self):
        injector = StreamingInjector()
        quartz_mock.CGEventPost.reset_mock()

        injector._type_text("hi")

        assert quartz_mock.CGEventPost.call_count == 4  # 2 per char (down+up)

    def test_type_text_sets_unicode_string(self):
        injector = StreamingInjector()
        quartz_mock.CGEventKeyboardSetUnicodeString.reset_mock()

        injector._type_text("a")

        assert quartz_mock.CGEventKeyboardSetUnicodeString.call_count >= 2


class TestStreamingInjectorMaxBackspace:
    """Test max_backspace_chars behavior."""

    def test_max_backspace_limits_deletion(self):
        injector = StreamingInjector(max_backspace_chars=5)
        injector._typed_text = "a" * 100
        injector._last_update_time = 0

        result = injector.update("completely different")
        assert result is True

    def test_max_backspace_zero_means_no_backspace(self):
        injector = StreamingInjector(max_backspace_chars=0)
        injector._typed_text = "old text"
        injector._last_update_time = 0

        result = injector.update("new text")
        assert result is True


class TestStreamingInjectorThreadSafety:
    """Test thread safety of StreamingInjector."""

    def test_concurrent_updates(self):
        injector = StreamingInjector(max_updates_per_sec=1000)
        errors = []

        def update_text(text):
            try:
                for i in range(20):
                    injector.update(f"{text} {i}", force=True)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=update_text, args=(f"thread{i}",))
            for i in range(3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_reset_and_update(self):
        injector = StreamingInjector()
        errors = []

        def do_updates():
            try:
                for i in range(50):
                    injector.update(f"text {i}", force=True)
            except Exception as e:
                errors.append(e)

        def do_resets():
            try:
                for _ in range(50):
                    injector.reset()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=do_updates)
        t2 = threading.Thread(target=do_resets)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0
