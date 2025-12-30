"""HID text injection using Quartz Event Services."""

import time
import threading

from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventKeyboardSetUnicodeString,
    CGEventPost,
    CGEventSourceCreate,
    kCGEventSourceStateHIDSystemState,
    kCGHIDEventTap,
)

from .logger import log


class StreamingInjector:
    """Diff-based text injector for live streaming transcription."""

    def __init__(
        self,
        max_updates_per_sec: int = 4,
        max_backspace_chars: int = 30,
        keystroke_delay_seconds: float = 0.002,
        backspace_delay_seconds: float = 0.001,
    ):
        self._source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
        self._typed_text = ""
        self._last_update_time = 0.0
        self._lock = threading.Lock()
        self._max_updates_per_sec = max(1, max_updates_per_sec)
        self._max_backspace_chars = max(0, max_backspace_chars)
        self._keystroke_delay = keystroke_delay_seconds
        self._backspace_delay = backspace_delay_seconds

    def reset(self) -> None:
        """Reset state for new session."""
        with self._lock:
            self._typed_text = ""
            self._last_update_time = 0.0

    def update(self, new_text: str, force: bool = False) -> bool:
        """
        Update the injected text using minimal diff.

        Args:
            new_text: The full text that should be visible
            force: Bypass throttling (for final update)

        Returns:
            True if text was updated, False if throttled
        """
        if not new_text:
            return False

        with self._lock:
            # Throttle check
            now = time.time()
            if not force and (now - self._last_update_time) < (1.0 / self._max_updates_per_sec):
                return False

            # Compute diff
            old_text = self._typed_text
            if old_text == new_text:
                return False

            prefix_keep = ""
            old_tail = old_text
            new_tail = new_text
            if self._max_backspace_chars is not None and len(old_text) > self._max_backspace_chars:
                prefix_keep = old_text[:-self._max_backspace_chars]
                if not new_text.startswith(prefix_keep):
                    log.warning(f"Injector: update blocked (needs backspacing > {self._max_backspace_chars} chars)")
                    return False
                old_tail = old_text[len(prefix_keep):]
                new_tail = new_text[len(prefix_keep):]

            # Find common prefix length in the editable tail
            common_len = 0
            for i, (c1, c2) in enumerate(zip(old_tail, new_tail)):
                if c1 == c2:
                    common_len = i + 1
                else:
                    break

            # How many chars to delete from tail
            delete_count = len(old_tail) - common_len
            # What to type
            suffix = new_tail[common_len:]

            # Send backspaces
            if delete_count > 0:
                self._send_backspaces(delete_count)

            # Type new suffix
            if suffix:
                self._type_text(suffix)

            self._last_update_time = now
            self._typed_text = prefix_keep + new_tail
            return True

    def _send_backspaces(self, count: int) -> None:
        """Send backspace key events."""
        for _ in range(count):
            # Keycode 51 = Backspace on macOS
            key_down = CGEventCreateKeyboardEvent(self._source, 51, True)
            key_up = CGEventCreateKeyboardEvent(self._source, 51, False)
            CGEventPost(kCGHIDEventTap, key_down)
            CGEventPost(kCGHIDEventTap, key_up)
            time.sleep(self._backspace_delay)

    def _type_text(self, text: str) -> None:
        """Type text characters."""
        for char in text:
            key_down = CGEventCreateKeyboardEvent(self._source, 0, True)
            key_up = CGEventCreateKeyboardEvent(self._source, 0, False)
            CGEventKeyboardSetUnicodeString(key_down, len(char), char)
            CGEventKeyboardSetUnicodeString(key_up, len(char), char)
            CGEventPost(kCGHIDEventTap, key_down)
            CGEventPost(kCGHIDEventTap, key_up)
            time.sleep(self._keystroke_delay)

    @property
    def typed_text(self) -> str:
        """Get currently typed text."""
        with self._lock:
            return self._typed_text
