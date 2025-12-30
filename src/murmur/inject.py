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


class StreamingInjector:
    """Diff-based text injector for live streaming transcription."""

    # Throttle: max updates per second
    MAX_UPDATES_PER_SEC = 4
    KEYSTROKE_DELAY = 0.002
    BACKSPACE_DELAY = 0.001

    def __init__(self):
        self._source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
        self._typed_text = ""
        self._last_update_time = 0.0
        self._lock = threading.Lock()

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
            if not force and (now - self._last_update_time) < (1.0 / self.MAX_UPDATES_PER_SEC):
                return False

            self._last_update_time = now

            # Compute diff
            old_text = self._typed_text
            if old_text == new_text:
                return False

            # Find common prefix length
            common_len = 0
            for i, (c1, c2) in enumerate(zip(old_text, new_text)):
                if c1 == c2:
                    common_len = i + 1
                else:
                    break

            # How many chars to delete from old
            delete_count = len(old_text) - common_len
            # What to type
            suffix = new_text[common_len:]

            # Send backspaces
            if delete_count > 0:
                self._send_backspaces(delete_count)

            # Type new suffix
            if suffix:
                self._type_text(suffix)

            self._typed_text = new_text
            return True

    def _send_backspaces(self, count: int) -> None:
        """Send backspace key events."""
        for _ in range(count):
            # Keycode 51 = Backspace on macOS
            key_down = CGEventCreateKeyboardEvent(self._source, 51, True)
            key_up = CGEventCreateKeyboardEvent(self._source, 51, False)
            CGEventPost(kCGHIDEventTap, key_down)
            CGEventPost(kCGHIDEventTap, key_up)
            time.sleep(self.BACKSPACE_DELAY)

    def _type_text(self, text: str) -> None:
        """Type text characters."""
        for char in text:
            key_down = CGEventCreateKeyboardEvent(self._source, 0, True)
            key_up = CGEventCreateKeyboardEvent(self._source, 0, False)
            CGEventKeyboardSetUnicodeString(key_down, len(char), char)
            CGEventKeyboardSetUnicodeString(key_up, len(char), char)
            CGEventPost(kCGHIDEventTap, key_down)
            CGEventPost(kCGHIDEventTap, key_up)
            time.sleep(self.KEYSTROKE_DELAY)

    @property
    def typed_text(self) -> str:
        """Get currently typed text."""
        with self._lock:
            return self._typed_text

