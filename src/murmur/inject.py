"""HID text injection using Quartz Event Services."""

import time

from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventKeyboardSetUnicodeString,
    CGEventPost,
    CGEventSourceCreate,
    kCGEventSourceStateHIDSystemState,
    kCGHIDEventTap,
)


class TextInjector:
    """Injects text as keyboard events into the focused application."""

    # Delay between keystrokes (seconds)
    # Too fast can overwhelm some applications
    KEYSTROKE_DELAY = 0.002

    def __init__(self):
        self._source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)

    def inject(self, text: str) -> None:
        """
        Inject text as keyboard events.

        Each character is sent as a key-down and key-up event
        with Unicode character override.
        """
        if not text:
            return

        for char in text:
            self._type_char(char)
            time.sleep(self.KEYSTROKE_DELAY)

    def _type_char(self, char: str) -> None:
        """Type a single character."""
        # Create key events (keycode 0 is a placeholder, we override with unicode)
        key_down = CGEventCreateKeyboardEvent(self._source, 0, True)
        key_up = CGEventCreateKeyboardEvent(self._source, 0, False)

        # Override with the actual unicode character
        CGEventKeyboardSetUnicodeString(key_down, len(char), char)
        CGEventKeyboardSetUnicodeString(key_up, len(char), char)

        # Post events to the system
        CGEventPost(kCGHIDEventTap, key_down)
        CGEventPost(kCGHIDEventTap, key_up)
