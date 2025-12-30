"""macOS notifications for Murmur."""

import subprocess


def notify(title: str, message: str, sound: bool = False) -> None:
    """
    Show a macOS notification.

    Uses osascript which is available on all macOS versions.
    """
    sound_part = 'sound name "default"' if sound else ""
    script = f'display notification "{message}" with title "{title}" {sound_part}'

    subprocess.Popen(
        ["osascript", "-e", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
