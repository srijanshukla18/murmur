"""Menu bar application for Murmur."""

import subprocess
import threading
import time
from enum import Enum
from pathlib import Path

import rumps
from pynput import keyboard
from PyObjCTools import AppHelper

from .audio import AudioRecorder
from .config import Config
from .inject import TextInjector
from .logger import log
from .notify import notify
from .transcribe import Transcriber


class State(Enum):
    """Application states."""

    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"


# Menu bar icons for each state
ICONS = {
    State.IDLE: "🎤",
    State.RECORDING: "🔴",
    State.TRANSCRIBING: "⏳",
}

# macOS system sounds
SOUNDS = {
    "start": "/System/Library/Sounds/Funk.aiff",
    "stop": "/System/Library/Sounds/Blow.aiff",
    "error": "/System/Library/Sounds/Basso.aiff",
}

# Timeout for transcription (seconds)
TRANSCRIPTION_TIMEOUT = 45


class MurmurApp(rumps.App):
    """Murmur menu bar application."""

    def __init__(self, config: Config):
        super().__init__("Murmur", quit_button=None)

        self.config = config
        self.state = State.IDLE
        self.title = ICONS[State.IDLE]
        self._transcription_start_time = 0
        self._last_toggle_time = 0

        # Initialize components
        self.recorder = AudioRecorder()
        self.transcriber = Transcriber(
            config.whisper_binary,
            config.model_path,
        )
        self.injector = TextInjector()

        # Setup menu
        self.status_item = rumps.MenuItem("Status: Idle")
        self.status_item.set_callback(None)  # Not clickable
        self.menu = [
            self.status_item,
            None,  # Separator
            rumps.MenuItem("Quit", callback=self._quit),
        ]

        # Start hotkey listener in background thread
        self._start_hotkey_listener()

        log.info(f"Murmur initialized (hotkey={config.hotkey}, model={config.model})")

    def _start_hotkey_listener(self) -> None:
        """Start listening for the configured hotkey."""
        hotkey = self.config.hotkey

        # Map config hotkey to pynput key
        key_map = {
            "alt_r": keyboard.Key.alt_r,
            "alt_l": keyboard.Key.alt_l,
            "caps_lock": keyboard.Key.caps_lock,
            "f8": keyboard.Key.f8,
            "f9": keyboard.Key.f9,
            "f10": keyboard.Key.f10,
        }

        target_key = key_map.get(hotkey)
        if not target_key:
            log.warning(f"Unknown hotkey '{hotkey}', using Right Option")
            target_key = keyboard.Key.alt_r

        # Track state and debounce (macOS modifier keys generate multiple events)
        key_state = {"pressed": False, "last_release": 0.0}

        def on_press(key):
            if key == target_key and not key_state["pressed"]:
                key_state["pressed"] = True
                log.debug("Hotkey pressed (down) - triggering toggle")
                # Dispatch to main thread immediately on press
                AppHelper.callAfter(self._toggle)

        def on_release(key):
            if key == target_key and key_state["pressed"]:
                key_state["pressed"] = False
                log.debug("Hotkey released (up)")

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()
        log.debug(f"Hotkey listener started for {hotkey} (toggle on release)")

    def _toggle(self) -> None:
        """Toggle between idle and recording states."""
        # Debounce: ignore rapid key repeats (200ms minimum)
        now = time.time()
        if now - self._last_toggle_time < 0.2:
            log.debug(f"Toggle ignored (debounce: {now - self._last_toggle_time:.3f}s)")
            return
        
        log.debug(f"Toggle triggered. Current state: {self.state}")
        self._last_toggle_time = now

        if self.state == State.IDLE:
            self._start_recording()
        elif self.state == State.RECORDING:
            self._stop_recording()
        elif self.state == State.TRANSCRIBING:
            # Check for stuck state (timeout protection)
            elapsed = time.time() - self._transcription_start_time
            if elapsed > TRANSCRIPTION_TIMEOUT:
                log.warning(f"Transcription stuck for {elapsed:.0f}s, forcing reset")
                notify("Murmur", "Transcription timed out", sound=False)
                self._set_state(State.IDLE)

    def _start_recording(self) -> None:
        """Start recording audio."""
        log.info("Recording started")
        self._set_state(State.RECORDING)
        self.recorder.start()  # Start mic IMMEDIATELY
        self._play_sound("start")  # Sound plays in background

    def _stop_recording(self) -> None:
        """Stop recording and begin transcription."""
        log.info("Recording stopped, starting transcription")
        self._set_state(State.TRANSCRIBING)
        self._transcription_start_time = time.time()
        self._play_sound("stop")

        # Get audio data
        audio_data = self.recorder.stop()
        audio_duration = len(audio_data) / (16000 * 2) if audio_data else 0
        log.debug(f"Audio captured: {len(audio_data)} bytes (~{audio_duration:.1f}s)")

        if not audio_data or len(audio_data) < 1000:
            log.warning("Audio too short, skipping transcription")
            notify("Murmur", "Recording too short", sound=False)
            self._set_state(State.IDLE)
            return

        # Transcribe in background thread
        def transcribe():
            try:
                text = self.transcriber.transcribe(audio_data)
                # Dispatch completion to main thread
                AppHelper.callAfter(lambda: self._on_transcription_complete(text))
            except Exception as e:
                log.error(f"Transcription thread error: {e}")
                AppHelper.callAfter(lambda: self._on_transcription_complete(None, str(e)))

        thread = threading.Thread(target=transcribe, daemon=True)
        thread.start()

    def _on_transcription_complete(
        self, text: str | None, error: str | None = None
    ) -> None:
        """Handle completed transcription."""
        elapsed = time.time() - self._transcription_start_time

        if text:
            log.info(f"Transcription complete in {elapsed:.1f}s: {text[:50]}...")
            self.injector.inject(text)
        else:
            self._play_sound("error")
            if error:
                log.error(f"Transcription failed: {error}")
                notify("Murmur", f"Error: {error[:50]}", sound=False)
            else:
                log.warning("Transcription returned empty (silence or error)")
                notify("Murmur", "No speech detected", sound=False)

        self._set_state(State.IDLE)

    def _set_state(self, state: State) -> None:
        """Update application state and UI."""
        self.state = state
        self.title = ICONS[state]

        status_text = {
            State.IDLE: "Status: Idle",
            State.RECORDING: "Status: Recording...",
            State.TRANSCRIBING: "Status: Transcribing...",
        }
        self.status_item.title = status_text[state]

    def _play_sound(self, sound_name: str, wait: bool = False) -> None:
        """Play a system sound."""
        if not self.config.sound:
            return

        sound_path = SOUNDS.get(sound_name)
        if sound_path and Path(sound_path).exists():
            if wait:
                # Use run for blocking (waits for sound to finish)
                subprocess.run(
                    ["afplay", sound_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                # Fire and forget
                subprocess.Popen(
                    ["afplay", sound_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

    def _quit(self, _) -> None:
        """Quit the application."""
        log.info("Murmur shutting down")
        rumps.quit_application()


def main():
    """Entry point for murmur."""
    try:
        config = Config.load()
        log.info("=" * 40)
        log.info("Murmur starting")
        log.info(f"  Hotkey: {config.hotkey}")
        log.info(f"  Model: {config.model}")
        log.info(f"  Sound: {config.sound}")

        print("Murmur starting...")
        print(f"  Hotkey: {config.hotkey}")
        print(f"  Model: {config.model}")
        print(f"  Sound: {config.sound}")
        print()
        print("Press the hotkey to start/stop recording.")
        print("Text will be typed at your cursor position.")
        print(f"Logs: ~/Library/Logs/Murmur/")
        print()

        app = MurmurApp(config)
        notify("Murmur", "Murmur is ready", sound=False)
        app.run()
    except FileNotFoundError as e:
        log.error(f"Setup error: {e}")
        print(f"Error: {e}")
        print("\nRun ./setup.sh first to build whisper.cpp and download the model.")
        exit(1)
    except Exception as e:
        log.error(f"Fatal error: {e}")
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
