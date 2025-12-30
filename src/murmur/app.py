"""Menu bar application for Murmur."""

import subprocess
import threading
import time
from enum import Enum
from pathlib import Path

import rumps
from pynput import keyboard
from PyObjCTools import AppHelper

from .audio import StreamingRecorder
from .config import Config
from .inject import StreamingInjector
from .logger import log
from .notify import notify
from .transcribe import StreamingTranscriber, StreamingResult


class State(Enum):
    """Application states."""

    LOADING = "loading"
    IDLE = "idle"
    TRANSCRIBING = "transcribing"
    LIVE = "live"


# Menu bar icons for each state
ICONS = {
    State.LOADING: "⏳",
    State.IDLE: "🎤",
    State.TRANSCRIBING: "⏳",
    State.LIVE: "🔴",
}

# macOS system sounds
SOUNDS = {
    "start": "/System/Library/Sounds/Funk.aiff",
    "stop": "/System/Library/Sounds/Blow.aiff",
    "error": "/System/Library/Sounds/Basso.aiff",
}

class MurmurApp(rumps.App):
    """Murmur menu bar application."""

    # Streaming config
    INFERENCE_INTERVAL = 0.5  # Run inference every 500ms
    AUDIO_WINDOW = 10.0       # Use last 10s of audio for inference

    def __init__(self, config: Config):
        super().__init__("Murmur", quit_button=None)

        self.config = config
        self.state = State.LOADING
        self.title = ICONS[State.LOADING]
        self._transcription_start_time = 0
        self._last_toggle_time = 0

        # Initialize audio/injection (fast)
        self.streaming_recorder = StreamingRecorder(
            buffer_seconds=12.0,
            vad_threshold=0.01,
        )
        self.streaming_injector = StreamingInjector()

        # Transcriber loaded async (slow - model load)
        self.streaming_transcriber: StreamingTranscriber | None = None

        # Streaming state
        self._streaming_thread: threading.Thread | None = None
        self._streaming_stop = threading.Event()

        # Setup menu
        self.status_item = rumps.MenuItem("Status: Loading model...")
        self.status_item.set_callback(None)
        self.menu = [
            self.status_item,
            None,
            rumps.MenuItem("Quit", callback=self._quit),
        ]

        # Start hotkey listener
        self._start_hotkey_listener()

        # Load model in background thread
        def load_model():
            try:
                transcriber = StreamingTranscriber(model_path=config.model_path)
                AppHelper.callAfter(lambda: self._on_model_loaded(transcriber))
            except Exception as e:
                log.error(f"Model load failed: {e}")
                AppHelper.callAfter(lambda: self._on_model_load_failed(str(e)))

        threading.Thread(target=load_model, daemon=True).start()
        log.info(f"Murmur starting (hotkey={config.hotkey}, model={config.model})")

    def _on_model_loaded(self, transcriber: StreamingTranscriber) -> None:
        """Called when model finishes loading."""
        self.streaming_transcriber = transcriber
        self._set_state(State.IDLE)
        notify("Murmur", "Ready", sound=False)
        log.info("Model loaded, ready")

    def _on_model_load_failed(self, error: str) -> None:
        """Called if model fails to load."""
        self.status_item.title = f"Error: {error[:30]}"
        notify("Murmur", f"Failed: {error[:50]}", sound=False)
        log.error(f"Model load failed: {error}")

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
        log.debug(f"Hotkey listener started for {hotkey} (toggle on press)")

    def _toggle(self) -> None:
        """Toggle between idle and live streaming states."""
        # Debounce: ignore rapid key repeats (200ms minimum)
        now = time.time()
        if now - self._last_toggle_time < 0.2:
            log.debug(f"Toggle ignored (debounce: {now - self._last_toggle_time:.3f}s)")
            return

        log.debug(f"Toggle triggered. Current state: {self.state}")
        self._last_toggle_time = now

        if self.state == State.LOADING:
            log.debug("Model still loading, ignoring toggle")
            return
        elif self.state == State.IDLE:
            self._start_live_streaming()
        elif self.state == State.LIVE:
            self._stop_live_streaming()

    def _start_live_streaming(self) -> None:
        """Start live streaming transcription."""
        log.info("Live streaming started")
        self._set_state(State.LIVE)
        self._play_sound("start")

        # Reset all streaming components
        self.streaming_transcriber.reset()
        self.streaming_injector.reset()
        self._streaming_stop.clear()

        # Start audio capture
        self.streaming_recorder.start()

        # Start inference loop in background thread
        self._streaming_thread = threading.Thread(
            target=self._streaming_loop,
            daemon=True,
        )
        self._streaming_thread.start()

    def _stop_live_streaming(self) -> None:
        """Stop live streaming and do final transcription."""
        log.info("Live streaming stopped, running final pass")
        self._set_state(State.TRANSCRIBING)  # Immediate UI feedback
        self._play_sound("stop")

        # Signal streaming loop to stop
        self._streaming_stop.set()

        # Wait for streaming thread to finish (with timeout)
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=1.0)

        # Get full audio for final pass (as numpy)
        full_audio = self.streaming_recorder.stop(as_numpy=True)

        if len(full_audio) > 1600:  # >0.1s of audio
            # Run final transcription on full audio
            def final_transcribe():
                try:
                    result = self.streaming_transcriber.process_audio(
                        full_audio,
                        is_final=True,
                    )
                    AppHelper.callAfter(lambda: self._on_streaming_complete(result))
                except Exception as e:
                    log.error(f"Final transcription error: {e}")
                    AppHelper.callAfter(lambda: self._on_streaming_complete(None))

            thread = threading.Thread(target=final_transcribe, daemon=True)
            thread.start()
        else:
            self._set_state(State.IDLE)

    def _streaming_loop(self) -> None:
        """Background loop that runs inference periodically."""
        log.debug("Streaming inference loop started")
        last_inference = 0.0

        while not self._streaming_stop.is_set():
            now = time.time()

            # Check if it's time for inference
            if now - last_inference >= self.INFERENCE_INTERVAL:
                # Only run inference if there's speech activity or buffer has content
                if self.streaming_recorder.is_speech_active() or self.streaming_recorder.buffer_duration > 1.0:
                    audio = self.streaming_recorder.get_audio_window(self.AUDIO_WINDOW)

                    if len(audio) > 1600:  # >0.1s of audio
                        silence = self.streaming_recorder.silence_duration()
                        result = self.streaming_transcriber.process_audio(
                            audio,
                            silence_duration=silence,
                            is_final=False,
                        )

                        if result and result.full_text:
                            # Update injection on main thread
                            AppHelper.callAfter(
                                lambda r=result: self._on_streaming_update(r)
                            )

                last_inference = now

            # Sleep briefly to avoid busy-waiting
            time.sleep(0.05)

        log.debug("Streaming inference loop stopped")

    def _on_streaming_update(self, result: StreamingResult) -> None:
        """Handle streaming transcription update."""
        if self.state != State.LIVE:
            return

        # Inject the current full text (diff-based)
        if result.full_text:
            self.streaming_injector.update(result.full_text)
            log.debug(f"Live: {result.full_text[:50]}...")

    def _on_streaming_complete(self, result: StreamingResult | None) -> None:
        """Handle final transcription completion."""
        if result and result.full_text:
            # Force final update to ensure all text is typed
            self.streaming_injector.update(result.full_text, force=True)
            log.info(f"Final transcription: {result.full_text[:50]}...")
        else:
            log.warning("No final transcription result")

        self._set_state(State.IDLE)

    def _set_state(self, state: State) -> None:
        """Update application state and UI."""
        self.state = state
        self.title = ICONS[state]

        status_text = {
            State.LOADING: "Status: Loading model...",
            State.IDLE: "Status: Idle",
            State.TRANSCRIBING: "Status: Transcribing...",
            State.LIVE: "Status: 🔴 Live",
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

        print("Murmur starting (LIVE MODE)...")
        print(f"  Hotkey: {config.hotkey}")
        print(f"  Model: {config.model}")
        print(f"  Sound: {config.sound}")
        print()
        print("Press the hotkey to start/stop live streaming.")
        print("Text will appear as you speak (with corrections).")
        print(f"Logs: ~/Library/Logs/Murmur/")
        print()

        app = MurmurApp(config)
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
