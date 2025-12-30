"""Audio capture for Murmur."""

import io
import wave
import threading
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioRecorder:
    """Records audio from the microphone."""

    SAMPLE_RATE = 16000  # Whisper expects 16kHz
    CHANNELS = 1  # Mono

    def __init__(self):
        self._buffer: list[np.ndarray] = []
        self._recording = False
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start recording audio."""
        with self._lock:
            if self._recording:
                return

            self._buffer = []
            self._recording = True

            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype=np.float32,
                callback=self._audio_callback,
            )
            self._stream.start()

    def stop(self) -> bytes:
        """Stop recording and return WAV data."""
        with self._lock:
            if not self._recording:
                return b""

            self._recording = False

            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            if not self._buffer:
                return b""

            # Concatenate all audio chunks
            audio = np.concatenate(self._buffer)
            self._buffer = []

            # Convert to WAV format
            return self._to_wav(audio)

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """Callback for audio stream."""
        if self._recording:
            self._buffer.append(indata.copy())

    def _to_wav(self, audio: np.ndarray) -> bytes:
        """Convert float32 audio to WAV bytes."""
        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # Write to WAV buffer
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        return buffer.getvalue()

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording
