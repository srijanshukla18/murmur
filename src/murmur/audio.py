"""Audio capture for Murmur."""

import io
import wave
import threading
import time
from collections import deque
from typing import Optional, Callable

import numpy as np
import sounddevice as sd


class RingBuffer:
    """Thread-safe ring buffer for audio samples."""

    def __init__(self, max_seconds: float, sample_rate: int = 16000):
        self.max_samples = int(max_seconds * sample_rate)
        self.sample_rate = sample_rate
        self._buffer: deque[np.ndarray] = deque()
        self._total_samples = 0
        self._lock = threading.Lock()

    def append(self, chunk: np.ndarray) -> None:
        """Add audio chunk, discarding oldest if over capacity."""
        with self._lock:
            self._buffer.append(chunk.copy())
            self._total_samples += len(chunk)
            # Trim oldest chunks if over capacity
            while self._total_samples > self.max_samples and self._buffer:
                oldest = self._buffer.popleft()
                self._total_samples -= len(oldest)

    def get_audio(self, last_seconds: Optional[float] = None) -> np.ndarray:
        """Get audio from buffer (last N seconds or all)."""
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            audio = np.concatenate(list(self._buffer))
            if last_seconds is not None:
                max_samples = int(last_seconds * self.sample_rate)
                if len(audio) > max_samples:
                    audio = audio[-max_samples:]
            return audio

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._total_samples = 0

    @property
    def duration(self) -> float:
        """Current buffer duration in seconds."""
        with self._lock:
            return self._total_samples / self.sample_rate


class VAD:
    """Simple Voice Activity Detection using RMS energy."""

    def __init__(
        self,
        threshold: float = 0.01,
        speech_pad_ms: int = 300,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.speech_pad_samples = int(speech_pad_ms * sample_rate / 1000)
        self.sample_rate = sample_rate
        self._last_speech_time = 0.0
        self._is_speaking = False

    def process(self, audio: np.ndarray) -> bool:
        """Check if audio contains speech. Returns True if speech detected."""
        if len(audio) == 0:
            return False
        rms = np.sqrt(np.mean(audio**2))
        now = time.time()
        if rms > self.threshold:
            self._last_speech_time = now
            self._is_speaking = True
            return True
        # Pad silence after speech
        if self._is_speaking and (now - self._last_speech_time) < (self.speech_pad_samples / self.sample_rate):
            return True
        self._is_speaking = False
        return False

    def silence_duration(self) -> float:
        """How long since last speech detected."""
        if self._is_speaking:
            return 0.0
        return time.time() - self._last_speech_time

    def reset(self) -> None:
        """Reset VAD state."""
        self._last_speech_time = 0.0
        self._is_speaking = False

    @property
    def is_speaking(self) -> bool:
        """Whether speech is currently detected."""
        return self._is_speaking


class StreamingRecorder:
    """Records audio with ring buffer for live streaming transcription."""

    SAMPLE_RATE = 16000
    CHANNELS = 1

    def __init__(
        self,
        buffer_seconds: float = 12.0,
        vad_threshold: float = 0.01,
        vad_speech_pad_ms: int = 300,
        audio_chunk_ms: int = 100,
        on_audio_chunk: Optional[Callable[[np.ndarray], None]] = None,
    ):
        self.ring_buffer = RingBuffer(buffer_seconds, self.SAMPLE_RATE)
        self.vad = VAD(
            threshold=vad_threshold,
            speech_pad_ms=vad_speech_pad_ms,
            sample_rate=self.SAMPLE_RATE,
        )
        self._on_audio_chunk = on_audio_chunk
        self._recording = False
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._full_buffer: list[np.ndarray] = []  # Keep all audio for final pass
        self._audio_chunk_ms = audio_chunk_ms

    def start(self) -> None:
        """Start streaming audio capture."""
        with self._lock:
            if self._recording:
                return
            self.ring_buffer.clear()
            self.vad.reset()
            self._full_buffer = []
            self._recording = True
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=max(
                    1, int(self.SAMPLE_RATE * (self._audio_chunk_ms / 1000.0))
                ),
            )
            self._stream.start()

    def stop(self, as_numpy: bool = True) -> np.ndarray | bytes:
        """Stop recording and return full audio.

        Args:
            as_numpy: If True, return float32 numpy array. If False, return WAV bytes.
        """
        with self._lock:
            if not self._recording:
                return np.array([], dtype=np.float32) if as_numpy else b""
            self._recording = False
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            if not self._full_buffer:
                return np.array([], dtype=np.float32) if as_numpy else b""
            audio = np.concatenate(self._full_buffer)
            self._full_buffer = []
            return audio if as_numpy else self._to_wav(audio)

    def get_audio_window(self, last_seconds: Optional[float] = None) -> np.ndarray:
        """Get audio from ring buffer for inference."""
        return self.ring_buffer.get_audio(last_seconds)

    def consume_audio(self, seconds: float) -> None:
        """Remove old audio from the buffer (it has been transcribed)."""
        # We don't actually remove it from the ring buffer implementation easily,
        # but for this specific "clean slate" fix, clearing the buffer
        # when a major commit happens is the safest way to prevent loops.
        # However, a hard clear might lose the very start of the next word.
        # A better approach for the ring buffer is to just reset it if we trust
        # the prompt to carry the context.
        with self._lock:
            self.ring_buffer.clear()
            self.vad.reset()  # Reset VAD too so we don't carry over old speech state

    def is_speech_active(self) -> bool:
        """Check if speech is currently detected."""
        return self.vad.is_speaking

    def silence_duration(self) -> float:
        """Get duration of current silence."""
        return self.vad.silence_duration()

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """Audio stream callback."""
        chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        with self._lock:
            if not self._recording:
                return
            self.ring_buffer.append(chunk)
            self._full_buffer.append(chunk.copy())
            self.vad.process(chunk)
        if self._on_audio_chunk:
            self._on_audio_chunk(chunk)

    def _to_wav(self, audio: np.ndarray) -> bytes:
        """Convert float32 audio to WAV bytes."""
        audio_int16 = (audio * 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        return buffer.getvalue()

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    @property
    def buffer_duration(self) -> float:
        """Current ring buffer duration."""
        return self.ring_buffer.duration
