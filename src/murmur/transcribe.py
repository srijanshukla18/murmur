"""Whisper transcription interface for Murmur."""

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from pywhispercpp.model import Model

from .logger import log


@dataclass
class StreamingResult:
    """Result from streaming transcription."""
    committed_text: str  # Stable text that won't change
    pending_text: str    # Unstable text that may be revised
    full_text: str       # committed + pending
    is_final: bool       # True on final pass after stop


class StreamingTranscriber:
    """Embedded streaming transcriber using pywhispercpp."""

    # Stability rules
    STABILITY_COUNT = 2       # Passes before text is committed
    SILENCE_COMMIT_MS = 600   # Commit after this much silence

    def __init__(
        self,
        model_path: Path,
        on_update: Optional[Callable[[StreamingResult], None]] = None,
    ):
        self.model_path = model_path
        self._on_update = on_update

        # Load model once, keep in memory
        log.info(f"Loading whisper model: {model_path}")
        self._model = Model(
            str(model_path),
            print_realtime=False,
            print_progress=False,
            redirect_whispercpp_logs_to=os.devnull,
            n_threads=4,
        )
        log.info("Whisper model loaded (Metal GPU)")

        # State
        self._committed_text = ""
        self._pending_text = ""
        self._last_full_text = ""
        self._stability_count = 0
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Reset state for new session."""
        with self._lock:
            self._committed_text = ""
            self._pending_text = ""
            self._last_full_text = ""
            self._stability_count = 0

    def process_audio(
        self,
        audio: np.ndarray,
        silence_duration: float = 0.0,
        is_final: bool = False,
    ) -> Optional[StreamingResult]:
        """
        Process audio and return transcription result.

        Args:
            audio: float32 numpy array (16kHz mono)
            silence_duration: Current silence duration in seconds
            is_final: True for final pass after recording stops
        """
        if audio is None or len(audio) < 1600:  # <0.1s
            return None

        # Run inference with prompt for continuity
        text = self._transcribe(audio)
        if text is None:
            return None

        with self._lock:
            return self._update_stability(text, silence_duration, is_final)

    def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Run whisper inference on audio."""
        try:
            # Use committed text as prompt for continuity
            prompt = None
            with self._lock:
                if self._committed_text:
                    words = self._committed_text.split()[-50:]
                    prompt = " ".join(words)

            # Transcribe with callback for segments
            segments = []
            def on_segment(seg):
                segments.append(seg.text)

            if prompt:
                self._model.transcribe(
                    audio,
                    new_segment_callback=on_segment,
                    prompt=prompt,
                )
            else:
                self._model.transcribe(
                    audio,
                    new_segment_callback=on_segment,
                )

            if not segments:
                return None

            text = " ".join(segments)
            return self._clean_output(text)

        except Exception as e:
            log.debug(f"Transcription error: {e}")
            return None

    def _update_stability(
        self,
        new_text: str,
        silence_duration: float,
        is_final: bool,
    ) -> StreamingResult:
        """Update committed/pending text based on stability rules."""
        new_text = new_text.strip()

        # Final pass: commit everything
        if is_final:
            self._committed_text = new_text
            self._pending_text = ""
            return StreamingResult(
                committed_text=self._committed_text,
                pending_text="",
                full_text=new_text,
                is_final=True,
            )

        # Check stability
        if new_text == self._last_full_text:
            self._stability_count += 1
        else:
            self._stability_count = 0
        self._last_full_text = new_text

        # Determine what to commit
        should_commit = (
            self._stability_count >= self.STABILITY_COUNT
            or silence_duration >= (self.SILENCE_COMMIT_MS / 1000.0)
        )

        if should_commit and new_text:
            if new_text.startswith(self._committed_text):
                self._committed_text = new_text
                self._pending_text = ""
            elif self._committed_text and new_text != self._committed_text:
                common = self._find_common_prefix(self._committed_text, new_text)
                if len(common) >= len(self._committed_text) * 0.8:
                    self._committed_text = new_text
                    self._pending_text = ""
                else:
                    self._pending_text = new_text[len(self._committed_text):].strip()
            else:
                self._committed_text = new_text
                self._pending_text = ""
        else:
            if new_text.startswith(self._committed_text):
                self._pending_text = new_text[len(self._committed_text):].strip()
            else:
                self._pending_text = new_text

        result = StreamingResult(
            committed_text=self._committed_text,
            pending_text=self._pending_text,
            full_text=new_text,
            is_final=False,
        )

        if self._on_update:
            self._on_update(result)

        return result

    def _find_common_prefix(self, s1: str, s2: str) -> str:
        """Find common prefix between two strings, word-aligned."""
        words1 = s1.split()
        words2 = s2.split()
        common = []
        for w1, w2 in zip(words1, words2):
            if w1 == w2:
                common.append(w1)
            else:
                break
        return " ".join(common)

    def _clean_output(self, text: str) -> Optional[str]:
        """Clean whisper output."""
        if not text:
            return None

        # Remove bracket artifacts
        text = re.sub(r"\[.*?\]", "", text)

        # Remove hallucinations
        hallucinations = [
            "(music)", "(Music)", "[Music]", "(silence)", "(Silence)",
            "Thank you.", "Thanks for watching!", "Subscribe",
            "[BLANK_AUDIO]", "(BLANK_AUDIO)",
        ]
        for h in hallucinations:
            text = text.replace(h, "")

        text = " ".join(text.split())  # Normalize whitespace
        return text.strip() if text.strip() else None

    @property
    def committed_text(self) -> str:
        """Get current committed text."""
        with self._lock:
            return self._committed_text

    @property
    def pending_text(self) -> str:
        """Get current pending text."""
        with self._lock:
            return self._pending_text
