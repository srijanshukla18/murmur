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

    def __init__(
        self,
        model_path: Path,
        stability_count: int = 2,
        silence_commit_ms: int = 600,
        prompt_max_words: int = 50,
        overlap_max_words: int = 20,
        min_audio_seconds: float = 0.1,
        use_initial_prompt: bool = True,
        on_update: Optional[Callable[[StreamingResult], None]] = None,
    ):
        self.model_path = model_path
        self._on_update = on_update
        self._stability_count_required = stability_count
        self._silence_commit_seconds = silence_commit_ms / 1000.0
        self._prompt_max_words = prompt_max_words
        self._overlap_max_words = overlap_max_words
        self._min_audio_samples = max(1, int(16000 * min_audio_seconds))
        self._use_initial_prompt = use_initial_prompt

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
        self._last_committed_at_clear = ""
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
        if audio is None or len(audio) < self._min_audio_samples:
            return None

        # Run inference on the sliding window
        text = self._transcribe(audio)

        # Silence handling: update stability even if text is None
        # to allow silence-based commits of pending text.
        with self._lock:
            return self._update_stability(text, silence_duration, is_final)

    def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Run whisper inference on audio."""
        try:
            # Use committed text as prompt for continuity
            prompt = None
            with self._lock:
                if self._committed_text:
                    words = self._committed_text.split()[-self._prompt_max_words:]
                    prompt = " ".join(words)

            segments = []
            def on_segment(seg):
                segments.append(seg.text)

            # Transcribe with prompt for context (if enabled)
            if prompt and self._use_initial_prompt:
                self._model.transcribe(
                    audio,
                    new_segment_callback=on_segment,
                    initial_prompt=prompt,
                )
            else:
                self._model.transcribe(
                    audio,
                    new_segment_callback=on_segment,
                )

            if not segments:
                return None

            text = " ".join(segments)
            cleaned = self._clean_output(text)
            
            # Anti-loop safety: if the output starts with the prompt (or suffix of it),
            # strip it. Whisper often repeats the prompt in the output.
            if cleaned and prompt:
                 # Check for near-exact suffix match of prompt in output
                 prompt_words = prompt.split()
                 output_words = cleaned.split()
                 
                 # Only check against the words we actually sent as prompt
                 # (Limit check to prompt length)
                 max_check = min(len(prompt_words), len(output_words), 20)
                 
                 overlap_len = 0
                 for i in range(max_check, 0, -1):
                     if output_words[:i] == prompt_words[-i:]:
                         overlap_len = i
                         break
                 
                 if overlap_len > 0:
                     # Strip the repetitve prefix
                     cleaned = " ".join(output_words[overlap_len:])

            return cleaned if cleaned else None

        except Exception as e:
            log.debug(f"Transcription error: {e}")
            return None

    def _update_stability(
        self,
        new_text: Optional[str],
        silence_duration: float,
        is_final: bool,
    ) -> StreamingResult:
        """Update committed/pending text based on stability rules."""
        if new_text is None:
            new_text = ""
        new_text = new_text.strip()

        # Merge with already committed prefix
        merged_text = self._merge_with_committed(self._committed_text, new_text)

        # Final pass: commit everything
        if is_final:
            self._committed_text = merged_text
            self._pending_text = ""
            return StreamingResult(
                committed_text=self._committed_text,
                pending_text="",
                full_text=self._committed_text,
                is_final=True,
            )

        # Check stability
        if merged_text == self._last_full_text:
            self._stability_count += 1
        else:
            self._stability_count = 0
        self._last_full_text = merged_text

        # Determine what to commit
        should_commit = (
            self._stability_count >= self._stability_count_required
            or silence_duration >= self._silence_commit_seconds
        )

        if should_commit and merged_text:
            log.debug(f"Commit: '{merged_text[-30:]}' (stability={self._stability_count}, silence={silence_duration:.1f}s)")
            self._committed_text = merged_text
            self._pending_text = ""
        else:
            if merged_text.startswith(self._committed_text):
                self._pending_text = merged_text[len(self._committed_text):].strip()
            else:
                self._pending_text = merged_text

        result = StreamingResult(
            committed_text=self._committed_text,
            pending_text=self._pending_text,
            full_text=merged_text,
            is_final=False,
        )

        if self._on_update:
            self._on_update(result)

        return result

    def _merge_with_committed(self, committed: str, new_text: str) -> str:
        """Merge new text with committed prefix using word overlap."""
        if not committed:
            return new_text
        if not new_text:
            return committed
        if new_text.startswith(committed):
            return new_text

        committed_words = committed.split()
        new_words = new_text.split()
        max_overlap = min(self._overlap_max_words, len(committed_words), len(new_words))

        for overlap in range(max_overlap, 0, -1):
            if committed_words[-overlap:] == new_words[:overlap]:
                merged_words = committed_words + new_words[overlap:]
                log.debug(f"Merge: overlap of {overlap} words found")
                return " ".join(merged_words)

        log.debug(f"Merge: no overlap found, forcing append")
        return f"{committed} {new_text}".strip()

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
