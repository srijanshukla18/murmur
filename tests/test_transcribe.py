"""Tests for src/murmur/transcribe.py"""

import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Mock pywhispercpp before importing
pywhispercpp_mock = MagicMock()
sys.modules['pywhispercpp'] = pywhispercpp_mock
sys.modules['pywhispercpp.model'] = pywhispercpp_mock.model
pywhispercpp_mock.model.Model = MagicMock()

from murmur.transcribe import StreamingResult, StreamingTranscriber


class TestStreamingResult:
    """Test StreamingResult dataclass."""

    def test_streaming_result_creation(self):
        result = StreamingResult(
            committed_text="hello",
            pending_text="world",
            full_text="hello world",
            is_final=False,
        )
        assert result.committed_text == "hello"
        assert result.pending_text == "world"
        assert result.full_text == "hello world"
        assert result.is_final is False

    def test_streaming_result_final(self):
        result = StreamingResult(
            committed_text="final text",
            pending_text="",
            full_text="final text",
            is_final=True,
        )
        assert result.is_final is True

    def test_streaming_result_empty(self):
        result = StreamingResult(
            committed_text="",
            pending_text="",
            full_text="",
            is_final=False,
        )
        assert result.committed_text == ""
        assert result.pending_text == ""
        assert result.full_text == ""


class TestStreamingTranscriberInit:
    """Test StreamingTranscriber initialization."""

    def test_init_stores_model_path(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        assert transcriber.model_path == model_path

    def test_init_default_stability_count(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        assert transcriber._stability_count_required == 2

    def test_init_custom_stability_count(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path, stability_count=5)
        assert transcriber._stability_count_required == 5

    def test_init_default_silence_commit(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        assert transcriber._silence_commit_seconds == 0.6

    def test_init_custom_silence_commit(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path, silence_commit_ms=1000)
        assert transcriber._silence_commit_seconds == 1.0

    def test_init_default_prompt_max_words(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        assert transcriber._prompt_max_words == 50

    def test_init_custom_prompt_max_words(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path, prompt_max_words=100)
        assert transcriber._prompt_max_words == 100

    def test_init_default_overlap_max_words(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        assert transcriber._overlap_max_words == 20

    def test_init_custom_overlap_max_words(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path, overlap_max_words=30)
        assert transcriber._overlap_max_words == 30

    def test_init_default_use_initial_prompt(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        assert transcriber._use_initial_prompt is True

    def test_init_custom_use_initial_prompt(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path, use_initial_prompt=False)
        assert transcriber._use_initial_prompt is False

    def test_init_min_audio_samples(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path, min_audio_seconds=0.5)
        assert transcriber._min_audio_samples == 8000

    def test_init_on_update_callback(self, temp_dir):
        model_path = temp_dir / "model.bin"
        callback = MagicMock()
        transcriber = StreamingTranscriber(model_path=model_path, on_update=callback)
        assert transcriber._on_update is callback

    def test_init_loads_model(self, temp_dir):
        model_path = temp_dir / "model.bin"
        pywhispercpp_mock.model.Model.reset_mock()
        transcriber = StreamingTranscriber(model_path=model_path)
        pywhispercpp_mock.model.Model.assert_called_once()

    def test_init_empty_state(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        assert transcriber._committed_text == ""
        assert transcriber._pending_text == ""
        assert transcriber._last_full_text == ""
        assert transcriber._stability_count == 0


class TestStreamingTranscriberReset:
    """Test StreamingTranscriber reset method."""

    def test_reset_clears_committed_text(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        transcriber._committed_text = "some text"
        transcriber.reset()
        assert transcriber._committed_text == ""

    def test_reset_clears_pending_text(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        transcriber._pending_text = "pending"
        transcriber.reset()
        assert transcriber._pending_text == ""

    def test_reset_clears_last_full_text(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        transcriber._last_full_text = "last"
        transcriber.reset()
        assert transcriber._last_full_text == ""

    def test_reset_clears_stability_count(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        transcriber._stability_count = 5
        transcriber.reset()
        assert transcriber._stability_count == 0


class TestStreamingTranscriberProperties:
    """Test StreamingTranscriber properties."""

    def test_committed_text_property(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        transcriber._committed_text = "committed"
        assert transcriber.committed_text == "committed"

    def test_pending_text_property(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        transcriber._pending_text = "pending"
        assert transcriber.pending_text == "pending"

    def test_properties_thread_safe(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        transcriber._committed_text = "test"
        transcriber._pending_text = "pending"

        results = []

        def read_props():
            for _ in range(100):
                results.append((transcriber.committed_text, transcriber.pending_text))

        threads = [threading.Thread(target=read_props) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 300


class TestStreamingTranscriberProcessAudio:
    """Test process_audio method."""

    def test_process_audio_none_returns_none(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        result = transcriber.process_audio(None)
        assert result is None

    def test_process_audio_too_short_returns_none(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path, min_audio_seconds=1.0)
        short_audio = np.zeros(100, dtype=np.float32)
        result = transcriber.process_audio(short_audio)
        assert result is None

    def test_process_audio_returns_streaming_result(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)

        mock_model = MagicMock()

        def mock_transcribe(audio, new_segment_callback=None, initial_prompt=None):
            if new_segment_callback:
                seg = MagicMock()
                seg.text = "hello world"
                new_segment_callback(seg)

        mock_model.transcribe = mock_transcribe
        transcriber._model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        result = transcriber.process_audio(audio)
        assert isinstance(result, StreamingResult)

    def test_process_audio_is_final_commits_all(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)

        mock_model = MagicMock()

        def mock_transcribe(audio, new_segment_callback=None, initial_prompt=None):
            if new_segment_callback:
                seg = MagicMock()
                seg.text = "final text"
                new_segment_callback(seg)

        mock_model.transcribe = mock_transcribe
        transcriber._model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        result = transcriber.process_audio(audio, is_final=True)
        assert result.is_final is True
        assert result.committed_text == "final text"
        assert result.pending_text == ""


class TestStreamingTranscriberCleanOutput:
    """Test _clean_output method."""

    def test_clean_output_none_returns_none(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        assert transcriber._clean_output(None) is None

    def test_clean_output_empty_returns_none(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        assert transcriber._clean_output("") is None

    def test_clean_output_whitespace_only_returns_none(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        assert transcriber._clean_output("   ") is None

    def test_clean_output_removes_brackets(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        result = transcriber._clean_output("hello [noise] world")
        assert "[noise]" not in result
        assert "hello" in result
        assert "world" in result

    def test_clean_output_removes_music_hallucination(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        result = transcriber._clean_output("(music) hello world")
        assert "(music)" not in result
        assert "hello world" in result

    def test_clean_output_removes_silence_hallucination(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        result = transcriber._clean_output("(silence) test")
        assert "(silence)" not in result

    def test_clean_output_removes_thank_you(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        result = transcriber._clean_output("Thank you. actual text")
        assert "Thank you." not in result

    def test_clean_output_removes_blank_audio(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        result = transcriber._clean_output("[BLANK_AUDIO] text")
        assert "[BLANK_AUDIO]" not in result

    def test_clean_output_normalizes_whitespace(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        result = transcriber._clean_output("hello    world")
        assert result == "hello world"


class TestStreamingTranscriberMergeWithCommitted:
    """Test _merge_with_committed method."""

    def test_merge_empty_committed_returns_new(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        result = transcriber._merge_with_committed("", "new text")
        assert result == "new text"

    def test_merge_empty_new_returns_committed(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        result = transcriber._merge_with_committed("committed", "")
        assert result == "committed"

    def test_merge_new_starts_with_committed(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        result = transcriber._merge_with_committed("hello", "hello world")
        assert result == "hello world"

    def test_merge_finds_word_overlap(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        result = transcriber._merge_with_committed("hello world", "world how are you")
        assert "hello world how are you" in result

    def test_merge_no_overlap_appends(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        result = transcriber._merge_with_committed("abc", "xyz")
        assert result == "abc xyz"


class TestStreamingTranscriberUpdateStability:
    """Test _update_stability method."""

    def test_update_stability_increments_on_same_text(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        transcriber._last_full_text = "same text"
        transcriber._stability_count = 0

        transcriber._update_stability("same text", silence_duration=0.0, is_final=False)
        assert transcriber._stability_count == 1

    def test_update_stability_resets_on_different_text(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)
        transcriber._last_full_text = "old text"
        transcriber._stability_count = 5

        transcriber._update_stability("new text", silence_duration=0.0, is_final=False)
        assert transcriber._stability_count == 0

    def test_update_stability_commits_on_threshold(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path, stability_count=2)
        transcriber._last_full_text = "stable text"
        transcriber._stability_count = 1

        result = transcriber._update_stability("stable text", silence_duration=0.0, is_final=False)
        assert transcriber._committed_text == "stable text"

    def test_update_stability_commits_on_silence(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path, silence_commit_ms=500)
        transcriber._last_full_text = ""
        transcriber._stability_count = 0

        result = transcriber._update_stability("silent commit", silence_duration=0.6, is_final=False)
        assert transcriber._committed_text == "silent commit"

    def test_update_stability_final_commits_all(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path)

        result = transcriber._update_stability("final text", silence_duration=0.0, is_final=True)
        assert result.is_final is True
        assert result.committed_text == "final text"
        assert result.pending_text == ""

    def test_update_stability_calls_on_update(self, temp_dir):
        model_path = temp_dir / "model.bin"
        callback = MagicMock()
        transcriber = StreamingTranscriber(model_path=model_path, on_update=callback)

        transcriber._update_stability("some text", silence_duration=0.0, is_final=False)
        callback.assert_called_once()

    def test_update_stability_sets_pending_text(self, temp_dir):
        model_path = temp_dir / "model.bin"
        transcriber = StreamingTranscriber(model_path=model_path, stability_count=10)
        transcriber._committed_text = "hello"
        transcriber._last_full_text = ""
        transcriber._stability_count = 0

        result = transcriber._update_stability("hello world", silence_duration=0.0, is_final=False)
        assert transcriber._pending_text == "world"
