"""Tests for src/murmur/audio.py"""

import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from murmur.audio import RingBuffer, VAD, StreamingRecorder


class TestRingBuffer:
    """Test RingBuffer class."""

    def test_init_default_sample_rate(self):
        buffer = RingBuffer(max_seconds=5.0)
        assert buffer.sample_rate == 16000

    def test_init_custom_sample_rate(self):
        buffer = RingBuffer(max_seconds=5.0, sample_rate=44100)
        assert buffer.sample_rate == 44100

    def test_init_max_samples_calculation(self):
        buffer = RingBuffer(max_seconds=5.0, sample_rate=16000)
        assert buffer.max_samples == 80000

    def test_duration_empty_buffer(self):
        buffer = RingBuffer(max_seconds=5.0)
        assert buffer.duration == 0.0

    def test_append_single_chunk(self):
        buffer = RingBuffer(max_seconds=5.0)
        chunk = np.zeros(16000, dtype=np.float32)
        buffer.append(chunk)
        assert buffer.duration == 1.0

    def test_append_multiple_chunks(self):
        buffer = RingBuffer(max_seconds=5.0)
        chunk = np.zeros(8000, dtype=np.float32)
        buffer.append(chunk)
        buffer.append(chunk)
        assert buffer.duration == 1.0

    def test_append_over_capacity_discards_oldest(self):
        buffer = RingBuffer(max_seconds=1.0)
        chunk = np.zeros(16000, dtype=np.float32)
        buffer.append(chunk)
        buffer.append(chunk)
        assert buffer.duration <= 1.0

    def test_get_audio_empty_buffer(self):
        buffer = RingBuffer(max_seconds=5.0)
        audio = buffer.get_audio()
        assert len(audio) == 0
        assert audio.dtype == np.float32

    def test_get_audio_returns_all(self):
        buffer = RingBuffer(max_seconds=5.0)
        chunk = np.ones(16000, dtype=np.float32)
        buffer.append(chunk)
        audio = buffer.get_audio()
        assert len(audio) == 16000
        assert np.allclose(audio, chunk)

    def test_get_audio_last_seconds(self):
        buffer = RingBuffer(max_seconds=5.0)
        chunk1 = np.ones(16000, dtype=np.float32) * 1.0
        chunk2 = np.ones(16000, dtype=np.float32) * 2.0
        buffer.append(chunk1)
        buffer.append(chunk2)
        audio = buffer.get_audio(last_seconds=1.0)
        assert len(audio) == 16000
        assert np.allclose(audio, chunk2)

    def test_get_audio_last_seconds_more_than_available(self):
        buffer = RingBuffer(max_seconds=5.0)
        chunk = np.ones(8000, dtype=np.float32)
        buffer.append(chunk)
        audio = buffer.get_audio(last_seconds=1.0)
        assert len(audio) == 8000

    def test_clear(self):
        buffer = RingBuffer(max_seconds=5.0)
        chunk = np.zeros(16000, dtype=np.float32)
        buffer.append(chunk)
        buffer.clear()
        assert buffer.duration == 0.0

    def test_prune_removes_oldest(self):
        buffer = RingBuffer(max_seconds=5.0)
        chunk1 = np.ones(16000, dtype=np.float32) * 1.0
        chunk2 = np.ones(16000, dtype=np.float32) * 2.0
        buffer.append(chunk1)
        buffer.append(chunk2)
        buffer.prune(1.0)
        audio = buffer.get_audio()
        assert len(audio) == 16000
        assert np.allclose(audio, chunk2)

    def test_prune_partial_chunk(self):
        buffer = RingBuffer(max_seconds=5.0)
        chunk = np.ones(16000, dtype=np.float32)
        buffer.append(chunk)
        buffer.prune(0.5)
        audio = buffer.get_audio()
        assert len(audio) == 8000

    def test_prune_empty_buffer(self):
        buffer = RingBuffer(max_seconds=5.0)
        buffer.prune(1.0)
        assert buffer.duration == 0.0

    def test_thread_safety_append(self):
        buffer = RingBuffer(max_seconds=10.0)
        errors = []

        def append_chunks():
            try:
                for _ in range(100):
                    chunk = np.zeros(160, dtype=np.float32)
                    buffer.append(chunk)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=append_chunks) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_append_copies_data(self):
        buffer = RingBuffer(max_seconds=5.0)
        chunk = np.ones(16000, dtype=np.float32)
        buffer.append(chunk)
        chunk[:] = 2.0
        audio = buffer.get_audio()
        assert np.allclose(audio, np.ones(16000))


class TestVAD:
    """Test VAD (Voice Activity Detection) class."""

    def test_init_default_values(self):
        vad = VAD()
        assert vad.threshold == 0.01
        assert vad.sample_rate == 16000

    def test_init_custom_threshold(self):
        vad = VAD(threshold=0.02)
        assert vad.threshold == 0.02

    def test_init_custom_speech_pad(self):
        vad = VAD(speech_pad_ms=500)
        assert vad.speech_pad_samples == 8000

    def test_process_empty_audio(self):
        vad = VAD()
        result = vad.process(np.array([], dtype=np.float32))
        assert result is False

    def test_process_silent_audio(self):
        vad = VAD(threshold=0.01)
        silent = np.zeros(16000, dtype=np.float32)
        result = vad.process(silent)
        assert result is False

    def test_process_loud_audio(self):
        vad = VAD(threshold=0.01)
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        loud = 0.5 * np.sin(2 * np.pi * 440 * t)
        result = vad.process(loud)
        assert result is True

    def test_is_speaking_after_speech(self):
        vad = VAD(threshold=0.01)
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        loud = 0.5 * np.sin(2 * np.pi * 440 * t)
        vad.process(loud)
        assert vad.is_speaking is True

    def test_is_speaking_after_silence(self):
        vad = VAD(threshold=0.01, speech_pad_ms=0)
        silent = np.zeros(16000, dtype=np.float32)
        vad.process(silent)
        assert vad.is_speaking is False

    def test_speech_pad_extends_speaking(self):
        vad = VAD(threshold=0.01, speech_pad_ms=500)
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        loud = 0.5 * np.sin(2 * np.pi * 440 * t)
        vad.process(loud)

        silent = np.zeros(1600, dtype=np.float32)
        result = vad.process(silent)
        assert result is True

    def test_silence_duration_during_speech(self):
        vad = VAD(threshold=0.01)
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        loud = 0.5 * np.sin(2 * np.pi * 440 * t)
        vad.process(loud)
        assert vad.silence_duration() == 0.0

    def test_silence_duration_after_silence(self):
        vad = VAD(threshold=0.01, speech_pad_ms=0)
        silent = np.zeros(16000, dtype=np.float32)
        vad.process(silent)
        time.sleep(0.1)
        assert vad.silence_duration() >= 0.1

    def test_reset(self):
        vad = VAD()
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        loud = 0.5 * np.sin(2 * np.pi * 440 * t)
        vad.process(loud)
        vad.reset()
        assert vad.is_speaking is False


class TestStreamingRecorder:
    """Test StreamingRecorder class."""

    def test_init_defaults(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder()
            assert recorder.SAMPLE_RATE == 16000
            assert recorder.CHANNELS == 1
            assert recorder._recording is False

    def test_init_custom_buffer_seconds(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder(buffer_seconds=20.0)
            assert recorder.ring_buffer.max_samples == 320000

    def test_init_custom_vad_threshold(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder(vad_threshold=0.02)
            assert recorder.vad.threshold == 0.02

    def test_is_recording_false_initially(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder()
            assert recorder.is_recording is False

    def test_start_sets_recording_true(self):
        mock_stream = MagicMock()
        with patch('sounddevice.InputStream', return_value=mock_stream):
            recorder = StreamingRecorder()
            recorder.start()
            assert recorder.is_recording is True
            mock_stream.start.assert_called_once()

    def test_start_clears_buffers(self):
        mock_stream = MagicMock()
        with patch('sounddevice.InputStream', return_value=mock_stream):
            recorder = StreamingRecorder()
            recorder.ring_buffer.append(np.zeros(1600, dtype=np.float32))
            recorder.start()
            assert recorder.ring_buffer.duration == 0.0

    def test_start_idempotent(self):
        mock_stream = MagicMock()
        with patch('sounddevice.InputStream', return_value=mock_stream):
            recorder = StreamingRecorder()
            recorder.start()
            recorder.start()
            assert mock_stream.start.call_count == 1

    def test_stop_returns_empty_if_not_recording(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder()
            audio = recorder.stop()
            assert len(audio) == 0

    def test_stop_sets_recording_false(self):
        mock_stream = MagicMock()
        with patch('sounddevice.InputStream', return_value=mock_stream):
            recorder = StreamingRecorder()
            recorder.start()
            recorder.stop()
            assert recorder.is_recording is False

    def test_stop_closes_stream(self):
        mock_stream = MagicMock()
        with patch('sounddevice.InputStream', return_value=mock_stream):
            recorder = StreamingRecorder()
            recorder.start()
            recorder.stop()
            mock_stream.stop.assert_called_once()
            mock_stream.close.assert_called_once()

    def test_get_audio_window_returns_ring_buffer_audio(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder()
            recorder.ring_buffer.append(np.ones(16000, dtype=np.float32))
            audio = recorder.get_audio_window()
            assert len(audio) == 16000

    def test_get_audio_window_with_seconds(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder()
            recorder.ring_buffer.append(np.ones(32000, dtype=np.float32))
            audio = recorder.get_audio_window(last_seconds=1.0)
            assert len(audio) == 16000

    def test_consume_audio_prunes_buffer(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder()
            recorder.ring_buffer.append(np.ones(32000, dtype=np.float32))
            recorder.consume_audio(1.0)
            assert recorder.ring_buffer.duration == 1.0

    def test_consume_audio_zero_does_nothing(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder()
            recorder.ring_buffer.append(np.ones(16000, dtype=np.float32))
            recorder.consume_audio(0)
            assert recorder.ring_buffer.duration == 1.0

    def test_consume_audio_negative_does_nothing(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder()
            recorder.ring_buffer.append(np.ones(16000, dtype=np.float32))
            recorder.consume_audio(-1.0)
            assert recorder.ring_buffer.duration == 1.0

    def test_is_speech_active_delegates_to_vad(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder()
            t = np.linspace(0, 1, 16000, dtype=np.float32)
            loud = 0.5 * np.sin(2 * np.pi * 440 * t)
            recorder.vad.process(loud)
            assert recorder.is_speech_active() is True

    def test_silence_duration_delegates_to_vad(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder()
            assert recorder.silence_duration() >= 0.0

    def test_buffer_duration_property(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder()
            recorder.ring_buffer.append(np.ones(16000, dtype=np.float32))
            assert recorder.buffer_duration == 1.0

    def test_audio_callback_appends_to_buffers(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder()
            recorder._recording = True
            chunk = np.ones((1600, 1), dtype=np.float32)
            recorder._audio_callback(chunk, 1600, None, None)
            assert recorder.ring_buffer.duration > 0

    def test_audio_callback_invokes_on_audio_chunk(self):
        callback_received = []

        def on_chunk(chunk):
            callback_received.append(chunk)

        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder(on_audio_chunk=on_chunk)
            recorder._recording = True
            chunk = np.ones((1600, 1), dtype=np.float32)
            recorder._audio_callback(chunk, 1600, None, None)
            assert len(callback_received) == 1

    def test_audio_callback_ignores_when_not_recording(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder()
            recorder._recording = False
            chunk = np.ones((1600, 1), dtype=np.float32)
            recorder._audio_callback(chunk, 1600, None, None)
            assert recorder.ring_buffer.duration == 0

    def test_to_wav_converts_to_bytes(self):
        with patch('sounddevice.InputStream'):
            recorder = StreamingRecorder()
            audio = np.zeros(16000, dtype=np.float32)
            wav_bytes = recorder._to_wav(audio)
            assert isinstance(wav_bytes, bytes)
            assert len(wav_bytes) > 0
            assert wav_bytes[:4] == b'RIFF'

    def test_stop_as_wav_bytes(self):
        mock_stream = MagicMock()
        with patch('sounddevice.InputStream', return_value=mock_stream):
            recorder = StreamingRecorder()
            recorder.start()
            recorder._full_buffer.append(np.zeros(16000, dtype=np.float32))
            wav_bytes = recorder.stop(as_numpy=False)
            assert isinstance(wav_bytes, bytes)
            assert wav_bytes[:4] == b'RIFF'

    def test_stop_returns_concatenated_audio(self):
        mock_stream = MagicMock()
        with patch('sounddevice.InputStream', return_value=mock_stream):
            recorder = StreamingRecorder()
            recorder.start()
            recorder._full_buffer.append(np.ones(8000, dtype=np.float32))
            recorder._full_buffer.append(np.ones(8000, dtype=np.float32) * 2)
            audio = recorder.stop()
            assert len(audio) == 16000
            assert audio[0] == 1.0
            assert audio[8000] == 2.0
