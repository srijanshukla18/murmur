"""Pytest configuration and shared fixtures for Murmur tests."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_audio():
    """Generate sample audio data for testing."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio


@pytest.fixture
def silent_audio():
    """Generate silent audio data for testing."""
    sample_rate = 16000
    duration = 1.0
    return np.zeros(int(sample_rate * duration), dtype=np.float32)


@pytest.fixture
def loud_audio():
    """Generate loud audio data for testing (above VAD threshold)."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio


@pytest.fixture
def quiet_audio():
    """Generate quiet audio data for testing (below VAD threshold)."""
    sample_rate = 16000
    duration = 1.0
    return np.full(int(sample_rate * duration), 0.001, dtype=np.float32)


@pytest.fixture
def mock_quartz():
    """Mock Quartz framework for inject module tests."""
    with patch.dict('sys.modules', {
        'Quartz': MagicMock(),
    }):
        yield


@pytest.fixture
def mock_pynput():
    """Mock pynput for app module tests."""
    with patch.dict('sys.modules', {
        'pynput': MagicMock(),
        'pynput.keyboard': MagicMock(),
    }):
        yield


@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice for audio module tests."""
    with patch('sounddevice.InputStream') as mock_stream:
        mock_stream.return_value = MagicMock()
        yield mock_stream


@pytest.fixture
def sample_toml_config(temp_dir):
    """Create a sample TOML configuration file."""
    config_content = """
[murmur]
hotkey = "f8"
model = "tiny.en"
sound = false
toggle_debounce_seconds = 0.3

[streaming]
buffer_seconds = 15.0
audio_window_seconds = 8.0
inference_interval_seconds = 0.4
stability_count = 3
silence_commit_ms = 500

[injector]
max_updates_per_sec = 5
max_backspace_chars = 25
"""
    config_file = temp_dir / "murmur.toml"
    config_file.write_text(config_content)
    return config_file
