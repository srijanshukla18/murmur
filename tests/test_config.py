"""Tests for src/murmur/config.py"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from murmur.config import Config


class TestConfigDefaults:
    """Test Config default values."""

    def test_default_hotkey(self):
        config = Config()
        assert config.hotkey == "alt_r"

    def test_default_model(self):
        config = Config()
        assert config.model == "small.en"

    def test_default_sound(self):
        config = Config()
        assert config.sound is True

    def test_default_toggle_debounce(self):
        config = Config()
        assert config.toggle_debounce_seconds == 0.2

    def test_default_buffer_seconds(self):
        config = Config()
        assert config.buffer_seconds == 12.0

    def test_default_audio_window_seconds(self):
        config = Config()
        assert config.audio_window_seconds == 10.0

    def test_default_inference_interval_seconds(self):
        config = Config()
        assert config.inference_interval_seconds == 0.5

    def test_default_audio_chunk_ms(self):
        config = Config()
        assert config.audio_chunk_ms == 100

    def test_default_min_audio_seconds(self):
        config = Config()
        assert config.min_audio_seconds == 0.1

    def test_default_vad_threshold(self):
        config = Config()
        assert config.vad_threshold == 0.01

    def test_default_vad_speech_pad_ms(self):
        config = Config()
        assert config.vad_speech_pad_ms == 300

    def test_default_stability_count(self):
        config = Config()
        assert config.stability_count == 2

    def test_default_silence_commit_ms(self):
        config = Config()
        assert config.silence_commit_ms == 600

    def test_default_prompt_max_words(self):
        config = Config()
        assert config.prompt_max_words == 50

    def test_default_overlap_max_words(self):
        config = Config()
        assert config.overlap_max_words == 20

    def test_default_use_initial_prompt(self):
        config = Config()
        assert config.use_initial_prompt is True

    def test_default_consume_audio_on_commit(self):
        config = Config()
        assert config.consume_audio_on_commit is True

    def test_default_batch_mode(self):
        config = Config()
        assert config.batch_mode is False

    def test_default_batch_silence_threshold_ms(self):
        config = Config()
        assert config.batch_silence_threshold_ms == 500

    def test_default_max_updates_per_sec(self):
        config = Config()
        assert config.max_updates_per_sec == 4

    def test_default_max_backspace_chars(self):
        config = Config()
        assert config.max_backspace_chars == 30

    def test_default_keystroke_delay_seconds(self):
        config = Config()
        assert config.keystroke_delay_seconds == 0.002

    def test_default_backspace_delay_seconds(self):
        config = Config()
        assert config.backspace_delay_seconds == 0.001


class TestConfigNormalizeHotkey:
    """Test _normalize_hotkey static method."""

    def test_normalize_right_option(self):
        assert Config._normalize_hotkey("right_option") == "alt_r"

    def test_normalize_right_alt(self):
        assert Config._normalize_hotkey("right_alt") == "alt_r"

    def test_normalize_left_option(self):
        assert Config._normalize_hotkey("left_option") == "alt_l"

    def test_normalize_left_alt(self):
        assert Config._normalize_hotkey("left_alt") == "alt_l"

    def test_normalize_caps_lock(self):
        assert Config._normalize_hotkey("caps_lock") == "caps_lock"

    def test_normalize_f8(self):
        assert Config._normalize_hotkey("f8") == "f8"

    def test_normalize_f9(self):
        assert Config._normalize_hotkey("f9") == "f9"

    def test_normalize_f10(self):
        assert Config._normalize_hotkey("f10") == "f10"

    def test_normalize_unknown_key(self):
        assert Config._normalize_hotkey("unknown_key") == "unknown_key"

    def test_normalize_case_insensitive(self):
        assert Config._normalize_hotkey("RIGHT_OPTION") == "alt_r"
        assert Config._normalize_hotkey("Right_Option") == "alt_r"


class TestConfigMergeDicts:
    """Test _merge_dicts static method."""

    def test_merge_empty_dicts(self):
        result = Config._merge_dicts({}, {})
        assert result == {}

    def test_merge_with_empty_base(self):
        result = Config._merge_dicts({}, {"a": 1})
        assert result == {"a": 1}

    def test_merge_with_empty_override(self):
        result = Config._merge_dicts({"a": 1}, {})
        assert result == {"a": 1}

    def test_merge_simple_override(self):
        result = Config._merge_dicts({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_merge_non_overlapping(self):
        result = Config._merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_merge_nested_dicts(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = Config._merge_dicts(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_merge_nested_replaces_non_dict(self):
        base = {"a": 1}
        override = {"a": {"x": 2}}
        result = Config._merge_dicts(base, override)
        assert result == {"a": {"x": 2}}

    def test_merge_deeply_nested(self):
        base = {"a": {"b": {"c": 1}}}
        override = {"a": {"b": {"d": 2}}}
        result = Config._merge_dicts(base, override)
        assert result == {"a": {"b": {"c": 1, "d": 2}}}


class TestConfigLoadToml:
    """Test _load_toml static method."""

    def test_load_nonexistent_file(self, temp_dir):
        result = Config._load_toml(temp_dir / "nonexistent.toml")
        assert result == {}

    def test_load_valid_toml(self, temp_dir):
        toml_content = """
[murmur]
hotkey = "f9"
model = "base.en"
"""
        toml_file = temp_dir / "test.toml"
        toml_file.write_text(toml_content)
        result = Config._load_toml(toml_file)
        assert result == {"murmur": {"hotkey": "f9", "model": "base.en"}}

    def test_load_empty_toml(self, temp_dir):
        toml_file = temp_dir / "empty.toml"
        toml_file.write_text("")
        result = Config._load_toml(toml_file)
        assert result == {}


class TestConfigModelPath:
    """Test model_path property."""

    def test_model_path_default(self):
        config = Config()
        expected = config.whisper_path / "models" / "ggml-small.en.bin"
        assert config.model_path == expected

    def test_model_path_custom_model(self):
        config = Config()
        config.model = "tiny.en"
        expected = config.whisper_path / "models" / "ggml-tiny.en.bin"
        assert config.model_path == expected

    def test_model_path_custom_whisper_path(self, temp_dir):
        config = Config()
        config.whisper_path = temp_dir
        expected = temp_dir / "models" / "ggml-small.en.bin"
        assert config.model_path == expected


class TestConfigLoad:
    """Test Config.load() method."""

    def test_load_returns_config_instance(self):
        with patch.object(Config, '_config_paths', return_value=[]):
            with patch.dict(os.environ, {}, clear=True):
                config = Config.load()
                assert isinstance(config, Config)

    def test_load_env_hotkey_override(self):
        with patch.object(Config, '_config_paths', return_value=[]):
            with patch.dict(os.environ, {'MURMUR_HOTKEY': 'f10'}, clear=True):
                config = Config.load()
                assert config.hotkey == 'f10'

    def test_load_env_model_override(self):
        with patch.object(Config, '_config_paths', return_value=[]):
            with patch.dict(os.environ, {'MURMUR_MODEL': 'medium.en'}, clear=True):
                config = Config.load()
                assert config.model == 'medium.en'

    def test_load_env_sound_false(self):
        with patch.object(Config, '_config_paths', return_value=[]):
            with patch.dict(os.environ, {'MURMUR_SOUND': 'false'}, clear=True):
                config = Config.load()
                assert config.sound is False

    def test_load_env_sound_zero(self):
        with patch.object(Config, '_config_paths', return_value=[]):
            with patch.dict(os.environ, {'MURMUR_SOUND': '0'}, clear=True):
                config = Config.load()
                assert config.sound is False

    def test_load_env_sound_no(self):
        with patch.object(Config, '_config_paths', return_value=[]):
            with patch.dict(os.environ, {'MURMUR_SOUND': 'no'}, clear=True):
                config = Config.load()
                assert config.sound is False

    def test_load_env_sound_true(self):
        with patch.object(Config, '_config_paths', return_value=[]):
            with patch.dict(os.environ, {'MURMUR_SOUND': 'true'}, clear=True):
                config = Config.load()
                assert config.sound is True

    def test_load_from_toml_file(self, temp_dir):
        toml_content = """
[murmur]
hotkey = "f8"
model = "tiny.en"
sound = false
toggle_debounce_seconds = 0.3

[streaming]
buffer_seconds = 15.0
audio_window_seconds = 8.0
stability_count = 3

[injector]
max_updates_per_sec = 5
"""
        toml_file = temp_dir / "murmur.toml"
        toml_file.write_text(toml_content)

        with patch.object(Config, '_config_paths', return_value=[toml_file]):
            with patch.dict(os.environ, {}, clear=True):
                config = Config.load()
                assert config.hotkey == "f8"
                assert config.model == "tiny.en"
                assert config.sound is False
                assert config.toggle_debounce_seconds == 0.3
                assert config.buffer_seconds == 15.0
                assert config.audio_window_seconds == 8.0
                assert config.stability_count == 3
                assert config.max_updates_per_sec == 5

    def test_load_multiple_config_files_merged(self, temp_dir):
        toml1_content = """
[murmur]
hotkey = "f8"
model = "tiny.en"
"""
        toml2_content = """
[murmur]
model = "base.en"
sound = false
"""
        toml1 = temp_dir / "config1.toml"
        toml2 = temp_dir / "config2.toml"
        toml1.write_text(toml1_content)
        toml2.write_text(toml2_content)

        with patch.object(Config, '_config_paths', return_value=[toml1, toml2]):
            with patch.dict(os.environ, {}, clear=True):
                config = Config.load()
                assert config.hotkey == "f8"
                assert config.model == "base.en"
                assert config.sound is False

    def test_load_whisper_path_from_config(self, temp_dir):
        toml_content = f"""
[murmur]
whisper_path = "{temp_dir}/custom_whisper"
"""
        toml_file = temp_dir / "murmur.toml"
        toml_file.write_text(toml_content)

        with patch.object(Config, '_config_paths', return_value=[toml_file]):
            with patch.dict(os.environ, {}, clear=True):
                config = Config.load()
                assert config.whisper_path == temp_dir / "custom_whisper"

    def test_load_streaming_config(self, temp_dir):
        toml_content = """
[streaming]
inference_interval_seconds = 0.3
audio_chunk_ms = 50
min_audio_seconds = 0.2
vad_threshold = 0.02
vad_speech_pad_ms = 400
silence_commit_ms = 800
prompt_max_words = 60
overlap_max_words = 25
use_initial_prompt = false
consume_audio_on_commit = false
batch_mode = true
batch_silence_threshold_ms = 600
"""
        toml_file = temp_dir / "murmur.toml"
        toml_file.write_text(toml_content)

        with patch.object(Config, '_config_paths', return_value=[toml_file]):
            with patch.dict(os.environ, {}, clear=True):
                config = Config.load()
                assert config.inference_interval_seconds == 0.3
                assert config.audio_chunk_ms == 50
                assert config.min_audio_seconds == 0.2
                assert config.vad_threshold == 0.02
                assert config.vad_speech_pad_ms == 400
                assert config.silence_commit_ms == 800
                assert config.prompt_max_words == 60
                assert config.overlap_max_words == 25
                assert config.use_initial_prompt is False
                assert config.consume_audio_on_commit is False
                assert config.batch_mode is True
                assert config.batch_silence_threshold_ms == 600

    def test_load_injector_config(self, temp_dir):
        toml_content = """
[injector]
max_backspace_chars = 40
keystroke_delay_seconds = 0.003
backspace_delay_seconds = 0.002
"""
        toml_file = temp_dir / "murmur.toml"
        toml_file.write_text(toml_content)

        with patch.object(Config, '_config_paths', return_value=[toml_file]):
            with patch.dict(os.environ, {}, clear=True):
                config = Config.load()
                assert config.max_backspace_chars == 40
                assert config.keystroke_delay_seconds == 0.003
                assert config.backspace_delay_seconds == 0.002


class TestConfigPaths:
    """Test _config_paths static method."""

    def test_config_paths_returns_list(self):
        paths = Config._config_paths()
        assert isinstance(paths, list)
        assert len(paths) > 0

    def test_config_paths_includes_home_config(self):
        paths = Config._config_paths()
        home_config = Path.home() / ".config" / "murmur" / "murmur.toml"
        assert home_config in paths

    def test_config_paths_are_path_objects(self):
        paths = Config._config_paths()
        for path in paths:
            assert isinstance(path, Path)
