"""Tests for src/murmur/app.py"""

import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from enum import Enum

import pytest
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock dependencies before importing app
quartz_mock = MagicMock()
sys.modules['Quartz'] = quartz_mock
quartz_mock.CGEventCreateKeyboardEvent = MagicMock(return_value=MagicMock())
quartz_mock.CGEventKeyboardSetUnicodeString = MagicMock()
quartz_mock.CGEventPost = MagicMock()
quartz_mock.CGEventSourceCreate = MagicMock(return_value=MagicMock())
quartz_mock.kCGEventSourceStateHIDSystemState = 1
quartz_mock.kCGHIDEventTap = 0

pynput_mock = MagicMock()
sys.modules['pynput'] = pynput_mock
sys.modules['pynput.keyboard'] = pynput_mock.keyboard

pywhispercpp_mock = MagicMock()
sys.modules['pywhispercpp'] = pywhispercpp_mock
sys.modules['pywhispercpp.model'] = pywhispercpp_mock.model
pywhispercpp_mock.model.Model = MagicMock()

from murmur.app import State, SOUNDS, MurmurApp, main
from murmur.config import Config


class TestState:
    """Test State enum."""

    def test_state_loading_value(self):
        assert State.LOADING.value == "loading"

    def test_state_idle_value(self):
        assert State.IDLE.value == "idle"

    def test_state_transcribing_value(self):
        assert State.TRANSCRIBING.value == "transcribing"

    def test_state_live_value(self):
        assert State.LIVE.value == "live"

    def test_state_is_enum(self):
        assert isinstance(State.LOADING, Enum)


class TestSounds:
    """Test SOUNDS dictionary."""

    def test_sounds_has_start(self):
        assert "start" in SOUNDS

    def test_sounds_has_stop(self):
        assert "stop" in SOUNDS

    def test_sounds_has_error(self):
        assert "error" in SOUNDS

    def test_sounds_are_aiff_paths(self):
        for name, path in SOUNDS.items():
            assert path.endswith(".aiff")

    def test_sounds_are_system_library(self):
        for name, path in SOUNDS.items():
            assert path.startswith("/System/Library/Sounds/")


class TestMurmurAppInit:
    """Test MurmurApp initialization."""

    def test_init_stores_config(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                assert app.config is config

    def test_init_state_is_loading(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                assert app.state == State.LOADING

    def test_init_creates_streaming_recorder(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                assert app.streaming_recorder is not None

    def test_init_creates_streaming_injector(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                assert app.streaming_injector is not None

    def test_init_transcriber_is_none(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                assert app.streaming_transcriber is None

    def test_init_starts_hotkey_listener(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener') as mock_listener:
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                mock_listener.assert_called_once()


class TestMurmurAppOnModelLoaded:
    """Test _on_model_loaded method."""

    def test_on_model_loaded_sets_transcriber(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                mock_transcriber = MagicMock()
                app._on_model_loaded(mock_transcriber)
                assert app.streaming_transcriber is mock_transcriber

    def test_on_model_loaded_sets_state_idle(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                mock_transcriber = MagicMock()
                app._on_model_loaded(mock_transcriber)
                assert app.state == State.IDLE


class TestMurmurAppSetState:
    """Test _set_state method."""

    def test_set_state_updates_state(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                app._set_state(State.IDLE)
                assert app.state == State.IDLE

    def test_set_state_to_live(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                app._set_state(State.LIVE)
                assert app.state == State.LIVE


class TestMurmurAppToggle:
    """Test _toggle method."""

    def test_toggle_debounce(self):
        config = Config()
        config.toggle_debounce_seconds = 0.5
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                app.state = State.IDLE
                app._last_toggle_time = time.time()

                with patch.object(app, '_start_live_streaming') as mock_start:
                    app._toggle()
                    mock_start.assert_not_called()

    def test_toggle_from_idle_starts_streaming(self):
        config = Config()
        config.toggle_debounce_seconds = 0.0
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                app.state = State.IDLE
                app._last_toggle_time = 0

                with patch.object(app, '_start_live_streaming') as mock_start:
                    app._toggle()
                    mock_start.assert_called_once()

    def test_toggle_from_live_stops_streaming(self):
        config = Config()
        config.toggle_debounce_seconds = 0.0
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                app.state = State.LIVE
                app._last_toggle_time = 0
                app.streaming_transcriber = MagicMock()

                with patch.object(app, '_stop_live_streaming') as mock_stop:
                    app._toggle()
                    mock_stop.assert_called_once()

    def test_toggle_from_loading_ignored(self):
        config = Config()
        config.toggle_debounce_seconds = 0.0
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                app.state = State.LOADING
                app._last_toggle_time = 0

                with patch.object(app, '_start_live_streaming') as mock_start:
                    app._toggle()
                    mock_start.assert_not_called()


class TestMurmurAppPlaySound:
    """Test _play_sound method."""

    def test_play_sound_disabled(self):
        config = Config()
        config.sound = False
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                with patch('subprocess.Popen') as mock_popen:
                    app = MurmurApp(config)
                    app._play_sound("start")
                    mock_popen.assert_not_called()

    def test_play_sound_unknown_sound(self):
        config = Config()
        config.sound = True
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                with patch('subprocess.Popen') as mock_popen:
                    app = MurmurApp(config)
                    app._play_sound("nonexistent")
                    mock_popen.assert_not_called()

    def test_play_sound_valid_sound(self):
        config = Config()
        config.sound = True
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                with patch('subprocess.Popen') as mock_popen:
                    with patch.object(Path, 'exists', return_value=True):
                        app = MurmurApp(config)
                        app._play_sound("start")
                        mock_popen.assert_called_once()

    def test_play_sound_wait_uses_run(self):
        config = Config()
        config.sound = True
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                with patch('subprocess.run') as mock_run:
                    with patch.object(Path, 'exists', return_value=True):
                        app = MurmurApp(config)
                        app._play_sound("start", wait=True)
                        mock_run.assert_called_once()


class TestMurmurAppStartLiveStreaming:
    """Test _start_live_streaming method."""

    def test_start_live_streaming_sets_state(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                app.streaming_transcriber = MagicMock()
                with patch.object(app, '_play_sound'):
                    with patch.object(app.streaming_recorder, 'start'):
                        app._start_live_streaming()
                        assert app.state == State.LIVE

    def test_start_live_streaming_resets_components(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                mock_transcriber = MagicMock()
                mock_injector = MagicMock()
                app.streaming_transcriber = mock_transcriber
                app.streaming_injector = mock_injector
                with patch.object(app, '_play_sound'):
                    with patch.object(app.streaming_recorder, 'start'):
                        app._start_live_streaming()
                        mock_transcriber.reset.assert_called_once()
                        mock_injector.reset.assert_called_once()

    def test_start_live_streaming_starts_recorder(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                app.streaming_transcriber = MagicMock()
                with patch.object(app, '_play_sound'):
                    with patch.object(app.streaming_recorder, 'start') as mock_start:
                        app._start_live_streaming()
                        mock_start.assert_called_once()


class TestMurmurAppStopLiveStreaming:
    """Test _stop_live_streaming method."""

    def test_stop_live_streaming_sets_transcribing_state(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                app.streaming_transcriber = MagicMock()
                app._streaming_stop = threading.Event()
                app._streaming_thread = None
                with patch.object(app, '_play_sound'):
                    with patch.object(app.streaming_recorder, 'stop', return_value=np.zeros(100)):
                        app._stop_live_streaming()
                        assert app.state in [State.TRANSCRIBING, State.IDLE]

    def test_stop_live_streaming_signals_stop(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                app.streaming_transcriber = MagicMock()
                app._streaming_stop = threading.Event()
                app._streaming_thread = None
                with patch.object(app, '_play_sound'):
                    with patch.object(app.streaming_recorder, 'stop', return_value=np.zeros(100)):
                        app._stop_live_streaming()
                        assert app._streaming_stop.is_set()


class TestMurmurAppShutdown:
    """Test shutdown method."""

    def test_shutdown_stops_live_streaming(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                app.state = State.LIVE
                app.streaming_transcriber = MagicMock()
                app._streaming_stop = threading.Event()
                app._streaming_thread = None
                app._listener = MagicMock()

                with patch.object(app, '_stop_live_streaming') as mock_stop:
                    with patch.object(app, '_play_sound'):
                        with patch.object(app.streaming_recorder, 'stop', return_value=np.zeros(100)):
                            app.shutdown()
                            mock_stop.assert_called_once()

    def test_shutdown_stops_listener(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                app.state = State.IDLE
                mock_listener = MagicMock()
                app._listener = mock_listener
                app.shutdown()
                mock_listener.stop.assert_called_once()


class TestMurmurAppOnStreamingUpdate:
    """Test _on_streaming_update method."""

    def test_on_streaming_update_ignores_non_live(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                app.state = State.IDLE

                mock_result = MagicMock()
                mock_result.full_text = "test"
                with patch.object(app.streaming_injector, 'update') as mock_update:
                    app._on_streaming_update(mock_result)
                    mock_update.assert_not_called()

    def test_on_streaming_update_injects_text(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)
                app.state = State.LIVE

                mock_result = MagicMock()
                mock_result.full_text = "hello world"
                with patch.object(app.streaming_injector, 'update') as mock_update:
                    app._on_streaming_update(mock_result)
                    mock_update.assert_called_once_with("hello world")


class TestMurmurAppOnStreamingComplete:
    """Test _on_streaming_complete method."""

    def test_on_streaming_complete_injects_final(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)

                mock_result = MagicMock()
                mock_result.full_text = "final text"
                with patch.object(app.streaming_injector, 'update') as mock_update:
                    app._on_streaming_complete(mock_result)
                    mock_update.assert_called_once_with("final text", force=True)

    def test_on_streaming_complete_sets_idle(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)

                mock_result = MagicMock()
                mock_result.full_text = "test"
                app._on_streaming_complete(mock_result)
                assert app.state == State.IDLE

    def test_on_streaming_complete_none_result(self):
        config = Config()
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                app = MurmurApp(config)

                app._on_streaming_complete(None)
                assert app.state == State.IDLE


class TestMain:
    """Test main function."""

    def test_main_creates_app(self):
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                with patch.object(Config, 'load', return_value=Config()):
                    with patch('builtins.print'):
                        with patch('time.sleep', side_effect=KeyboardInterrupt):
                            with pytest.raises(SystemExit):
                                main()

    def test_main_handles_keyboard_interrupt(self):
        with patch.object(MurmurApp, '_start_hotkey_listener'):
            with patch('sounddevice.InputStream'):
                with patch.object(Config, 'load', return_value=Config()):
                    with patch('builtins.print'):
                        with patch('time.sleep', side_effect=KeyboardInterrupt):
                            try:
                                main()
                            except SystemExit:
                                pass

    def test_main_handles_file_not_found(self):
        with patch.object(Config, 'load', side_effect=FileNotFoundError("test")):
            with patch('builtins.print'):
                with pytest.raises(SystemExit):
                    main()

    def test_main_handles_generic_exception(self):
        with patch.object(Config, 'load', side_effect=Exception("test error")):
            with patch('builtins.print'):
                with pytest.raises(SystemExit):
                    main()
