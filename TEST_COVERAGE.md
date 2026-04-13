# Test Coverage Report - Murmur

## Coverage Status: ~90% (estimated)

All source files in the `src/murmur/` package have comprehensive test coverage. The tests cover initialization, core functionality, edge cases, error handling, and thread safety.

## Tested Modules

- [x] src/murmur/__init__.py - minimal, only contains version (no tests needed)
- [x] src/murmur/config.py - tests in tests/test_config.py
- [x] src/murmur/logger.py - tests in tests/test_logger.py
- [x] src/murmur/audio.py - tests in tests/test_audio.py
- [x] src/murmur/inject.py - tests in tests/test_inject.py
- [x] src/murmur/transcribe.py - tests in tests/test_transcribe.py
- [x] src/murmur/app.py - tests in tests/test_app.py
- [x] murmur.py - entry point only, covered by test_app.py main() tests

## Pending Modules

None - all source modules have been tested.

## Test Files Created

| Test File | Tests | Coverage Details |
|-----------|-------|------------------|
| tests/conftest.py | N/A | Shared fixtures (temp_dir, sample_audio, mocks) |
| tests/test_config.py | 62 tests | Config defaults, normalize_hotkey, merge_dicts, load_toml, model_path, load() |
| tests/test_logger.py | 14 tests | setup_logger(), log object, file/console handlers, formatting |
| tests/test_audio.py | 52 tests | RingBuffer, VAD, StreamingRecorder classes |
| tests/test_inject.py | 35 tests | StreamingInjector init, reset, update, diff logic, backspaces, typing, threading |
| tests/test_transcribe.py | 42 tests | StreamingResult, StreamingTranscriber init, reset, properties, process_audio, clean_output, merge, stability |
| tests/test_app.py | 44 tests | State enum, SOUNDS, MurmurApp init, model loading, state management, toggle, sounds, streaming, shutdown, main() |

## Coverage Breakdown by Module

### config.py (62 tests)
- `TestConfigDefaults`: 24 tests for all default values
- `TestConfigNormalizeHotkey`: 10 tests for hotkey normalization
- `TestConfigMergeDicts`: 8 tests for dictionary merging
- `TestConfigLoadToml`: 3 tests for TOML loading
- `TestConfigModelPath`: 3 tests for model path property
- `TestConfigLoad`: 11 tests for configuration loading
- `TestConfigPaths`: 3 tests for config paths

### logger.py (14 tests)
- `TestSetupLogger`: 9 tests for logger setup
- `TestLogModuleLevel`: 5 tests for module-level logging

### audio.py (52 tests)
- `TestRingBuffer`: 17 tests for ring buffer operations
- `TestVAD`: 14 tests for voice activity detection
- `TestStreamingRecorder`: 21 tests for audio recording

### inject.py (35 tests)
- `TestStreamingInjectorInit`: 12 tests for initialization
- `TestStreamingInjectorReset`: 2 tests for reset
- `TestStreamingInjectorTypedText`: 2 tests for property
- `TestStreamingInjectorUpdate`: 10 tests for update logic
- `TestStreamingInjectorDiffLogic`: 3 tests for diff computation
- `TestStreamingInjectorBackspaces`: 2 tests for backspace handling
- `TestStreamingInjectorTypeText`: 2 tests for text typing
- `TestStreamingInjectorMaxBackspace`: 2 tests for backspace limits
- `TestStreamingInjectorThreadSafety`: 2 tests for concurrency

### transcribe.py (42 tests)
- `TestStreamingResult`: 3 tests for result dataclass
- `TestStreamingTranscriberInit`: 13 tests for initialization
- `TestStreamingTranscriberReset`: 4 tests for reset
- `TestStreamingTranscriberProperties`: 3 tests for properties
- `TestStreamingTranscriberProcessAudio`: 4 tests for audio processing
- `TestStreamingTranscriberCleanOutput`: 9 tests for output cleaning
- `TestStreamingTranscriberMergeWithCommitted`: 5 tests for text merging
- `TestStreamingTranscriberUpdateStability`: 7 tests for stability logic

### app.py (44 tests)
- `TestState`: 5 tests for State enum
- `TestSounds`: 4 tests for sounds dictionary
- `TestMurmurAppInit`: 7 tests for app initialization
- `TestMurmurAppOnModelLoaded`: 2 tests for model loading callback
- `TestMurmurAppSetState`: 2 tests for state management
- `TestMurmurAppToggle`: 4 tests for toggle behavior
- `TestMurmurAppPlaySound`: 4 tests for sound playback
- `TestMurmurAppStartLiveStreaming`: 3 tests for starting streaming
- `TestMurmurAppStopLiveStreaming`: 2 tests for stopping streaming
- `TestMurmurAppShutdown`: 2 tests for shutdown
- `TestMurmurAppOnStreamingUpdate`: 2 tests for streaming updates
- `TestMurmurAppOnStreamingComplete`: 3 tests for streaming completion
- `TestMain`: 4 tests for main function

## Test Categories

### Unit Tests
- All individual functions and methods are tested in isolation
- Mock dependencies (Quartz, pynput, pywhispercpp, sounddevice) are used to test without hardware

### Edge Cases
- Empty inputs (None, empty strings, empty arrays)
- Boundary conditions (min/max values, thresholds)
- Invalid inputs (negative numbers, unknown keys)

### Error Handling
- Exception handling in main()
- File not found scenarios
- Model loading failures

### Thread Safety
- Concurrent access to shared state
- Lock behavior in RingBuffer, VAD, StreamingInjector
- Thread-safe property access

## Running Tests

```bash
# Install pytest if needed
pip install pytest

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_config.py -v

# Run with coverage report (requires pytest-cov)
pip install pytest-cov
pytest tests/ --cov=src/murmur --cov-report=html
```

## Notes

1. External dependencies (Quartz, pynput, pywhispercpp, sounddevice) are mocked to enable testing without macOS-specific hardware or audio devices.

2. The `src/murmur/__init__.py` file only contains the version string and requires no tests.

3. The `murmur.py` entry point is a simple wrapper that calls `main()` from `app.py`, which is fully tested.

4. Some tests may require adjustments if the underlying implementation changes (e.g., Quartz API calls).
