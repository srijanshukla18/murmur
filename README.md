# Murmur

**Global voice-to-text injection for macOS** — speak instead of type, anywhere.

Murmur runs [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with CoreML acceleration on Apple Silicon. Press a hotkey, speak, press again — your words appear at the cursor in any application.

## Why Murmur?

You're in **Claude Code** (or any CLI tool, browser, editor). You want to type a long prompt or explain a complex bug. Instead of typing, you:

1. Tap **Right Option (⌥)**
2. Speak naturally
3. Tap **Right Option (⌥)** again
4. Text appears where your cursor is

Murmur was built specifically to accelerate long-form prompting in high-velocity CLI environments like `claude`, `git`, or `vim`. It works everywhere you can type.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         macOS                                   │
│                                                                 │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────────┐   │
│  │ Menu Bar │◄───│   Murmur    │───►│ Quartz Event Services│   │
│  │   Icon   │    │  (Python)   │    │   (HID Injection)    │   │
│  └──────────┘    └──────┬──────┘    └──────────────────────┘   │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    whisper.cpp                            │  │
│  │         (CoreML backend on Apple Neural Engine)           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         ▲                                       │
│                         │                                       │
│  ┌──────────────────────┴───────────────────────────────────┐  │
│  │              Microphone (16kHz PCM capture)               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Inference Engine | whisper.cpp + CoreML | Speech-to-text on Apple Neural Engine |
| Orchestrator | Python + rumps | Menu bar app, hotkey handling, state management |
| Audio Capture | sounddevice | Record microphone to WAV buffer |
| Text Injection | Quartz (PyObjC) | Type text into any focused application |
| Hotkey | pynput | Global Right Option key detection |

---

## UX Design

### Interaction Model: Toggle (Tap-Tap)

```
[Tap Right ⌥] ──► 🔴 Recording ──► [Tap Right ⌥] ──► ⏳ Transcribing ──► 💬 Text Injected
     │                                    │
     └── "Funk" sound                     └── "Blow" sound
```

**Why toggle instead of hold-to-talk?**
- Holding a key for 30+ seconds while dictating a long prompt is tiring.
- Toggle lets you rest your hands while speaking.
- Double-tap is harder to do accidentally than a long press.

### Menu Bar States

| Icon | State | Meaning |
|------|-------|---------|
| 🎤 | Idle | Ready to record |
| 🔴 | Recording | Listening to your voice |
| ⏳ | Processing | Transcribing audio |

### Audio Feedback

- **Start recording**: macOS "Funk" sound (High-audibility synth)
- **Stop recording**: macOS "Blow" sound

This lets you know the system shifted states without needing to look at the menu bar.

---

## Technical Decisions

### Why whisper.cpp + CoreML?

| Option | Pros | Cons |
|--------|------|------|
| **whisper.cpp + CoreML** ✓ | Runs on Neural Engine, low power, fast, mature | Requires compilation |
| OpenAI Whisper (Python) | Easy install | Slow, high battery, Python GIL |
| MLX Whisper | Native Apple | Less mature, fewer options |
| macOS Dictation | Zero setup | Less accurate, requires internet |

whisper.cpp with CoreML runs inference on Apple's Neural Engine (ANE), giving:
- ~200ms latency for short utterances.
- Tiny battery footprint (ANE is highly efficient).
- 100% Offline operation (privacy and speed).

### Why Batch Mode (not Streaming)?

**Streaming**: Text appears as you speak (real-time feedback).
**Batch**: Text appears after you stop speaking.

We chose **batch mode** because:
1. CLI prompts are typically short (< 60 seconds).
2. Batch is simpler, more reliable, and allows for global context correction.
3. Avoids partial/jittery text appearing mid-sentence in sensitive terminals.
4. Single clean injection is less taxing on the UI bridge.

### Why Right Option Key?

- Reachable without moving hands from home-row.
- Rarely used in normal typing (unlike Cmd, Ctrl, Shift).
- Doesn't conflict with most application-level hotkeys.
- Can be remapped via Karabiner-Elements or environment variables if needed.

### Why Not Docker?

Docker cannot be used for this project because:
1. **CoreML requires macOS** — no CoreML in Linux containers.
2. **Metal needs direct GPU access** — containers can't access Apple Silicon GPU directly.
3. **Quartz APIs are macOS-only** — HID injection requires direct OS access.
4. **Microphone access** — hardware capture from containers is brittle on macOS.

This is a native macOS application by design.

---

## Model Choice

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| tiny.en | ~75MB | Fastest | Lower | Quick commands |
| **base.en** ✓ | ~150MB | Fast | Good | General use (default) |
| small.en | ~500MB | Medium | Better | Complex technical dictation |
| medium.en | ~1.5GB | Slower | High | Accuracy-critical |

**Default: base.en** — the best balance of speed and accuracy for technical/CLI use.

---

## Requirements

### Hardware
- Apple Silicon Mac (M1/M2/M3/M4)
- Microphone (built-in or external)

### Software
- macOS 12.0+ (Monterey or later)
- Xcode Command Line Tools (`xcode-select --install`)
- **cmake** (required for building whisper.cpp)
- **uv** (high-performance Python manager)

### Permissions Required
- **Microphone**: Allow Terminal/iTerm to record audio.
- **Accessibility**: Allow Terminal/iTerm to "control your computer" (required for HID injection).

---

## Installation

### 1. Install Prerequisites

```bash
# Install cmake (required for building whisper.cpp)
brew install cmake

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup

```bash
git clone https://github.com/srijanshukla/murmur.git
cd murmur

# Run setup script (clones whisper.cpp, builds with CoreML, downloads base model)
./setup.sh
```

### 3. Install Python Dependencies

```bash
uv sync
```

---

## Usage

### Start Murmur

```bash
uv run python murmur.py
```

A microphone icon (🎤) appears in your menu bar.

### Basic Workflow

1. Focus on any text field (Terminal, browser, editor).
2. **Tap Right Option (⌥)** — icon turns red (🔴), you hear "Funk".
3. Speak your prompt naturally.
4. **Tap Right Option (⌥)** — icon shows hourglass (⏳), you hear "Blow".
5. Text appears at cursor position.

---

## Configuration

Configuration is handled via environment variables or `~/.config/murmur/config.toml`.

```bash
# Example overrides
export MURMUR_HOTKEY="f8"
export MURMUR_MODEL="small.en"
export MURMUR_SOUND="false"
```

---

## Troubleshooting

### "Permission denied" when recording
Grant microphone permission:
`System Settings → Privacy & Security → Microphone → Enable for your terminal`

### Text not appearing in applications
Grant accessibility permission:
`System Settings → Privacy & Security → Accessibility → Enable for your terminal`

### Transcription says "Okay." or "What?"
This usually happens if you record silence or background noise. Murmur includes built-in hallucination filtering for these common artifacts.

---

## How It Works (Technical Deep Dive)

### 1. Hotkey Detection
Uses `pynput` to listen for global key events. We listen for `on_press` specifically to ensure the UI feels instantaneous.

### 2. Audio Capture
Captured mono at 16kHz (Whisper's native format) via `sounddevice`. Audio is held in a RAM buffer to avoid disk I/O while you speak.

### 3. Transcription
When stop is triggered, the buffer is saved to a temp WAV and passed to the compiled `whisper-cli` binary:
```bash
./whisper.cpp/build/bin/whisper-cli -m models/ggml-base.en.bin -f /tmp/audio.wav --no-timestamps
```

### 4. Text Injection
Uses **Quartz Event Services** to inject keystrokes. This is higher level than standard keyboard drivers, allowing us to type Unicode characters directly into the focused window's event tap.

---

## License
MIT

---

## Acknowledgments
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — Georgi Gerganov's C++ port.
- [rumps](https://github.com/jaredks/rumps) — macOS Statusbar library.
