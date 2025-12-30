# Murmur

**Live streaming voice-to-text for macOS** — speak and watch text appear in real-time, anywhere.

Murmur uses [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with Metal GPU acceleration via embedded Python bindings. Press a hotkey, speak, and watch your words appear live at the cursor. Press again to stop.

## Quick Start

```bash
# Prerequisites
brew install cmake
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup
git clone https://github.com/srijanshukla/murmur.git
cd murmur
./setup.sh
uv sync

# Run
uv run murmur
```

Press **Right Option (⌥)** to start/stop live transcription.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  Hotkey (Right ⌥)                                               │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐     ┌──────────────────┐                  │
│  │ StreamingRecorder│────►│ Ring Buffer (12s)│                  │
│  │  (16kHz mono)   │     │ + VAD Detection  │                  │
│  └─────────────────┘     └────────┬─────────┘                  │
│                                   │ numpy float32              │
│                                   ▼ every 500ms                │
│                    ┌──────────────────────────┐                │
│                    │  StreamingTranscriber    │                │
│                    │  (pywhispercpp embedded) │                │
│                    │  Metal GPU acceleration  │                │
│                    └────────────┬─────────────┘                │
│                                 │ StreamingResult              │
│                                 ▼                              │
│                    ┌──────────────────────────┐                │
│                    │   StreamingInjector      │                │
│                    │  (diff-based, backspace) │                │
│                    └────────────┬─────────────┘                │
│                                 │ keystrokes                   │
│                                 ▼                              │
│                         Focused Application                    │
└─────────────────────────────────────────────────────────────────┘
```

### Live Streaming Pipeline

1. **Audio Capture**: Ring buffer keeps last 12 seconds, VAD detects speech
2. **Inference**: Every 500ms, whisper runs on the audio window (~60ms on M2)
3. **Stability**: Text is "committed" after 2 stable passes or 600ms silence
4. **Injection**: Diff-based — only types changes, uses backspaces for corrections

---

## Features

- **Live streaming** — text appears as you speak, not after
- **Embedded whisper.cpp** — model stays in GPU memory, no process spawning
- **Metal acceleration** — ~60ms inference for 10s audio on Apple Silicon
- **Smart corrections** — backspaces and retypes when whisper revises words
- **VAD gating** — only runs inference when speech detected
- **Stability tracking** — prevents jarring rewrites of committed text

---

## Requirements

### Hardware
- Apple Silicon Mac (M1/M2/M3/M4)
- Microphone

### Software
- macOS 12.0+ (Monterey or later)
- Xcode Command Line Tools: `xcode-select --install`
- cmake: `brew install cmake`
- uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Permissions
- **Microphone**: System Settings → Privacy & Security → Microphone → Enable Terminal
- **Accessibility**: System Settings → Privacy & Security → Accessibility → Enable Terminal

---

## Installation

```bash
# 1. Clone
git clone https://github.com/srijanshukla/murmur.git
cd murmur

# 2. Build whisper.cpp and download model
./setup.sh

# 3. Install Python dependencies
uv sync
```

---

## Usage

```bash
uv run murmur
```

### Workflow

1. Focus any text field (terminal, browser, editor)
2. **Tap Right Option (⌥)** — recording starts, you hear "Funk"
3. Speak — text appears live as you talk
4. **Tap Right Option (⌥)** — final pass runs, you hear "Blow"

### Menu Bar States

| Icon | State | Meaning |
|------|-------|---------|
| 🎤 | Idle | Ready |
| 🔴 | Live | Streaming transcription active |

---

## Configuration

Environment variables or `~/.config/murmur/config.toml`:

```bash
export MURMUR_HOTKEY="alt_r"     # alt_r, alt_l, f8, f9, f10, caps_lock
export MURMUR_MODEL="base.en"   # tiny.en, base.en, small.en, medium.en
export MURMUR_SOUND="true"      # true/false
```

### Models

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| tiny.en | ~75MB | Fastest | Quick commands |
| **base.en** | ~150MB | Fast | General use (default) |
| small.en | ~500MB | Medium | Technical dictation |
| medium.en | ~1.5GB | Slower | High accuracy |

---

## Architecture

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Audio | sounddevice + ring buffer | Continuous 16kHz capture |
| VAD | RMS energy threshold | Detect speech vs silence |
| Transcription | pywhispercpp (embedded) | Metal GPU inference |
| Injection | Quartz Events | Diff-based keystroke injection |
| Hotkey | pynput | Global key detection |

### Key Design Decisions

**Why embedded whisper.cpp?**
- Model loads once, stays in GPU memory
- No process spawn overhead (~100ms saved per inference)
- Direct numpy input (no temp files)
- ~60ms inference latency on M2 Pro

**Why streaming instead of batch?**
- Immediate feedback while speaking
- Better UX for long-form dictation
- Corrections happen in real-time

**Why diff-based injection?**
- Whisper can revise recent words as more context arrives
- Instead of clearing and retyping, we compute minimal edits
- Backspaces only the changed portion, types the correction

---

## Troubleshooting

### Text not appearing
Grant accessibility permission to your terminal app.

### No audio captured
Grant microphone permission to your terminal app.

### Transcription hallucinations
Common artifacts like "Thank you." or "[BLANK_AUDIO]" are filtered automatically.

### High latency
Ensure Metal GPU is being used (check logs at `~/Library/Logs/Murmur/`).

---

## Logs

```bash
# View today's logs
cat ~/Library/Logs/Murmur/murmur-$(date +%Y-%m-%d).log

# Tail live
tail -f ~/Library/Logs/Murmur/murmur-$(date +%Y-%m-%d).log
```

---

## License

MIT

## Acknowledgments

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — Georgi Gerganov
- [pywhispercpp](https://github.com/absadiki/pywhispercpp) — Python bindings
- [rumps](https://github.com/jaredks/rumps) — macOS menu bar
