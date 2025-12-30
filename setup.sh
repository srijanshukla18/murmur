#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_DIR="$SCRIPT_DIR/whisper.cpp"
MODEL="${1:-base.en}"

echo "╔════════════════════════════════════════════╗"
echo "║         Murmur Setup Script                ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Check for Xcode Command Line Tools
if ! xcode-select -p &>/dev/null; then
    echo "❌ Xcode Command Line Tools not found."
    echo "   Run: xcode-select --install"
    exit 1
fi
echo "✓ Xcode Command Line Tools found"

# Check for cmake
if ! command -v cmake &>/dev/null; then
    echo "❌ cmake not found."
    echo "   Run: brew install cmake"
    exit 1
fi
echo "✓ cmake found"

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠️  Warning: Not running on Apple Silicon."
    echo "   CoreML acceleration may not work optimally."
fi

# Clone or update whisper.cpp
if [ -d "$WHISPER_DIR" ]; then
    echo "→ Updating whisper.cpp..."
    cd "$WHISPER_DIR"
    git pull --quiet
else
    echo "→ Cloning whisper.cpp..."
    git clone --depth 1 https://github.com/ggerganov/whisper.cpp "$WHISPER_DIR"
    cd "$WHISPER_DIR"
fi

# Build with cmake (Metal is enabled by default on macOS)
echo "→ Building whisper.cpp with Metal support..."
echo "  (This may take a few minutes)"

# Configure with cmake
cmake -B build \
    -DWHISPER_METAL=ON \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(sysctl -n hw.ncpu) --config Release

if [ ! -f "$WHISPER_DIR/build/bin/whisper-cli" ]; then
    echo "❌ Build failed. Check cmake/Xcode installation."
    exit 1
fi
echo "✓ whisper.cpp built successfully"

# Download model
MODEL_FILE="$WHISPER_DIR/models/ggml-${MODEL}.bin"
if [ -f "$MODEL_FILE" ]; then
    echo "✓ Model ggml-${MODEL}.bin already exists"
else
    echo "→ Downloading model: ${MODEL}..."
    cd "$WHISPER_DIR"
    bash ./models/download-ggml-model.sh "$MODEL"

    if [ ! -f "$MODEL_FILE" ]; then
        echo "❌ Model download failed."
        exit 1
    fi
    echo "✓ Model downloaded"
fi

# Verify installation
echo ""
echo "→ Verifying installation..."
cd "$WHISPER_DIR"

if ./build/bin/whisper-cli --help &>/dev/null; then
    echo "✓ whisper-cli is working"
else
    echo "❌ whisper-cli verification failed"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║         Setup Complete!                    ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "Model: ggml-${MODEL}.bin"
echo "Location: $WHISPER_DIR"
echo ""
echo "Next steps:"
echo "  1. uv sync"
echo "  2. Grant permissions (Microphone + Accessibility)"
echo "  3. uv run python murmur.py"
echo ""
