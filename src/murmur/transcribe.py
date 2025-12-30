"""Whisper transcription interface for Murmur."""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class Transcriber:
    """Transcribes audio using whisper.cpp."""

    def __init__(self, whisper_binary: Path, model_path: Path):
        self.whisper_binary = whisper_binary
        self.model_path = model_path

        # Verify paths exist
        if not self.whisper_binary.exists():
            raise FileNotFoundError(
                f"whisper.cpp binary not found: {self.whisper_binary}\n"
                "Run ./setup.sh to build whisper.cpp"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Run ./setup.sh to download the model"
            )

    def transcribe(self, audio_data: bytes, timeout: float = 30.0) -> Optional[str]:
        """
        Transcribe audio data to text.

        Args:
            audio_data: WAV audio data as bytes
            timeout: Maximum time to wait for transcription

        Returns:
            Transcribed text, or None if failed
        """
        if not audio_data:
            return None

        # Write audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = Path(f.name)

        try:
            result = subprocess.run(
                [
                    str(self.whisper_binary),
                    "-m", str(self.model_path),
                    "-f", str(temp_path),
                    "--no-timestamps",
                    "-t", "4",  # threads
                    "--no-prints",  # suppress progress output
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                # Log error but don't crash
                print(f"Whisper error: {result.stderr}")
                return None

            return self._clean_output(result.stdout)

        except subprocess.TimeoutExpired:
            print("Transcription timed out")
            return None
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
        finally:
            # Clean up temp file
            temp_path.unlink(missing_ok=True)

    def _clean_output(self, raw_output: str) -> Optional[str]:
        """Clean whisper output, removing artifacts."""
        if not raw_output:
            return None

        lines = []
        for line in raw_output.strip().split("\n"):
            # Remove timestamp patterns like [00:00:00.000 --> 00:00:01.000]
            line = re.sub(r"\[[\d:.]+\s*-->\s*[\d:.]+\]\s*", "", line)
            # Remove other bracket artifacts
            line = re.sub(r"\[.*?\]", "", line)
            line = line.strip()
            if line:
                lines.append(line)

        text = " ".join(lines)

        # Remove common whisper hallucinations on silence
        hallucinations = [
            "(music)",
            "(Music)",
            "[Music]",
            "(silence)",
            "(Silence)",
            "Thank you.",
            "Thanks for watching!",
            "Subscribe",
        ]
        for h in hallucinations:
            text = text.replace(h, "")

        # Handle specific short hallucinations like "Okay." or "What?" on silence/noise
        if text.strip().lower() in ("okay.", "okay", "what?", "what"):
            return None

        text = text.strip()
        return text if text else None
