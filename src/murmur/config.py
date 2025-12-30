"""Configuration handling for Murmur."""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass
class Config:
    """Murmur configuration."""

    hotkey: str = "alt_r"  # pynput key name for Right Option
    model: str = "base.en"
    sound: bool = True
    whisper_path: Path = Path(__file__).parent.parent.parent / "whisper.cpp"

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment and config file."""
        config = cls()

        # Load from config file if exists
        config_file = Path.home() / ".config" / "murmur" / "config.toml"
        if config_file.exists():
            with open(config_file, "rb") as f:
                data = tomllib.load(f)
                murmur_config = data.get("murmur", {})
                if "hotkey" in murmur_config:
                    config.hotkey = cls._normalize_hotkey(murmur_config["hotkey"])
                if "model" in murmur_config:
                    config.model = murmur_config["model"]
                if "sound" in murmur_config:
                    config.sound = murmur_config["sound"]

        # Environment variables override config file
        if env_hotkey := os.environ.get("MURMUR_HOTKEY"):
            config.hotkey = cls._normalize_hotkey(env_hotkey)
        if env_model := os.environ.get("MURMUR_MODEL"):
            config.model = env_model
        if env_sound := os.environ.get("MURMUR_SOUND"):
            config.sound = env_sound.lower() not in ("false", "0", "no")

        return config

    @staticmethod
    def _normalize_hotkey(key: str) -> str:
        """Normalize hotkey names to pynput format."""
        mappings = {
            "right_option": "alt_r",
            "right_alt": "alt_r",
            "left_option": "alt_l",
            "left_alt": "alt_l",
            "caps_lock": "caps_lock",
            "f8": "f8",
            "f9": "f9",
            "f10": "f10",
        }
        return mappings.get(key.lower(), key.lower())

    @property
    def model_path(self) -> Path:
        """Get full path to the model file."""
        return self.whisper_path / "models" / f"ggml-{self.model}.bin"
