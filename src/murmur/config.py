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
    model: str = "small.en"
    sound: bool = True
    toggle_debounce_seconds: float = 0.2

    buffer_seconds: float = 12.0
    audio_window_seconds: float = 10.0
    inference_interval_seconds: float = 0.5
    audio_chunk_ms: int = 100
    min_audio_seconds: float = 0.1
    vad_threshold: float = 0.01
    vad_speech_pad_ms: int = 300
    stability_count: int = 2
    silence_commit_ms: int = 600
    prompt_max_words: int = 50
    overlap_max_words: int = 20
    use_initial_prompt: bool = True
    consume_audio_on_commit: bool = True
    batch_mode: bool = False
    batch_silence_threshold_ms: int = 500

    max_updates_per_sec: int = 4
    max_backspace_chars: int = 30
    keystroke_delay_seconds: float = 0.002
    backspace_delay_seconds: float = 0.001

    whisper_path: Path = Path(__file__).parent.parent.parent / "whisper.cpp"

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment and config file."""
        config = cls()

        data: dict = {}
        for path in cls._config_paths():
            data = cls._merge_dicts(data, cls._load_toml(path))

        murmur_config = data.get("murmur", {})
        streaming_config = data.get("streaming", {})
        injector_config = data.get("injector", {})

        if "hotkey" in murmur_config:
            config.hotkey = cls._normalize_hotkey(murmur_config["hotkey"])
        if "model" in murmur_config:
            config.model = murmur_config["model"]
        if "sound" in murmur_config:
            config.sound = murmur_config["sound"]
        if "toggle_debounce_seconds" in murmur_config:
            config.toggle_debounce_seconds = float(murmur_config["toggle_debounce_seconds"])
        if "whisper_path" in murmur_config:
            config.whisper_path = Path(murmur_config["whisper_path"]).expanduser()

        if "buffer_seconds" in streaming_config:
            config.buffer_seconds = float(streaming_config["buffer_seconds"])
        if "audio_window_seconds" in streaming_config:
            config.audio_window_seconds = float(streaming_config["audio_window_seconds"])
        if "inference_interval_seconds" in streaming_config:
            config.inference_interval_seconds = float(streaming_config["inference_interval_seconds"])
        if "audio_chunk_ms" in streaming_config:
            config.audio_chunk_ms = int(streaming_config["audio_chunk_ms"])
        if "min_audio_seconds" in streaming_config:
            config.min_audio_seconds = float(streaming_config["min_audio_seconds"])
        if "vad_threshold" in streaming_config:
            config.vad_threshold = float(streaming_config["vad_threshold"])
        if "vad_speech_pad_ms" in streaming_config:
            config.vad_speech_pad_ms = int(streaming_config["vad_speech_pad_ms"])
        if "stability_count" in streaming_config:
            config.stability_count = int(streaming_config["stability_count"])
        if "silence_commit_ms" in streaming_config:
            config.silence_commit_ms = int(streaming_config["silence_commit_ms"])
        if "prompt_max_words" in streaming_config:
            config.prompt_max_words = int(streaming_config["prompt_max_words"])
        if "overlap_max_words" in streaming_config:
            config.overlap_max_words = int(streaming_config["overlap_max_words"])
        if "use_initial_prompt" in streaming_config:
            config.use_initial_prompt = bool(streaming_config["use_initial_prompt"])
        if "consume_audio_on_commit" in streaming_config:
            config.consume_audio_on_commit = bool(streaming_config["consume_audio_on_commit"])
        if "batch_mode" in streaming_config:
            config.batch_mode = bool(streaming_config["batch_mode"])
        if "batch_silence_threshold_ms" in streaming_config:
            config.batch_silence_threshold_ms = int(streaming_config["batch_silence_threshold_ms"])

        if "max_updates_per_sec" in injector_config:
            config.max_updates_per_sec = int(injector_config["max_updates_per_sec"])
        if "max_backspace_chars" in injector_config:
            config.max_backspace_chars = int(injector_config["max_backspace_chars"])
        if "keystroke_delay_seconds" in injector_config:
            config.keystroke_delay_seconds = float(injector_config["keystroke_delay_seconds"])
        if "backspace_delay_seconds" in injector_config:
            config.backspace_delay_seconds = float(injector_config["backspace_delay_seconds"])

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

    @staticmethod
    def _load_toml(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "rb") as f:
            return tomllib.load(f)

    @staticmethod
    def _merge_dicts(base: dict, override: dict) -> dict:
        merged = dict(base)
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = Config._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _config_paths() -> list[Path]:
        repo_root = Path(__file__).resolve().parents[2]
        return [
            repo_root / "murmur.toml",
            repo_root / "murmur.conf",  # Legacy support
            Path.home() / ".config" / "murmur" / "murmur.toml",
            Path.home() / ".config" / "murmur" / "config.toml",
            Path.home() / ".config" / "murmur" / "murmur.conf",
        ]

    @property
    def model_path(self) -> Path:
        """Get full path to the model file."""
        return self.whisper_path / "models" / f"ggml-{self.model}.bin"
