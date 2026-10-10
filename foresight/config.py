"""Paths and user settings. Everything lives under one data directory so the
whole app state can be backed up or wiped by deleting a single folder."""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_NAME = "Foresight"


def data_dir() -> Path:
    override = os.environ.get("FORESIGHT_HOME")
    if override:
        root = Path(override)
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_NAME
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "foresight"
    root.mkdir(parents=True, exist_ok=True)
    return root


def models_dir() -> Path:
    p = data_dir() / "models"
    p.mkdir(exist_ok=True)
    return p


def bin_dir() -> Path:
    p = data_dir() / "bin"
    p.mkdir(exist_ok=True)
    return p


def meetings_dir() -> Path:
    p = data_dir() / "meetings"
    p.mkdir(exist_ok=True)
    return p


def logs_dir() -> Path:
    p = data_dir() / "logs"
    p.mkdir(exist_ok=True)
    return p


def db_path() -> Path:
    return data_dir() / "foresight.db"


# Languages offered for transcription. Gemma 4's ASR prompt names the language
# explicitly, so this is a user choice rather than auto-detection.
LANGUAGES = ["English", "Danish", "Swedish", "Norwegian", "German", "Dutch", "French",
             "Spanish", "Italian", "Portuguese", "Polish", "Finnish", "Japanese", "Korean", "Chinese"]


@dataclass
class Settings:
    language: str = "English"
    # "gemma-4-e4b" or "gemma-4-e2b"; same model handles transcription, notes and answers
    llm: str = "gemma-4-e4b"
    # "auto" picks cuda when an NVIDIA GPU is present, otherwise vulkan
    backend: str = "auto"
    ctx_size: int = 65536
    mic_device: int | None = None  # PortAudio index; None = Windows default input
    speaker_device: int | None = None  # output device whose loopback is captured; None = default
    capture_system_audio: bool = True
    # With speakers (no headphones) the mic hears the other side too; we then drop
    # mic lines that duplicate what the system-audio channel already transcribed.
    headphones: bool = False
    save_audio: bool = True
    folders: list[str] = field(default_factory=list)
    your_name: str = "You"

    @classmethod
    def load(cls) -> "Settings":
        path = data_dir() / "settings.json"
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        path = data_dir() / "settings.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(path)


_lock = threading.Lock()
_settings: Settings | None = None


def settings() -> Settings:
    global _settings
    with _lock:
        if _settings is None:
            _settings = Settings.load()
        return _settings


def update_settings(**changes) -> Settings:
    global _settings
    with _lock:
        current = _settings or Settings.load()
        known = {f.name for f in fields(Settings)}
        bad = set(changes) - known
        if bad:
            raise KeyError(f"unknown settings: {', '.join(sorted(bad))}")
        data = asdict(current) | changes
        _settings = Settings(**data)
        _settings.save()
        return _settings
