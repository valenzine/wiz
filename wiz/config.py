"""Persistent user configuration for wiz.

Config lives at ~/.config/wiz/config.toml (created on demand).
Only Python 3.11+ stdlib tomllib is used for reading; writing is a tiny
hand-rolled TOML emitter so we don't depend on a third-party package.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

CONFIG_DIR = Path(os.environ.get("WIZ_CONFIG_DIR", Path.home() / ".config" / "wiz"))
CONFIG_PATH = CONFIG_DIR / "config.toml"
CACHE_DIR = Path.home() / ".cache" / "wiz"

# Where this state lived when the project was called whiz (honoring the
# old WHIZ_CONFIG_DIR override, so a custom location is migrated too).
LEGACY_CONFIG_DIR = Path(os.environ.get("WHIZ_CONFIG_DIR", Path.home() / ".config" / "whiz"))
LEGACY_CACHE_DIR = Path.home() / ".cache" / "whiz"


def migrate_legacy_dirs(
    on_copied: Callable[[Path, Path], None] | None = None,
) -> list[tuple[Path, Path]]:
    """Copy the whiz-era config and cache dirs to their wiz locations, once.

    Copied, not moved: an older whiz install keeps working, and nothing is
    lost if wiz is removed. A destination that already exists is never
    touched. Each copy lands in a private temporary sibling and is renamed
    into place, so an interrupted copy leaves nothing behind and two wiz
    commands started together can't trip over each other: whichever
    finishes second sees the destination already there and stands down.
    An explicit WIZ_CONFIG_DIR opts the config dir out of migration.

    ``on_copied(old, new)`` is called right after each successful copy, so a
    later failure can't hide one that already happened. Returns the pairs
    copied on this call.
    """
    pairs = [(LEGACY_CACHE_DIR, CACHE_DIR)]
    if "WIZ_CONFIG_DIR" not in os.environ:
        pairs.insert(0, (LEGACY_CONFIG_DIR, CONFIG_DIR))
    migrated = []
    for old, new in pairs:
        if new.exists() or not old.is_dir():
            continue
        tmp = None
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix=f".{new.name}.migrating-", dir=new.parent))
            shutil.copytree(old, tmp, symlinks=True, dirs_exist_ok=True)
            try:
                tmp.rename(new)
            except OSError:
                if not new.is_dir():
                    raise
                # Another wiz command finished the same copy first.
                shutil.rmtree(tmp, ignore_errors=True)
                continue
        except BaseException as e:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
            if not isinstance(e, OSError):
                raise
            raise RuntimeError(
                f"Could not copy {old} to {new}: {e}. Nothing was removed from {old}. "
                f"To skip this copy, create {new} yourself (mkdir -p {new}) and run wiz again."
            ) from e
        migrated.append((old, new))
        if on_copied is not None:
            on_copied(old, new)
    return migrated


@dataclass
class Config:
    # Model alias or absolute path preferred by default (empty => auto-pick best).
    model: str = ""
    # Extra directories to scan for models.
    model_dirs: list[str] = field(default_factory=list)
    # whisper-cli binary path. Empty => auto-detect on PATH.
    whisper_cli: str = ""
    # ffmpeg binary path. Empty => auto-detect on PATH.
    ffmpeg: str = ""
    # Number of CPU threads (0 => auto: min(8, cpu_count)).
    threads: int = 0
    # Spoken language code or "auto".
    language: str = "auto"
    # Enable VAD by default.
    vad: bool = True
    # Path to Silero VAD model (empty => auto-discover or download ggml-silero-vad.bin).
    vad_model: str = ""
    # VAD threshold.
    vad_threshold: float = 0.5
    # Output formats to produce by default.
    outputs: list[str] = field(default_factory=lambda: ["srt", "json"])
    # Print progress to stderr.
    verbose: bool = True
    # Additional flags passed verbatim to whisper-cli.
    extra_args: list[str] = field(default_factory=list)
    # --- Diarization (sherpa-onnx) ---
    # Enable speaker diarization by default.
    diarize: bool = False
    # Known number of speakers (0 => auto-detect via cluster_threshold).
    num_speakers: int = 0
    # Clustering threshold when auto-detecting (larger = fewer speakers; default 0.9 per sherpa-onnx guidance).
    cluster_threshold: float = 0.9
    # Explicit paths to diarization models (empty => auto-discover).
    diarization_segmentation_model: str = ""
    diarization_embedding_model: str = ""
    # Preserve sherpa-onnx's historical CPU / one-thread behavior unless the
    # user explicitly chooses a different execution setting.
    diarization_provider: str = "cpu"
    diarization_threads: int = 1
    # Pyannote segmentation window shift, as a fraction of the window
    # (0 < x <= 1). 0.1 is sherpa-onnx's own default; larger is faster.
    diarization_window_shift: float = 0.1
    # Remembered answer to the one-time diarization auto-setup prompt.
    # None (unset) => ask on a TTY / proceed automatically when scripted;
    # true/false answers permanently for both. Written by the prompt and
    # settable by hand: wiz config set auto_diarization_setup=false
    auto_diarization_setup: bool | None = None
    # --- AI analysis (Ollama / OpenAI-compatible) ---
    # Base URL of the chat completions endpoint (without /chat/completions).
    ai_base_url: str = "http://localhost:11434/v1"
    # Model name (e.g. 'llava', 'qwen2.5-vl', 'gpt-4o-mini'). Empty => error with hint.
    ai_model: str = ""
    # API key (Ollama ignores this; set for cloud OpenAI-compatible providers).
    ai_api_key: str = ""
    # Max frames sent to a vision model (spread evenly across the video).
    ai_max_frames: int = 50
    # --- Speaker voice profiles ---
    # Cosine-similarity threshold for auto-matching a cluster to a stored profile.
    # Higher = stricter (fewer auto-assignments); 0.8 suits 3D-Speaker embeddings.
    speaker_match_threshold: float = 0.8
    # When True, save a voice profile for each named speaker after transcription/merge.
    save_voice_profiles: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_MODEL_SEARCH_DIRS: list[Path] = [
    Path.home() / ".cache" / "whisper",
    Path.home() / "Library" / "Application Support" / "com.unspoken.app" / "WhisperModels",
    Path.home() / "Library" / "Caches" / "whisper",
    Path("/usr/local/share/whisper"),
    Path("/opt/homebrew/share/whisper"),
    Path("/usr/share/whisper"),
]


def _escape_toml_string(value: str) -> str:
    """Escape a string for a basic TOML double-quoted literal.

    Order matters: backslash first (or its own output would be re-escaped
    by later steps), then the quote, then control characters — a raw
    ``\n`` inside a quoted string makes the file invalid TOML for the
    NEXT ``load()`` of every command (C1, wave-1 audit). Multi-line values
    such as AI prompts are the known trigger.
    """
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _emit_toml(data: dict[str, Any]) -> str:
    """Minimal TOML writer for our flat config schema."""
    lines: list[str] = []
    for key, value in data.items():
        if value is None:
            # Tri-state fields (e.g. auto_diarization_setup) are "unset" by
            # None — an emitted `= None` would be invalid TOML and break the
            # NEXT load() for every command, not just the one that saved.
            continue
        if isinstance(value, bool):
            lines.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, int):
            lines.append(f"{key} = {value}")
        elif isinstance(value, float):
            lines.append(f"{key} = {value}")
        elif isinstance(value, str):
            lines.append(f'{key} = "{_escape_toml_string(value)}"')
        elif isinstance(value, list):
            if not value:
                lines.append(f"{key} = []")
            elif all(isinstance(v, str) for v in value):
                items = ", ".join(
                    '"' + _escape_toml_string(v) + '"' for v in value
                )
                lines.append(f"{key} = [{items}]")
            else:
                items = ", ".join(str(v) for v in value)
                lines.append(f"{key} = [{items}]")
        else:
            lines.append(f"{key} = {value!r}")
    return "\n".join(lines) + "\n"


def load() -> Config:
    """Load config from disk, falling back to defaults (missing file only).

    A MISSING file means defaults — fine. A CORRUPT file must not be
    silently swallowed into defaults either: that would make every
    ``wiz config set`` read-modify-write from an empty table and rewrite
    the file, permanently deleting every key the user had (C1, wave-1
    audit). Raise a RuntimeError naming the file with the fix instead;
    ``main()`` catches RuntimeError and prints it as a clean error.
    """
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as e:
            raise RuntimeError(
                f"config.toml is corrupt and could not be read: {e}\n"
                f"Fix or delete the file by hand: {CONFIG_PATH}\n"
                "(a deleted file regenerates from defaults; a corrupted one "
                "left in place breaks every command until fixed)"
            ) from e
        # Only keep known keys so older configs don't break dataclass init.
        known = {k: v for k, v in data.items() if k in Config.__dataclass_fields__}
        return Config(**known)
    return Config()


DIARIZATION_PROVIDERS = frozenset({"cpu", "coreml"})


def validate_diarization_execution_settings(config: Config) -> None:
    """Reject invalid persisted sherpa-onnx execution settings."""
    provider = config.diarization_provider
    if not isinstance(provider, str) or provider not in DIARIZATION_PROVIDERS:
        raise RuntimeError(
            "Invalid diarization_provider="
            f"{provider!r}. Must be one of: {', '.join(sorted(DIARIZATION_PROVIDERS))}"
        )
    threads = config.diarization_threads
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise RuntimeError(
            f"Invalid diarization_threads={threads!r}. Must be an integer >= 1"
        )
    shift = config.diarization_window_shift
    if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not 0 < shift <= 1:
        raise RuntimeError(
            f"Invalid diarization_window_shift={shift!r}. Must be a number with 0 < x <= 1"
        )


def save(cfg: Config) -> Path:
    """Write config to disk, preserving keys this version does not know about.

    Read-modify-write rather than a plain overwrite. ``load()`` already filters
    to known dataclass fields, so a naive ``write_text(_emit_toml(cfg.to_dict()))``
    silently deletes every key the running build has never heard of.

    That is not hypothetical. The config file is shared by several writers that
    do not agree on the schema: older installs left their own keys, feature
    branches add theirs (``ocr_*``), and a user may be running a pipx
    install that is older or newer than the checkout. Any ``wiz config set``
    from the wrong one wiped the others' settings without a word.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    merged: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("rb") as fh:
                merged.update(tomllib.load(fh))
        except (OSError, tomllib.TOMLDecodeError):
            # An unreadable file should not block saving; fall back to a
            # clean write rather than refusing to persist the change.
            merged = {}
    # Skip fields the in-memory config never answered (None): they must not
    # clobber a value a previous session persisted (a fresh Config() has
    # auto_diarization_setup=None even when the file says true).
    updates = {k: v for k, v in cfg.to_dict().items() if v is not None}
    merged.update(updates)

    CONFIG_PATH.write_text(_emit_toml(merged), encoding="utf-8")
    return CONFIG_PATH


def model_search_dirs(cfg: Config) -> list[Path]:
    """Built-in defaults plus user-configured extra dirs."""
    dirs = list(DEFAULT_MODEL_SEARCH_DIRS)
    for d in cfg.model_dirs:
        dirs.append(Path(d).expanduser())
    # De-dup preserving order.
    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        key = str(d.expanduser().resolve())
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out
