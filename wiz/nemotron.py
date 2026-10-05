"""Native NeMo-Speech.cpp adapter for automatic speaker diarization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from wiz import config as cfg
from wiz.diarize import DiarSegment, DiarizationUnavailable

MODEL_NAME = "Nemotron-3-Diarization.q8_0.gguf"
MODEL_REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
MODEL_SHA256 = "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1"
MODEL_SIZE = 107_012_128
MODEL_URL = (
    "https://huggingface.co/nvidia/Nemotron-3-Diarization-GGUF/resolve/"
    f"{MODEL_REVISION}/{MODEL_NAME}"
)
PRESET = "v3-offline"
CACHE_VERSION = 1
MAX_RTTM_ROWS = 100_000
OFFICIAL_RUNTIME_DIR = (
    Path.home() / "Library" / "Application Support" / "NeMoSpeech"
)


def _default_model_path() -> Path:
    return cfg.CACHE_DIR / "diarization" / MODEL_NAME


def find_runtime(config: cfg.Config) -> Path | None:
    """Return the configured native executable, without accepting a bad override."""
    configured = getattr(config, "nemo_speech_cli", "")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise DiarizationUnavailable(
            f"Configured nemo_speech_cli is not an executable file: {path}"
        )
    found = shutil.which("nemo-speech")
    if found:
        return Path(found)
    # The official bundle has a platform-specific directory below this stable
    # prefix. Keep the bundle intact: its binary loads sibling libraries.
    candidates = [OFFICIAL_RUNTIME_DIR / "bin" / "nemo-speech"]
    candidates.extend(sorted(OFFICIAL_RUNTIME_DIR.glob("*/bin/nemo-speech")))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def find_model(config: cfg.Config) -> Path | None:
    """Return the configured model, without falling back from a bad override."""
    configured = getattr(config, "nemotron_model", "")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path
        raise DiarizationUnavailable(
            f"Configured nemotron_model is not a file: {path}"
        )
    path = _default_model_path()
    return path if path.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_support_sha256(runtime: Path) -> str:
    """Fingerprint sibling libraries loaded by the bundled executable."""
    library_dir = runtime.parent.parent / "lib"
    digest = hashlib.sha256()
    if not library_dir.is_dir():
        digest.update(b"no-sibling-libraries")
        return digest.hexdigest()
    for library in sorted(path for path in library_dir.rglob("*") if path.is_file()):
        digest.update(str(library.relative_to(library_dir)).encode("utf-8"))
        digest.update(_sha256(library).encode("ascii"))
    return digest.hexdigest()


def download_model(dest_dir: Path | None = None) -> Path:
    """Atomically download the pinned official Nemotron model and verify it."""
    target = (dest_dir or _default_model_path().parent) / MODEL_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == MODEL_SIZE and _sha256(target) == MODEL_SHA256:
        return target

    fd, tmp_name = tempfile.mkstemp(prefix=f".{MODEL_NAME}.", suffix=".download", dir=target.parent)
    os.close(fd)
    temporary = Path(tmp_name)
    try:
        # Keep the shared downloader as the single HTTP implementation.
        from wiz.diarize import _download

        _download(MODEL_URL, temporary)
        if temporary.stat().st_size != MODEL_SIZE or _sha256(temporary) != MODEL_SHA256:
            raise RuntimeError("Downloaded Nemotron model failed size or SHA256 verification.")
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def cache_path(source: Path) -> Path:
    """Native cache path, deliberately distinct from legacy sherpa cache files."""
    return Path(str(source) + ".nemotron.diar.json")


def _source_fingerprint(source: Path) -> dict[str, int | str]:
    stat = source.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(source),
    }


def _validate_segments(rows: object) -> list[DiarSegment] | None:
    if not isinstance(rows, list) or not rows or len(rows) > MAX_RTTM_ROWS:
        return None
    segments: list[DiarSegment] = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        if (
            not isinstance(row.get("start"), (int, float))
            or isinstance(row.get("start"), bool)
            or not isinstance(row.get("end"), (int, float))
            or isinstance(row.get("end"), bool)
            or not isinstance(row.get("speaker"), int)
            or isinstance(row.get("speaker"), bool)
        ):
            return None
        try:
            start = float(row["start"])
            end = float(row["end"])
            speaker = int(row["speaker"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(value) for value in (start, end)) or start < 0 or end <= start or speaker < 0:
            return None
        segments.append(DiarSegment(start=start, end=end, speaker=speaker))
    return segments


def _cache_identity(source: Path, runtime: Path, model: Path, device: str) -> dict[str, object]:
    return {
        "version": CACHE_VERSION,
        "source": _source_fingerprint(source),
        "runtime": str(runtime),
        "runtime_sha256": _sha256(runtime),
        "runtime_support_sha256": _runtime_support_sha256(runtime),
        "model": str(model),
        "model_sha256": _sha256(model),
        "device": device,
        "preset": PRESET,
    }


def _load_cache(source: Path, identity: dict[str, object]) -> list[DiarSegment] | None:
    path = cache_path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("identity") != identity:
        return None
    return _validate_segments(payload.get("segments"))


def _write_cache(source: Path, identity: dict[str, object], segments: list[DiarSegment]) -> Path:
    path = cache_path(source)
    payload = {
        "identity": identity,
        "segments": [{"start": s.start, "end": s.end, "speaker": s.speaker} for s in segments],
    }
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp_name).replace(path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def _validate_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as wav:
            valid = (
                wav.getframerate() == 16_000
                and wav.getnchannels() == 1
                and wav.getsampwidth() == 2
                and wav.getcomptype() == "NONE"
            )
    except (OSError, wave.Error) as exc:
        raise RuntimeError(f"Nemotron requires a readable 16kHz mono PCM WAV: {path}") from exc
    if not valid:
        raise RuntimeError(f"Nemotron requires a 16kHz mono 16-bit PCM WAV: {path}")


def parse_rttm(path: Path) -> list[DiarSegment]:
    """Parse NeMo RTTM output into stable integer speaker IDs."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Nemotron did not produce RTTM output: {path}") from exc
    if len(lines) > MAX_RTTM_ROWS:
        raise RuntimeError(f"Nemotron RTTM exceeds {MAX_RTTM_ROWS} rows.")
    labels: dict[str, int] = {}
    segments: list[DiarSegment] = []
    for line_no, line in enumerate(lines, 1):
        parts = line.split()
        # NeMo writes the recording basename verbatim. It can contain spaces,
        # so parse the fixed RTTM suffix from the right instead of assuming a
        # single token recording ID.
        if len(parts) < 10 or parts[0] != "SPEAKER":
            raise RuntimeError(f"Malformed Nemotron RTTM row {line_no}.")
        recording_id = " ".join(parts[1:-8])
        channel, start_text, duration_text, orthography, subtype, label, confidence, slat = parts[-8:]
        if (
            not recording_id
            or not channel.isdecimal()
            or int(channel) < 1
            or orthography != "<NA>"
            or subtype != "<NA>"
            or confidence != "<NA>"
            or slat != "<NA>"
        ):
            raise RuntimeError(f"Malformed Nemotron RTTM row {line_no}.")
        try:
            start = float(start_text)
            duration = float(duration_text)
        except ValueError as exc:
            raise RuntimeError(f"Malformed Nemotron RTTM timing at row {line_no}.") from exc
        end = start + duration
        if not all(math.isfinite(value) for value in (start, duration, end)) or start < 0 or duration <= 0:
            raise RuntimeError(f"Invalid Nemotron RTTM timing at row {line_no}.")
        if not label or label == "<NA>":
            raise RuntimeError(f"Malformed Nemotron RTTM speaker at row {line_no}.")
        speaker = labels.setdefault(label, len(labels))
        segments.append(DiarSegment(start=start, end=end, speaker=speaker))
    if not segments:
        raise RuntimeError("Nemotron produced an empty RTTM file.")
    return sorted(segments, key=lambda segment: (segment.start, segment.end, segment.speaker))


def run(
    wav: Path,
    config: cfg.Config,
    *,
    dry_run: bool = False,
    use_cache: bool = True,
    cache_source: Path | None = None,
) -> list[DiarSegment]:
    """Run native Nemotron diarization with validated cache reuse."""
    runtime = find_runtime(config)
    if runtime is None:
        raise DiarizationUnavailable(
            "nemo-speech is not installed or on PATH; set nemo_speech_cli to its executable."
        )
    model = find_model(config)
    if model is None:
        if dry_run:
            model = _default_model_path()
        else:
            raise DiarizationUnavailable(
                "Nemotron model not found. Run `wiz models download-diarization` first."
            )
    device = getattr(config, "nemotron_device", "auto") or "auto"
    if device not in cfg.NEMOTRON_DEVICES:
        raise DiarizationUnavailable(
            f"Invalid nemotron_device={device!r}. Choose: {', '.join(sorted(cfg.NEMOTRON_DEVICES))}"
        )
    command_prefix = [str(runtime), "diarize", str(wav), "--model", str(model), "--device", device,
                      "--preset", PRESET, "--format", "rttm"]
    if dry_run:
        print("DRY-RUN Nemotron diarization:")
        print("  command: " + " ".join(command_prefix + ["--output", "RESULT.rttm"]))
        return []

    _validate_wav(wav)
    source = cache_source or wav
    identity = _cache_identity(source, runtime, model, device)
    if use_cache:
        cached = _load_cache(source, identity)
        if cached is not None:
            from wiz import ui
            ui.muted(f"Reusing Nemotron diarization cache ({len(cached)} segments): {cache_path(source)}")
            return cached

    from wiz import ui
    ui.phase("diarizing with Nemotron")
    ui.muted(f"  device: {device}")
    ui.muted(f"  preset: {PRESET}")
    with tempfile.TemporaryDirectory(prefix="wiz-nemotron-") as temporary_dir:
        rttm = Path(temporary_dir) / "result.rttm"
        completed = subprocess.run(
            command_prefix + ["--output", str(rttm)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"nemo-speech diarization failed (exit {completed.returncode}): {details}")
        segments = parse_rttm(rttm)
    saved_cache = _write_cache(source, identity, segments)
    ui.muted(
        f"Nemotron found {len(segments)} segments across "
        f"{len({segment.speaker for segment in segments})} speaker clusters."
    )
    ui.muted(f"Saved Nemotron diarization cache: {saved_cache}")
    return segments
