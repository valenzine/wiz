"""Native Nemotron adapter contracts, without a real runtime or model."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import shlex
import wave

import pytest

from wiz import config as cfg
from wiz import diarize as D
from wiz import nemotron as N


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16_000)
        out.writeframes(b"\0\0" * 160)


def _config(**values):
    config = cfg.Config()
    config.diarization_backend = "nemotron"
    config.nemo_speech_cli = ""
    config.nemotron_model = ""
    config.nemotron_device = "auto"
    for key, value in values.items():
        setattr(config, key, value)
    return config


def _ready(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime" / "bin" / "nemo-speech"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"runtime")
    runtime.chmod(0o755)
    library = runtime.parent.parent / "lib" / "libnemo.dylib"
    library.parent.mkdir()
    library.write_bytes(b"library")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    monkeypatch.setattr(N, "find_runtime", lambda _config: runtime)
    monkeypatch.setattr(N, "find_model", lambda _config: model)
    return runtime, model


def _rttm_process(calls):
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        output = Path(argv[argv.index("--output") + 1])
        output.write_text(
            "SPEAKER recording 1 1.5 0.5 <NA> <NA> guest <NA> <NA>\n"
            "SPEAKER recording 1 0.0 1.0 <NA> <NA> host <NA> <NA>\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return run


def test_native_executes_fixed_command_and_maps_arbitrary_ids(tmp_path, monkeypatch):
    wav = tmp_path / "episode.wav"
    _wav(wav)
    _ready(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(N.subprocess, "run", _rttm_process(calls))

    segments = D.run_diarization(wav, _config(), use_cache=False)

    assert [(segment.start, segment.end, segment.speaker) for segment in segments] == [
        (0.0, 1.0, 1), (1.5, 2.0, 0),
    ]
    argv = calls[0][0]
    assert argv[1:3] == ["diarize", str(wav)]
    assert ["--preset", N.PRESET, "--format", "rttm"] == argv[argv.index("--preset"):argv.index("--preset") + 4]
    assert "--output" in argv
    assert calls[0][1]["check"] is False
    assert "shell" not in calls[0][1]


def test_native_cache_uses_source_and_runtime_model_device_and_preset(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp3"
    source.write_bytes(b"original audio")
    wav = tmp_path / "normalized.wav"
    _wav(wav)
    runtime, model = _ready(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(N.subprocess, "run", _rttm_process(calls))

    assert D.run_diarization(wav, _config(), cache_source=source)
    assert D.run_diarization(wav, _config(), cache_source=source)
    assert len(calls) == 1
    assert N.cache_path(source).exists()
    assert not N.cache_path(wav).exists()

    model.write_bytes(b"new model")
    assert D.run_diarization(wav, _config(), cache_source=source)
    runtime.write_bytes(b"new runtime")
    assert D.run_diarization(wav, _config(), cache_source=source)
    (runtime.parent.parent / "lib" / "libnemo.dylib").write_bytes(b"new library")
    assert D.run_diarization(wav, _config(), cache_source=source)
    assert D.run_diarization(wav, _config(nemotron_device="metal"), cache_source=source)
    original_stat = source.stat()
    source.write_bytes(b"changed audio!")  # same 14-byte length as the original source
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert D.run_diarization(wav, _config(nemotron_device="metal"), cache_source=source)
    assert len(calls) == 6


def test_native_invalid_cache_rows_are_never_reused(tmp_path, monkeypatch):
    wav = tmp_path / "episode.wav"
    _wav(wav)
    _ready(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(N.subprocess, "run", _rttm_process(calls))
    assert D.run_diarization(wav, _config())
    cached = N.cache_path(wav)
    payload = json.loads(cached.read_text(encoding="utf-8"))
    payload["segments"] = [{"start": 0, "end": float("inf"), "speaker": 0}]
    cached.write_text(json.dumps(payload), encoding="utf-8")
    assert D.run_diarization(wav, _config())
    assert len(calls) == 2


def test_native_cache_rejects_numeric_overflow():
    assert N._validate_segments([
        {"start": 10 ** 400, "end": 1, "speaker": 0},
    ]) is None


@pytest.mark.parametrize("corruption", ["empty", "invalid_utf8"])
def test_native_empty_or_non_utf8_cache_is_a_miss(tmp_path, monkeypatch, corruption):
    wav = tmp_path / "episode.wav"
    _wav(wav)
    _ready(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(N.subprocess, "run", _rttm_process(calls))
    assert D.run_diarization(wav, _config())
    cache = N.cache_path(wav)
    if corruption == "empty":
        payload = json.loads(cache.read_text(encoding="utf-8"))
        payload["segments"] = []
        cache.write_text(json.dumps(payload), encoding="utf-8")
    else:
        cache.write_bytes(b"\xff\xfe")
    assert D.run_diarization(wav, _config())
    assert len(calls) == 2


@pytest.mark.parametrize("row", ["", "SPEAKER too short", "SPEAKER x 1 nan 1 <NA> <NA> person <NA> <NA>\n"])
def test_native_rejects_empty_or_malformed_rttm(tmp_path, row):
    path = tmp_path / "result.rttm"
    path.write_text(row, encoding="utf-8")
    with pytest.raises(RuntimeError, match="Nemotron"):
        N.parse_rttm(path)


def test_native_parses_official_rttm_with_a_spaced_unicode_recording_id(tmp_path):
    path = tmp_path / "result.rttm"
    path.write_text(
        "SPEAKER Perón con espacios 1 0.000 1.099 <NA> <NA> speaker_1 <NA> <NA>\n"
        "SPEAKER Perón con espacios 1 1.171 4.818 <NA> <NA> speaker_2 <NA> <NA>\n",
        encoding="utf-8",
    )
    assert N.parse_rttm(path) == [
        D.DiarSegment(start=0.0, end=1.099, speaker=0),
        D.DiarSegment(start=1.171, end=5.989, speaker=1),
    ]


def test_native_rejects_forced_speaker_count_and_explicit_threshold(tmp_path):
    config = _config()
    with pytest.raises(ValueError, match="speaker count"):
        D.run_diarization(tmp_path / "missing.wav", config, num_speakers=2)
    with pytest.raises(ValueError, match="threshold"):
        D.run_diarization(tmp_path / "missing.wav", config, threshold=0.9)


def test_dispatcher_rejects_an_unknown_backend_before_sherpa_setup(tmp_path, monkeypatch):
    config = _config(diarization_backend="nemotorn")
    monkeypatch.setattr(D, "_import_sherpa", lambda: pytest.fail("wrong backend must not select sherpa"))
    with pytest.raises(D.DiarizationUnavailable, match="diarization_backend"):
        D.run_diarization(tmp_path / "missing.wav", config)


def test_native_dry_run_needs_setup_but_not_a_real_wav(tmp_path, monkeypatch, capsys):
    runtime, model = _ready(tmp_path / "runtime with spaces", monkeypatch)
    wav = tmp_path / "temporary audio does not exist.wav"
    assert D.run_diarization(wav, _config(), dry_run=True) == []
    text = capsys.readouterr().out
    assert "--preset v3-offline --format rttm" in text
    command = text.split("  command: ", 1)[1].strip()
    argv = shlex.split(command)
    assert argv[0] == str(runtime)
    assert argv[2] == str(wav)
    assert argv[argv.index("--model") + 1] == str(model)


def test_native_dry_run_does_not_download_a_missing_model(tmp_path, monkeypatch, capsys):
    runtime = tmp_path / "nemo-speech"
    runtime.write_bytes(b"runtime")
    runtime.chmod(0o755)
    monkeypatch.setattr(N, "find_runtime", lambda _config: runtime)
    monkeypatch.setattr(N, "find_model", lambda _config: None)
    monkeypatch.setattr(N, "download_model", lambda: pytest.fail("dry run must not download"))
    assert N.run(tmp_path / "missing.wav", _config(), dry_run=True) == []
    assert str(N._default_model_path()) in capsys.readouterr().out


def test_native_direct_call_rejects_invalid_device(tmp_path, monkeypatch):
    _ready(tmp_path, monkeypatch)
    with pytest.raises(D.DiarizationUnavailable, match="nemotron_device"):
        N.run(tmp_path / "missing.wav", _config(nemotron_device="broken"), dry_run=True)


def test_explicit_invalid_runtime_and_model_do_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr(N.shutil, "which", lambda _name: str(tmp_path / "ignored"))
    with pytest.raises(D.DiarizationUnavailable, match="nemo_speech_cli"):
        N.find_runtime(_config(nemo_speech_cli=str(tmp_path / "missing")))
    with pytest.raises(D.DiarizationUnavailable, match="nemotron_model"):
        N.find_model(_config(nemotron_model=str(tmp_path / "missing.gguf")))


def test_runtime_discovery_accepts_the_official_stable_prefix(tmp_path, monkeypatch):
    runtime = tmp_path / "NeMoSpeech" / "bin" / "nemo-speech"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"runtime")
    runtime.chmod(0o755)
    monkeypatch.setattr(N, "OFFICIAL_RUNTIME_DIR", runtime.parent.parent)
    monkeypatch.setattr(N.shutil, "which", lambda _name: None)
    assert N.find_runtime(_config()) == runtime


def test_embedding_download_does_not_provision_a_segmentation_model(tmp_path, monkeypatch):
    calls = []

    def download(url, target):
        calls.append((url, target))
        target.write_bytes(b"embedding")

    monkeypatch.setattr(D, "_download", download)
    result = D.download_embedding_model(tmp_path)
    assert result == tmp_path / D.EMB_MODEL_FILE
    assert result.read_bytes() == b"embedding"
    assert len(calls) == 1
    assert calls[0][0] == D.EMB_URL
    assert calls[0][1].parent == result.parent
    assert calls[0][1].name.startswith(f".{result.name}.")
    assert not (tmp_path / D.SEG_DIR_NAME).exists()


def test_native_process_failure_is_loud(tmp_path, monkeypatch):
    wav = tmp_path / "episode.wav"
    _wav(wav)
    _ready(tmp_path, monkeypatch)
    monkeypatch.setattr(
        N.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=3, stdout="", stderr="metal unavailable"),
    )
    with pytest.raises(RuntimeError, match="exit 3.*metal unavailable"):
        D.run_diarization(wav, _config(), use_cache=False)


def test_native_returns_segments_when_cache_write_fails(tmp_path, monkeypatch):
    wav = tmp_path / "episode.wav"
    _wav(wav)
    _ready(tmp_path, monkeypatch)
    monkeypatch.setattr(N.subprocess, "run", _rttm_process([]))

    def fail_cache_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(N, "_write_cache", fail_cache_write)

    assert D.run_diarization(wav, _config(), use_cache=False) == [
        D.DiarSegment(start=0.0, end=1.0, speaker=1),
        D.DiarSegment(start=1.5, end=2.0, speaker=0),
    ]


def test_native_execution_oserror_is_diarization_unavailable(tmp_path, monkeypatch):
    wav = tmp_path / "episode.wav"
    _wav(wav)
    _ready(tmp_path, monkeypatch)

    def fail_exec(*_args, **_kwargs):
        raise OSError("executable format error")

    monkeypatch.setattr(N.subprocess, "run", fail_exec)

    with pytest.raises(
        D.DiarizationUnavailable,
        match="Could not execute nemo-speech.*executable format error",
    ):
        D.run_diarization(wav, _config(), use_cache=False)


def test_native_invalid_utf8_process_output_is_reported_as_failure(tmp_path):
    wav = tmp_path / "episode.wav"
    _wav(wav)
    runtime = tmp_path / "nemo-speech"
    runtime.write_text("#!/bin/sh\nprintf '\\377' >&2\nexit 23\n", encoding="utf-8")
    runtime.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")

    with pytest.raises(RuntimeError, match="exit 23"):
        N.run(
            wav,
            _config(nemo_speech_cli=str(runtime), nemotron_model=str(model)),
            use_cache=False,
        )


def test_native_finds_model_in_configured_model_dirs(tmp_path, monkeypatch):
    custom_models = tmp_path / "custom-models"
    custom_models.mkdir()
    model = custom_models / N.MODEL_NAME
    model.write_bytes(b"model")
    monkeypatch.setattr(
        N, "_default_model_path", lambda: tmp_path / "missing" / N.MODEL_NAME
    )

    assert N.find_model(_config(model_dirs=[str(custom_models)])) == model


def test_native_ignores_blank_padded_rttm_rows(tmp_path):
    path = tmp_path / "result.rttm"
    path.write_text(
        " \t \n"
        "SPEAKER recording 1 0.0 1.0 <NA> <NA> host <NA> <NA>\n"
        "\t\n",
        encoding="utf-8",
    )

    assert N.parse_rttm(path) == [D.DiarSegment(start=0.0, end=1.0, speaker=0)]


def test_native_model_download_preserves_existing_model_after_failed_verification(
    tmp_path, monkeypatch
):
    target = tmp_path / N.MODEL_NAME
    target.write_bytes(b"known-good")
    expected = b"expected-model"
    monkeypatch.setattr(N, "MODEL_SIZE", len(expected))
    monkeypatch.setattr(N, "MODEL_SHA256", hashlib.sha256(expected).hexdigest())
    monkeypatch.setattr(
        D, "_download", lambda _url, temporary: temporary.write_bytes(b"truncated")
    )

    with pytest.raises(RuntimeError, match="size or SHA256"):
        N.download_model(tmp_path)

    assert target.read_bytes() == b"known-good"
    assert list(tmp_path.glob(f".{N.MODEL_NAME}.*.download")) == []
