"""CLI contracts for the native Nemotron diarization backend."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from wiz import cli, config as cfg


def _native_config(**changes) -> cfg.Config:
    config = cfg.Config(diarization_backend="nemotron")
    for key, value in changes.items():
        setattr(config, key, value)
    return config


@pytest.mark.parametrize(
    "argv",
    [
        ["transcribe", "episode.wav"],
        ["merge", "episode.wav"],
        ["speakers", "match", "episode.wav"],
    ],
)
def test_native_execution_flags_are_shared_by_public_diarization_commands(argv):
    args = cli.build_parser().parse_args(
        argv
        + [
            "--diarization-backend", "sherpa",
            "--nemo-speech-cli", "/opt/nemo-speech",
            "--nemotron-model", "/models/diar.gguf",
            "--nemotron-device", "metal",
            "--diarization-provider", "coreml",
            "--diarization-threads", "8",
            "--diarization-window-shift", "0.4",
        ]
    )
    config = cfg.Config()
    cli._apply_diarization_execution_overrides(args, config)

    assert config.diarization_backend == "sherpa"
    assert config.nemo_speech_cli == "/opt/nemo-speech"
    assert config.nemotron_model == "/models/diar.gguf"
    assert config.nemotron_device == "metal"
    assert config.diarization_provider == "coreml"
    assert config.diarization_threads == 8
    assert config.diarization_window_shift == 0.4


def test_native_is_the_default_and_does_not_use_sherpa_threshold():
    args = SimpleNamespace(cluster_threshold=None)
    assert cfg.Config().diarization_backend == "nemotron"
    assert cli._diarization_threshold(args, cfg.Config()) is None
    assert cli._diarization_threshold(
        args, cfg.Config(diarization_backend="sherpa")
    ) == 0.9


@pytest.mark.parametrize(
    "field, value",
    [
        ("speakers", 2),
        ("cluster_threshold", 0.85),
        ("diarization_window_shift", 0.4),
    ],
)
def test_native_rejects_unsupported_explicit_tuning(field, value):
    args = SimpleNamespace(speakers=0, cluster_threshold=None, diarization_window_shift=None)
    setattr(args, field, value)
    with pytest.raises(SystemExit, match="Nemotron detects speakers automatically"):
        cli._validate_diarization_options(args, _native_config())


def test_transcribe_command_rejects_native_forced_count_before_setup(monkeypatch):
    args = cli.build_parser().parse_args(["transcribe", "episode.wav", "--speakers", "2"])
    monkeypatch.setattr(cli.cfg, "load", lambda: cfg.Config())
    monkeypatch.setattr(cli, "_ensure_diarization_ready", lambda *_a, **_k: pytest.fail("setup ran before validation"))
    monkeypatch.setattr(cli, "_build_transcribe_args", lambda *_a, **_k: pytest.fail("build ran before validation"))

    with pytest.raises(SystemExit, match="Nemotron detects speakers automatically"):
        cli.cmd_transcribe(args)


def test_native_setup_does_not_install_or_download_sherpa(monkeypatch):
    config = _native_config()
    monkeypatch.setattr(cli.N, "find_runtime", lambda _config: Path("/usr/local/bin/nemo-speech"))
    monkeypatch.setattr(cli.N, "find_model", lambda _config: Path("/models/diar.gguf"))
    monkeypatch.setattr(cli, "_install_sherpa_onnx", lambda: pytest.fail("sherpa install ran"))
    monkeypatch.setattr(cli.D, "download_diarization_models", lambda: pytest.fail("sherpa model download ran"))

    assert cli._ensure_diarization_ready(config, setup_allowed=True)


def test_native_dry_run_does_not_download_missing_model(monkeypatch, capsys):
    config = _native_config()
    monkeypatch.setattr(cli.N, "find_runtime", lambda _config: Path("/usr/local/bin/nemo-speech"))
    monkeypatch.setattr(cli.N, "find_model", lambda _config: None)
    monkeypatch.setattr(cli.N, "download_model", lambda: pytest.fail("model download ran"))

    assert not cli._ensure_diarization_ready(config, dry_run=True, setup_allowed=True)
    assert "DRY-RUN" in capsys.readouterr().err


def test_native_configured_paths_fail_without_fallback(tmp_path):
    bad_runtime = tmp_path / "missing-nemo-speech"
    bad_model = tmp_path / "missing-model.gguf"
    config = _native_config(
        nemo_speech_cli=str(bad_runtime),
        nemotron_model=str(bad_model),
    )
    with pytest.raises(cli.D.DiarizationUnavailable, match="nemo_speech_cli"):
        cli._ensure_diarization_ready(config, setup_allowed=True)

    valid_runtime = tmp_path / "nemo-speech"
    valid_runtime.write_text("#!/bin/sh\n", encoding="utf-8")
    valid_runtime.chmod(0o755)
    with pytest.raises(cli.D.DiarizationUnavailable, match="nemotron_model"):
        cli._ensure_diarization_ready(
            _native_config(
                nemo_speech_cli=str(valid_runtime),
                nemotron_model=str(bad_model),
            ),
            setup_allowed=True,
        )


def test_native_setup_hint_is_backend_specific():
    hint = cli._diarization_setup_hint(_native_config())
    assert "NeMo-Speech.cpp" in hint
    assert cli.D.DIARIZE_INJECT not in hint


def test_native_missing_runtime_never_triggers_package_setup(monkeypatch):
    monkeypatch.setattr(cli.N, "find_runtime", lambda _config: None)
    monkeypatch.setattr(cli, "_auto_setup_consent", lambda _config: pytest.fail("runtime must be installed explicitly"))
    monkeypatch.setattr(cli, "_install_sherpa_onnx", lambda: pytest.fail("native runtime is not sherpa"))
    monkeypatch.setattr(cli.N, "download_model", lambda: pytest.fail("model download before runtime"))
    assert not cli._ensure_diarization_ready(cfg.Config())


@pytest.mark.parametrize("allowed", [False, True])
def test_native_model_setup_respects_opt_out_and_uses_only_native_download(monkeypatch, allowed):
    config = cfg.Config()
    downloads = []
    monkeypatch.setattr(cli.N, "find_runtime", lambda _config: Path("/bin/nemo-speech"))
    monkeypatch.setattr(cli.N, "find_model", lambda _config: Path("/model.gguf") if downloads else None)
    monkeypatch.setattr(cli.N, "download_model", lambda: downloads.append("native"))
    monkeypatch.setattr(cli, "_auto_setup_consent", lambda _config: True if allowed else pytest.fail("opt-out must bypass consent"))
    monkeypatch.setattr(cli.D, "download_diarization_models", lambda: pytest.fail("Pyannote download"))
    monkeypatch.setattr(cli, "_install_sherpa_onnx", lambda: pytest.fail("unnecessary package install"))
    assert cli._ensure_diarization_ready(config, setup_allowed=allowed) is allowed
    assert downloads == (["native"] if allowed else [])


def test_native_voice_setup_downloads_embedding_without_pyannote(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace())
    monkeypatch.setattr(cli.D, "find_embedding_model", lambda _config: None)
    monkeypatch.setattr(cli, "_auto_setup_consent", lambda _config: True)
    monkeypatch.setattr(cli.D, "download_diarization_models", lambda: pytest.fail("native profiles must not need Pyannote"))
    downloads = []
    monkeypatch.setattr(cli.D, "download_embedding_model", lambda: downloads.append("embedding"))
    cli._ensure_voice_profiles_ready(cfg.Config(), setup_allowed=True)
    assert downloads == ["embedding"]


@pytest.mark.parametrize("runtime_state", ["installed", "missing", "invalid_override"])
def test_native_model_download_reports_completed_downloads(tmp_path, monkeypatch, capsys, runtime_state):
    config = _native_config()
    if runtime_state == "invalid_override":
        config.nemo_speech_cli = str(tmp_path / "missing-runtime")
    else:
        monkeypatch.setattr(cli.N, "find_runtime", lambda _config: tmp_path / "runtime" if runtime_state == "installed" else None)
    monkeypatch.setattr(cli.cfg, "load", lambda: config)
    native_model = tmp_path / cli.N.MODEL_NAME
    embedding_model = tmp_path / cli.D.EMB_MODEL_FILE
    monkeypatch.setattr(cli.N, "download_model", lambda dest_dir: native_model.write_bytes(b"native model"))
    monkeypatch.setattr(cli.D, "download_embedding_model", lambda dest_dir: embedding_model.write_bytes(b"embedding model"))

    rc = cli.cmd_models_download_diarization(SimpleNamespace(dest=str(tmp_path), diarization_backend=None))

    output = capsys.readouterr()
    assert native_model.read_bytes() == b"native model"
    assert embedding_model.read_bytes() == b"embedding model"
    assert rc == 0
    assert "Done." in output.out
    assert "Download failed" not in output.err
    assert "download-diarization" not in output.err
    if runtime_state == "missing":
        assert "Install NeMo-Speech.cpp" in output.err
    elif runtime_state == "invalid_override":
        assert "Configured nemo_speech_cli" in output.err
        assert "missing-runtime" in output.err
    else:
        assert "runtime missing" not in output.err


def test_native_model_download_failure_remains_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.cfg, "load", lambda: _native_config())
    def fail_download(dest_dir):
        raise RuntimeError("model checksum mismatch")
    monkeypatch.setattr(cli.N, "download_model", fail_download)
    monkeypatch.setattr(cli.D, "download_embedding_model", lambda **_kwargs: pytest.fail("download continued after failure"))

    assert cli.cmd_models_download_diarization(SimpleNamespace(dest=str(tmp_path), diarization_backend=None)) == 2
    assert "model checksum mismatch" in capsys.readouterr().err
