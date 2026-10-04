"""Execution settings passed to sherpa's speaker inference models."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from wiz import config as cfg
from wiz import diarize as D
from wiz import profiles as P
from wiz import cli


@pytest.mark.parametrize(
    ("provider", "threads"),
    [("cpu", 1), ("coreml", 8)],
)
def test_diarization_passes_settings_to_both_models(
    tmp_path, monkeypatch, provider, threads,
):
    seen = {}
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    monkeypatch.setattr(D, "find_segmentation_model", lambda _config: model)
    monkeypatch.setattr(D, "find_embedding_model", lambda _config: model)
    monkeypatch.setattr(D, "_read_wav_pcm", lambda _wav: ([0.0], 16000))
    monkeypatch.setattr(D, "_write_diarization_cache", lambda *_a, **_k: tmp_path / "cache.json")

    def segmentation(**kwargs):
        seen["segmentation"] = kwargs
        return object()

    def embedding(*args, **kwargs):
        seen["embedding"] = (args, kwargs)
        return object()

    class FakeDiarizer:
        sample_rate = 16000

        def __init__(self, _config):
            pass

        def process(self, _samples, callback):
            return SimpleNamespace(sort_by_start_time=lambda: [])

    fake = SimpleNamespace(
        OfflineSpeakerSegmentationPyannoteModelConfig=lambda *_a, **_k: object(),
        OfflineSpeakerSegmentationModelConfig=segmentation,
        SpeakerEmbeddingExtractorConfig=embedding,
        FastClusteringConfig=lambda **_k: object(),
        OfflineSpeakerDiarizationConfig=lambda **_k: SimpleNamespace(validate=lambda: True),
        OfflineSpeakerDiarization=FakeDiarizer,
    )
    monkeypatch.setattr(D, "_import_sherpa", lambda: fake)

    config = cfg.Config(diarization_provider=provider, diarization_threads=threads)
    assert D.run_diarization(tmp_path / "episode.wav", config, use_cache=False) == []
    assert seen["segmentation"]["provider"] == provider
    assert seen["segmentation"]["num_threads"] == threads
    assert seen["embedding"][1]["provider"] == provider
    assert seen["embedding"][1]["num_threads"] == threads


@pytest.mark.parametrize(
    ("provider", "threads"), [("cpu", 1), ("coreml", 8)],
)
def test_voice_profile_embeddings_pass_execution_settings(
    tmp_path, monkeypatch, provider, threads,
):
    seen = {}
    model = tmp_path / "embedding.onnx"
    model.write_bytes(b"model")
    monkeypatch.setattr(P, "find_embedding_model", lambda _config: model)
    monkeypatch.setattr(P, "_read_wav_pcm", lambda _wav: ([0.0] * 16000, 16000))

    class FakeStream:
        def accept_waveform(self, _sample_rate, _samples):
            pass

        def input_finished(self):
            pass

    class FakeExtractor:
        dim = 2

        def __init__(self, _config):
            pass

        def create_stream(self):
            return FakeStream()

        def is_ready(self, _stream):
            return True

        def compute(self, _stream):
            return [0.5, 0.5]

    def embedding_config(*args, **kwargs):
        seen["config"] = (args, kwargs)
        return object()

    monkeypatch.setattr(P, "_import_sherpa", lambda: SimpleNamespace(
        SpeakerEmbeddingExtractorConfig=embedding_config,
        SpeakerEmbeddingExtractor=FakeExtractor,
    ))

    config = cfg.Config(diarization_provider=provider, diarization_threads=threads)
    result = P.compute_speaker_embeddings(
        tmp_path / "episode.wav", [D.DiarSegment(0, 1, 0)], config,
    )
    assert result == {0: [0.5, 0.5]}
    assert seen["config"][1]["provider"] == provider
    assert seen["config"][1]["num_threads"] == threads


def test_config_defaults_and_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    defaults = cfg.load()
    assert (defaults.diarization_provider, defaults.diarization_threads) == ("cpu", 1)

    cfg.save({"diarization_provider": "coreml", "diarization_threads": 4})
    loaded = cfg.load()
    assert (loaded.diarization_provider, loaded.diarization_threads) == ("coreml", 4)


def test_config_set_persists_only_the_requested_execution_setting(tmp_path, monkeypatch):
    """An explicit default must remain explicit without materializing defaults.

    ``config set`` is a surgical update: existing known and future keys stay
    intact, while unset defaults do not become sticky configuration.
    """
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(cfg, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    config_path.write_text(
        'ai_base_url = "https://configured.example/v1"\nfuture_key = "keep"\n',
        encoding="utf-8",
    )

    assert cli.cmd_config_set(SimpleNamespace(assignment="diarization_threads=1")) == 0

    assert config_path.read_text(encoding="utf-8") == (
        'ai_base_url = "https://configured.example/v1"\n'
        'future_key = "keep"\n'
        "diarization_threads = 1\n"
    )


@pytest.mark.parametrize("command", ["transcribe", "merge", "match"])
def test_cli_overrides_persistent_execution_settings(tmp_path, monkeypatch, command):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    (tmp_path / "config.toml").write_text(
        'diarization_provider = "cpu"\ndiarization_threads = 4\n', encoding="utf-8"
    )
    source = tmp_path / "episode.wav"
    source.write_bytes(b"wav")
    prefix = ["speakers", "match"] if command == "match" else [command]
    args = cli.build_parser().parse_args(
        prefix + [str(source), "--speakers", "2", "--diarization-provider", "coreml",
                  "--diarization-threads", "8"]
    )

    class Captured(Exception):
        pass

    def capture(config, *args, **kwargs):
        assert (config.diarization_provider, config.diarization_threads) == ("coreml", 8)
        raise Captured

    if command == "transcribe":
        monkeypatch.setattr(cli, "_build_transcribe_args", lambda _args, config, *, timings=None: capture(config))
    else:
        monkeypatch.setattr(cli, "_ensure_diarization_ready", capture)
    with pytest.raises(Captured):
        args.func(args)


@pytest.mark.parametrize("command", ["transcribe", "merge", "match"])
def test_cli_uses_persistent_execution_settings_without_overrides(tmp_path, monkeypatch, command):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text(
        'diarization_provider = "coreml"\ndiarization_threads = 4\n', encoding="utf-8"
    )
    source = tmp_path / "episode.wav"
    source.write_bytes(b"wav")
    prefix = ["speakers", "match"] if command == "match" else [command]
    args = cli.build_parser().parse_args(prefix + [str(source), "--speakers", "2"])

    class Captured(Exception):
        pass

    def capture(config, *args, **kwargs):
        assert (config.diarization_provider, config.diarization_threads) == ("coreml", 4)
        raise Captured

    if command == "transcribe":
        monkeypatch.setattr(cli, "_build_transcribe_args", lambda _args, config, *, timings=None: capture(config))
    else:
        monkeypatch.setattr(cli, "_ensure_diarization_ready", capture)
    with pytest.raises(Captured):
        args.func(args)


@pytest.mark.parametrize(
    "option,value", [
        ("--diarization-threads", "0"),
        ("--diarization-threads", "-1"),
        ("--diarization-provider", "nonsense"),
    ],
)
def test_invalid_cli_execution_settings_are_rejected(option, value, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["transcribe", "episode.wav", option, value])
    assert exc.value.code == 2
    assert option in capsys.readouterr().err


@pytest.mark.parametrize(
    "line,key", [
        ('diarization_provider = "nonsense"\n', "diarization_provider"),
        ('diarization_threads = 0\n', "diarization_threads"),
        ('diarization_threads = -1\n', "diarization_threads"),
    ],
)
def test_invalid_persistent_execution_settings_are_rejected(tmp_path, monkeypatch, line, key):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text(line, encoding="utf-8")
    with pytest.raises(RuntimeError, match=key):
        D.run_diarization(tmp_path / "episode.wav", cfg.load())


@pytest.mark.parametrize(
    "assignment", [
        "diarization_provider=nonsense",
        "diarization_threads=0",
        "diarization_threads=-1",
    ],
)
def test_config_set_rejects_invalid_execution_settings(tmp_path, monkeypatch, assignment):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    with pytest.raises(SystemExit, match="Invalid diarization_"):
        cli.cmd_config_set(SimpleNamespace(assignment=assignment))
    assert not (tmp_path / "config.toml").exists()


def test_explicit_provider_initialization_failure_names_provider(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    monkeypatch.setattr(D, "find_segmentation_model", lambda _config: model)
    monkeypatch.setattr(D, "find_embedding_model", lambda _config: model)

    def fail(_config):
        raise RuntimeError("CoreML session unavailable")

    fake = SimpleNamespace(
        OfflineSpeakerSegmentationPyannoteModelConfig=lambda *_a, **_k: object(),
        OfflineSpeakerSegmentationModelConfig=lambda **_k: object(),
        SpeakerEmbeddingExtractorConfig=lambda *_a, **_k: object(),
        FastClusteringConfig=lambda **_k: object(),
        OfflineSpeakerDiarizationConfig=lambda **_k: SimpleNamespace(validate=lambda: True),
        OfflineSpeakerDiarization=fail,
    )
    monkeypatch.setattr(D, "_import_sherpa", lambda: fake)
    with pytest.raises(RuntimeError, match="coreml.*CoreML session unavailable") as exc:
        D.run_diarization(
            tmp_path / "episode.wav",
            cfg.Config(diarization_provider="coreml", diarization_threads=8),
            use_cache=False,
        )
    assert not isinstance(exc.value, D.DiarizationUnavailable)


def test_execution_settings_do_not_invalidate_diarization_cache(tmp_path, monkeypatch):
    wav = tmp_path / "episode.wav"
    wav.write_bytes(b"audio")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    segments = [D.DiarSegment(0, 1, 0)]
    D._write_diarization_cache(
        wav, segments, num_speakers=2, threshold=0.9,
        seg_model=model, emb_model=model,
    )
    monkeypatch.setattr(D, "find_segmentation_model", lambda _config: model)
    monkeypatch.setattr(D, "find_embedding_model", lambda _config: model)
    monkeypatch.setattr(D, "_import_sherpa", lambda: pytest.fail("cache should avoid inference"))
    result = D.run_diarization(
        wav, cfg.Config(diarization_provider="coreml", diarization_threads=8),
        num_speakers=2, threshold=0.9,
    )
    assert result == segments


def test_normalized_audio_reuses_cache_keyed_on_compressed_source(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp3"
    source.write_bytes(b"compressed audio")
    first_wav = tmp_path / "episode.wav"
    first_wav.write_bytes(b"first temporary WAV")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    segments = [D.DiarSegment(0, 1, 0)]
    monkeypatch.setattr(D, "find_segmentation_model", lambda _config: model)
    monkeypatch.setattr(D, "find_embedding_model", lambda _config: model)
    monkeypatch.setattr(D, "_read_wav_pcm", lambda _wav: ([0.0], 16000))
    process_calls = []

    class FakeDiarizer:
        sample_rate = 16000

        def __init__(self, _config):
            pass

        def process(self, _samples, callback):
            process_calls.append(1)
            return SimpleNamespace(sort_by_start_time=lambda: segments)

    fake = SimpleNamespace(
        OfflineSpeakerSegmentationPyannoteModelConfig=lambda *_a, **_k: object(),
        OfflineSpeakerSegmentationModelConfig=lambda **_k: object(),
        SpeakerEmbeddingExtractorConfig=lambda *_a, **_k: object(),
        FastClusteringConfig=lambda **_k: object(),
        OfflineSpeakerDiarizationConfig=lambda **_k: SimpleNamespace(validate=lambda: True),
        OfflineSpeakerDiarization=FakeDiarizer,
    )
    monkeypatch.setattr(D, "_import_sherpa", lambda: fake)

    config = cfg.Config()
    assert D.run_diarization(first_wav, config, num_speakers=2, threshold=0.9,
                             cache_source=source) == segments
    cache = D.diar_cache_path(source)
    assert cache.exists()
    assert not D.diar_cache_path(first_wav).exists()

    first_wav.unlink()
    second_wav = tmp_path / "episode.diarize.wav"
    second_wav.write_bytes(b"second temporary WAV")
    monkeypatch.setattr(D, "_import_sherpa", lambda: pytest.fail("cache hit should avoid inference"))
    assert D.run_diarization(second_wav, config, num_speakers=2, threshold=0.9,
                             cache_source=source) == segments
    assert len(process_calls) == 1

    source.write_bytes(b"different compressed audio")
    monkeypatch.setattr(D, "_import_sherpa", lambda: fake)
    assert D.run_diarization(second_wav, config, num_speakers=2, threshold=0.9,
                             cache_source=source) == segments
    assert len(process_calls) == 2


# ---------- Pyannote window shift ----------

def _window_shift_sherpa(tmp_path, monkeypatch, pyannote, segments=()):
    """Stub models + sherpa so run_diarization reaches the Pyannote config."""
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    monkeypatch.setattr(D, "find_segmentation_model", lambda _config: model)
    monkeypatch.setattr(D, "find_embedding_model", lambda _config: model)
    monkeypatch.setattr(D, "_read_wav_pcm", lambda _wav: ([0.0], 16000))
    process_calls = []

    class FakeDiarizer:
        sample_rate = 16000

        def __init__(self, _config):
            pass

        def process(self, _samples, callback):
            process_calls.append(1)
            return SimpleNamespace(sort_by_start_time=lambda: list(segments))

    monkeypatch.setattr(D, "_import_sherpa", lambda: SimpleNamespace(
        OfflineSpeakerSegmentationPyannoteModelConfig=pyannote,
        OfflineSpeakerSegmentationModelConfig=lambda **_k: object(),
        SpeakerEmbeddingExtractorConfig=lambda *_a, **_k: object(),
        FastClusteringConfig=lambda **_k: object(),
        OfflineSpeakerDiarizationConfig=lambda **_k: SimpleNamespace(validate=lambda: True),
        OfflineSpeakerDiarization=FakeDiarizer,
    ))
    return model, process_calls


def test_window_shift_defaults_to_0_2_and_is_always_passed(tmp_path, monkeypatch):
    assert cfg.Config().diarization_window_shift == 0.2
    args = cli.build_parser().parse_args(["transcribe", "episode.wav"])
    assert args.diarization_window_shift is None  # config default applies
    seen = []
    model, _ = _window_shift_sherpa(
        tmp_path, monkeypatch, lambda *a, **k: seen.append((a, k)) or object(),
    )
    D.run_diarization(tmp_path / "episode.wav", cfg.Config(), use_cache=False)
    assert seen == [((), {"model": str(model), "window_shift_ratio": 0.2})]


def test_window_shift_reaches_pyannote_config(tmp_path, monkeypatch):
    seen = []
    model, _ = _window_shift_sherpa(
        tmp_path, monkeypatch, lambda *a, **k: seen.append((a, k)) or object(),
    )
    D.run_diarization(
        tmp_path / "episode.wav", cfg.Config(diarization_window_shift=0.35), use_cache=False,
    )
    assert seen == [((), {"model": str(model), "window_shift_ratio": 0.35})]


def test_too_old_sherpa_says_how_to_upgrade(tmp_path, monkeypatch):
    def old_pyannote(model):  # sherpa-onnx < 1.13.6: no window_shift_ratio kwarg
        return object()

    _window_shift_sherpa(tmp_path, monkeypatch, old_pyannote)
    with pytest.raises(D.DiarizationUnavailable, match=r"1\.13\.6.*pipx inject --force transcript-wiz 'sherpa-onnx>=1\.13\.6'"):
        D.run_diarization(tmp_path / "episode.wav", cfg.Config(), use_cache=False)


@pytest.mark.parametrize("command", ["transcribe", "merge", "match"])
@pytest.mark.parametrize(("cli_value", "expected"), [(["--diarization-window-shift", "0.35"], 0.35), ([], 0.3)])
def test_cli_window_shift_overrides_config(tmp_path, monkeypatch, command, cli_value, expected):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text("diarization_window_shift = 0.3\n", encoding="utf-8")
    source = tmp_path / "episode.wav"
    source.write_bytes(b"wav")
    prefix = ["speakers", "match"] if command == "match" else [command]
    args = cli.build_parser().parse_args(prefix + [str(source), "--speakers", "2"] + cli_value)

    class Captured(Exception):
        pass

    def capture(config, *args, **kwargs):
        assert config.diarization_window_shift == expected
        # Provider/thread defaults are untouched by the new setting.
        assert (config.diarization_provider, config.diarization_threads) == ("cpu", 1)
        raise Captured

    if command == "transcribe":
        monkeypatch.setattr(cli, "_build_transcribe_args", lambda _args, config, *, timings=None: capture(config))
    else:
        monkeypatch.setattr(cli, "_ensure_diarization_ready", capture)
    with pytest.raises(Captured):
        args.func(args)


@pytest.mark.parametrize("command", [["transcribe"], ["merge"], ["speakers", "match"]])
@pytest.mark.parametrize("value", ["0", "-0.1", "1.1", "nan", "abc"])
def test_invalid_cli_window_shift_is_rejected(command, value, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(
            command + ["episode.wav", "--diarization-window-shift", value]
        )
    assert exc.value.code == 2
    # Our validator's message, not argparse's "unrecognized arguments".
    assert "--diarization-window-shift: must be a number with 0 < x <= 1" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1"])
def test_invalid_persistent_window_shift_is_rejected(tmp_path, monkeypatch, value):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text(f"diarization_window_shift = {value}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="diarization_window_shift"):
        D.run_diarization(tmp_path / "episode.wav", cfg.load())


def test_config_set_window_shift(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    with pytest.raises(SystemExit, match="Invalid diarization_window_shift"):
        cli.cmd_config_set(SimpleNamespace(assignment="diarization_window_shift=1.5"))
    assert not (tmp_path / "config.toml").exists()
    cli.cmd_config_set(SimpleNamespace(assignment="diarization_window_shift=0.25"))
    assert cfg.load().diarization_window_shift == 0.25


def test_dry_run_shows_window_shift(tmp_path, monkeypatch, capsys):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    monkeypatch.setattr(D, "find_segmentation_model", lambda _config: model)
    monkeypatch.setattr(D, "find_embedding_model", lambda _config: model)
    D.run_diarization(
        tmp_path / "episode.wav", cfg.Config(diarization_window_shift=0.2), dry_run=True,
    )
    assert "window shift ratio: 0.2" in capsys.readouterr().out


def test_window_shift_is_part_of_cache_key(tmp_path, monkeypatch):
    source = tmp_path / "episode.mp3"
    source.write_bytes(b"compressed audio")
    wav = tmp_path / "episode.wav"
    wav.write_bytes(b"temporary WAV")
    segments = [D.DiarSegment(0, 1, 0)]
    _, process_calls = _window_shift_sherpa(
        tmp_path, monkeypatch, lambda *_a, **_k: object(), segments,
    )

    def run(shift):
        return D.run_diarization(
            wav, cfg.Config(diarization_window_shift=shift),
            num_speakers=2, threshold=0.9, cache_source=source,
        )

    assert run(0.1) == segments and len(process_calls) == 1
    assert run(0.2) == segments and len(process_calls) == 2   # 0.1 cache not reused
    assert run(0.2) == segments and len(process_calls) == 2   # 0.2 cache reused
    assert run(0.1) == segments and len(process_calls) == 3   # 0.2 cache not reused
    # Still keyed on the original source, never the temporary WAV.
    assert D.diar_cache_path(source).exists()
    assert not D.diar_cache_path(wav).exists()


def test_cache_without_window_shift_field_is_a_miss(tmp_path, monkeypatch):
    import json

    wav = tmp_path / "episode.wav"
    wav.write_bytes(b"audio")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    D._write_diarization_cache(
        wav, [D.DiarSegment(0, 1, 0)], num_speakers=2, threshold=0.9,
        seg_model=model, emb_model=model,
    )
    cache = D.diar_cache_path(wav)
    data = json.loads(cache.read_text(encoding="utf-8"))
    del data["window_shift_ratio"]  # as written before the field existed
    cache.write_text(json.dumps(data), encoding="utf-8")
    assert D.load_diarization_cache(
        wav, num_speakers=2, threshold=0.9, seg_model=model, emb_model=model,
    ) is None
