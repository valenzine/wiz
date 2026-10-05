"""Tests for wiz.cli helpers — model-picker recommendation heuristic,
vision resolution, output fallbacks (HTML without diarization), and the
proactive diarization auto-setup (user decision, 2026-09-05).

Run with: pytest tests/test_cli.py
"""

from __future__ import annotations

import argparse
import builtins
import json
import re
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wiz import cli
from wiz.diarize import DiarSegment


def test_stage_timings_render_elapsed_and_skipped_stages(monkeypatch):
    """The always-on timing table retains subsecond precision and makes
    skipped work explicit, so it is useful for a resume benchmark."""
    clock = iter([10.0, 10.25, 10.25, 10.75])
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(clock))
    calls: list[tuple[object, object, object]] = []
    monkeypatch.setattr(cli.ui, "table", lambda title, columns, rows: calls.append((title, columns, rows)))

    timings = cli._StageTimings()
    with timings.measure("Audio preparation"):
        pass
    timings.skip("Transcription", "resumed")
    with timings.measure("Diarization"):
        pass
    timings.skip("Speaker profiles", "no speaker labels")
    timings.skip("Writing outputs / frames", "nothing requested")
    timings.render(total_wall=0.75)

    assert calls == [(
        "Stage timing",
        [("Stage", "left"), ("Time", "right")],
        [
            ["Audio preparation", "0.2s"],
            ["Transcription", "skipped (resumed)"],
            ["Diarization", "0.5s"],
            ["Speaker profiles", "skipped (no speaker labels)"],
            ["Writing outputs / frames", "skipped (nothing requested)"],
            ["Total (wall)", "0.8s"],
        ],
    )]


def test_stage_timings_pause_accumulate_and_preserve_skip_reason(monkeypatch):
    clock = iter([0.0, 0.2, 1.0, 1.3, 2.0, 2.4, 10.0, 10.4, 30.0, 30.2])
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(clock))
    calls: list[list[list[str]]] = []
    monkeypatch.setattr(cli.ui, "table", lambda _title, _columns, rows: calls.append(rows))

    timings = cli._StageTimings()
    with timings.measure("Diarization"):
        pass
    with timings.measure("Diarization"):
        pass
    timings.start("Writing outputs / frames")
    timings.pause("Writing outputs / frames")
    timings.resume("Writing outputs / frames")
    timings.stop("Writing outputs / frames")
    timings.skip("Speaker profiles", "disabled")
    timings.skip("Speaker profiles", "no speaker labels")
    timings.render(total_wall=119.99)

    values = dict(calls[0])
    assert values["Diarization"] == "0.5s"
    assert values["Writing outputs / frames"] == "0.8s"
    assert values["Speaker profiles"] == "skipped (disabled)"
    assert values["Total (wall)"] == "2:00.0"


def test_transcribe_preflight_failure_does_not_render_timing(monkeypatch, tmp_path):
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    clock = iter([0.0, 0.4])
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    monkeypatch.setattr(
        cli, "_build_transcribe_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ffmpeg failed")),
    )
    tables: list[tuple[str | None, list[list[str]]]] = []
    monkeypatch.setattr(cli.ui, "table", lambda title, _columns, rows: tables.append((title, rows)))

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        cli.cmd_transcribe(_transcribe_args(audio))

    assert tables == []


def test_transcribe_prepare_failure_renders_failed_stage_timing(monkeypatch, tmp_path):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"fake audio")
    _prepare_diarization_build(monkeypatch)
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa", vad=False))
    now = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: now[0])

    def fail_prepare(*_args, **_kwargs):
        now[0] += 0.4
        raise RuntimeError("ffmpeg failed")

    monkeypatch.setattr(cli.aud, "prepare_diarization_audio", fail_prepare)
    tables: list[tuple[str | None, list[list[str]]]] = []
    monkeypatch.setattr(cli.ui, "table", lambda title, _columns, rows: tables.append((title, rows)))

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        cli.cmd_transcribe(_transcribe_args(source, outputs="srt", speakers=1))

    timing = dict(tables[0][1])
    assert timing["Audio preparation"] == "0.4s"
    assert timing["Total (wall)"] == "0.4s"


@pytest.mark.parametrize(
    ("returncode", "resume", "expected"),
    [
        (1, False, "skipped (transcription failed)"),
        (0, True, "skipped (resumed)"),
    ],
)
def test_transcribe_timing_labels_unwritten_outputs_truthfully(
    monkeypatch, tmp_path, returncode, resume, expected,
):
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=False)
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda _cmd: SimpleNamespace(returncode=returncode))
    tables: list[tuple[str | None, list[list[str]]]] = []
    monkeypatch.setattr(cli.ui, "table", lambda title, _columns, rows: tables.append((title, rows)))
    args = _transcribe_args(audio, outputs="srt", speakers=None)
    args.resume = resume

    assert cli.cmd_transcribe(args) == returncode

    timing = dict(next(rows for title, rows in tables if title == "Stage timing"))
    assert timing["Audio preparation"] == "skipped (not needed)"
    assert timing["Writing outputs / frames"] == expected


def test_transcribe_missing_input_does_not_render_timing(monkeypatch, tmp_path):
    tables: list[tuple[str | None, list[list[str]]]] = []
    monkeypatch.setattr(cli.ui, "table", lambda title, _columns, rows: tables.append((title, rows)))

    with pytest.raises(SystemExit, match="Input file not found"):
        cli.cmd_transcribe(_transcribe_args(tmp_path / "missing.m4a"))

    assert tables == []


def test_transcribe_timing_reports_empty_whisper_json_not_written(monkeypatch, tmp_path):
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    (tmp_path / "meeting.m4a.json").write_text('{"transcription": []}', encoding="utf-8")
    monkeypatch.setattr(
        cli, "_run_diarize_or_fallback",
        lambda *_args: [DiarSegment(start=0.0, end=1.0, speaker=0)],
    )
    tables: list[tuple[str | None, list[list[str]]]] = []
    monkeypatch.setattr(cli.ui, "table", lambda title, _columns, rows: tables.append((title, rows)))

    assert cli.cmd_transcribe(_transcribe_args(audio, outputs="srt", speakers=1)) == 1

    timing = dict(next(rows for title, rows in tables if title == "Stage timing"))
    assert timing["Writing outputs / frames"] == "skipped (no transcript segments)"


def test_transcribe_timing_prioritizes_degraded_resume_over_resume_label(monkeypatch, tmp_path):
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli, "_run_diarize_or_fallback", lambda *_args: [])
    tables: list[tuple[str | None, list[list[str]]]] = []
    monkeypatch.setattr(cli.ui, "table", lambda title, _columns, rows: tables.append((title, rows)))
    args = _transcribe_args(audio, outputs="srt", speakers=1)
    args.resume = True

    assert cli.cmd_transcribe(args) == 1

    timing = dict(next(rows for title, rows in tables if title == "Stage timing"))
    assert timing["Writing outputs / frames"] == "skipped (no speaker labels)"


def test_recommend_model_empty_returns_zero():
    assert cli._recommend_model([], prefer_vision=False) == 0


def test_recommend_model_prefers_non_cloud():
    models = ["gpt-4o-mini:cloud", "qwen2.5:3b", "llava:latest"]
    # qwen2.5:3b (non-cloud + text token) beats gpt-4o-mini (cloud + text token)
    idx = cli._recommend_model(models, prefer_vision=False)
    assert models[idx] == "qwen2.5:3b"


def test_recommend_model_prefers_vision_when_requested():
    models = ["gpt-4o-mini:cloud", "qwen2.5:3b", "llava:latest"]
    # llava matches vision tokens; qwen2.5:3b does not (prefer_vision=True)
    idx = cli._recommend_model(models, prefer_vision=True)
    assert models[idx] == "llava:latest"


def test_recommend_model_all_cloud_picks_best_token_match():
    models = ["devstral-small-2:24b-cloud", "glm-5.1:cloud", "qwen3-coder-next:cloud"]
    idx = cli._recommend_model(models, prefer_vision=False)
    # All cloud (score 0 base); text-token matches add 5. First one with a text
    # token wins. 'devstral' contains 'devstral' token -> score 5.
    assert models[idx] == "devstral-small-2:24b-cloud"


def test_recommend_model_first_wins_on_ties():
    models = ["alpha:cloud", "beta:cloud", "gamma:cloud"]
    # All cloud, no token matches -> tie at score 0 -> first wins (index 0).
    assert cli._recommend_model(models, prefer_vision=False) == 0


def test_recommend_model_prefers_cloud_vision_when_requested():
    models = ["gpt-oss:20b-cloud", "qwen3.5:cloud", "glm-5.1:cloud"]
    # qwen3.5 is cloud vision-capable; gpt-oss and glm-5.1 are not.
    idx = cli._recommend_model(models, prefer_vision=True)
    assert models[idx] == "qwen3.5:cloud"


def test_model_picker_persists_only_the_chosen_model(tmp_path, monkeypatch):
    """Choosing a model must not persist command-scoped configuration.

    The picker receives the active config, which can include CLI overrides;
    only the user's model choice belongs in persistent configuration.
    """
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", config_path)
    config_path.write_text('future_key = "keep"\n', encoding="utf-8")
    monkeypatch.setattr(cli.AI, "list_ollama_models", lambda _url: ["llama3.1"])
    monkeypatch.setattr(cli.AI, "probe_model", lambda *_args: (True, ""))
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "")
    config = cli.cfg.Config(diarization_backend="sherpa", ai_base_url="https://command-line.example/v1", ai_api_key="ephemeral")

    assert cli._pick_model_interactive(config, prefer_vision=False) == "llama3.1"
    assert config.ai_model == "llama3.1"
    assert config_path.read_text(encoding="utf-8") == (
        'future_key = "keep"\n'
        'ai_model = "llama3.1"\n'
    )


# ---------- _looks_vision_capable ----------

def test_looks_vision_capable_true_for_known_vision_models():
    for name in ("llava", "llava:latest", "qwen2.5-vl", "minicpm-v", "gpt-4o",
                "gpt-4o-mini", "pixtral-12b", "internvl2", "phi-3.5-vision",
                # Cloud vision-capable models (no 'vl'/'vision' in name).
                "qwen3.5:cloud", "qwen3.5:397b", "kimi-k2.6:cloud",
                "kimi-k2.7-code:cloud", "gemma4:31b", "gemma4:31b-cloud",
                "mistral-large-3:675b", "minimax-m3:cloud"):
        assert cli._looks_vision_capable(name) is True, name


def test_looks_vision_capable_false_for_text_models():
    for name in ("gpt-oss:20b-cloud", "gpt-oss:120b", "llama3.1", "qwen2.5:3b",
                "deepseek-coder", "devstral-small", "gpt-3.5-turbo",
                "glm-5.1:cloud", "qwen3-coder-next:cloud"):
        assert cli._looks_vision_capable(name) is False, name


def test_looks_vision_capable_empty_or_none():
    assert cli._looks_vision_capable("") is False
    assert cli._looks_vision_capable(None) is False


# ---------- _resolve_vision ----------

def _resolve(explicit=False, no=False, frames=False, model="llava"):
    return cli._resolve_vision(
        explicit_vision=explicit, no_vision=no,
        has_frames=frames, model=model,
    )


def test_resolve_vision_no_vision_always_disables():
    # --no-vision wins even if --vision was also set and frames exist.
    use, kind, msg = _resolve(explicit=True, no=True, frames=True, model="llava")
    assert use is False
    assert msg == ""


def test_resolve_vision_explicit_with_frames_enables():
    use, kind, msg = _resolve(explicit=True, no=False, frames=True, model="llava")
    assert use is True
    assert kind == ""
    assert msg == ""


def test_resolve_vision_explicit_without_frames_warns():
    use, kind, msg = _resolve(explicit=True, no=False, frames=False, model="llava")
    assert use is False
    assert kind == "warn"
    assert "no frames manifest" in msg


def test_resolve_vision_explicit_but_text_model_overrides():
    # Explicit --vision is a user override: even with a text-looking model we
    # send the frames (the HTTP layer surfaces a rejection hint if it fails).
    use, kind, msg = _resolve(explicit=True, no=False, frames=True, model="gpt-oss:20b")
    assert use is True
    assert kind == ""
    assert msg == ""


def test_resolve_vision_auto_enables_for_vision_model_with_frames():
    use, kind, msg = _resolve(explicit=False, no=False, frames=True, model="llava")
    assert use is True
    assert kind == "info"
    assert "auto-enabling" in msg


def test_resolve_vision_auto_enables_for_cloud_qwen3_5():
    use, kind, msg = _resolve(explicit=False, no=False, frames=True, model="qwen3.5:cloud")
    assert use is True
    assert kind == "info"
    assert "auto-enabling" in msg


def test_resolve_vision_auto_enables_for_cloud_kimi_k2_6():
    use, kind, msg = _resolve(explicit=False, no=False, frames=True, model="kimi-k2.6:cloud")
    assert use is True
    assert kind == "info"


def test_resolve_vision_text_model_with_frames_stays_text_only_with_hint():
    use, kind, msg = _resolve(explicit=False, no=False, frames=True, model="gpt-oss:20b")
    assert use is False
    assert kind == "hint"
    assert "vision-capable" in msg


def test_resolve_vision_no_frames_is_text_only_silently():
    use, kind, msg = _resolve(explicit=False, no=False, frames=False, model="llava")
    assert use is False
    assert kind == ""
    assert msg == ""


def test_resolve_vision_no_vision_overrides_auto_enable():
    use, kind, msg = _resolve(explicit=False, no=True, frames=True, model="llava")
    assert use is False
    assert kind == ""
    assert msg == ""


# ---------- output fallbacks (HTML without diarization) ----------

# whisper-cli -oj fixture: two segments the merge/HTML path can parse.
_WHISPER_JSON = (
    '{"transcription": ['
    '{"timestamps":{"from":"00:00:00,000","to":"00:00:02,000"},"text":"hello world"},'
    '{"timestamps":{"from":"00:00:02,000","to":"00:00:04,500"},"text":"second line"}'
    "]}"
)


def _transcribe_args(file, outputs="srt,html", speakers=1):
    return SimpleNamespace(
        file=str(file),
        diarization_backend="sherpa",
        output="",
        outputs=outputs,
        model="",
        threads=0,
        language="",
        vad=False,
        vad_threshold=None,
        no_timestamps=False,
        print_progress=False,
        no_progress=True,
        keep_wav=False,
        no_auto_vad_download=True,
        no_auto_diarization_setup=False,
        translate=False,
        speakers=speakers,
        no_speakers=False,
        cluster_threshold=None,
        name_speakers=False,
        no_name_speakers=True,
        speakers_names=None,
        screenshots=False,
        no_screenshots=True,
        screenshot_width=None,
        no_voice_profiles=True,
        resume=False,
        verbose=False,
        extra=[],
        dry_run=False,
        analyze=False,
        vision=False,
        no_vision=False,
    )


def _setup_transcribe(monkeypatch, tmp_path, *, diarize_enabled, screenshots=False, name="meeting"):
    """Create a fake audio input + whisper JSON and stub out the heavy machinery.

    Returns the input Path. ``diarize_enabled``/``screenshots`` are what the
    stubbed _build_transcribe_args reports (the real one derives them from
    --speakers/video detection). The proactive diarization auto-setup is
    stubbed ready=True so no test ever runs pip or downloads models.
    """
    audio = tmp_path / f"{name}.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / f"{name}.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")

    def fake_build(args, config, *, timings=None):
        return (["whisper-cli"], "model.bin", audio, audio, False,
                audio.with_suffix(""), diarize_enabled, screenshots)

    monkeypatch.setattr(cli, "_build_transcribe_args", fake_build)
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda cmd: SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    monkeypatch.setattr(cli, "_ensure_diarization_ready", lambda config, dry_run=False, setup_allowed=True: True)
    return audio


def test_transcribe_html_fallback_when_diarization_unavailable_returns_error(tmp_path, monkeypatch, capsys):
    """--speakers with --outputs html must not silently skip the HTML when
    diarization is unavailable: it degrades to generic 'Speaker' labels.
    Uses the real _run_diarize_or_fallback (with run_diarization stubbed to
    raise the sherpa-missing error) so the 'diarization unavailable' warn is
    exercised end-to-end."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=1))

    assert rc == 1
    html_path = tmp_path / "meeting.speakers.html"
    assert html_path.exists(), "HTML transcript was silently skipped"
    content = html_path.read_text(encoding="utf-8")
    assert "hello world" in content
    assert ">Speaker<" in content  # generic label, not 'Speaker A'
    assert "Speaker A" not in content
    # The labeled SRT is NOT faked — it needs real diarization. (The
    # generic-label .speakers.txt IS written on this audio run: `wiz analyze`
    # needs a frames manifest or a .speakers.txt to find a transcript.)
    assert not (tmp_path / "meeting.speakers.srt").exists()
    txt = (tmp_path / "meeting.speakers.txt").read_text(encoding="utf-8")
    assert "Speaker (00:00:00):" in txt
    # The degraded page self-identifies with a muted note line.
    assert 'class="note"' in content
    assert "No speaker diarization" in content
    # Loud degradation via the real fallback helper, honest about what
    # happens next (audio run, explicit html → generic labels written).
    err = capsys.readouterr().err
    assert "diarization unavailable" in err
    assert "Falling back to generic 'Speaker' labels" in err


def test_transcribe_html_without_speakers(tmp_path, monkeypatch, capsys):
    """--outputs html on an audio run without diarization still writes the
    HTML (previously silently skipped); no speaker warning is needed."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=False)

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=None))

    assert rc == 0
    assert (tmp_path / "meeting.speakers.html").exists()
    assert (tmp_path / "meeting.speakers.txt").exists()
    assert not (tmp_path / "meeting.speakers.srt").exists()
    err = capsys.readouterr().err
    assert "diarization unavailable" not in err  # diarization was never on


def test_transcribe_html_and_frames_fallback_for_video(tmp_path, monkeypatch, capsys):
    """Video + --speakers + --outputs html with diarization unavailable:
    the frames manifest AND the HTML are written with generic labels, and
    frames are still inlined into the HTML."""
    video = tmp_path / "recording.mov"
    video.write_bytes(b"fake video")
    (tmp_path / "recording.wav.json").write_text(_WHISPER_JSON, encoding="utf-8")
    frames_dir = tmp_path / "recording.frames"
    frames_dir.mkdir()
    (frames_dir / "seg0001.jpg").write_bytes(b"\xff\xd8jpeg\xff\xd9")
    manifest = tmp_path / "recording.frames.json"

    def fake_build(args, config, *, timings=None):
        wav = tmp_path / "recording.wav"
        return (["whisper-cli"], "model.bin", wav, video, False,
                tmp_path / "recording", True, True)

    monkeypatch.setattr(cli, "_build_transcribe_args", fake_build)
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda cmd: SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    monkeypatch.setattr(
        cli, "_extract_and_manifest_screenshots",
        lambda in_path, merged, of_base, ffmpeg, width, dry_run: (frames_dir, manifest, False),
    )
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))

    rc = cli.cmd_transcribe(_transcribe_args(video, outputs="html", speakers=1))

    assert rc == 1
    html = (tmp_path / "recording.speakers.html").read_text(encoding="utf-8")
    assert "hello world" in html
    assert "<img" in html  # frame inlined even without speaker labels
    assert 'class="note"' in html  # degraded page self-identifies
    # Explicit --speakers + artifacts to write → loud warn via the real
    # fallback helper, honest that generic labels are being written.
    err = capsys.readouterr().err
    assert "diarization unavailable" in err
    assert "Falling back to generic 'Speaker' labels" in err
    # Labeled outputs are not faked; video runs have a frames manifest, so
    # no generic-label .speakers.txt is needed for `wiz analyze`.
    assert not (tmp_path / "recording.speakers.srt").exists()
    assert not (tmp_path / "recording.speakers.txt").exists()


def test_transcribe_no_crash_when_json_missing(tmp_path, monkeypatch):
    """A missing whisper JSON on the unlabeled path must warn, not crash
    (the old screenshots-only block read an unbound 'result')."""
    video = tmp_path / "recording.mov"
    video.write_bytes(b"fake video")

    def fake_build(args, config, *, timings=None):
        wav = tmp_path / "recording.wav"
        return (["whisper-cli"], "model.bin", wav, video, False,
                tmp_path / "recording", False, True)

    monkeypatch.setattr(cli, "_build_transcribe_args", fake_build)
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda cmd: SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))

    rc = cli.cmd_transcribe(_transcribe_args(video, outputs="html", speakers=None))

    assert rc == 0
    assert not (tmp_path / "recording.speakers.html").exists()


def test_build_args_forces_json_with_html_output(tmp_path, monkeypatch):
    """--outputs html must force -oj so the HTML can be rendered even when
    diarization is unavailable (segments are needed to build the page)."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    monkeypatch.setattr(cli.M, "pick_best", lambda config: Path("/models/turbo.bin"))
    monkeypatch.setattr(cli, "_find_whisper_cli", lambda configured="": "whisper-cli")

    cmd, *_rest = cli._build_transcribe_args(
        _transcribe_args(audio, outputs="html", speakers=None),
        cli.cfg.Config(diarization_backend="sherpa", vad=False),
    )

    assert "-oj" in cmd


# ---------- diarization audio normalization ----------

def _prepare_diarization_build(monkeypatch):
    """Keep the real input-preparation path isolated from external tools."""
    monkeypatch.setattr(cli.M, "pick_best", lambda config: Path("/models/turbo.bin"))
    monkeypatch.setattr(cli, "_find_whisper_cli", lambda configured="": "whisper-cli")
    monkeypatch.setattr(cli, "_ensure_diarization_ready", lambda *a, **k: True)
    monkeypatch.setattr(cli.aud, "find_ffmpeg", lambda configured="": "ffmpeg")


def _write_pcm_wav(path, *, rate=16_000, channels=1, width=2):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(b"\0" * channels * width * 160)


def test_transcribe_mp3_speakers_normalizes_for_whisper_and_diarization_then_cleans(tmp_path, monkeypatch, capsys):
    source = tmp_path / "recording.mp3"
    source.write_bytes(b"fake mp3")
    (tmp_path / "recording.mp3.json").write_text(_WHISPER_JSON, encoding="utf-8")
    _prepare_diarization_build(monkeypatch)
    extracted: list[Path] = []
    whisper_cmds: list[list[str]] = []
    diarized: list[tuple[Path, Path | None]] = []

    def fake_extract(src, ffmpeg, dest_dir=None, dry_run=False, output=None):
        assert src == source
        assert ffmpeg == "ffmpeg"
        assert output is not None
        extracted.append(output)
        output.write_bytes(b"normalized wav")
        return output

    monkeypatch.setattr(cli.aud, "extract_audio", fake_extract)
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda cmd: whisper_cmds.append(cmd) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, *_a, **_k: diarized.append((wav, _k.get("cache_source"))) or [DiarSegment(start=0, end=1, speaker=0)],
    )
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa", vad=False))

    rc = cli.cmd_transcribe(_transcribe_args(source, outputs="srt", speakers=1))

    normalized = tmp_path / "recording.wav"
    assert rc == 0
    assert extracted == [normalized]
    assert whisper_cmds[0][whisper_cmds[0].index("-f") + 1] == str(normalized)
    assert whisper_cmds[0][whisper_cmds[0].index("-of") + 1] == str(source)
    assert diarized == [(normalized, source)]
    assert not normalized.exists()
    assert (tmp_path / "recording.speakers.txt").exists()
    assert not (tmp_path / "recording.mp3.speakers.txt").exists()
    rendered = " ".join(capsys.readouterr().err.split())
    assert "Input" in rendered and "Audio" in rendered
    assert "Video" not in rendered


def test_transcribe_mp3_degraded_html_keeps_speaker_stem(tmp_path, monkeypatch):
    source = tmp_path / "recording.mp3"
    source.write_bytes(b"fake mp3")
    (tmp_path / "recording.mp3.json").write_text(_WHISPER_JSON, encoding="utf-8")
    _prepare_diarization_build(monkeypatch)

    def fake_extract(_src, _ffmpeg, dest_dir=None, dry_run=False, output=None):
        assert output is not None
        output.write_bytes(b"normalized wav")
        return output

    monkeypatch.setattr(cli.aud, "extract_audio", fake_extract)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda _cmd: SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa", vad=False))

    assert cli.cmd_transcribe(_transcribe_args(source, outputs="html", speakers=1)) == 1
    assert (tmp_path / "recording.speakers.html").exists()
    assert not (tmp_path / "recording.mp3.speakers.html").exists()


def test_transcribe_mp3_without_speakers_remains_direct(tmp_path, monkeypatch):
    source = tmp_path / "recording.mp3"
    source.write_bytes(b"fake mp3")
    _prepare_diarization_build(monkeypatch)
    monkeypatch.setattr(cli.aud, "extract_audio", lambda *_a, **_k: pytest.fail("plain MP3 must stay direct"))

    cmd, _model, wav, *_rest = cli._build_transcribe_args(
        _transcribe_args(source, outputs="srt", speakers=None), cli.cfg.Config(diarization_backend="sherpa", vad=False),
    )

    assert wav == source
    assert cmd[cmd.index("-f") + 1] == str(source)


def test_transcribe_mp3_speakers_keep_wav_retains_normalized_audio(tmp_path, monkeypatch):
    source = tmp_path / "recording.mp3"
    source.write_bytes(b"fake mp3")
    (tmp_path / "recording.mp3.json").write_text(_WHISPER_JSON, encoding="utf-8")
    _prepare_diarization_build(monkeypatch)

    def fake_extract(_src, _ffmpeg, dest_dir=None, dry_run=False, output=None):
        assert output is not None
        output.write_bytes(b"normalized wav")
        return output

    monkeypatch.setattr(cli.aud, "extract_audio", fake_extract)
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda cmd: SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli.D, "run_diarization", lambda *_a, **_k: [DiarSegment(start=0, end=1, speaker=0)])
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa", vad=False))
    args = _transcribe_args(source, outputs="srt", speakers=1)
    args.keep_wav = True

    assert cli.cmd_transcribe(args) == 0
    assert (tmp_path / "recording.wav").exists()


def test_transcribe_mp3_resume_reuses_source_named_json(tmp_path, monkeypatch):
    source = tmp_path / "recording.mp3"
    source.write_bytes(b"fake mp3")
    (tmp_path / "recording.mp3.json").write_text(_WHISPER_JSON, encoding="utf-8")
    _prepare_diarization_build(monkeypatch)

    def fake_extract(_src, _ffmpeg, dest_dir=None, dry_run=False, output=None):
        assert output is not None
        output.write_bytes(b"normalized wav")
        return output

    monkeypatch.setattr(cli.aud, "extract_audio", fake_extract)
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda _cmd: pytest.fail("resume must skip whisper"))
    monkeypatch.setattr(cli.D, "run_diarization", lambda *_a, **_k: [DiarSegment(start=0, end=1, speaker=0)])
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa", vad=False))
    args = _transcribe_args(source, outputs="srt", speakers=1)
    args.resume = True

    assert cli.cmd_transcribe(args) == 0
    assert not (tmp_path / "recording.wav").exists()


def test_transcribe_mp3_interrupt_removes_normalized_audio(tmp_path, monkeypatch):
    source = tmp_path / "recording.mp3"
    source.write_bytes(b"fake mp3")
    _prepare_diarization_build(monkeypatch)

    def fake_extract(_src, _ffmpeg, dest_dir=None, dry_run=False, output=None):
        assert output is not None
        output.write_bytes(b"normalized wav")
        return output

    def interrupt(_cmd):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.aud, "extract_audio", fake_extract)
    monkeypatch.setattr(cli.D, "run_diarization", lambda *_a, **_k: [DiarSegment(start=0, end=1, speaker=0)])
    monkeypatch.setattr(cli, "_run_whisper_streaming", interrupt)
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa", vad=False))

    with pytest.raises(KeyboardInterrupt):
        cli.cmd_transcribe(_transcribe_args(source, outputs="srt", speakers=1))
    assert not (tmp_path / "recording.wav").exists()


def test_diarization_compatible_wav_dry_run_needs_no_ffmpeg(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    _write_pcm_wav(source)
    _prepare_diarization_build(monkeypatch)
    monkeypatch.setattr(cli.aud, "find_ffmpeg", lambda *_a: pytest.fail("compatible WAV needs no ffmpeg"))
    args = _transcribe_args(source, outputs="srt", speakers=1)
    args.dry_run = True

    cmd, _model, wav, *_rest = cli._build_transcribe_args(args, cli.cfg.Config(diarization_backend="sherpa", vad=False))
    assert wav == source
    assert cmd[cmd.index("-f") + 1] == str(source)


def test_invalid_outputs_fail_before_normalizing_mp3(tmp_path, monkeypatch):
    source = tmp_path / "recording.mp3"
    source.write_bytes(b"fake mp3")
    _prepare_diarization_build(monkeypatch)
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa", vad=False))
    monkeypatch.setattr(cli.aud, "extract_audio", lambda *_a, **_k: pytest.fail("ffmpeg should not run"))

    with pytest.raises(SystemExit, match="Unknown output format 'bogus'"):
        cli.cmd_transcribe(_transcribe_args(source, outputs="srt,bogus", speakers=1))
    assert not (tmp_path / "recording.wav").exists()


def test_missing_whisper_cli_fails_before_normalizing_mp3(tmp_path, monkeypatch):
    source = tmp_path / "recording.mp3"
    source.write_bytes(b"fake mp3")
    _prepare_diarization_build(monkeypatch)
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa", vad=False))

    def fake_prepare(src, ffmpeg, *, dry_run=False):
        out = src.with_suffix(".wav")
        _write_pcm_wav(out)
        return out

    def missing_whisper_cli(configured=""):
        raise RuntimeError("whisper-cli not found on PATH")

    monkeypatch.setattr(cli.aud, "prepare_diarization_audio", fake_prepare)
    monkeypatch.setattr(cli, "_find_whisper_cli", missing_whisper_cli)

    with pytest.raises(RuntimeError, match="whisper-cli not found"):
        cli.cmd_transcribe(_transcribe_args(source, outputs="srt", speakers=1))
    assert sorted(p.name for p in tmp_path.glob("*.wav")) == []


@pytest.mark.parametrize("error", [RuntimeError("ffmpeg failed"), KeyboardInterrupt()])
def test_prepare_diarization_audio_removes_partial_output(tmp_path, monkeypatch, error):
    source = tmp_path / "recording.mp3"
    source.write_bytes(b"fake mp3")

    def failing_extract(src, ffmpeg, dest_dir=None, dry_run=False, output=None):
        output.write_bytes(b"partial wav")
        raise error

    monkeypatch.setattr(cli.aud, "extract_audio", failing_extract)

    with pytest.raises(type(error)):
        cli.aud.prepare_diarization_audio(source, "ffmpeg")
    assert not (tmp_path / "recording.wav").exists()
    assert source.exists()


def test_diarization_compatible_wav_stays_direct(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    _write_pcm_wav(source)
    _prepare_diarization_build(monkeypatch)
    monkeypatch.setattr(cli.aud, "extract_audio", lambda *_a, **_k: pytest.fail("compatible WAV must stay direct"))

    cmd, _model, wav, *_rest = cli._build_transcribe_args(
        _transcribe_args(source, outputs="srt", speakers=1), cli.cfg.Config(diarization_backend="sherpa", vad=False),
    )

    assert wav == source
    assert cmd[cmd.index("-f") + 1] == str(source)


def test_diarization_incompatible_wav_normalizes_without_overwriting_source(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    _write_pcm_wav(source, rate=44_100, channels=2, width=2)
    original = source.read_bytes()
    _prepare_diarization_build(monkeypatch)
    captured: list[Path] = []

    def fake_extract(_src, _ffmpeg, dest_dir=None, dry_run=False, output=None):
        assert output is not None
        captured.append(output)
        output.write_bytes(b"normalized wav")
        return output

    monkeypatch.setattr(cli.aud, "extract_audio", fake_extract)
    _cmd, _model, wav, *_rest = cli._build_transcribe_args(
        _transcribe_args(source, outputs="srt", speakers=1), cli.cfg.Config(diarization_backend="sherpa", vad=False),
    )

    assert wav == tmp_path / "recording.diarize.wav"
    assert captured == [wav]
    assert source.read_bytes() == original


def test_incompatible_wav_keeps_compatible_wav_speaker_output_names(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    _write_pcm_wav(source, rate=44_100, channels=2)
    (tmp_path / "recording.wav.json").write_text(_WHISPER_JSON, encoding="utf-8")
    _prepare_diarization_build(monkeypatch)

    def fake_extract(_src, _ffmpeg, dest_dir=None, dry_run=False, output=None):
        assert output is not None
        output.write_bytes(b"normalized wav")
        return output

    monkeypatch.setattr(cli.aud, "extract_audio", fake_extract)
    monkeypatch.setattr(cli.D, "run_diarization", lambda *_a, **_k: [DiarSegment(start=0, end=1, speaker=0)])
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda _cmd: SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa", vad=False))

    assert cli.cmd_transcribe(_transcribe_args(source, outputs="srt", speakers=1)) == 0
    assert (tmp_path / "recording.speakers.srt").exists()
    assert not (tmp_path / "recording.wav.speakers.srt").exists()


def test_diarization_normalization_does_not_clobber_existing_wav_target(tmp_path, monkeypatch):
    source = tmp_path / "recording.mp3"
    source.write_bytes(b"fake mp3")
    existing = tmp_path / "recording.wav"
    existing.write_bytes(b"user-owned wav")
    captured: list[Path] = []

    def fake_extract(_src, _ffmpeg, dest_dir=None, dry_run=False, output=None):
        assert output is not None
        captured.append(output)
        output.write_bytes(b"normalized wav")
        return output

    monkeypatch.setattr(cli.aud, "extract_audio", fake_extract)

    normalized = cli.aud.prepare_diarization_audio(source, "ffmpeg")

    assert normalized == tmp_path / "recording.diarize.wav"
    assert captured == [normalized]
    assert existing.read_bytes() == b"user-owned wav"


def test_video_diarization_extraction_path_is_unchanged(tmp_path, monkeypatch):
    source = tmp_path / "recording.mov"
    source.write_bytes(b"fake video")
    _prepare_diarization_build(monkeypatch)
    extracted: list[Path] = []

    def fake_extract(src, ffmpeg, dest_dir=None, dry_run=False, output=None):
        assert output is None
        extracted.append(src)
        return src.with_suffix(".wav")

    monkeypatch.setattr(cli.aud, "extract_audio", fake_extract)
    _cmd, _model, wav, in_path, *_rest = cli._build_transcribe_args(
        _transcribe_args(source, outputs="srt", speakers=1), cli.cfg.Config(diarization_backend="sherpa", vad=False),
    )

    assert in_path == source
    assert wav == source.with_suffix(".wav")
    assert extracted == [source]


def test_find_whisper_json_dotted_output_stem(tmp_path):
    """-o /x/out.v2 -> the JSON is out.v2.json, not out.json (with_suffix
    would eat the dotted stem and read a stale transcript from an old run)."""
    of_base = tmp_path / "out.v2"
    wav = tmp_path / "in.wav"
    wanted = tmp_path / "out.v2.json"
    wanted.write_text("{}", encoding="utf-8")
    # A stale out.json (from a run of the old, buggy naming) must never win
    # over the run's real out.v2.json.
    (tmp_path / "out.json").write_text("{}", encoding="utf-8")
    found = cli._find_whisper_json(of_base, wav, of_passed=True)
    assert found == wanted


def test_find_whisper_json_of_passed_stale_only_never_wins(tmp_path):
    """with -of out.v2, whisper-cli writes out.v2.json and nothing else —
    when only the stale out.json exists, it must NOT be ingested: the
    caller warns on the missing out.v2.json instead of merging old data."""
    of_base = tmp_path / "out.v2"
    wav = tmp_path / "in.wav"
    (tmp_path / "out.json").write_text("{}", encoding="utf-8")  # stale
    found = cli._find_whisper_json(of_base, wav, of_passed=True)
    assert found == tmp_path / "out.v2.json"  # reported missing, not stale


# ---------- merge fallback ----------


def _merge_args(file, outputs="html", speakers=1, speakers_names=None, no_speakers=False):
    return SimpleNamespace(
        file=str(file), json="", outputs=outputs, speakers=speakers,
        diarization_backend="sherpa",
        no_speakers=no_speakers, no_auto_diarization_setup=False,
        cluster_threshold=None, name_speakers=False,
        no_name_speakers=True, speakers_names=speakers_names, screenshots=False,
        no_screenshots=False, screenshot_width=None, no_voice_profiles=True,
    )


def _raise_sherpa_missing(wav, config, num_speakers=0, threshold=0.9, **_kwargs):
    # M2 (wave-1 audit): unavailability arrives as the TYPED exception now;
    # the "sherpa_onnx" token stays in the message for the SystemExit
    # match assertions below.
    raise cli.D.DiarizationUnavailable("The 'sherpa_onnx' package is required for diarization")


def _stub_setup_unavailable(monkeypatch):
    """Simulate the proactive setup having run and failed (pip offline,
    model download error, ...) — the safety-net case every degraded path
    below still guards. Without this stub cmd_merge would attempt a REAL
    `pip install sherpa-onnx`, since merge now calls the setup before
    diarizing."""
    monkeypatch.setattr(
        cli, "_ensure_diarization_ready",
        lambda config, dry_run=False, setup_allowed=True: False,
    )
    monkeypatch.setattr(cli.aud, "find_ffmpeg", lambda configured="": "ffmpeg")
    monkeypatch.setattr(
        cli.aud, "prepare_diarization_audio",
        lambda src, ffmpeg, dry_run=False: src.with_suffix(".wav"),
    )


def _stub_setup_ready(monkeypatch):
    """Simulate a successful one-time setup (package + models ready)."""
    monkeypatch.setattr(
        cli, "_ensure_diarization_ready",
        lambda config, dry_run=False, setup_allowed=True: True,
    )
    monkeypatch.setattr(cli.aud, "find_ffmpeg", lambda configured="": "ffmpeg")
    monkeypatch.setattr(
        cli.aud, "prepare_diarization_audio",
        lambda src, ffmpeg, dry_run=False: src.with_suffix(".wav"),
    )


def test_merge_html_fallback_when_diarization_unavailable_returns_error(tmp_path, monkeypatch, capsys):
    """wiz merge --speakers --outputs html with sherpa-onnx missing still
    writes a generic-label HTML transcript, then exits 1 because the
    explicit speaker request was not fulfilled."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 1
    html_path = tmp_path / "meeting.m4a.speakers.html"
    assert html_path.exists(), "HTML transcript was silently skipped"
    content = html_path.read_text(encoding="utf-8")
    assert "hello world" in content
    assert ">Speaker<" in content
    assert 'class="note"' in content  # degraded page self-identifies
    # Labeled SRT is not faked (this previously asserted meeting.speakers.srt
    # — a file this code path never writes, so it guarded nothing).
    assert not (tmp_path / "meeting.m4a.speakers.srt").exists()
    # Audio fallback also writes a generic-label .speakers.txt so
    # `wiz analyze` finds a transcript.
    txt = (tmp_path / "meeting.m4a.speakers.txt").read_text(encoding="utf-8")
    assert "Speaker (00:00:00):" in txt
    assert "diarization unavailable" in capsys.readouterr().err


def test_merge_still_raises_when_nothing_else_requested(tmp_path, monkeypatch):
    """Without --outputs html (or screenshots) there is nothing to fall back
    to: an explicit --speakers merge against missing sherpa-onnx stays loud."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    with pytest.raises(SystemExit, match="sherpa_onnx"):
        cli.cmd_merge(_merge_args(audio, outputs="", speakers=1))


# ---------- command-level success paths and new fallback behaviors ----------


def test_transcribe_stage_timing_excludes_naming_wait_but_total_includes_analysis_and_cleanup(
    tmp_path, monkeypatch,
):
    """The command-level timer excludes human naming time from output writing,
    while its wall total includes chained analysis and cleanup."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa", save_voice_profiles=False))
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path / "profiles")
    now = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: now[0])

    def advance(seconds):
        now[0] += seconds

    monkeypatch.setattr(
        cli, "_run_diarize_or_fallback",
        lambda *_args: advance(2.0) or [DiarSegment(start=0.0, end=4.0, speaker=0)],
    )
    monkeypatch.setattr(
        cli, "_run_whisper_streaming",
        lambda _cmd: advance(3.0) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        cli.P, "compute_speaker_embeddings",
        lambda *_args: advance(4.0) or {0: [1.0, 0.0]},
    )
    monkeypatch.setattr(cli.P, "auto_assign_names", lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(
        cli, "_prompt_speaker_names",
        lambda *_args, **_kwargs: advance(50.0) or {"Speaker A": "Alice"},
    )
    original_srt = cli.MR.format_labeled_srt
    original_txt = cli.MR.format_dialogue_txt
    monkeypatch.setattr(
        cli.MR, "format_labeled_srt", lambda merged: advance(0.5) or original_srt(merged),
    )
    monkeypatch.setattr(
        cli.MR, "format_dialogue_txt", lambda merged: advance(0.5) or original_txt(merged),
    )
    monkeypatch.setattr(cli, "cmd_analyze", lambda _args: advance(7.0) or 0)
    original_cleanup = cli._remove_intermediate_audio
    monkeypatch.setattr(
        cli, "_remove_intermediate_audio",
        lambda wav, source: advance(6.0) or original_cleanup(wav, source),
    )
    tables: list[tuple[str | None, list[list[str]]]] = []
    monkeypatch.setattr(cli.ui, "table", lambda title, _columns, rows: tables.append((title, rows)))
    args = _transcribe_args(audio, outputs="srt", speakers=1)
    args.no_voice_profiles = False
    args.name_speakers = True
    args.no_name_speakers = False
    args.analyze = True

    assert cli.cmd_transcribe(args) == 0

    timing = dict(next(rows for title, rows in tables if title == "Stage timing"))
    assert timing["Diarization"] == "2.0s"
    assert timing["Transcription"] == "3.0s"
    assert timing["Speaker profiles"] == "4.0s"
    assert timing["Writing outputs / frames"] == "1.0s"
    assert timing["Analysis"] == "7.0s"
    assert timing["Total (wall)"] == "1:13.0"


def test_transcribe_diarized_success_writes_labeled_outputs(tmp_path, monkeypatch):
    """Happy path: diarization succeeds -> labeled .speakers.srt/.txt/.html
    all written with letterized labels, and no degraded-run note."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    diar = [
        DiarSegment(start=0.0, end=3.0, speaker=0),   # Speaker A
        DiarSegment(start=3.0, end=5.0, speaker=1),   # Speaker B
    ]
    monkeypatch.setattr(cli, "_run_diarize_or_fallback", lambda wav, config, args: diar)
    tables: list[tuple[str | None, list[list[object]]]] = []
    monkeypatch.setattr(
        cli.ui, "table", lambda title, _columns, rows: tables.append((title, rows)),
    )

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=2))

    assert rc == 0
    srt = (tmp_path / "meeting.speakers.srt").read_text(encoding="utf-8")
    assert "Speaker A:" in srt and "Speaker B:" in srt
    txt = (tmp_path / "meeting.speakers.txt").read_text(encoding="utf-8")
    assert "Speaker A (00:00:00):" in txt
    html = (tmp_path / "meeting.speakers.html").read_text(encoding="utf-8")
    assert "Speaker A" in html
    assert 'class="note"' not in html  # not a degraded run
    timing_rows = next(rows for title, rows in tables if title == "Stage timing")
    assert [row[0] for row in timing_rows] == [
        "Audio preparation", "Transcription", "Diarization", "Speaker profiles",
        "Writing outputs / frames", "Total (wall)",
    ]


def test_merge_diarized_success_writes_labeled_outputs(tmp_path, monkeypatch):
    """wiz merge happy path: diarization succeeds -> labeled srt/txt/html
    written under the JSON stem with letterized labels, no note."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9, **_kwargs: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
            DiarSegment(start=3.0, end=5.0, speaker=1),
        ],
    )
    tables: list[tuple[str | None, list[list[object]]]] = []
    monkeypatch.setattr(
        cli.ui, "table", lambda title, _columns, rows: tables.append((title, rows)),
    )

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=2))

    assert rc == 0
    srt = (tmp_path / "meeting.m4a.speakers.srt").read_text(encoding="utf-8")
    assert "Speaker A:" in srt and "Speaker B:" in srt
    txt = (tmp_path / "meeting.m4a.speakers.txt").read_text(encoding="utf-8")
    assert "Speaker A (00:00:00):" in txt
    html = (tmp_path / "meeting.m4a.speakers.html").read_text(encoding="utf-8")
    assert "Speaker A" in html
    assert 'class="note"' not in html
    timing_rows = next(rows for title, rows in tables if title == "Stage timing")
    assert [row[0] for row in timing_rows] == [
        "Audio preparation", "Transcription", "Diarization", "Speaker profiles",
        "Writing outputs / frames", "Total (wall)",
    ]


def test_merge_timing_labels_zero_diarization_outputs_not_written(tmp_path, monkeypatch):
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", lambda *_args, **_kwargs: [])
    tables: list[tuple[str | None, list[list[str]]]] = []
    monkeypatch.setattr(cli.ui, "table", lambda title, _columns, rows: tables.append((title, rows)))

    assert cli.cmd_merge(_merge_args(audio, outputs="", speakers=1)) == 1

    timing = dict(next(rows for title, rows in tables if title == "Stage timing"))
    assert timing["Writing outputs / frames"] == "skipped (no speaker labels)"


@pytest.mark.parametrize("command", ["transcribe", "merge"])
def test_profile_provider_failure_keeps_labeled_outputs(tmp_path, monkeypatch, capsys, command):
    diar = [DiarSegment(start=0.0, end=3.0, speaker=0)]
    def fail_profiles(*_args, **_kwargs):
        raise cli.D.DiarizationProviderError("CoreML embedding session unavailable")
    monkeypatch.setattr(cli.P, "compute_speaker_embeddings", fail_profiles)

    if command == "transcribe":
        audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
        monkeypatch.setattr(cli, "_run_diarize_or_fallback", lambda *_a: diar)
        args = _transcribe_args(audio, outputs="srt,html", speakers=1)
        args.no_voice_profiles = False
        assert cli.cmd_transcribe(args) == 0
        base = tmp_path / "meeting.speakers"
    else:
        audio = tmp_path / "meeting.m4a"
        audio.write_bytes(b"fake audio")
        (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
        monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
        _stub_setup_ready(monkeypatch)
        monkeypatch.setattr(cli.D, "run_diarization", lambda *_a, **_k: diar)
        args = _merge_args(audio, outputs="html", speakers=1)
        args.no_voice_profiles = False
        assert cli.cmd_merge(args) == 0
        base = tmp_path / "meeting.m4a.speakers"

    assert Path(str(base) + ".srt").exists()
    assert Path(str(base) + ".txt").exists()
    assert Path(str(base) + ".html").exists()
    assert "voice-profile matching skipped" in capsys.readouterr().err


def test_invalid_diarization_config_does_not_block_plain_transcription(tmp_path, monkeypatch):
    source = tmp_path / "recording.mp3"
    source.write_bytes(b"source")
    _prepare_diarization_build(monkeypatch)
    monkeypatch.setattr(
        cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa", diarization_provider="nonsense", vad=False),
    )
    monkeypatch.setattr(cli.aud, "prepare_diarization_audio", lambda *_a, **_k: pytest.fail("plain transcription must stay direct"))
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda _cmd: SimpleNamespace(returncode=0))
    assert cli.cmd_transcribe(_transcribe_args(source, outputs="srt", speakers=None)) == 0


def test_merge_mp3_finds_normalized_transcribe_json_and_cleans_audio(tmp_path, monkeypatch):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake audio")
    # The source-based output name is the same as direct whisper on MP3.
    (tmp_path / "meeting.mp3.json").write_text(_WHISPER_JSON, encoding="utf-8")
    (tmp_path / "meeting.json").write_text("stale transcript", encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    monkeypatch.setattr(cli, "_ensure_diarization_ready", lambda *a, **k: True)
    monkeypatch.setattr(cli.aud, "find_ffmpeg", lambda configured="": "ffmpeg")
    diarized: list[tuple[Path, Path | None]] = []

    def fake_extract(src, ffmpeg, dest_dir=None, dry_run=False, output=None):
        assert src == audio
        assert output is not None
        output.write_bytes(b"normalized wav")
        return output

    monkeypatch.setattr(cli.aud, "extract_audio", fake_extract)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, *_a, **_k: diarized.append((wav, _k.get("cache_source"))) or [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )

    assert cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1)) == 0
    normalized = tmp_path / "meeting.wav"
    assert diarized == [(normalized, audio)]
    assert not normalized.exists()


def test_merge_mp3_interrupt_removes_normalized_audio(tmp_path, monkeypatch):
    source = tmp_path / "meeting.mp3"
    source.write_bytes(b"fake audio")
    (tmp_path / "meeting.mp3.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    monkeypatch.setattr(cli, "_ensure_diarization_ready", lambda *a, **k: True)
    monkeypatch.setattr(cli.aud, "find_ffmpeg", lambda configured="": "ffmpeg")

    def fake_extract(_src, _ffmpeg, dest_dir=None, dry_run=False, output=None):
        assert output is not None
        output.write_bytes(b"normalized wav")
        return output

    def interrupt(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.aud, "extract_audio", fake_extract)
    monkeypatch.setattr(cli.D, "run_diarization", interrupt)

    with pytest.raises(KeyboardInterrupt):
        cli.cmd_merge(_merge_args(source, outputs="html", speakers=1))
    assert not (tmp_path / "meeting.wav").exists()


def test_transcribe_fallback_warns_discarded_speakers_names(tmp_path, monkeypatch, capsys):
    """The fallback must say --speakers-names was discarded, not let the
    user believe the names were applied."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    args = _transcribe_args(audio, outputs="html", speakers=1)
    args.speakers_names = ["Alice,Bob"]
    rc = cli.cmd_transcribe(args)

    assert rc == 1
    # Whitespace-normalized: rich wraps long status lines at the console
    # width, which may split the phrase across lines (assert on content,
    # not on wrap luck).
    assert "--speakers-names had no effect" in " ".join(capsys.readouterr().err.split())


def test_merge_fallback_warns_discarded_speakers_names(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1,
                                   speakers_names=["Alice,Bob"]))

    assert rc == 1
    # Wrap-insensitive (see the transcribe twin above).
    assert "--speakers-names had no effect" in " ".join(capsys.readouterr().err.split())


def test_merge_zero_segments_falls_back_to_unlabeled_html_returns_error(tmp_path, monkeypatch, capsys):
    """Diarization runs but finds no speech: warn + generic-label HTML
    (and .speakers.txt on audio runs) instead of crashing or skipping."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9, **_kwargs: [],
    )

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 1
    html = (tmp_path / "meeting.m4a.speakers.html").read_text(encoding="utf-8")
    assert ">Speaker<" in html
    assert 'class="note"' in html
    assert (tmp_path / "meeting.m4a.speakers.txt").exists()
    assert "Diarization produced no segments; writing unlabeled output" in capsys.readouterr().err


def test_merge_returns_1_when_nothing_written(tmp_path, monkeypatch, capsys):
    """No html/screenshots requested + no segments -> nothing written, AND
    nothing was kept either -> rc=1: a silent rc=0 would read as success.
    (The kept-only case is now rc=0 — see the no-clobber test above.)"""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9, **_kwargs: [],
    )

    rc = cli.cmd_merge(_merge_args(audio, outputs="", speakers=1))

    assert rc == 1
    assert "Diarization produced no segments; nothing to merge." in capsys.readouterr().err


def _capture_status(monkeypatch):
    calls: list[tuple[str, str, str | None]] = []

    def fake(msg, kind="info", detail=None):
        calls.append((msg, kind, detail))

    monkeypatch.setattr(cli.ui, "status", fake)
    return calls


def test_diarize_fallback_warns_for_explicit_speakers(tmp_path, monkeypatch):
    """Explicit --speakers degrades loudly (warn), not with the quiet hint
    used for merely auto-enabled diarization."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    calls = _capture_status(monkeypatch)

    cli._run_diarize_or_fallback(audio, cli.cfg.Config(diarization_backend="sherpa"), _transcribe_args(audio, speakers=1))

    kinds = [k for _m, k, _d in calls]
    assert "warn" in kinds


def test_diarize_fallback_stays_hint_when_auto_enabled(tmp_path, monkeypatch):
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    calls = _capture_status(monkeypatch)

    cli._run_diarize_or_fallback(audio, cli.cfg.Config(diarization_backend="sherpa"), _transcribe_args(audio, speakers=None))

    kinds = [k for _m, k, _d in calls]
    assert "hint" in kinds
    assert "warn" not in kinds


# ---------- blocker fixes: no-clobber + explicit-html gating ----------


def test_transcribe_fallback_never_clobbers_existing_named_outputs_returns_error(tmp_path, monkeypatch, capsys):
    """Blocker 2 regression: a diarized run (with real names) left a named
    .speakers.txt/.html; a later degraded run must keep them, not collapse
    them to a one-line generic 'Speaker' wall."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    # Earlier diarized run's outputs, with real speaker names.
    named_txt = tmp_path / "meeting.speakers.txt"
    named_html = tmp_path / "meeting.speakers.html"
    named_srt = tmp_path / "meeting.speakers.srt"
    named_txt.write_text("Vadim (00:00:00): real named content\n", encoding="utf-8")
    named_html.write_text("<html>named run</html>", encoding="utf-8")
    named_srt.write_text("1\n00:00:00,000 --> ...\nVadim: real named content\n", encoding="utf-8")

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=1))

    assert rc == 1
    assert named_txt.read_text(encoding="utf-8") == "Vadim (00:00:00): real named content\n"
    assert named_html.read_text(encoding="utf-8") == "<html>named run</html>"
    assert "real named content" in named_srt.read_text(encoding="utf-8")
    err = capsys.readouterr().err
    assert "kept" in err and "meeting.speakers.txt" in err
    assert "kept" in err and "meeting.speakers.html" in err


def test_merge_fallback_never_clobbers_existing_named_outputs_returns_error(tmp_path, monkeypatch, capsys):
    """The merge fallback also preserves named artifacts, while returning
    an error because explicit speaker labels were not produced."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    named_txt = tmp_path / "meeting.m4a.speakers.txt"
    named_html = tmp_path / "meeting.m4a.speakers.html"
    named_txt.write_text("Vadim (00:00:00): real named content\n", encoding="utf-8")
    named_html.write_text("\u003chtml\u003enamed run\u003c/html\u003e", encoding="utf-8")

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 1
    assert named_txt.read_text(encoding="utf-8") == "Vadim (00:00:00): real named content\n"
    assert named_html.read_text(encoding="utf-8") == "\u003chtml\u003enamed run\u003c/html\u003e"
    err = capsys.readouterr().err
    assert "kept" in err and "meeting.m4a.speakers.txt" in err
    assert "kept" in err and "meeting.m4a.speakers.html" in err
    assert "no speaker labels were produced" in " ".join(err.split())


def _merge_auto_diarized(monkeypatch, tmp_path):
    """An audio merge that behaves like video auto-enabled diarization
    (speakers_auto=True, no --speakers) with sherpa-onnx missing."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    monkeypatch.setattr(cli, "_video_auto_flags", lambda args, in_path: (False, True))
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    return audio


def test_merge_auto_diarization_degrade_stays_success(tmp_path, monkeypatch, capsys):
    """Only an EXPLICIT --speakers fails when no labels are produced; an
    auto-enabled diarization that degrades to generic labels exits 0."""
    audio = _merge_auto_diarized(monkeypatch, tmp_path)

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=None))

    assert rc == 0
    assert ">Speaker<" in (tmp_path / "meeting.m4a.speakers.html").read_text(encoding="utf-8")
    assert "no speaker labels were produced" not in " ".join(capsys.readouterr().err.split())


def test_merge_auto_diarization_kept_only_stays_success(tmp_path, monkeypatch):
    """Auto-enabled diarization that only keeps named outputs from an
    earlier diarized run is a no-op success, not a failure."""
    audio = _merge_auto_diarized(monkeypatch, tmp_path)
    named_txt = tmp_path / "meeting.m4a.speakers.txt"
    named_html = tmp_path / "meeting.m4a.speakers.html"
    named_txt.write_text("Vadim (00:00:00): real named content\n", encoding="utf-8")
    named_html.write_text("<html>named run</html>", encoding="utf-8")

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=None))

    assert rc == 0
    assert named_html.read_text(encoding="utf-8") == "<html>named run</html>"


def test_transcribe_missing_whisper_json_with_explicit_speakers_returns_error(tmp_path, monkeypatch, capsys):
    """Diarization segments alone are not speaker labels: when the whisper
    JSON is missing, an explicit --speakers run wrote nothing labeled and
    must exit nonzero, with the reason printed last."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    (tmp_path / "meeting.m4a.json").unlink()
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9, **_kwargs: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt", speakers=1))

    assert rc == 1
    assert not (tmp_path / "meeting.speakers.srt").exists()
    assert "no speaker labels were produced" in " ".join(capsys.readouterr().err.split())


def test_transcribe_fallback_rerun_overwrites_existing_degraded_outputs(tmp_path, monkeypatch, capsys):
    """Review follow-up (idempotence): the second identical degraded run
    must NOT say "kept" for its own degraded output — that file has no
    speaker names to destroy, and keeping it would leave a stale transcript
    forever whenever --model/--language/the audio change. The degraded
    file is cheaply detectable (provenance note in the HTML, all-generic
    label lines in the txt), so the run rewrites both files but exits nonzero."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    # Run 1: writes the degraded outputs.
    rc1 = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=1))
    assert rc1 == 1
    degraded_html = tmp_path / "meeting.speakers.html"
    degraded_txt = tmp_path / "meeting.speakers.txt"
    assert degraded_html.exists() and degraded_txt.exists()
    assert cli._GENERIC_LABEL_NOTE in degraded_html.read_text(encoding="utf-8")

    # Run 2: identical command. A naive .exists() guard keeps the stale
    # degraded files; the fix rewrites them.
    rc2 = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=1))

    assert rc2 == 1
    html2 = degraded_html.read_text(encoding="utf-8")
    txt2 = degraded_txt.read_text(encoding="utf-8")
    assert "hello world" in html2  # fresh content, not the stale run-1 file
    assert cli._GENERIC_LABEL_NOTE in html2
    assert "Speaker (00:00:00):" in txt2
    err = capsys.readouterr().err
    assert "overwriting" in err  # says what it did to the degraded files
    assert "kept" not in err      # ...and does NOT claim to keep them


def test_merge_fallback_rerun_overwrites_existing_degraded_outputs(tmp_path, monkeypatch, capsys):
    """Merge half of the idempotence fix: the second identical degraded
    merge rewrites its own degraded outputs (with an "overwriting" note)
    while returning an error for the unfulfilled request."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    rc1 = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))
    assert rc1 == 1
    degraded_html = tmp_path / "meeting.m4a.speakers.html"
    degraded_txt = tmp_path / "meeting.m4a.speakers.txt"
    assert degraded_html.exists() and degraded_txt.exists()

    rc2 = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc2 == 1
    assert "hello world" in degraded_html.read_text(encoding="utf-8")
    assert "Speaker (00:00:00):" in degraded_txt.read_text(encoding="utf-8")
    err = capsys.readouterr().err
    assert "overwriting" in err
    assert "kept" not in err


def test_transcribe_fallback_mixed_named_html_kept_degraded_txt_overwritten(tmp_path, monkeypatch, capsys):
    """Per-file decision (review follow-up): an earlier diarized run left a
    NAMED html but a degraded txt on disk; the fallback must keep the
    named file and still refresh the degraded one — the guard protects
    speaker names, not wiz's own fallback output."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    named_html = tmp_path / "meeting.speakers.html"
    degraded_txt = tmp_path / "meeting.speakers.txt"
    named_html.write_text("<html>named run</html>", encoding="utf-8")
    degraded_txt.write_text("Speaker (00:00:00): stale degraded content\n", encoding="utf-8")

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt,html", speakers=1))

    assert rc == 1
    assert named_html.read_text(encoding="utf-8") == "<html>named run</html>"  # kept
    assert "stale degraded content" not in degraded_txt.read_text(encoding="utf-8")  # refreshed
    assert "Speaker (00:00:00):" in degraded_txt.read_text(encoding="utf-8")
    err = capsys.readouterr().err
    assert "kept" in err and "meeting.speakers.html" in err
    assert "overwriting" in err


def test_transcribe_config_html_is_not_degraded(tmp_path, monkeypatch, capsys):
    """Blocker 3: html in config.outputs alone must NOT trigger the
    degraded fallback on a failed-diarization run (a typed --outputs html
    is a promise; a config default describes the success path)."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    def fake_load():
        config = cli.cfg.Config(diarization_backend="sherpa")
        config.outputs = ["srt", "html"]
        return config

    monkeypatch.setattr(cli.cfg, "load", fake_load)

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="", speakers=1))

    # L (wave-1 audit): rc symmetry with merge — the explicit --speakers
    # degraded and nothing speaker-related was written, so the run exits
    # nonzero instead of reporting success.
    assert rc == 1
    assert not (tmp_path / "meeting.speakers.html").exists()
    assert not (tmp_path / "meeting.speakers.txt").exists()
    # The diarization-unavailable status still fires (it is honest about the
    # run) — it just does not promise degraded artifacts it will not write.
    err = capsys.readouterr().err
    assert "diarization unavailable" in err
    assert "Falling back to generic" not in err


def test_merge_config_html_is_not_degraded(tmp_path, monkeypatch, capsys):
    """Blocker 3 on the merge path: config-only html + explicit --speakers
    with sherpa missing stays loud (SystemExit), exactly like master."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")

    def fake_load():
        config = cli.cfg.Config(diarization_backend="sherpa")
        config.outputs = ["srt", "html"]
        return config

    monkeypatch.setattr(cli.cfg, "load", fake_load)
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    with pytest.raises(SystemExit, match="sherpa_onnx"):
        cli.cmd_merge(_merge_args(audio, outputs="", speakers=1))


def test_transcribe_fallback_honest_skip_when_nothing_to_write(tmp_path, monkeypatch, capsys):
    """Message fidelity: audio run, NO explicit html (and no video frames):
    diarization fails → the message must say "skipping", not promise a
    generic-label fallback that will not happen."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    rc = cli.cmd_transcribe(_transcribe_args(audio, outputs="srt", speakers=1))

    # L (wave-1 audit): rc symmetry with merge — nothing speaker-related was
    # written for the explicit --speakers, so nonzero. (A degraded run that
    # DOES write generic-label outputs still exits 0 — see the html fallback
    # tests above.)
    assert rc == 1
    err = capsys.readouterr().err
    assert "Skipping speaker labels" in err
    assert "Falling back to generic" not in err
    assert not (tmp_path / "meeting.speakers.html").exists()


def test_merge_sherpa_missing_does_not_double_warn(tmp_path, monkeypatch, capsys):
    """Review fix: the except branch already said 'falling back to generic
    labels'; the follow-up 'produced no segments' block must not repeat it.
    It stays for the genuinely-new case (ran and returned nothing) — covered
    by test_merge_zero_segments_falls_back_to_unlabeled_html."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 1
    err = capsys.readouterr().err
    assert err.count("diarization unavailable") == 1
    assert err.count("Diarization produced no segments") == 0


# ---------- proactive diarization auto-setup (user decision, 2026-09-05) ----------
#
# Policy: when diarization is about to run — auto-enabled for video or
# explicitly requested — wiz performs the one-time setup itself (pip
# install sherpa-onnx + ~90 MB model download) instead of degrading; the
# fallbacks above remain only as the safety net for a failed setup or
# --no-auto-diarization-setup. These tests run the REAL setup helpers by
# stubbing their inputs (availability probe, pip, model download) so no
# test ever touches the network or mutates the venv.


def test_no_auto_diarization_setup_flags_registered():
    """--no-auto-diarization-setup exists on transcribe, merge AND speakers
    match (review round 3: cmd_speakers_match read the flag via getattr but
    the subparser never registered it, so the default always won and setup
    was unconditionally allowed on a command whose help calls itself a dry
    run). Defaults to False everywhere: setup-on-first-use is on by default."""
    parser = cli.build_parser()
    args = parser.parse_args(["transcribe", "x.wav", "--no-auto-diarization-setup"])
    assert args.no_auto_diarization_setup is True
    args = parser.parse_args(["transcribe", "x.wav"])
    assert args.no_auto_diarization_setup is False
    args = parser.parse_args(["merge", "x.wav", "--no-auto-diarization-setup"])
    assert args.no_auto_diarization_setup is True
    args = parser.parse_args(["merge", "x.wav"])
    assert args.no_auto_diarization_setup is False
    args = parser.parse_args(["speakers", "match", "x.wav", "--no-auto-diarization-setup"])
    assert args.no_auto_diarization_setup is True
    args = parser.parse_args(["speakers", "match", "x.wav"])
    assert args.no_auto_diarization_setup is False


def test_ensure_diarization_ready_short_circuits_when_available(monkeypatch):
    """Already-available diarization returns True with no install, no
    download, and no status output — every later run stays quiet."""
    calls = _capture_status(monkeypatch)

    def _boom(*_a, **_k):
        raise AssertionError("nothing may run when diarization is ready")

    monkeypatch.setattr(cli, "_diarization_available", lambda config: True)
    monkeypatch.setattr(cli, "_install_sherpa_onnx", _boom)
    monkeypatch.setattr(cli.D, "download_diarization_models", _boom)

    assert cli._ensure_diarization_ready(cli.cfg.Config(diarization_backend="sherpa")) is True
    assert calls == []


def test_ensure_diarization_ready_happy_path_installs_then_downloads(monkeypatch, capsys):
    """Fresh machine: install the package FIRST, then download the models —
    models without the package that runs them would leave half a setup —
    then confirm readiness with a final availability re-check."""
    events: list[str] = []
    # Not ready at entry; ready once the models are on disk — the re-check
    # then sees the freshly installed package + downloaded models.
    monkeypatch.setattr(cli, "_diarization_available", lambda config: "download" in events)
    # A None entry makes `import sherpa_onnx` raise ImportError, forcing the
    # install branch deterministically regardless of the host venv.
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    monkeypatch.setattr(cli, "_install_sherpa_onnx", lambda: events.append("install") or True)
    monkeypatch.setattr(cli.D, "download_diarization_models", lambda: events.append("download"))
    monkeypatch.setattr(cli.D, "find_segmentation_model", lambda config: None)
    monkeypatch.setattr(cli.D, "find_embedding_model", lambda config: None)
    # Consent auto-allows on non-tty stdin; pin it so `pytest -s` (a real
    # terminal stdin) can't turn this wiring test into a live y/N prompt.
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))

    assert cli._ensure_diarization_ready(cli.cfg.Config(diarization_backend="sherpa")) is True
    assert events == ["install", "download"]
    assert "Diarization models downloaded" in capsys.readouterr().err


def test_ensure_diarization_ready_opt_out_skips_setup(monkeypatch, capsys):
    """--no-auto-diarization-setup (setup_allowed=False): report the honest
    False but install and download NOTHING — silent degradation is the
    user's explicit choice."""
    monkeypatch.setattr(cli.D, "find_segmentation_model", lambda config: None)
    monkeypatch.setattr(cli.D, "find_embedding_model", lambda config: None)

    def _boom(*_a, **_k):
        raise AssertionError("opted-out setup must not install or download")

    monkeypatch.setattr(cli, "_install_sherpa_onnx", _boom)
    monkeypatch.setattr(cli.D, "download_diarization_models", _boom)
    # --no-auto-diarization-setup short-circuits BEFORE any prompt (consent
    # ordering): an opted-out run must never sit at a y/N question.
    monkeypatch.setattr(builtins, "input", _boom_input)

    assert cli._ensure_diarization_ready(cli.cfg.Config(diarization_backend="sherpa"), setup_allowed=False) is False
    assert capsys.readouterr().err == ""


def test_ensure_diarization_ready_dry_run_never_sets_up(monkeypatch, capsys):
    """dry_run announces the setup it WOULD perform and runs none of it."""
    monkeypatch.setattr(cli.D, "find_segmentation_model", lambda config: None)
    monkeypatch.setattr(cli.D, "find_embedding_model", lambda config: None)

    def _boom(*_a, **_k):
        raise AssertionError("dry-run must not install or download")

    monkeypatch.setattr(cli, "_install_sherpa_onnx", _boom)
    monkeypatch.setattr(cli.D, "download_diarization_models", _boom)
    # dry_run reports and returns BEFORE consent too: a dry-run must never
    # sit at a prompt (or pip-install) anything.
    monkeypatch.setattr(builtins, "input", _boom_input)

    assert cli._ensure_diarization_ready(cli.cfg.Config(diarization_backend="sherpa"), dry_run=True) is False
    err = capsys.readouterr().err
    assert "DRY-RUN" in err
    assert "~90 MB" in err


def test_install_sherpa_onnx_targets_running_venv_and_reports_progress(monkeypatch, capsys):
    """The installer runs pip via sys.executable (installs into the RUNNING
    venv — dev uv venv, pipx venv, anything; a bare `pipx` binary would
    miss dev venvs), announces itself with the --no-auto-diarization-setup
    opt-out, and verifies the fresh wheel is importable."""
    seen: dict[str, list[str]] = {}

    def fake_run(cmd, check=False):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())

    assert cli._install_sherpa_onnx() is True
    # The spec the diarize extra declares, not a bare package name (review
    # round 3): a bare `pip install sherpa-onnx` could land an older version
    # than the documented manual path (`cli.D.DIARIZE_INJECT`).
    assert seen["cmd"] == [sys.executable, "-m", "pip", "install", cli.D.DIARIZE_REQUIREMENT]
    assert ">=" in cli.D.DIARIZE_REQUIREMENT
    err = capsys.readouterr().err
    assert "installing the diarize extra" in err    # status line up front
    assert "--no-auto-diarization-setup" in err      # the opt-out is surfaced
    assert "sherpa-onnx installed" in err


def test_ensure_diarization_ready_install_failure_returns_false(monkeypatch, capsys):
    """A failed pip install (offline, disk full, ...) returns False so the
    caller stays on its degraded path — with the manual remediation hint.
    No model download may follow the miss."""
    monkeypatch.setattr(cli, "_diarization_available", lambda config: False)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)  # force the install branch
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, check=False: SimpleNamespace(returncode=1))

    def _boom(*_a, **_k):
        raise AssertionError("models must not download when the install failed")

    monkeypatch.setattr(cli.D, "download_diarization_models", _boom)
    # Pin non-tty stdin: consent must auto-allow (plain pytest already is
    # non-tty; `-s` on a terminal would otherwise hit the live prompt).
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))

    assert cli._ensure_diarization_ready(cli.cfg.Config(diarization_backend="sherpa")) is False
    err = capsys.readouterr().err
    assert "pip install sherpa-onnx failed" in err
    assert cli.D.DIARIZE_INJECT in err


def _fresh_machine_stubs(monkeypatch, events):
    """Shared wiring-test setup: stub the setup's inputs so the REAL
    _ensure_diarization_ready runs inside the real command paths, with
    ``events`` recording the (stubbed) install + download steps."""
    monkeypatch.setattr(cli, "_diarization_available", lambda config: "download" in events)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    monkeypatch.setattr(cli, "_install_sherpa_onnx", lambda: events.append("install") or True)
    monkeypatch.setattr(cli.D, "download_diarization_models", lambda: events.append("download"))
    monkeypatch.setattr(cli.D, "find_segmentation_model", lambda config: None)
    monkeypatch.setattr(cli.D, "find_embedding_model", lambda config: None)
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    monkeypatch.setattr(cli.M, "pick_best", lambda config: Path("/models/turbo.bin"))
    monkeypatch.setattr(cli, "_find_whisper_cli", lambda configured="": "whisper-cli")
    monkeypatch.setattr(cli.aud, "find_ffmpeg", lambda configured="": "ffmpeg")
    monkeypatch.setattr(
        cli.aud, "extract_audio",
        lambda src, ffmpeg, dest_dir=None, dry_run=False, output=None: output or src.with_suffix(".wav"),
    )
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda cmd: SimpleNamespace(returncode=0))
    # The real _ensure_diarization_ready now consults _auto_setup_consent,
    # which auto-allows on non-tty stdin. Pin it so the wiring tests behave
    # identically under `pytest -s` (real terminal stdin) as under capture.
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))


def test_transcribe_auto_diarization_setup_success_writes_speakers_html(tmp_path, monkeypatch, capsys):
    """The headline behavior: `wiz transcribe recording.mov --outputs html`
    on a fresh machine just works — the setup runs (real
    _build_transcribe_args, real _ensure_diarization_ready), diarization
    produces real labels, and the old quiet auto-skip hint never appears."""
    events: list[str] = []
    _fresh_machine_stubs(monkeypatch, events)
    video = tmp_path / "recording.mov"
    video.write_bytes(b"fake video")
    (tmp_path / "recording.wav.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9, **_kwargs: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )
    # Frames: stub the extractor (the diarized HTML path inlines frames);
    # the real one would run ffmpeg on the fake video bytes.
    frames_dir = tmp_path / "recording.frames"
    frames_dir.mkdir()
    monkeypatch.setattr(
        cli, "_extract_and_manifest_screenshots",
        lambda in_path, merged, of_base, ffmpeg, width, dry_run: (frames_dir, tmp_path / "recording.frames.json", False),
    )

    rc = cli.cmd_transcribe(_transcribe_args(video, outputs="html", speakers=None))

    assert rc == 0
    assert events == ["install", "download"]  # the one-time setup ran
    html = (tmp_path / "recording.speakers.html").read_text(encoding="utf-8")
    assert "Speaker A" in html  # real labels, not the degraded generic 'Speaker'
    err = capsys.readouterr().err
    assert "skipping speaker labels" not in err  # the skip hint is obsolete now


def test_transcribe_auto_diarization_skips_after_failed_setup(tmp_path, monkeypatch, capsys):
    """Setup failed + merely auto-enabled diarization: the run still
    completes, and the hint says the setup is incomplete and offers
    --no-speakers — a hint, never a crash."""
    video = tmp_path / "recording.mov"
    video.write_bytes(b"fake video")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    monkeypatch.setattr(cli.M, "pick_best", lambda config: Path("/models/turbo.bin"))
    monkeypatch.setattr(cli, "_find_whisper_cli", lambda configured="": "whisper-cli")
    monkeypatch.setattr(cli.aud, "find_ffmpeg", lambda configured="": "ffmpeg")
    monkeypatch.setattr(
        cli.aud, "extract_audio",
        lambda src, ffmpeg, dest_dir=None, dry_run=False, output=None: output or src.with_suffix(".wav"),
    )
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda cmd: SimpleNamespace(returncode=0))
    _stub_setup_unavailable(monkeypatch)
    ran = []
    monkeypatch.setattr(cli.D, "run_diarization", lambda *a, **k: ran.append(1) or [])

    rc = cli.cmd_transcribe(_transcribe_args(video, outputs="srt", speakers=None))

    assert rc == 0
    assert ran == []  # diarization never ran after the failed setup
    err = capsys.readouterr().err
    assert "setup incomplete" in err
    assert "--no-speakers" in err


def test_merge_auto_diarization_setup_success_writes_labeled_outputs(tmp_path, monkeypatch):
    """cmd_merge's half of the wiring: explicit --speakers triggers the
    setup, and labeled outputs land afterwards — the first `wiz merge`
    on a fresh machine just works."""
    events: list[str] = []
    _fresh_machine_stubs(monkeypatch, events)
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9, **_kwargs: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 0
    assert events == ["install", "download"]
    html = (tmp_path / "meeting.m4a.speakers.html").read_text(encoding="utf-8")
    assert "Speaker A" in html
    srt = (tmp_path / "meeting.m4a.speakers.srt").read_text(encoding="utf-8")
    assert "Speaker A:" in srt


def test_merge_zero_segments_message_is_actionable(tmp_path, monkeypatch, capsys):
    """Zero segments is an audio-content verdict, not a defect: the warning
    must say what to DO next, not just what happened."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9, **_kwargs: [],
    )

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 1
    # Wrap-insensitive: the hint lines are muted lines near the console
    # width, so any phrase may be split by rich's wrapping.
    err = " ".join(capsys.readouterr().err.split())
    assert "Diarization produced no segments" in err
    assert "--speakers N" in err           # the biggest accuracy lever first
    assert "--cluster-threshold" in err     # loosen the clustering
    assert "silence" in err                # silence is content, not a defect


def test_speakers_match_setup_failure_exits_with_hint(tmp_path, monkeypatch):
    """`wiz speakers match` needs diarization by definition: when the
    setup cannot make it work, exit loudly with the manual command —
    there is no degraded path to fall back to here."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_unavailable(monkeypatch)

    args = SimpleNamespace(file=str(audio), speakers=1, cluster_threshold=None,
                           no_auto_diarization_setup=False)
    with pytest.raises(SystemExit, match=re.escape(cli.D.DIARIZE_INJECT)):
        cli.cmd_speakers_match(args)


def test_speakers_match_runs_after_setup_success(tmp_path, monkeypatch, capsys):
    """Setup made diarization work → the match command proceeds (and
    reports honestly when there are no profiles yet)."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9, **_kwargs: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )
    monkeypatch.setattr(cli.P, "load_profiles", lambda: [])

    args = SimpleNamespace(file=str(audio), speakers=1, cluster_threshold=None,
                           no_auto_diarization_setup=False)
    rc = cli.cmd_speakers_match(args)

    assert rc == 0
    assert "No stored voice profiles" in capsys.readouterr().err


def test_speakers_match_mp3_normalizes_for_diarization_then_cleans(tmp_path, monkeypatch):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake audio")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    monkeypatch.setattr(cli, "_ensure_diarization_ready", lambda *a, **k: True)
    monkeypatch.setattr(cli.aud, "find_ffmpeg", lambda configured="": "ffmpeg")
    diarized: list[tuple[Path, Path | None]] = []

    def fake_extract(src, ffmpeg, dest_dir=None, dry_run=False, output=None):
        assert src == audio
        assert output is not None
        output.write_bytes(b"normalized wav")
        return output

    monkeypatch.setattr(cli.aud, "extract_audio", fake_extract)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, *_a, **_k: diarized.append((wav, _k.get("cache_source"))) or [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )
    monkeypatch.setattr(cli.P, "load_profiles", lambda: [])
    args = SimpleNamespace(file=str(audio), speakers=1, cluster_threshold=None,
                           no_auto_diarization_setup=False)

    assert cli.cmd_speakers_match(args) == 0
    normalized = tmp_path / "meeting.wav"
    assert diarized == [(normalized, audio)]
    assert not normalized.exists()


def test_speakers_match_cleans_normalized_audio_when_matching_fails(tmp_path, monkeypatch):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake audio")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    monkeypatch.setattr(cli, "_ensure_diarization_ready", lambda *a, **k: True)
    monkeypatch.setattr(cli.aud, "find_ffmpeg", lambda configured="": "ffmpeg")

    def fake_extract(_src, _ffmpeg, dest_dir=None, dry_run=False, output=None):
        assert output is not None
        output.write_bytes(b"normalized wav")
        return output

    monkeypatch.setattr(cli.aud, "extract_audio", fake_extract)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda *_a, **_k: [DiarSegment(start=0.0, end=3.0, speaker=0)],
    )
    monkeypatch.setattr(
        cli.P, "load_profiles",
        lambda: [cli.P.Profile("Alice", [1.0], 1, "2026-01-01T00:00:00Z")],
    )
    monkeypatch.setattr(cli.P, "compute_speaker_embeddings", lambda *_a, **_k: {0: [1.0]})
    def fail_matching(*_args, **_kwargs):
        raise RuntimeError("matching failed")

    monkeypatch.setattr(cli.P, "match_speakers", fail_matching)
    args = SimpleNamespace(file=str(audio), speakers=1, cluster_threshold=None,
                           no_auto_diarization_setup=False)

    with pytest.raises(RuntimeError, match="matching failed"):
        cli.cmd_speakers_match(args)
    assert not (tmp_path / "meeting.wav").exists()


# ---------- auto-setup consent (review round 3, 2026-09-07) ----------
#
# The auto-setup writes into site-packages (unlike the VAD model's cache-file
# download), so an interactive terminal is ASKED first (y/N) and the answer
# persists in auto_diarization_setup. Non-interactive sessions auto-allow so
# a scripted fresh machine still just works; --no-auto-diarization-setup
# short-circuits before any prompt.


class _FakeTtyErr:
    """A stderr stand-in that claims to be a terminal.

    _auto_setup_consent prompts only when BOTH stdin and stderr are ttys,
    and capsys's stderr is always non-tty, so the TTY branch is unreachable
    in tests without faking the stream. Output is collected (not asserted on
    unless a message matters) so rich can render into it freely.
    """

    encoding = "utf-8"

    def __init__(self):
        self.buf: list[str] = []

    def isatty(self) -> bool:
        return True

    def write(self, s: str) -> int:
        self.buf.append(s)
        return len(s)

    def flush(self) -> None:
        pass


def _pin_ttys(monkeypatch, *, stdin_tty: bool, stderr: _FakeTtyErr | None = None):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: stdin_tty))
    if stderr is not None:
        monkeypatch.setattr(sys, "stderr", stderr)


def _boom_input(prompt=""):
    raise AssertionError("the consent prompt must not run here")


def test_consent_answered_true_allows_without_prompting(monkeypatch):
    """auto_diarization_setup=true in config answers permanently — a TTY or
    not, no prompt, setup proceeds."""
    monkeypatch.setattr(builtins, "input", _boom_input)
    config = cli.cfg.Config(diarization_backend="sherpa")
    config.auto_diarization_setup = True
    assert cli._auto_setup_consent(config) is True


def test_consent_answered_false_declines_without_prompting(monkeypatch):
    """auto_diarization_setup=false answers permanently the other way."""
    monkeypatch.setattr(builtins, "input", _boom_input)
    config = cli.cfg.Config(diarization_backend="sherpa")
    config.auto_diarization_setup = False
    assert cli._auto_setup_consent(config) is False


def test_consent_tty_yes_persists_true(tmp_path, monkeypatch):
    """Interactive y: allow, and remember the answer so the question is
    once-ever (config object AND on-disk file)."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", config_path)
    config_path.write_text('future_key = "keep"\n', encoding="utf-8")
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=_FakeTtyErr())
    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    config = cli.cfg.Config(diarization_backend="sherpa", model="ephemeral-cli-override", diarization_provider="coreml")

    assert cli._auto_setup_consent(config) is True
    assert config.auto_diarization_setup is True
    assert config_path.read_text(encoding="utf-8") == (
        'future_key = "keep"\n'
        "auto_diarization_setup = true\n"
    )


def test_consent_tty_no_persists_false_with_way_back_hint(tmp_path, monkeypatch):
    """A decline must say how to change the answer later — a one-shot 'no'
    can't strand the user with no path back to auto-setup."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    err = _FakeTtyErr()
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=err)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
    config = cli.cfg.Config(diarization_backend="sherpa")

    assert cli._auto_setup_consent(config) is False
    saved = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert "auto_diarization_setup = false" in saved
    # Wrap-insensitive: rich wraps the long hint lines at console width.
    flat = " ".join("".join(err.buf).split())
    assert cli.D.DIARIZE_INJECT in flat  # the manual path
    assert "wiz config set auto_diarization_setup=true" in flat  # the way back


def test_consent_tty_eof_declines_and_persists_false(tmp_path, monkeypatch):
    """Piped stdin under a tty stderr (wiz t rec.mov < /dev/null): EOF is a
    decline, not a crash — and it persists like any other answer."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=_FakeTtyErr())

    def _eof(prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", _eof)
    config = cli.cfg.Config(diarization_backend="sherpa")

    assert cli._auto_setup_consent(config) is False
    saved = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert "auto_diarization_setup = false" in saved


def test_consent_tty_ctrl_c_declines_without_persisting(tmp_path, monkeypatch):
    """^C at the prompt is a plain decline: the question stays UNANSWERED so
    a reflexive interrupt doesn't permanently disable auto-setup."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=_FakeTtyErr())

    def _interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", _interrupt)
    config = cli.cfg.Config(diarization_backend="sherpa")

    assert cli._auto_setup_consent(config) is False
    assert config.auto_diarization_setup is None
    assert not (tmp_path / "config.toml").exists()


def test_consent_non_tty_allows_without_persisting(tmp_path, monkeypatch):
    """Scripts/cron (piped stdin): proceed automatically — a prompt would
    hang a headless run — but persist NOTHING (nobody answered)."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    _pin_ttys(monkeypatch, stdin_tty=False)  # capsys stderr is non-tty too
    monkeypatch.setattr(builtins, "input", _boom_input)
    config = cli.cfg.Config(diarization_backend="sherpa")

    assert cli._auto_setup_consent(config) is True
    assert config.auto_diarization_setup is None
    assert not (tmp_path / "config.toml").exists()


def test_consent_persist_failure_does_not_crash(tmp_path, monkeypatch):
    """An unwritable config must not turn a user's y/N into a crash — the
    answer still governs this run."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(cli.cfg, "save", lambda _config: (_ for _ in ()).throw(OSError("disk full")))
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=_FakeTtyErr())
    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    config = cli.cfg.Config(diarization_backend="sherpa")

    assert cli._auto_setup_consent(config) is True


def test_ensure_diarization_ready_tty_consent_yes_runs_setup(tmp_path, monkeypatch):
    """End-to-end: fresh machine on a real terminal — the user answers y at
    the prompt and the FULL setup runs (install, download), with the answer
    remembered for every future run."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    events: list[str] = []
    monkeypatch.setattr(cli, "_diarization_available", lambda config: "download" in events)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    monkeypatch.setattr(cli, "_install_sherpa_onnx", lambda: events.append("install") or True)
    monkeypatch.setattr(cli.D, "download_diarization_models", lambda: events.append("download"))
    monkeypatch.setattr(cli.D, "find_segmentation_model", lambda config: None)
    monkeypatch.setattr(cli.D, "find_embedding_model", lambda config: None)
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=_FakeTtyErr())
    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    config = cli.cfg.Config(diarization_backend="sherpa")

    assert cli._ensure_diarization_ready(config) is True
    assert events == ["install", "download"]
    saved = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert "auto_diarization_setup = true" in saved


def test_ensure_diarization_ready_tty_consent_no_skips_setup(tmp_path, monkeypatch):
    """End-to-end: a decline at the prompt installs/downloads NOTHING and
    lands the caller on its existing degraded/skip path (False return)."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")

    def _boom(*_a, **_k):
        raise AssertionError("declined consent must not install or download")

    monkeypatch.setattr(cli, "_diarization_available", lambda config: False)
    monkeypatch.setattr(cli, "_install_sherpa_onnx", _boom)
    monkeypatch.setattr(cli.D, "download_diarization_models", _boom)
    monkeypatch.setattr(cli.D, "find_segmentation_model", lambda config: None)
    monkeypatch.setattr(cli.D, "find_embedding_model", lambda config: None)
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=_FakeTtyErr())
    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
    config = cli.cfg.Config(diarization_backend="sherpa")

    assert cli._ensure_diarization_ready(config) is False
    assert config.auto_diarization_setup is False  # declined and remembered
    saved = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert "auto_diarization_setup = false" in saved


def test_config_set_auto_diarization_setup_roundtrip(tmp_path, monkeypatch):
    """`wiz config set auto_diarization_setup=false` must store a real bool:
    the tri-state 'bool | None' type string once missed _coerce's bool branch,
    which would persist the STRING 'false' — truthy on every later load."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    args = SimpleNamespace(assignment="auto_diarization_setup=false")

    assert cli.cmd_config_set(args) == 0
    assert cli.cfg.load().auto_diarization_setup is False

    args = SimpleNamespace(assignment="auto_diarization_setup=true")
    assert cli.cmd_config_set(args) == 0
    assert cli.cfg.load().auto_diarization_setup is True


def test_config_save_tri_state_none_semantics(tmp_path, monkeypatch):
    """Unset (None) is omitted from the file — an emitted `= None` would be
    invalid TOML that breaks the NEXT load for every command — and a fresh
    never-answered session must not clobber a previously persisted answer."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")

    cli.cfg.save({"auto_diarization_setup": None})  # a session that never answered
    assert "auto_diarization_setup" not in (tmp_path / "config.toml").read_text(encoding="utf-8")

    cli.cfg.save({"auto_diarization_setup": True})
    assert "auto_diarization_setup = true" in (tmp_path / "config.toml").read_text(encoding="utf-8")

    cli.cfg.save({"auto_diarization_setup": None})  # an unanswered value must not win
    assert cli.cfg.load().auto_diarization_setup is True

    cli.cfg.save({})  # an omitted setting must not win either
    assert cli.cfg.load().auto_diarization_setup is True


def test_config_show_renders_tri_state_unset_cleanly(tmp_path, monkeypatch, capsys):
    """`wiz config show` renders the unset tri-state as <unset>, not the
    Python None repr."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")

    assert cli.cmd_config_show(SimpleNamespace()) == 0
    assert "auto_diarization_setup = <unset>" in capsys.readouterr().out


# ---------- wave-1 audit fix batch (branch fix/wave1-cli, 2026-09-10) ----------
#
# Regression tests for the wave-1 silent-failure audit items: M1 (merge
# diarization gate), M2 (typed DiarizationUnavailable at both call sites),
# H4 (chained --analyze honors SystemExit), H5 (a degraded run never
# clobbers a NAMED frames manifest), M14 (degraded-output info without
# diarization), L-a (pip rc-0-but-not-importable), L-b (declined setup +
# cache hit), L-c (non-bool consent value).


def test_merge_no_speakers_gate_skips_diarization_entirely(tmp_path, monkeypatch):
    """M1: an audio merge with no --speakers never reaches run_diarization
    (or the one-time setup) — it goes straight to the JSON merge path and
    writes the generic-label fallback output."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))

    def _boom(*_a, **_k):
        raise AssertionError("diarization must not run when speakers were not requested")

    monkeypatch.setattr(cli.D, "run_diarization", _boom)
    monkeypatch.setattr(cli, "_ensure_diarization_ready", _boom)

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=None))

    assert rc == 0
    html = (tmp_path / "meeting.m4a.speakers.html").read_text(encoding="utf-8")
    assert ">Speaker<" in html  # explicit html degraded to generic labels


def test_merge_degraded_info_fires_without_diarization(tmp_path, monkeypatch, capsys):
    """M14: a merge requesting degraded output (explicit html) without
    speakers must explain WHY the artifacts carry generic labels — the old
    explanation only existed inside the diarization branch."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9, **_kwargs: [],
    )

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=None))

    assert rc == 0
    flat = " ".join(capsys.readouterr().err.split())
    assert "diarization not requested" in flat


def test_diarize_or_fallback_catches_typed_validate_failure(tmp_path, monkeypatch, capsys):
    """M2 (transcribe site): the config-validation DiarizationUnavailable —
    the path the old RuntimeError string matcher never matched — degrades
    with a hint instead of crashing."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")

    def _raise_validate(wav, config, num_speakers=0, threshold=0.9, **_kwargs):
        raise cli.D.DiarizationUnavailable(
            "sherpa-onnx diarization config validation failed; check model paths."
        )

    monkeypatch.setattr(cli.D, "run_diarization", _raise_validate)

    segs = cli._run_diarize_or_fallback(
        audio, cli.cfg.Config(diarization_backend="sherpa"), _transcribe_args(audio, speakers=1))

    assert segs == []
    flat = " ".join(capsys.readouterr().err.split())
    assert "diarization unavailable" in flat
    assert "config validation failed" in flat


def test_merge_validate_failure_raises_systemexit_with_hint(tmp_path, monkeypatch):
    """M2 (merge site): with nothing else requested, the typed catch turns
    the validate failure into the loud SystemExit + enable hint. (Under the
    old string matcher this escaped as a raw RuntimeError.)"""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_unavailable(monkeypatch)

    def _raise_validate(wav, config, num_speakers=0, threshold=0.9, **_kwargs):
        raise cli.D.DiarizationUnavailable(
            "sherpa-onnx diarization config validation failed; check model paths."
        )

    monkeypatch.setattr(cli.D, "run_diarization", _raise_validate)

    with pytest.raises(SystemExit) as excinfo:
        cli.cmd_merge(_merge_args(audio, outputs="", speakers=1))
    assert "config validation failed" in str(excinfo.value)
    assert cli.D.DIARIZE_INJECT in str(excinfo.value)


def test_transcribe_chained_analyze_failure_surfaces_message_and_rc(tmp_path, monkeypatch, capsys):
    """H4: a --analyze chain that raises SystemExit is not swallowed — the
    message is surfaced, the exit code honored, and the user told the
    transcription itself is fine."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=False)

    def _no_transcript(_args):
        raise SystemExit(
            "No transcript found for recording.mov (need a frames manifest or .speakers.txt)."
        )

    monkeypatch.setattr(cli, "cmd_analyze", _no_transcript)
    args = _transcribe_args(audio, outputs="srt", speakers=None)
    args.analyze = True

    rc = cli.cmd_transcribe(args)

    assert rc == 1
    flat = " ".join(capsys.readouterr().err.split())
    assert "Chained analysis failed" in flat
    assert "No transcript found" in flat
    assert "transcription itself succeeded" in flat


def test_transcribe_chained_analyze_honors_int_exit_code(tmp_path, monkeypatch, capsys):
    """H4 (bare code): SystemExit(3) from the chained analysis propagates
    as rc 3 — no fabricated 'Chained analysis failed: 3' message line."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=False)

    def _boom(_args):
        raise SystemExit(3)

    monkeypatch.setattr(cli, "cmd_analyze", _boom)
    args = _transcribe_args(audio, outputs="srt", speakers=None)
    args.analyze = True

    rc = cli.cmd_transcribe(args)

    assert rc == 3
    flat = " ".join(capsys.readouterr().err.split())
    assert "Chained analysis failed" not in flat


def test_transcribe_chained_analyze_runs_after_unfulfilled_speakers_request(tmp_path, monkeypatch):
    """An unavailable explicit diarization request remains nonzero, but does
    not skip separately requested analysis when fallback text is available."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    analyzed: list[str] = []

    def _analyze(args):
        assert "Speaker" in (tmp_path / "meeting.speakers.txt").read_text(encoding="utf-8")
        analyzed.append(args.file)
        return 0

    monkeypatch.setattr(cli, "cmd_analyze", _analyze)
    args = _transcribe_args(audio, outputs="html", speakers=1)
    args.analyze = True

    rc = cli.cmd_transcribe(args)

    assert rc == 1
    assert analyzed == [str(audio)]


def test_consent_non_bool_string_is_not_authoritative(tmp_path, monkeypatch):
    """L-c: a hand-edited `auto_diarization_setup = "false"` string is not a
    stored answer (it is truthy and used to read as consent) — it falls
    through to the interactive prompt, where the user's 'n' declines."""
    monkeypatch.setattr(cli.cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli.cfg, "CONFIG_PATH", tmp_path / "config.toml")
    _pin_ttys(monkeypatch, stdin_tty=True, stderr=_FakeTtyErr())
    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
    config = cli.cfg.Config(diarization_backend="sherpa")
    config.auto_diarization_setup = "false"

    assert cli._auto_setup_consent(config) is False
    saved = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert "auto_diarization_setup = false" in saved


def test_install_sherpa_rc0_but_not_importable_warns(monkeypatch, capsys):
    """L-a: pip exits 0 but sherpa_onnx still does not resolve (partial
    install, wrong venv) — a loud warning with the manual path, not a silent
    False that reads as a fresh-install loop on the next run."""
    monkeypatch.setattr(
        cli.subprocess, "run",
        lambda cmd, check=False: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    assert cli._install_sherpa_onnx() is False
    flat = " ".join(capsys.readouterr().err.split())
    assert "still not importable" in flat
    assert cli.D.DIARIZE_INJECT in flat


def test_merge_declined_setup_then_success_explains_cache_reuse(tmp_path, monkeypatch, capsys):
    """L-b: setup failed/declined yet diarization succeeded — only possible
    via the diarization cache; the run says so instead of leaving the
    surprise unexplained."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_unavailable(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9, **_kwargs: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )

    rc = cli.cmd_merge(_merge_args(audio, outputs="html", speakers=1))

    assert rc == 0
    flat = " ".join(capsys.readouterr().err.split())
    assert "ran WITHOUT the one-time setup" in flat
    assert ".diar.json" in flat


def test_manifest_is_named_detection(tmp_path):
    """H5 unit: a NAMED manifest reads as named (keep), an all-generic one
    as degraded (refresh), a missing one as writable."""
    frames_dir = tmp_path / "rec.frames"
    frames_dir.mkdir()
    manifest = tmp_path / "rec.frames.json"
    named = [cli.SC.FrameEntry(index=1, start=0.0, end=2.0, speaker="Alice",
                               text="hi", frame="seg0001.jpg")]
    cli.SC.write_manifest(named, frames_dir, manifest)
    assert cli._manifest_is_named(manifest) is True

    degraded = [cli.SC.FrameEntry(index=1, start=0.0, end=2.0, speaker="Speaker",
                                  text="hi", frame="seg0001.jpg")]
    cli.SC.write_manifest(degraded, frames_dir, manifest)
    assert cli._manifest_is_named(manifest) is False

    assert cli._manifest_is_named(tmp_path / "missing.frames.json") is False


def test_extract_manifests_degraded_run_never_clobbers_named_manifest(tmp_path, monkeypatch, capsys):
    """H5 unit: an incoming all-generic shot list KEEPS a NAMED manifest
    (with a warning) and never runs the frame extraction."""
    video = tmp_path / "recording.mov"
    video.write_bytes(b"fake video")
    frames_dir = tmp_path / "recording.frames"
    frames_dir.mkdir()
    manifest = tmp_path / "recording.frames.json"
    named = [cli.SC.FrameEntry(index=1, start=0.0, end=2.0, speaker="Alice",
                               text="hi", frame="seg0001.jpg")]
    cli.SC.write_manifest(named, frames_dir, manifest)

    def _boom(*_a, **_k):
        raise AssertionError("extraction must not run when the named manifest is kept")

    monkeypatch.setattr(cli.SC, "extract_segment_frames", _boom)
    merged = [(cli.MR.WhisperSeg(start=0.0, end=2.0, text="hi"), "Speaker")]

    result = cli._extract_and_manifest_screenshots(
        video, merged, tmp_path / "recording", ffmpeg="ffmpeg", width=1280,
    )

    assert result == (frames_dir, manifest, True)
    loaded = cli.SC.load_manifest(manifest)
    assert loaded[0].speaker == "Alice"  # untouched
    flat = " ".join(capsys.readouterr().err.split())
    assert "already exists with named speakers" in flat


def test_extract_manifests_refreshes_existing_degraded_manifest(tmp_path, monkeypatch):
    """H5 flip side: an existing DEGRADED manifest has no speaker names to
    protect — the degraded re-run refreshes it (idempotence), so re-runs
    with a different --model/--language can update the transcript."""
    video = tmp_path / "recording.mov"
    video.write_bytes(b"fake video")
    frames_dir = tmp_path / "recording.frames"
    frames_dir.mkdir()
    manifest = tmp_path / "recording.frames.json"
    degraded = [cli.SC.FrameEntry(index=1, start=0.0, end=2.0, speaker="Speaker",
                                  text="stale", frame="seg0001.jpg")]
    cli.SC.write_manifest(degraded, frames_dir, manifest)
    fresh = [cli.SC.FrameEntry(index=1, start=0.0, end=2.0, speaker="Speaker",
                               text="fresh", frame="seg0001.jpg")]
    monkeypatch.setattr(
        cli.SC, "extract_segment_frames",
        lambda video, merged, out_dir, ffmpeg="ffmpeg", width=1280, dry_run=False: fresh,
    )
    merged = [(cli.MR.WhisperSeg(start=0.0, end=2.0, text="fresh"), "Speaker")]

    result = cli._extract_and_manifest_screenshots(
        video, merged, tmp_path / "recording", ffmpeg="ffmpeg", width=1280,
    )

    assert result == (frames_dir, manifest, False)
    loaded = cli.SC.load_manifest(manifest)
    assert loaded[0].text == "fresh"  # refreshed, not kept


def test_transcribe_degraded_run_keeps_named_frames_manifest_returns_error(tmp_path, monkeypatch, capsys):
    """H5 end-to-end: a degraded video re-run (explicit --speakers, sherpa
    missing) KEEPS the named manifest an earlier diarized run wrote, while
    still reporting that the requested diarization was unfulfilled."""
    video = tmp_path / "recording.mov"
    video.write_bytes(b"fake video")
    (tmp_path / "recording.wav.json").write_text(_WHISPER_JSON, encoding="utf-8")
    wav = tmp_path / "recording.wav"
    frames_dir = tmp_path / "recording.frames"
    frames_dir.mkdir()
    manifest = tmp_path / "recording.frames.json"
    named = [cli.SC.FrameEntry(index=1, start=0.0, end=2.0, speaker="Alice",
                               text="hello world", frame="seg0001.jpg")]
    cli.SC.write_manifest(named, frames_dir, manifest)

    def fake_build(args, config, *, timings=None):
        return (["whisper-cli"], "model.bin", wav, video, False,
                tmp_path / "recording", True, True)

    monkeypatch.setattr(cli, "_build_transcribe_args", fake_build)
    monkeypatch.setattr(cli, "_run_whisper_streaming", lambda cmd: SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli.D, "run_diarization", _raise_sherpa_missing)
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))

    def _boom(*_a, **_k):
        raise AssertionError("extraction must not run when the named manifest is kept")

    monkeypatch.setattr(cli.SC, "extract_segment_frames", _boom)

    rc = cli.cmd_transcribe(_transcribe_args(video, outputs="", speakers=1))

    assert rc == 1
    loaded = cli.SC.load_manifest(manifest)
    assert loaded[0].speaker == "Alice"
    flat = " ".join(capsys.readouterr().err.split())
    assert "already exists with named speakers" in flat


# ---------- wave-1 M3 adoption: profiles API at the CLI call sites ----------
#
# py-support-2 (fbd04d5) gave profiles.py an auto-match provenance API:
# cosine_similarity returns None on dim mismatch, and save_profile(auto_match)
# creates-but-never-merges. These tests pin the CLI's ADOPTION of it — the
# speakers-match score table under a dim-mismatched stored profile, and the
# auto-vs-confirmed provenance threaded through _write_labeled_outputs →
# _save_named_profiles.


def test_speakers_match_dim_mismatch_renders_na_not_crash(tmp_path, monkeypatch, capsys):
    """`wiz speakers match` against a stored profile saved with a DIFFERENT
    embedding dim (embedding model swapped): cosine_similarity returns None
    and the score table must render an honest n/a — the old code crashed
    sorting None among floats, and read scores[0][0] on what could be an
    empty list."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9, **_kwargs: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    cli.P.save_profile("OldDim", [1.0, 2.0, 3.0, 4.0], samples=2)  # dim 4
    monkeypatch.setattr(
        cli.P, "compute_speaker_embeddings",
        lambda wav, segments, config: {0: [1.0, 1.0]},  # dim 2
    )
    tables: list[tuple[str | None, list[list[str]]]] = []
    monkeypatch.setattr(cli.ui, "table", lambda title, _columns, rows: tables.append((title, rows)))

    args = SimpleNamespace(file=str(audio), speakers=1, cluster_threshold=None,
                           no_auto_diarization_setup=False)
    rc = cli.cmd_speakers_match(args)

    assert rc == 0
    err = capsys.readouterr().err
    assert "dimension mismatch" in err  # match_speakers' skip warning fired
    match_rows = next(rows for title, rows in tables if title == "Speaker match (dry run)")
    assert match_rows[0][3] == "OldDim=n/a"  # incomparable pair rendered, not crashed
    assert match_rows[0][2] == "n/a"  # best score is n/a too — no scores existed
    timing = dict(next(rows for title, rows in tables if title == "Stage timing"))
    assert timing["Transcription"] == "skipped (not applicable)"
    assert timing["Writing outputs / frames"] == "skipped (not applicable)"
    assert "s" in timing["Speaker profiles"]


@pytest.mark.parametrize("command", ["transcribe", "merge", "speakers match"])
def test_commands_match_split_clusters_without_changing_profiles(tmp_path, monkeypatch, command):
    """Use the real matcher and naming flow with isolated synthetic profiles."""
    audio = _setup_transcribe(monkeypatch, tmp_path, diarize_enabled=True)
    _stub_setup_ready(monkeypatch)
    texts = ["first fragment", "other person", "second fragment", "unknown guest"]
    payload = {"transcription": [
        {"timestamps": {"from": f"00:00:{i:02d},000", "to": f"00:00:{i + 1:02d},000"},
         "text": text}
        for i, text in enumerate(texts)
    ]}
    (tmp_path / "meeting.m4a.json").write_text(json.dumps(payload), encoding="utf-8")
    diar = [DiarSegment(start=float(i), end=float(i + 1), speaker=i) for i in range(4)]
    monkeypatch.setattr(cli.D, "run_diarization", lambda *_args, **_kwargs: diar)
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path / "profiles")
    cli.P.save_profile("Alice", [1.0, 0.0, 0.0], samples=4)
    cli.P.save_profile("Bob", [0.0, 1.0, 0.0], samples=3, auto_match=True)
    before = {p.name: p.read_bytes() for p in cli.P.profiles_dir().glob("*.json")}
    monkeypatch.setattr(
        cli.P, "compute_speaker_embeddings",
        lambda *_args: {
            0: [1.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0],
            2: [0.99, 0.01, 0.0], 3: [0.0, 0.0, 1.0],
        },
    )
    tables = []
    monkeypatch.setattr(cli.ui, "table", lambda title, _columns, rows: tables.append((title, rows)))

    if command == "transcribe":
        args = _transcribe_args(audio, speakers=0)
        args.no_voice_profiles = False
        rc = cli.cmd_transcribe(args)
        output_base = tmp_path / "meeting"
    elif command == "merge":
        args = _merge_args(audio, speakers=0)
        args.no_voice_profiles = False
        rc = cli.cmd_merge(args)
        output_base = tmp_path / "meeting.m4a"
    else:
        args = SimpleNamespace(file=str(audio), speakers=0, cluster_threshold=None,
                               no_auto_diarization_setup=False)
        rc = cli.cmd_speakers_match(args)

    assert rc == 0
    expected = ["Alice", "Bob", "Alice", "Speaker D"]
    if command == "speakers match":
        rows = next(rows for title, rows in tables if title == "Speaker match (dry run)")
        assert [row[1] for row in rows] == ["Alice", "Bob", "Alice", "(no match)"]
    else:
        srt = Path(str(output_base) + ".speakers.srt").read_text(encoding="utf-8")
        txt = Path(str(output_base) + ".speakers.txt").read_text(encoding="utf-8")
        html = Path(str(output_base) + ".speakers.html").read_text(encoding="utf-8")
        assert [line for line in srt.splitlines() if ": " in line] == [
            f"{label}: {text}" for label, text in zip(expected, texts)
        ]
        assert [line.split(" (", 1)[0] for line in txt.splitlines() if line] == expected
        assert re.findall(r'<span class="speaker"[^>]*>(.*?)</span>', html) == expected
    assert {p.name: p.read_bytes() for p in cli.P.profiles_dir().glob("*.json")} == before


def test_merge_auto_match_never_merges_existing_profile(tmp_path, monkeypatch, capsys):
    """M3 cli adoption: a name that arrived via VOICE-PROFILE auto-match
    never merges into an existing profile — save_profile's guard keeps the
    stored centroid byte-identical and the run says so. (Before the adoption
    every named speaker was merged, so a self-confirming auto-match chain
    silently drifted the stored centroid.)"""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9, **_kwargs: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    cli.P.save_profile("Alice", [0.5, 0.5], samples=4)  # user-confirmed base
    before = (tmp_path / "Alice.json").read_text(encoding="utf-8")
    # The run's cluster embedding auto-matches Alice at threshold 0.8.
    monkeypatch.setattr(
        cli.P, "compute_speaker_embeddings",
        lambda wav, segments, config: {0: [0.5, 0.5]},
    )

    args = _merge_args(audio, outputs="html", speakers=1)
    args.no_voice_profiles = False  # enable the feature under test
    rc = cli.cmd_merge(args)

    assert rc == 0
    # The stored centroid is untouched — no merge, no replace.
    assert (tmp_path / "Alice.json").read_text(encoding="utf-8") == before
    flat = " ".join(capsys.readouterr().err.split())
    assert "auto-match not merged" in flat
    assert "Merged voice profile" not in flat  # no merge that didn't happen


def test_merge_confirmed_name_merges_existing_profile(tmp_path, monkeypatch, capsys):
    """M3 flip side: --speakers-names is a HUMAN confirmation — it overrides
    any auto-match for the same label and merges into the stored profile
    normally (auto_match=False), upgrading provenance to 'user'."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9, **_kwargs: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    cli.P.save_profile("Alice", [0.0, 0.0], samples=2)
    monkeypatch.setattr(
        cli.P, "compute_speaker_embeddings",
        lambda wav, segments, config: {0: [2.0, 2.0]},
    )

    args = _merge_args(audio, outputs="html", speakers=1, speakers_names=["Alice"])
    args.no_voice_profiles = False
    rc = cli.cmd_merge(args)

    assert rc == 0
    data = json.loads((tmp_path / "Alice.json").read_text(encoding="utf-8"))
    assert data["samples"] == 3                       # merged: 2 stored + 1 new
    # (0*2 + 2*1)/3 = 2/3 — plain abs checks, matching test_profiles.py's
    # idiom (pytest.approx routes through np.isscalar, gone in numpy 2.x).
    assert all(abs(v - 2/3) < 1e-9 for v in data["embedding"])
    assert data["source"] == "user"
    flat = " ".join(capsys.readouterr().err.split())
    assert "Merged voice profile: Alice" in flat


def test_write_labeled_outputs_prompt_confirmation_upgrades_auto_label(tmp_path, monkeypatch, capsys):
    """M3 provenance threading: interactive confirmation — even just pressing
    Enter on the auto-matched suggestion — upgrades the label from auto to
    user-confirmed, so the save MERGES instead of no-clobbering."""
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    merged = [(cli.MR.WhisperSeg(start=0.0, end=2.0, text="hi"), "Speaker A")]
    # The interactive prompt returns the auto-suggested name (Enter accepted).
    monkeypatch.setattr(
        cli, "_prompt_speaker_names",
        lambda merged, default_names=None: {"Speaker A": "Alice"},
    )
    cli.P.save_profile("Alice", [0.0, 0.0], samples=2, auto_match=True)  # source 'auto'

    _srt, _txt, _html, name_map = cli._write_labeled_outputs(
        merged, tmp_path / "rec", name_speakers=True,
        profile_names={"Speaker A": "Alice"},
        cluster_embeddings={0: [2.0, 2.0]},
        save_profiles=True,
    )

    assert name_map == {"Speaker A": "Alice"}
    data = json.loads((tmp_path / "Alice.json").read_text(encoding="utf-8"))
    assert data["samples"] == 3   # merged, not skipped — Enter is confirmation
    assert data["source"] == "user"  # provenance upgraded from 'auto'


@pytest.mark.parametrize(
    "listed_names,answers,expected_names",
    [
        ("Alice,Bob", ["Alicia", ""], ["Alicia", "Bob"]),
        ("Alice,Alice", ["Alicia", ""], ["Alicia", "Alice"]),
        ("Alice,Alice", ["", ""], ["Alice", "Alice"]),
        ("Alice,Bob", ["Bob", "Alice"], ["Bob", "Alice"]),
    ],
)
def test_combined_speaker_naming_preserves_cluster_identity_for_profile_saves(
    tmp_path, monkeypatch, listed_names, answers, expected_names,
):
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path / "profiles")
    for name in ("Alice", "Bob"):
        cli.P.save_profile(name, [0.0, 0.0], samples=2, auto_match=True)
    before = {path.stem: path.read_bytes() for path in cli.P.profiles_dir().glob("*.json")}
    merged = [
        (cli.MR.WhisperSeg(start=0.0, end=4.0, text="first voice"), "Speaker A"),
        (cli.MR.WhisperSeg(start=4.0, end=6.0, text="second voice"), "Speaker B"),
    ]
    prompts = []
    responses = iter(answers)

    def respond(prompt):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(builtins, "input", respond)
    embeddings = {0: [3.0, 0.0], 1: [0.0, 3.0]}
    srt, txt, html, name_map = cli._write_labeled_outputs(
        merged, tmp_path / "recording", name_speakers=True,
        speakers_names=[listed_names], html=True,
        cluster_embeddings=embeddings, save_profiles=True,
    )

    defaults = listed_names.split(",")
    assert prompts == [f"Name for Speaker {label} [{name}]: " for label, name in zip("AB", defaults)]
    assert name_map == dict(zip(["Speaker A", "Speaker B"], expected_names))
    assert [line.split(": ", 1)[0] for line in srt.read_text().splitlines() if ": " in line] == expected_names
    # Dialogue TXT combines consecutive cues belonging to the same speaker.
    turn_names = [name for index, name in enumerate(expected_names) if not index or name != expected_names[index - 1]]
    assert [line.split(" (", 1)[0] for line in txt.read_text().splitlines() if line] == turn_names
    assert re.findall(r'<span class="speaker"[^>]*>(.*?)</span>', html.read_text()) == turn_names
    for output in (srt, txt, html):
        assert "first voice" in output.read_text()
        assert "second voice" in output.read_text()
    for name in set(expected_names):
        data = json.loads((cli.P.profiles_dir() / f"{name}.json").read_text())
        selected_cluster = expected_names.index(name)  # A has the most talk time.
        expected_samples = 3 if name in before else 1
        assert data["name"] == name
        assert data["samples"] == expected_samples
        assert data["source"] == "user"
        assert data["embedding"] == [value / expected_samples for value in embeddings[selected_cluster]]
    for name in before.keys() - set(expected_names):
        assert (cli.P.profiles_dir() / f"{name}.json").read_bytes() == before[name]


def test_write_labeled_outputs_saves_duplicate_confirmed_name_once_from_longest_cluster(
    tmp_path, monkeypatch,
):
    """Two Enter confirmations for Alice save only the longest cluster once."""
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    # Speaker A has the longest individual utterance, but Speaker B has more
    # total transcript talk time across two fragments. Enter accepts Alice
    # for both suggested defaults.
    merged = [
        (cli.MR.WhisperSeg(start=0.0, end=5.0, text="longest utterance"), "Speaker A"),
        (cli.MR.WhisperSeg(start=5.0, end=8.0, text="first B fragment"), "Speaker B"),
        (cli.MR.WhisperSeg(start=8.0, end=11.0, text="second B fragment"), "Speaker B"),
    ]
    cli.P.save_profile("Alice", [0.0, 0.0], samples=2, auto_match=True)
    answers = iter(["", ""])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    cli._write_labeled_outputs(
        merged, tmp_path / "rec", name_speakers=True,
        profile_names={"Speaker A": "Alice", "Speaker B": "Alice"},
        cluster_embeddings={0: [1.0, 0.0], 1: [0.0, 1.0]},
        save_profiles=True,
    )

    data = json.loads((tmp_path / "Alice.json").read_text(encoding="utf-8"))
    assert data["samples"] == 3
    assert data["source"] == "user"
    assert data["embedding"] == [0.0, 1 / 3]


def test_write_labeled_outputs_prefers_confirmed_duplicate_over_longer_auto_cluster(
    tmp_path, monkeypatch, capsys,
):
    """A shorter human confirmation wins over an automatic duplicate."""
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    merged = [
        (cli.MR.WhisperSeg(start=0.0, end=10.0, text="long automatic fragment"), "Speaker A"),
        (cli.MR.WhisperSeg(start=10.0, end=11.0, text="short confirmed fragment"), "Speaker B"),
    ]
    cli.P.save_profile("Alice", [0.0, 0.0], samples=2)
    monkeypatch.setattr(
        cli, "_prompt_speaker_names",
        lambda _merged, default_names=None: {"Speaker B": "Alice"},
    )

    cli._write_labeled_outputs(
        merged, tmp_path / "rec", name_speakers=True,
        profile_names={"Speaker A": "Alice", "Speaker B": "Alice"},
        cluster_embeddings={0: [1.0, 0.0], 1: [0.0, 1.0]},
        save_profiles=True,
    )

    data = json.loads((tmp_path / "Alice.json").read_text(encoding="utf-8"))
    assert data["samples"] == 3
    assert data["source"] == "user"
    assert data["embedding"] == [0.0, 1 / 3]
    assert "auto-match not merged" not in capsys.readouterr().err


def test_write_labeled_outputs_reports_one_keep_hint_for_duplicate_auto_matches(
    tmp_path, monkeypatch, capsys,
):
    """Repeated auto matches leave an existing profile untouched once per name."""
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    merged = [
        (cli.MR.WhisperSeg(start=0.0, end=2.0, text="first fragment"), "Speaker A"),
        (cli.MR.WhisperSeg(start=2.0, end=5.0, text="second fragment"), "Speaker B"),
    ]
    cli.P.save_profile("Alice", [0.0, 0.0], samples=2)
    before = (tmp_path / "Alice.json").read_bytes()

    cli._write_labeled_outputs(
        merged, tmp_path / "rec",
        profile_names={"Speaker A": "Alice", "Speaker B": "Alice"},
        cluster_embeddings={0: [1.0, 0.0], 1: [0.0, 1.0]},
        save_profiles=True,
    )

    assert (tmp_path / "Alice.json").read_bytes() == before
    err = capsys.readouterr().err
    assert err.count("auto-match not merged") == 1
    assert "Auto-matched 2 cluster(s) to 1 distinct profile name(s)" in err


def test_write_labeled_outputs_deduplicates_sanitized_profile_path(tmp_path, monkeypatch):
    """Distinct display names that sanitize alike add only one sample."""
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    merged = [
        (cli.MR.WhisperSeg(start=0.0, end=5.0, text="long first name"), "Speaker A"),
        (cli.MR.WhisperSeg(start=5.0, end=7.0, text="short underscore name"), "Speaker B"),
    ]
    cli.P.save_profile("Alice Smith", [0.0, 0.0], samples=2)
    monkeypatch.setattr(
        cli, "_prompt_speaker_names",
        lambda _merged, default_names=None: {
            "Speaker A": "Alice Smith", "Speaker B": "Alice_Smith",
        },
    )

    cli._write_labeled_outputs(
        merged, tmp_path / "rec", name_speakers=True,
        cluster_embeddings={0: [1.0, 0.0], 1: [0.0, 1.0]},
        save_profiles=True,
    )

    data = json.loads((tmp_path / "Alice_Smith.json").read_text(encoding="utf-8"))
    assert data["samples"] == 3
    assert data["embedding"] == [1 / 3, 0.0]


@pytest.mark.parametrize("existing", [False, True])
def test_write_labeled_outputs_deduplicates_case_aliases_on_native_volume(
    tmp_path, monkeypatch, existing,
):
    """Case aliases share one target if native storage aliases them, new or existing."""
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    probe_dir = tmp_path / "case-sensitivity-probe"
    probe_dir.mkdir()
    (probe_dir / "Alice").write_text("probe", encoding="utf-8")
    native_aliases = (probe_dir / "alice").exists()
    merged = [
        (cli.MR.WhisperSeg(start=0.0, end=5.0, text="capitalized name"), "Speaker A"),
        (cli.MR.WhisperSeg(start=5.0, end=7.0, text="lowercase name"), "Speaker B"),
    ]
    if existing:
        cli.P.save_profile("Alice", [0.0, 0.0], samples=2)
    monkeypatch.setattr(
        cli, "_prompt_speaker_names",
        lambda _merged, default_names=None: {"Speaker A": "Alice", "Speaker B": "alice"},
    )

    cli._write_labeled_outputs(
        merged, tmp_path / "rec", name_speakers=True,
        cluster_embeddings={0: [1.0, 0.0], 1: [0.0, 1.0]},
        save_profiles=True,
    )

    profiles = sorted(tmp_path.glob("*.json"))
    assert len(profiles) == (1 if native_aliases else 2)
    if native_aliases:
        data = json.loads(profiles[0].read_text(encoding="utf-8"))
        assert data["samples"] == (3 if existing else 1)
        assert data["embedding"] == ([1 / 3, 0.0] if existing else [1.0, 0.0])
    else:
        assert len(profiles) == 2
        assert json.loads((tmp_path / "Alice.json").read_text(encoding="utf-8"))["samples"] == (
            3 if existing else 1
        )
        assert json.loads((tmp_path / "alice.json").read_text(encoding="utf-8"))["samples"] == 1


def test_write_labeled_outputs_prefers_confirmed_sanitized_alias_over_auto(
    tmp_path, monkeypatch, capsys,
):
    """A confirmed alias owns the shared profile path before an auto alias."""
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    merged = [
        (cli.MR.WhisperSeg(start=0.0, end=10.0, text="long auto alias"), "Speaker A"),
        (cli.MR.WhisperSeg(start=10.0, end=11.0, text="short confirmed alias"), "Speaker B"),
    ]
    cli.P.save_profile("Alice_Smith", [0.0, 0.0], samples=2)
    monkeypatch.setattr(
        cli, "_prompt_speaker_names",
        lambda _merged, default_names=None: {"Speaker B": "Alice_Smith"},
    )

    cli._write_labeled_outputs(
        merged, tmp_path / "rec", name_speakers=True,
        profile_names={"Speaker A": "Alice Smith", "Speaker B": "Alice_Smith"},
        cluster_embeddings={0: [1.0, 0.0], 1: [0.0, 1.0]},
        save_profiles=True,
    )

    data = json.loads((tmp_path / "Alice_Smith.json").read_text(encoding="utf-8"))
    assert data["samples"] == 3
    assert data["embedding"] == [0.0, 1 / 3]
    assert "auto-match not merged" not in capsys.readouterr().err


def test_speakers_names_keep_per_cluster_overrides_when_auto_names_repeat(tmp_path, monkeypatch, capsys):
    """Positional names may intentionally split clusters that auto-match alike."""
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    merged = [
        (cli.MR.WhisperSeg(start=0.0, end=10.0, text="Alice A"), "Speaker A"),
        (cli.MR.WhisperSeg(start=10.0, end=15.0, text="Bob B"), "Speaker B"),
        (cli.MR.WhisperSeg(start=15.0, end=21.0, text="Alice C explicitly named Bob"), "Speaker C"),
    ]
    cli.P.save_profile("Alice", [0.0, 0.0], samples=2)
    cli.P.save_profile("Bob", [0.0, 0.0], samples=2)

    _srt, _txt, _html, name_map = cli._write_labeled_outputs(
        merged, tmp_path / "rec", speakers_names=["Alice", "Bob"],
        profile_names={"Speaker A": "Alice", "Speaker C": "Alice", "Speaker B": "Bob"},
        cluster_embeddings={0: [1.0, 0.0], 1: [0.0, 1.0], 2: [1.0, 0.2]},
        save_profiles=True,
    )

    assert name_map == {"Speaker A": "Alice", "Speaker B": "Bob", "Speaker C": "Bob"}
    alice = json.loads((tmp_path / "Alice.json").read_text(encoding="utf-8"))
    assert alice["samples"] == 3
    assert alice["embedding"] == [1 / 3, 0.0]
    bob = json.loads((tmp_path / "Bob.json").read_text(encoding="utf-8"))
    assert bob["samples"] == 3
    assert bob["embedding"] == [1 / 3, 0.2 / 3]
    assert "auto-match not merged" not in capsys.readouterr().err


def test_write_labeled_outputs_skips_profile_cluster_without_transcript_cues(
    tmp_path, monkeypatch, capsys,
):
    """A matched diarization cluster may have no assigned Whisper cue."""
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    cli.P.save_profile("Alice", [1.0, 0.0], samples=2)
    before = (tmp_path / "Alice.json").read_bytes()
    merged = [(cli.MR.WhisperSeg(start=0.0, end=2.0, text="visible fragment"), "Speaker A")]

    srt, _txt, _html, _map = cli._write_labeled_outputs(
        merged, tmp_path / "rec",
        profile_names={"Speaker A": "Alice", "Speaker B": "Alice"},
        cluster_embeddings={0: [1.0, 0.0], 1: [0.99, 0.01]},
        save_profiles=True,
    )

    assert "Alice: visible fragment" in srt.read_text(encoding="utf-8")
    assert (tmp_path / "Alice.json").read_bytes() == before
    assert capsys.readouterr().err.count("auto-match not merged") == 1


def test_merge_speakers_names_overrides_wrong_auto_match(tmp_path, monkeypatch, capsys):
    """W2-M15: --speakers-names is a HUMAN confirmation and must override a
    wrong auto-match. A stored profile whose centroid sits exactly on the
    run's cluster embedding (cosine 1.0 >= threshold) makes the auto-match
    fire with the WRONG name — the user passing --speakers-names is
    correcting it, so the run must use the name they gave, merge into (or
    create) that profile, and not leave the wrong profile's keep-hint
    around."""
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")
    (tmp_path / "meeting.m4a.json").write_text(_WHISPER_JSON, encoding="utf-8")
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(diarization_backend="sherpa"))
    _stub_setup_ready(monkeypatch)
    monkeypatch.setattr(
        cli.D, "run_diarization",
        lambda wav, config, num_speakers=0, threshold=0.9, **_kwargs: [
            DiarSegment(start=0.0, end=3.0, speaker=0),
        ],
    )
    monkeypatch.setattr(cli.P, "profiles_dir", lambda: tmp_path)
    # A wrong auto-match target: centroid identical to the run's cluster
    # embedding, so cosine similarity is 1.0 and the auto-match fires.
    cli.P.save_profile("WrongName", [0.5, 0.5], samples=2)
    monkeypatch.setattr(
        cli.P, "compute_speaker_embeddings",
        lambda wav, segments, config: {0: [0.5, 0.5]},
    )

    args = _merge_args(audio, outputs="html", speakers=1, speakers_names=["Alice"])
    args.no_voice_profiles = False
    rc = cli.cmd_merge(args)

    assert rc == 0
    flat = " ".join(capsys.readouterr().err.split())
    # The human-supplied name won.
    assert "Alice" in flat
    assert "WrongName" in flat  # the wrong match is named in the auto-match note
    # Alice's profile was created by the human confirmation.
    alice = json.loads((tmp_path / "Alice.json").read_text(encoding="utf-8"))
    assert alice["samples"] == 1
    assert alice["source"] == "user"
    # The wrong profile was NOT merged into — its centroid is byte-identical.
    wrong = json.loads((tmp_path / "WrongName.json").read_text(encoding="utf-8"))
    assert wrong["samples"] == 2
    assert all(abs(v - 0.5) < 1e-9 for v in wrong["embedding"])


def test_diarize_install_hint_matches_the_extra_and_never_resolves_wiz():
    """The manual hint installs exactly what the [diarize] extra declares, and
    injects the requirement itself: an unrelated "wiz" package exists on PyPI."""
    import tomllib

    pyproject = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["optional-dependencies"]["diarize"] == [cli.D.DIARIZE_REQUIREMENT]
    assert pyproject["project"]["name"] == cli.D.PIPX_PACKAGE == "transcript-wiz"
    assert cli.D.DIARIZE_INJECT == f"pipx inject --force transcript-wiz '{cli.D.DIARIZE_REQUIREMENT}'"
    assert "wiz[" not in cli.D.DIARIZE_INJECT


def test_missing_sherpa_error_gives_the_same_install_command(monkeypatch):
    import builtins as _builtins

    real_import = _builtins.__import__

    def no_sherpa(name, *args, **kwargs):
        if name == "sherpa_onnx":
            raise ImportError("No module named 'sherpa_onnx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_builtins, "__import__", no_sherpa)
    with pytest.raises(cli.D.DiarizationUnavailable) as exc:
        cli.D._import_sherpa()
    assert cli.D.DIARIZE_INJECT in str(exc.value)


def test_upgrade_reinjects_the_diarize_requirement_into_the_pipx_package(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_diarize_extra_installed", lambda: True)
    monkeypatch.setattr(cli, "_run_live", lambda cmd: calls.append(cmd) or 0)
    assert cli.cmd_upgrade(argparse.Namespace()) == 0
    assert calls[0] == ["pipx", "install", "--force", cli._INSTALL_SOURCE]
    assert calls[1] == ["pipx", "inject", "--force", "transcript-wiz", cli.D.DIARIZE_REQUIREMENT]


def test_looks_degraded_txt_accepts_multi_paragraph_generic_turn(tmp_path):
    """A long generic-label turn continues in label-less paragraphs."""
    from wiz import merge as MR

    sentence = "This is a sentence that goes on for a while and ends."
    merged = [(MR.WhisperSeg(i, i + 1, sentence), "Speaker") for i in range(30)]
    degraded = tmp_path / "degraded.speakers.txt"
    degraded.write_text(MR.format_dialogue_txt(merged) + "\n", encoding="utf-8")
    assert "\n\n" in degraded.read_text(encoding="utf-8")
    assert cli._looks_degraded_txt(degraded)

    named = tmp_path / "named.speakers.txt"
    named.write_text(
        MR.format_dialogue_txt(merged + [(MR.WhisperSeg(30, 31, "Reply."), "Vadim")]) + "\n",
        encoding="utf-8",
    )
    assert not cli._looks_degraded_txt(named)

    foreign = tmp_path / "foreign.speakers.txt"
    foreign.write_text("Some notes\n\nSpeaker (00:00:01): hi\n", encoding="utf-8")
    assert not cli._looks_degraded_txt(foreign)


def test_degraded_fallback_keeps_named_txt_with_long_label(tmp_path):
    """A named turn longer than 80 characters must not be overwritten."""
    from wiz import merge as MR

    long_name = "A" * 81
    existing = MR.format_dialogue_txt([
        (MR.WhisperSeg(0, 1, "Generic."), "Speaker"),
        (MR.WhisperSeg(1, 2, "Named."), long_name),
    ]) + "\n"
    txt_out = tmp_path / "recording.speakers.txt"
    txt_out.write_text(existing, encoding="utf-8")

    written, kept = cli._write_html_transcript(
        [(MR.WhisperSeg(0, 1, "Replacement."), "Speaker")],
        tmp_path / "recording",
        None,
        "recording",
        note=cli._GENERIC_LABEL_NOTE,
        transcript_txt=True,
    )

    assert txt_out in kept
    assert txt_out not in written
    assert txt_out.read_text(encoding="utf-8") == existing
