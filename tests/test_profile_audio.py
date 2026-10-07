"""Regression coverage for streaming voice-profile WAV reads."""

from __future__ import annotations

from array import array
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from wiz import config as cfg
from wiz import diarize as D
from wiz import profiles as P


def _write_pcm_wav(path: Path, samples: array, *, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(16_000)
        out.writeframes(samples.tobytes())


def _fake_embedding_runtime(monkeypatch, tmp_path, *, compute=None):
    """Install a fake extractor that keeps every accepted waveform."""
    model = tmp_path / "embedding.onnx"
    model.write_bytes(b"model")
    streams = []

    class FakeStream:
        def accept_waveform(self, sample_rate, samples):
            self.sample_rate = sample_rate
            self.samples = samples
            streams.append(self)

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

        def compute(self, stream):
            if compute:
                return compute(stream)
            return [float(len(stream.samples)), stream.samples[0]]

    monkeypatch.setattr(P, "find_embedding_model", lambda _config: model)
    monkeypatch.setattr(P, "_import_sherpa", lambda: SimpleNamespace(
        SpeakerEmbeddingExtractorConfig=lambda *_a, **_k: object(),
        SpeakerEmbeddingExtractor=FakeExtractor,
    ))
    return streams


class _WaveSpy:
    def __init__(self, reader):
        self._reader = reader
        self.read_sizes: list[int] = []
        self.closed = False

    def __enter__(self):
        self._reader.__enter__()
        return self

    def __exit__(self, *args):
        self.closed = True
        return self._reader.__exit__(*args)

    def readframes(self, frames):
        self.read_sizes.append(frames)
        return self._reader.readframes(frames)

    def __getattr__(self, name):
        return getattr(self._reader, name)


def _spy_wave_open(monkeypatch):
    original_open = wave.open
    readers: list[_WaveSpy] = []

    def open_spy(*args, **kwargs):
        reader = _WaveSpy(original_open(*args, **kwargs))
        readers.append(reader)
        return reader

    monkeypatch.setattr(wave, "open", open_spy)
    return readers


def test_profile_embedding_reads_ordered_clipped_ranges_in_bounded_blocks(tmp_path, monkeypatch):
    total_frames = 31 * 16_000
    samples = array("h", [0]) * total_frames
    samples[0] = 1000
    samples[480_000] = 2000
    samples[490_880] = 3000
    samples[491_040] = 4000
    wav = tmp_path / "episode.wav"
    _write_pcm_wav(wav, samples)
    streams = _fake_embedding_runtime(monkeypatch, tmp_path)
    readers = _spy_wave_open(monkeypatch)

    result = P.compute_speaker_embeddings(
        wav,
        [
            D.DiarSegment(-1.0, 0.5, 2),
            D.DiarSegment(0.0, 30.35, 1),
            D.DiarSegment(30.0, 31.0, 1),  # Overlap remains a separate block.
            D.DiarSegment(30.68, 31.0, 3),
            D.DiarSegment(30.69, 31.0, 4),
            D.DiarSegment(30.71, 31.0, 5),  # 0.29 s tail is skipped.
        ],
        cfg.Config(),
    )

    assert [len(stream.samples) for stream in streams] == [8_000, 480_000, 5_600, 16_000, 5_120, 4_960]
    assert readers[0].read_sizes == [8_000, 480_000, 5_600, 16_000, 5_120, 4_960]
    assert max(readers[0].read_sizes) == 480_000 < total_frames
    assert list(result) == [2, 1, 3, 4]
    assert result[1] == pytest.approx([167_200.0, (1000 + 2000 + 2000) / 3 / 32768.0])
    assert 5 not in result


def test_profile_embedding_stream_reader_preserves_stereo_downmix(tmp_path, monkeypatch):
    frames = array("h", [1000, -1000, 1000, 3000])
    frames.extend([0, 0] * (4_800 - 2))
    wav = tmp_path / "stereo.wav"
    _write_pcm_wav(wav, frames, channels=2)
    streams = _fake_embedding_runtime(monkeypatch, tmp_path)

    result = P.compute_speaker_embeddings(
        wav, [D.DiarSegment(0.0, 0.3, 0)], cfg.Config(),
    )

    assert streams[0].samples[:2] == pytest.approx([0.0, 2000 / 32768.0])
    assert result == {0: pytest.approx([4800.0, 0.0])}


def test_profile_embedding_uses_available_audio_when_wav_is_truncated(tmp_path, monkeypatch):
    wav = tmp_path / "truncated.wav"
    _write_pcm_wav(wav, array("h", [0]) * 16_000)
    data = wav.read_bytes()
    wav.write_bytes(data[: len(data) - 8_000 * 2])  # Header still claims 1 s.
    _fake_embedding_runtime(monkeypatch, tmp_path)

    result = P.compute_speaker_embeddings(
        wav,
        [D.DiarSegment(0.0, 1.0, 0), D.DiarSegment(0.6, 1.0, 1)],
        cfg.Config(),
    )

    assert result == {0: [8000.0, 0.0]}


@pytest.mark.parametrize("channels", [1, 2, 3])
def test_wav_reader_keeps_complete_frames_in_truncated_pcm(tmp_path, channels):
    frames = [
        [32767, -32768, 12345][:channels],
        [-32768, 32767, -23456][:channels],
        [1200, 3400, -5600][:channels],
    ]
    samples = array("h", [sample for frame in frames for sample in frame])
    samples.extend([11] * channels)
    wav = tmp_path / "partial-frame.wav"
    _write_pcm_wav(wav, samples, channels=channels)
    wav.write_bytes(wav.read_bytes()[:-1])  # Last frame is incomplete.

    audio, sample_rate = D._read_wav_pcm(wav)

    assert sample_rate == 16_000
    assert audio == [sum(frame) / channels / 32768.0 for frame in frames]


def test_profile_embedding_closes_wav_when_extractor_fails(tmp_path, monkeypatch):
    wav = tmp_path / "episode.wav"
    _write_pcm_wav(wav, array("h", [0]) * 4_800)
    _fake_embedding_runtime(
        monkeypatch, tmp_path, compute=lambda _stream: (_ for _ in ()).throw(RuntimeError("inference failed")),
    )
    readers = _spy_wave_open(monkeypatch)

    with pytest.raises(RuntimeError, match="inference failed"):
        P.compute_speaker_embeddings(wav, [D.DiarSegment(0.0, 0.3, 0)], cfg.Config())

    assert readers[0].closed


@pytest.mark.parametrize(
    ("sample_rate", "sample_width", "message"),
    [
        (8_000, 2, "Expected 16000 Hz audio, got 8000 Hz."),
        (16_000, 1, "Expected 16-bit PCM WAV, got sample_width=1"),
    ],
)
def test_profile_embedding_rejects_invalid_wav_headers(
    tmp_path, monkeypatch, sample_rate, sample_width, message
):
    wav = tmp_path / "invalid.wav"
    with wave.open(str(wav), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(sample_width)
        out.setframerate(sample_rate)
        out.writeframes(b"\0" * sample_width * 4_800)
    _fake_embedding_runtime(monkeypatch, tmp_path)

    with pytest.raises(RuntimeError, match=message):
        P.compute_speaker_embeddings(wav, [D.DiarSegment(0.0, 0.3, 0)], cfg.Config())
