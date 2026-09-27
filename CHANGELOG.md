# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.17.1] - 2026-09-27

### Fixed

- When I explicitly request speaker diarization and it cannot produce real speaker
  labels, `transcribe` and `merge` now exit with an error even if they wrote
  generic-label outputs. Video's automatic diarization still degrades with a hint.

## [0.17.0] - 2026-09-27

### Changed

- Speaker diarization is about twice as fast by default: the window shift is now
  0.2 instead of 0.1. On a 50-minute, two-speaker episode, diarization took about
  3:50 instead of 7:46, and the share of words given to the wrong speaker was
  2.83% instead of 2.99% (0.25 was faster still, but less accurate at 3.17%).
  To get the old behaviour, run `wiz config set diarization_window_shift=0.1`.
- Diarization now needs sherpa-onnx 1.13.6 or newer. An older one is treated like
  a missing one: speaker labels are skipped with a warning that says how to upgrade:
  `pipx inject --force transcript-wiz 'sherpa-onnx>=1.13.6'`. The install hints
  and `wiz upgrade` now pass `--force`, since without it pipx leaves an older
  sherpa-onnx in place.
- Diarization caches written with the old default are recomputed once.

## [0.16.0] - 2026-09-27

### Changed

- Renamed to **wiz**: the command is `wiz` (there is no `whiz` alias), the Python
  package is `wiz`, settings live in `~/.config/wiz` (override with
  `WIZ_CONFIG_DIR`; the old `WHIZ_CONFIG_DIR` still works, and a custom
  location set either way is used as-is, not copied) and diarization models in
  `~/.cache/wiz/diarization`.
- The package is published as `transcript-wiz`, because `wiz` is taken on PyPI
  by an unrelated project. pipx lists it under that name, and `pipx inject` /
  `pipx uninstall` use it; the command is still `wiz`.
- New logo, and a wave instead of a lightning bolt in the terminal header.

### Added

- The first `wiz` command copies `~/.config/whiz` (settings and voice profiles)
  and `~/.cache/whiz` (diarization models) to their new locations. The originals
  are never modified, a location that already exists is never overwritten, and
  if the copy fails wiz stops with an error rather than starting with empty
  settings.

### Removed

- `wiz dictate` (alias `d`), which only pointed to upstream's separate dictation
  app, and the docs about it. Leftover `dictate_*` keys in an old config file are
  still preserved untouched.

### Upgrading

- Diarization caches written before this release don't match anymore, because
  the cache records where the models live and they moved to `~/.cache/wiz`.
  The first run on each recording diarizes again and writes a new cache.
- Running the old `whiz upgrade` installs `transcript-wiz` without the
  diarization library. wiz offers to install it the first time a run needs it,
  or install it yourself with `pipx inject transcript-wiz 'sherpa-onnx>=1.10'`.

Install wiz with `pipx install git+https://github.com/valenzine/wiz.git`. pipx
treats it as a new package (`transcript-wiz`), so the old `whiz` command stays until you run
`pipx uninstall whiz`.

## [0.15.0] - 2026-09-27

First release of **wiz**, my fork of
[whiz](https://github.com/ReidenXerx/whiz) by ReidenXerx (MIT).

### Added

- `--diarization-provider {cpu,coreml}` and `--diarization-threads N` on
  `transcribe`, `merge` and `speakers match`, with matching config keys
  `diarization_provider` (default `cpu`) and `diarization_threads` (default `1`).
  The defaults keep the previous behavior.
- `--diarization-window-shift` and the `diarization_window_shift` config key
  (0 < x <= 1, default `0.1`). Larger values make Pyannote segmentation faster
  but coarser. Values other than `0.1` need sherpa-onnx >= 1.13.6.
- Diarization and speaker-profile embedding runs now report how long they took,
  and show the provider, thread count and window shift in use.

### Fixed

- Speaker diarization now works on MP3 and other audio that isn't 16 kHz mono
  16-bit WAV. `transcribe`, `merge` and `speakers match` convert it to a
  temporary WAV for sherpa-onnx, never overwriting the source or an existing
  WAV. The temporary file is removed afterwards, including on errors and Ctrl-C.
- Whisper output for converted audio is named after the source
  (`recording.mp3.json`), not the temporary WAV.
- An unknown `--outputs` format or a missing whisper-cli now fails before any
  audio is extracted or converted, instead of leaving a stray WAV behind.
- The diarization cache for converted audio is keyed on the original file, so
  re-running `merge` or `transcribe --resume` reuses it.

### Changed

- Project identity: the license adds the fork's copyright line (the original is
  kept), and the README, package metadata and `whiz upgrade` point to
  `github.com/valenzine/wiz`. Upstream's website page (`docs/index.html`) is removed.
- The diarization cache also records the window shift. Caches written by
  earlier versions are recomputed once.
