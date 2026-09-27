# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.16.0] - 2026-09-27

### Changed

- Renamed to **wiz**: the command is `wiz` (there is no `whiz` alias), the Python
  package is `wiz`, settings live in `~/.config/wiz` (override with
  `WIZ_CONFIG_DIR`, formerly `WHIZ_CONFIG_DIR`) and diarization models in
  `~/.cache/wiz/diarization`.
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

Install wiz with `pipx install git+https://github.com/valenzine/wiz.git`. pipx
treats it as a new package, so the old `whiz` command stays until you run
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
