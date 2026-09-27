# North-stars — wiz

Falsifiable propositions about what this project IS. This file outranks every
other doc (README, ARCHITECTURE.md, code comments) — on conflict, the
north-star wins and the other source is stale. Owned by the user; agents
propose diffs, never edit silently.

## Invariants — must always hold

- **NS-1** — Whisper models are used UNQUANTIZED, always and everywhere — quantization corrupts transcription quality. Within each class the unquantized variant ranks first; a `-q*` variant resolves only when its OWN class's unquantized model is absent (preference is per-class, never blocking behind unrelated classes). `tiny` is excluded from KNOWN_MODELS/PREFERENCE entirely. Quantized files stay runnable when named explicitly. — src: wiz/models.py, tests/test_models.py *(carried over as old NS-15, unchanged in substance)*
- **NS-2** — A voice-profile save knows its provenance: an auto-match may CREATE a profile (`source: "auto"`) but may never MERGE into an existing one; only human confirmation merges and upgrades `auto` → `user`. A profile is discarded, not averaged, when the embedding dimension changes. — src: wiz/profiles.py, tests/test_profiles.py *(the wave-1 M3 contract, now a wiz invariant)*
- **NS-3** — The diarization cache is keyed on the parameters that produced it AND a fingerprint of the input; a stale cache can never be reused against different audio or settings. — src: wiz/diarize.py, tests/test_diarize_cache.py *(wave-1 H1)*
- **NS-4** — A degraded diarization run is loud, never silent: explicit `--speakers` writes no speaker-labeled artifacts → nonzero exit; an explicitly passed `--outputs html` is never dropped; real-labeled outputs are never clobbered by a degraded re-run; `--speakers-names` are never silently discarded. — src: wiz/cli.py, wiz/merge.py, tests/test_cli.py
- **NS-5** — `save()` preserves every config key it does not know. Unset tri-state (`None`) is omitted from emitted TOML, never written. — src: wiz/config.py, tests/test_config.py
- **NS-6** — A "tests pass" claim names the exact command actually run; a bare claim is NOT evidence. — src: local toolchain state *(carried over as old NS-8)*

## Settled — decided, do not relitigate

- **NS-7** — wiz is a transcription CLI for recorded audio and video. Live dictation is not part of it. — src: README.md *(replaces upstream's product-split record; NS-8, which only pointed dictation's segmentation rules at upstream's separate app, is retired)*

## Graveyard — tried and rejected / validated

- **NS-9** — VALIDATED (carried): voice-profile provenance contract — it caught PR #5's Swift port re-introducing centroid drift through the shared profile store (PR closed 2026-09-22, obsolete post-split). Keep the pattern.
- **NS-10** — REJECTED: runtime tuning-file dependency (old NS-12) — moot for wiz (no tuning file), recorded so the pattern stays rejected if a cross-implementation contract ever returns.

## Open — explicitly unresolved (do NOT assume either way)

- (none)
