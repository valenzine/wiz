---
name: whiz
description: "Skill for the Whiz area of whiz. 135 symbols across 9 files."
---

# Whiz

135 symbols | 9 files | Cohesion: 86%

## When to Use

- Working with code in `whiz/`
- Understanding how cmd_transcribe, cmd_merge, header work
- Modifying whiz-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `whiz/cli.py` | _outputs_include, _outputs_explicitly_include, _will_write_generic_labels, _run_whisper_streaming, _print_zero_segments_hints (+41) |
| `whiz/ai.py` | resolve_prompt_auto, transcript_text, _fmt_clock, chat_text, _frame_manifest (+17) |
| `whiz/merge.py` | _fmt_clock, format_dialogue_txt, flush, _speaker_color, speaker_palette (+11) |
| `whiz/ui.py` | _is_tty, header, rule, phase, status (+7) |
| `whiz/profiles.py` | load_profiles, cosine_similarity, match_speakers, auto_assign_names, compute_speaker_embeddings (+7) |
| `whiz/models.py` | _alias_from_name, _short_alias, discover, resolve, pick_best (+6) |
| `whiz/diarize.py` | _import_sherpa, find_embedding_model, run_diarization, _read_wav_pcm, _default_diarization_dir (+3) |
| `whiz/config.py` | to_dict, _escape_toml_string, _emit_toml, load, save |
| `whiz/screenshots.py` | _frame_name, extract_segment_frames, _extract_one |

## Entry Points

Start here when exploring this area:

- **`cmd_transcribe`** (Function) — `whiz/cli.py:1086`
- **`cmd_merge`** (Function) — `whiz/cli.py:1764`
- **`header`** (Function) — `whiz/ui.py:46`
- **`rule`** (Function) — `whiz/ui.py:76`
- **`phase`** (Function) — `whiz/ui.py:83`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `cmd_transcribe` | Function | `whiz/cli.py` | 1086 |
| `cmd_merge` | Function | `whiz/cli.py` | 1764 |
| `header` | Function | `whiz/ui.py` | 46 |
| `rule` | Function | `whiz/ui.py` | 76 |
| `phase` | Function | `whiz/ui.py` | 83 |
| `status` | Function | `whiz/ui.py` | 95 |
| `wrote` | Function | `whiz/ui.py` | 143 |
| `summary` | Function | `whiz/ui.py` | 182 |
| `spinner` | Function | `whiz/ui.py` | 217 |
| `write` | Function | `whiz/ui.py` | 249 |
| `format_dialogue_txt` | Function | `whiz/merge.py` | 204 |
| `flush` | Function | `whiz/merge.py` | 214 |
| `speaker_palette` | Function | `whiz/merge.py` | 248 |
| `format_speakers_html` | Function | `whiz/merge.py` | 267 |
| `speaker_label_line` | Function | `whiz/ui.py` | 162 |
| `tally` | Function | `whiz/ui.py` | 168 |
| `resolve_prompt_auto` | Function | `whiz/ai.py` | 439 |
| `transcript_text` | Function | `whiz/ai.py` | 478 |
| `chat_text` | Function | `whiz/ai.py` | 603 |
| `chunk_entries` | Function | `whiz/ai.py` | 644 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Cmd_transcribe → _diarization_available` | cross_community | 4 |
| `Cmd_transcribe → _auto_setup_consent` | cross_community | 4 |
| `Cmd_transcribe → _install_sherpa_onnx` | cross_community | 4 |
| `Cmd_transcribe → _outputs_explicitly_include` | intra_community | 4 |
| `Cmd_transcribe → _discarded_naming_flags` | intra_community | 4 |
| `Save_profile → Profiles_dir` | intra_community | 4 |
| `Auto_assign_names → Profiles_dir` | cross_community | 4 |
| `Cmd_transcribe → _video_auto_flags` | cross_community | 3 |
| `Cmd_transcribe → _auto_threads` | cross_community | 3 |
| `Cmd_transcribe → _find_whisper_cli` | cross_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Tests | 3 calls |

## How to Explore

1. `context({name: "cmd_transcribe"})` — see callers and callees
2. `query({search_query: "whiz"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
