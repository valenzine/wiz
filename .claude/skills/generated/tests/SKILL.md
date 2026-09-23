---
name: tests
description: "Skill for the Tests area of whiz. 131 symbols across 7 files."
---

# Tests

131 symbols | 7 files | Cohesion: 87%

## When to Use

- Working with code in `tests/`
- Understanding how test_assign_speakers_max_overlap, test_assign_speakers_no_diar, test_speakers_by_talk_time_orders_most_first work
- Modifying tests-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `tests/test_cli.py` | _transcribe_args, _setup_transcribe, test_transcribe_html_fallback_when_diarization_unavailable, test_transcribe_html_without_speakers, test_transcribe_html_and_frames_fallback_for_video (+66) |
| `tests/test_models.py` | _make_isolated_config, test_resolve_turbo_short_alias, test_pick_best_prefers_unquantized_over_quantized, test_pick_best_quantized_only_as_last_resort, test_pick_best_empty_returns_none (+18) |
| `tests/test_merge.py` | _seg, test_assign_speakers_max_overlap, test_assign_speakers_no_diar, test_speakers_by_talk_time_orders_most_first, test_speakers_in_order_of_appearance (+15) |
| `tests/test_diarize_cache.py` | test_diar_cache_path_uses_string_append, test_diar_cache_round_trip, test_diar_cache_miss_on_param_mismatch, test_diar_cache_missing_file_returns_none, test_diar_cache_threshold_epsilon_tolerance (+5) |
| `whiz/diarize.py` | diar_cache_path, load_diarization_cache, _write_diarization_cache |
| `tests/test_ai.py` | _mock_response, fake_urlopen |
| `tests/test_screenshots.py` | _seg, test_extract_frames_dry_run_names_entries |

## Entry Points

Start here when exploring this area:

- **`test_assign_speakers_max_overlap`** (Function) — `tests/test_merge.py:28`
- **`test_assign_speakers_no_diar`** (Function) — `tests/test_merge.py:45`
- **`test_speakers_by_talk_time_orders_most_first`** (Function) — `tests/test_merge.py:52`
- **`test_speakers_in_order_of_appearance`** (Function) — `tests/test_merge.py:63`
- **`test_relabel_replaces_names`** (Function) — `tests/test_merge.py:73`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `test_assign_speakers_max_overlap` | Function | `tests/test_merge.py` | 28 |
| `test_assign_speakers_no_diar` | Function | `tests/test_merge.py` | 45 |
| `test_speakers_by_talk_time_orders_most_first` | Function | `tests/test_merge.py` | 52 |
| `test_speakers_in_order_of_appearance` | Function | `tests/test_merge.py` | 63 |
| `test_relabel_replaces_names` | Function | `tests/test_merge.py` | 73 |
| `test_representative_quotes_picks_longest` | Function | `tests/test_merge.py` | 80 |
| `test_format_labeled_srt_structure` | Function | `tests/test_merge.py` | 92 |
| `test_format_dialogue_txt_merges_consecutive` | Function | `tests/test_merge.py` | 102 |
| `test_format_speakers_html_basic` | Function | `tests/test_merge.py` | 151 |
| `test_format_speakers_html_no_frames_when_dir_missing` | Function | `tests/test_merge.py` | 163 |
| `test_format_speakers_html_has_legend_and_search` | Function | `tests/test_merge.py` | 172 |
| `test_format_speakers_html_frame_clickable_opens_lightbox` | Function | `tests/test_merge.py` | 185 |
| `test_format_speakers_html_no_lightbox_without_frames` | Function | `tests/test_merge.py` | 205 |
| `test_format_speakers_html_note_renders_muted_and_escaped` | Function | `tests/test_merge.py` | 215 |
| `test_format_speakers_html_no_note_by_default` | Function | `tests/test_merge.py` | 224 |
| `test_assign_speakers_zero_overlap_uses_nearest_not_first` | Function | `tests/test_merge.py` | 233 |
| `test_assign_speakers_zero_overlap_warns_on_stderr` | Function | `tests/test_merge.py` | 246 |
| `test_assign_speakers_no_warning_when_all_overlap` | Function | `tests/test_merge.py` | 256 |
| `test_assign_speakers_zero_overlap_tie_breaks_earlier` | Function | `tests/test_merge.py` | 264 |
| `test_transcribe_html_fallback_when_diarization_unavailable` | Function | `tests/test_cli.py` | 237 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Run_diarization → Diar_cache_path` | cross_community | 3 |

## How to Explore

1. `context({name: "test_assign_speakers_max_overlap"})` — see callers and callees
2. `query({search_query: "tests"})` — find related execution flows
3. Read key files listed above for implementation details
4. `explain({target: "<file or symbol>"})` — persisted taint findings (source→sink data flows), when indexed with `--pdg`
