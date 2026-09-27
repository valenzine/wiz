#!/usr/bin/env bash
# Run a command with TMPDIR on project disk (not tmpfs /tmp).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="${GITNEXUS_TMPDIR:-$ROOT/.tmp-agent}"
export TMPDIR="$TMP"
export TEMP="$TMP"
export TMP="$TMP"
mkdir -p "$TMPDIR"
if [[ "${1:-}" == "gitnexus" ]]; then
  shift
  read -r -a GITNEXUS_CMD <<< "$(node "$ROOT/.bearing/lib/gitnexus-cmd.mjs")"
  exec "${GITNEXUS_CMD[@]}" "$@"
fi
exec "$@"
