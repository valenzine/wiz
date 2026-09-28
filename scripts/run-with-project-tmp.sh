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
  # Resolve the binary and the analyze --name alias the same way the JS callers do.
  shift
  cd "$ROOT"
  exec node "$ROOT/.bearing/lib/gitnexus-cmd.mjs" --exec "$@"
fi
exec "$@"
