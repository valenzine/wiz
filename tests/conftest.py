"""pytest config — ensure the wiz package is importable from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path

# Insert repo root so `import wiz` works without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture(autouse=True)
def _no_real_legacy_migration(tmp_path_factory, monkeypatch):
    """Never let a test copy the real ~/.config/whiz or ~/.cache/whiz."""
    from wiz import config as cfg

    empty = tmp_path_factory.mktemp("no-legacy")
    monkeypatch.setattr(cfg, "LEGACY_CONFIG_DIR", empty / "config-whiz")
    monkeypatch.setattr(cfg, "LEGACY_CACHE_DIR", empty / "cache-whiz")
