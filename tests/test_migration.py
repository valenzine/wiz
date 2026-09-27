"""One-time copy of whiz-era config/cache dirs to their wiz locations."""

from __future__ import annotations

import shutil

import pytest

from wiz import cli
from wiz import config as cfg


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.delenv("WIZ_CONFIG_DIR", raising=False)
    old_cfg, old_cache = tmp_path / ".config/whiz", tmp_path / ".cache/whiz"
    new_cfg, new_cache = tmp_path / ".config/wiz", tmp_path / ".cache/wiz"
    (old_cfg / "speakers").mkdir(parents=True)
    (old_cfg / "config.toml").write_text('model = "turbo"\n', encoding="utf-8")
    (old_cfg / "speakers" / "Axel.json").write_text('{"name": "Axel"}', encoding="utf-8")
    (old_cache / "diarization").mkdir(parents=True)
    (old_cache / "diarization" / "model.onnx").write_bytes(b"model")
    monkeypatch.setattr(cfg, "LEGACY_CONFIG_DIR", old_cfg)
    monkeypatch.setattr(cfg, "LEGACY_CACHE_DIR", old_cache)
    monkeypatch.setattr(cfg, "CONFIG_DIR", new_cfg)
    monkeypatch.setattr(cfg, "CACHE_DIR", new_cache)
    return old_cfg, old_cache, new_cfg, new_cache


def test_copies_profiles_config_and_models_and_keeps_originals(dirs):
    old_cfg, old_cache, new_cfg, new_cache = dirs
    assert cfg.migrate_legacy_dirs() == [(old_cfg, new_cfg), (old_cache, new_cache)]
    assert (new_cfg / "speakers" / "Axel.json").read_text(encoding="utf-8") == '{"name": "Axel"}'
    assert (new_cfg / "config.toml").read_text(encoding="utf-8") == 'model = "turbo"\n'
    assert (new_cache / "diarization" / "model.onnx").read_bytes() == b"model"
    # Copied, not moved: an old whiz install still finds its data.
    assert (old_cfg / "speakers" / "Axel.json").exists()
    assert (old_cache / "diarization" / "model.onnx").exists()
    # Runs once: a second call is a no-op.
    assert cfg.migrate_legacy_dirs() == []


def test_never_touches_an_existing_wiz_dir(dirs):
    _, _, new_cfg, _ = dirs
    new_cfg.mkdir(parents=True)
    (new_cfg / "config.toml").write_text('model = "mine"\n', encoding="utf-8")
    cfg.migrate_legacy_dirs()
    assert (new_cfg / "config.toml").read_text(encoding="utf-8") == 'model = "mine"\n'
    assert not (new_cfg / "speakers").exists()


def test_explicit_config_dir_opts_out_of_config_migration(dirs, monkeypatch):
    old_cfg, old_cache, new_cfg, new_cache = dirs
    monkeypatch.setenv("WIZ_CONFIG_DIR", str(new_cfg))
    assert cfg.migrate_legacy_dirs() == [(old_cache, new_cache)]
    assert not new_cfg.exists()


def test_leftover_from_interrupted_copy_is_replaced(dirs):
    _, _, new_cfg, _ = dirs
    stale = new_cfg.with_name("wiz.migrating")
    stale.mkdir(parents=True)
    (stale / "half-copied").write_text("x", encoding="utf-8")
    cfg.migrate_legacy_dirs()
    assert (new_cfg / "speakers" / "Axel.json").exists()
    assert not (new_cfg / "half-copied").exists()
    assert not stale.exists()


def test_failed_copy_is_loud_and_leaves_no_partial_dir(dirs, monkeypatch):
    old_cfg, _, new_cfg, _ = dirs

    def disk_full(src, dst, **_kwargs):
        (dst / "partial").mkdir(parents=True)
        raise OSError("No space left on device")

    monkeypatch.setattr(shutil, "copytree", disk_full)
    with pytest.raises(RuntimeError, match="No space left.*Nothing was removed"):
        cfg.migrate_legacy_dirs()
    assert not new_cfg.exists()
    assert not new_cfg.with_name("wiz.migrating").exists()
    assert (old_cfg / "speakers" / "Axel.json").exists()


def test_main_migrates_before_running_the_command(dirs, monkeypatch, capsys):
    _, _, new_cfg, _ = dirs
    monkeypatch.setattr(cfg, "CONFIG_PATH", new_cfg / "config.toml")
    with pytest.raises(SystemExit) as exc:
        cli.main(["config", "show"])
    assert exc.value.code == 0
    out, err = capsys.readouterr()
    assert "Copied your whiz data" in err
    # The command itself already sees the migrated settings.
    assert "turbo" in out
    assert (new_cfg / "speakers" / "Axel.json").exists()
