"""One-time copy of whiz-era config/cache dirs to their wiz locations."""

from __future__ import annotations

import shutil
from pathlib import Path

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


def _temp_siblings(new):
    return sorted(new.parent.glob(f".{new.name}.migrating-*"))


def test_interrupted_copy_leaves_nothing_and_is_retried(dirs, monkeypatch):
    _, _, new_cfg, _ = dirs
    real_copytree = shutil.copytree

    def interrupted(src, dst, *args, **kwargs):
        (dst / "half-copied").write_text("x", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(shutil, "copytree", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cfg.migrate_legacy_dirs()
    assert not new_cfg.exists() and _temp_siblings(new_cfg) == []

    monkeypatch.setattr(shutil, "copytree", real_copytree)
    cfg.migrate_legacy_dirs()
    assert (new_cfg / "speakers" / "Axel.json").exists()
    assert not (new_cfg / "half-copied").exists()


def test_concurrent_run_that_finished_first_is_not_an_error(dirs, monkeypatch):
    old_cfg, old_cache, new_cfg, new_cache = dirs
    real_copytree = shutil.copytree

    def other_wiz_wins(src, dst, *args, **kwargs):
        real_copytree(src, dst, *args, **kwargs)
        if src == old_cfg:  # another wiz command completes the same copy meanwhile
            real_copytree(src, new_cfg)

    monkeypatch.setattr(shutil, "copytree", other_wiz_wins)
    assert cfg.migrate_legacy_dirs() == [(old_cache, new_cache)]
    assert (new_cfg / "speakers" / "Axel.json").exists()
    assert _temp_siblings(new_cfg) == []


def test_notice_for_a_finished_copy_survives_a_later_failure(dirs, monkeypatch):
    old_cfg, old_cache, new_cfg, _ = dirs
    real_copytree = shutil.copytree
    copied = []

    def cache_fails(src, dst, *args, **kwargs):
        if src == old_cache:
            raise OSError("Permission denied")
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copytree", cache_fails)
    with pytest.raises(RuntimeError, match="Permission denied"):
        cfg.migrate_legacy_dirs(on_copied=lambda old, new: copied.append((old, new)))
    assert copied == [(old_cfg, new_cfg)]


def test_old_whiz_config_dir_override_is_migrated(tmp_path):
    import os
    import subprocess
    import sys

    env = {**os.environ, "WHIZ_CONFIG_DIR": str(tmp_path / "custom-whiz")}
    env.pop("WIZ_CONFIG_DIR", None)
    out = subprocess.run(
        [sys.executable, "-c", "from wiz import config; print(config.LEGACY_CONFIG_DIR)"],
        env=env, capture_output=True, text=True, check=True,
        cwd=Path(__file__).resolve().parent.parent,
    ).stdout.strip()
    assert out == str(tmp_path / "custom-whiz")


def test_failed_copy_is_loud_and_leaves_no_partial_dir(dirs, monkeypatch):
    old_cfg, _, new_cfg, _ = dirs

    def disk_full(src, dst, **_kwargs):
        (dst / "partial").mkdir(parents=True)
        raise OSError("No space left on device")

    monkeypatch.setattr(shutil, "copytree", disk_full)
    with pytest.raises(RuntimeError, match="No space left.*Nothing was removed.*mkdir -p"):
        cfg.migrate_legacy_dirs()
    assert not new_cfg.exists()
    assert _temp_siblings(new_cfg) == []
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
