"""One-time copy of whiz-era config/cache dirs to their wiz locations."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from wiz import cli
from wiz import config as cfg


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    # Independent of the caller's shell: no explicit config dir in effect.
    monkeypatch.setattr(cfg, "_EXPLICIT_CONFIG_DIR", None)
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


def _migrate():
    return list(cfg.migrate_legacy_dirs())


def test_copies_profiles_config_and_models_and_keeps_originals(dirs):
    old_cfg, old_cache, new_cfg, new_cache = dirs
    assert _migrate() == [(old_cfg, new_cfg), (old_cache, new_cache)]
    assert (new_cfg / "speakers" / "Axel.json").read_text(encoding="utf-8") == '{"name": "Axel"}'
    assert (new_cfg / "config.toml").read_text(encoding="utf-8") == 'model = "turbo"\n'
    assert (new_cache / "diarization" / "model.onnx").read_bytes() == b"model"
    # Copied, not moved: an old whiz install still finds its data.
    assert (old_cfg / "speakers" / "Axel.json").exists()
    assert (old_cache / "diarization" / "model.onnx").exists()
    # Runs once: a second call is a no-op.
    assert _migrate() == []


def test_never_touches_an_existing_wiz_dir(dirs):
    _, _, new_cfg, _ = dirs
    new_cfg.mkdir(parents=True)
    (new_cfg / "config.toml").write_text('model = "mine"\n', encoding="utf-8")
    _migrate()
    assert (new_cfg / "config.toml").read_text(encoding="utf-8") == 'model = "mine"\n'
    assert not (new_cfg / "speakers").exists()


def test_explicit_config_dir_opts_out_of_config_migration(dirs, monkeypatch):
    _, old_cache, new_cfg, new_cache = dirs
    monkeypatch.setattr(cfg, "_EXPLICIT_CONFIG_DIR", str(new_cfg))
    assert _migrate() == [(old_cache, new_cache)]
    assert not new_cfg.exists()


def _config_dir_with_env(**env_overrides):
    env = {k: v for k, v in os.environ.items() if k not in ("WIZ_CONFIG_DIR", "WHIZ_CONFIG_DIR")}
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", "from wiz import config; print(config.CONFIG_DIR, config._EXPLICIT_CONFIG_DIR)"],
        env=env, capture_output=True, text=True, check=True,
        cwd=Path(__file__).resolve().parent.parent,
    ).stdout.split()


def test_explicit_config_dir_is_used_as_is(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    assert _config_dir_with_env(WIZ_CONFIG_DIR=a) == [a, a]
    # The whiz-era variable keeps a custom location working...
    assert _config_dir_with_env(WHIZ_CONFIG_DIR=b) == [b, b]
    # ...but the new name wins when both are set.
    assert _config_dir_with_env(WIZ_CONFIG_DIR=a, WHIZ_CONFIG_DIR=b) == [a, a]
    assert _config_dir_with_env()[1] == "None"


def test_temp_dir_left_by_a_killed_run_is_replaced(dirs):
    _, _, new_cfg, _ = dirs
    stale = new_cfg.with_name(".wiz.migrating")
    stale.mkdir(parents=True)
    (stale / "half-copied").write_text("x", encoding="utf-8")
    _migrate()
    assert (new_cfg / "speakers" / "Axel.json").exists()
    assert not (new_cfg / "half-copied").exists()
    assert not stale.exists()


def test_interrupted_copy_leaves_nothing_and_is_retried(dirs, monkeypatch):
    _, _, new_cfg, _ = dirs
    real_copytree = shutil.copytree

    def interrupted(src, dst, *args, **kwargs):
        Path(dst).mkdir(parents=True)
        (Path(dst) / "half-copied").write_text("x", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(shutil, "copytree", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _migrate()
    assert not new_cfg.exists() and not new_cfg.with_name(".wiz.migrating").exists()

    monkeypatch.setattr(shutil, "copytree", real_copytree)
    _migrate()
    assert (new_cfg / "speakers" / "Axel.json").exists()


def test_failed_copy_is_loud_and_leaves_no_partial_dir(dirs, monkeypatch):
    old_cfg, _, new_cfg, _ = dirs

    def disk_full(src, dst, *args, **kwargs):
        (Path(dst) / "partial").mkdir(parents=True)
        raise OSError("No space left on device")

    monkeypatch.setattr(shutil, "copytree", disk_full)
    with pytest.raises(RuntimeError, match="No space left.*Nothing was removed.*mkdir -p"):
        _migrate()
    assert not new_cfg.exists()
    assert not new_cfg.with_name(".wiz.migrating").exists()
    assert (old_cfg / "speakers" / "Axel.json").exists()


def test_run_that_waited_for_another_wiz_does_not_copy_again(dirs, monkeypatch):
    old_cfg, _, new_cfg, _ = dirs
    real_copytree = shutil.copytree
    copied = []

    @contextlib.contextmanager
    def other_wiz_finished_while_waiting(directory):
        if directory == new_cfg.parent:
            real_copytree(old_cfg, new_cfg)
        yield

    monkeypatch.setattr(cfg, "_exclusive_lock", other_wiz_finished_while_waiting)
    monkeypatch.setattr(shutil, "copytree", lambda src, *a, **k: copied.append(src) or real_copytree(src, *a, **k))
    pairs = _migrate()
    assert old_cfg not in copied
    assert [old for old, _ in pairs] == [cfg.LEGACY_CACHE_DIR]


def test_copy_is_reported_before_a_later_failure(dirs, monkeypatch):
    old_cfg, old_cache, new_cfg, _ = dirs
    real_copytree = shutil.copytree

    def cache_fails(src, dst, *args, **kwargs):
        if src == old_cache:
            raise OSError("Permission denied")
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copytree", cache_fails)
    reported = []
    with pytest.raises(RuntimeError, match="Permission denied"):
        for pair in cfg.migrate_legacy_dirs():
            reported.append(pair)
    assert reported == [(old_cfg, new_cfg)]


def test_lock_held_by_another_process_excludes_us(tmp_path):
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl, os, sys; fd = os.open(sys.argv[1], os.O_RDONLY); "
         "fcntl.flock(fd, fcntl.LOCK_EX); print('held', flush=True); sys.stdin.readline()",
         str(tmp_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        acquired = threading.Event()

        def take_lock():
            with cfg._exclusive_lock(tmp_path):
                acquired.set()

        waiter = threading.Thread(target=take_lock, daemon=True)
        waiter.start()
        # Still blocked while the other process holds the lock. (A slow machine
        # only makes this wait longer; it can't make a working lock fail it.)
        assert not acquired.wait(0.2)
    finally:
        holder.stdin.write("release\n")
        holder.stdin.flush()
        holder.wait()
    waiter.join(5)
    assert acquired.is_set()  # and it gets the lock once the holder is gone


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
