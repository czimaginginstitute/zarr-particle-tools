"""Unit tests for the shm-dir safeguards (fs-type detection, space preflight, cleanup registry)."""

from pathlib import Path

import pytest

from zarr_particle_tools import subtomo_relion_job as srj


def test_fs_type_detects_disk_vs_tmpfs(tmp_path):
    # tmp_path is on a real filesystem (disk/network), never tmpfs/ramfs
    assert srj._fs_type(tmp_path) not in ("tmpfs", "ramfs")
    if Path("/dev/shm").is_dir():
        assert srj._fs_type(Path("/dev/shm")) == "tmpfs"


def test_preflight_raises_when_stack_wont_fit(tmp_path):
    with pytest.raises(RuntimeError, match="Not enough space"):
        srj._preflight_budget(tmp_path, [10**18], concurrency=1, all_at_once=True)


def test_cleanup_shm_unlinks_registered_files(tmp_path):
    f = tmp_path / "zpt_dummy.mrc"
    f.write_bytes(b"x")
    srj._ACTIVE_SHM.add(str(f))
    srj._cleanup_shm()
    assert not f.exists()
    assert str(f) not in srj._ACTIVE_SHM
