"""Unit tests for the shm-dir safeguards (fs-type detection, space preflight, cleanup registry)."""

from pathlib import Path

import numpy as np
import pandas as pd
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


def test_peak_budget_includes_temporary_s3_sources():
    outputs = [10, 20, 30]
    sources = [1, 2, 3]
    assert srj._peak_stage_bytes(outputs, concurrency=2, all_at_once=False, source_bytes=sources) == 55
    assert srj._peak_stage_bytes(outputs, concurrency=1, all_at_once=True, source_bytes=sources) == 63


def test_slug_prevents_path_escape_and_normalization_collisions():
    escaped = srj._slug("../outside")
    assert "/" not in escaped and escaped not in ("", ".", "..")
    assert srj._slug("A/B") != srj._slug("A_B")


def test_cleanup_shm_unlinks_registered_files(tmp_path):
    f = tmp_path / "zpt_dummy.mrc"
    f.write_bytes(b"x")
    srj._ACTIVE_SHM.add(str(f))
    srj._cleanup_shm()
    assert not f.exists()
    assert str(f) not in srj._ACTIVE_SHM


class _FakeReader:
    def __init__(self, shape=(1, 2, 2)):
        self.data = np.zeros(shape, dtype=np.float32)
        self.locator = "s3://bucket/source.mrc"
        self.closed = False

    def close(self):
        self.closed = True


def _one_tomogram(reader, micrograph_name):
    global_df = pd.DataFrame(
        {
            "rlnTomoName": ["../unsafe"],
            "rlnTomoTiltSeriesStarFile": ["unused.star"],
            "rlnTomoTiltSeriesPixelSize": [1.5],
        }
    )
    indiv = pd.DataFrame({"rlnMicrographName": [micrograph_name]})
    return global_df, indiv, reader


def test_materialize_validation_failure_closes_staged_reader(tmp_path, monkeypatch):
    reader = _FakeReader()
    global_df, indiv, _ = _one_tomogram(reader, "2@source.mrc")
    monkeypatch.setattr(srj, "_indiv_reader", lambda *a, **k: (indiv, reader))

    with pytest.raises(ValueError, match="outside stack range"):
        srj._materialize_tiltseries(global_df, tmp_path, tmp_path / "stage", tmp_path / "shm", None)

    assert reader.closed
    assert not list((tmp_path / "shm").glob("zpt_*.mrc"))


def test_materialize_write_failure_removes_partial_output(tmp_path, monkeypatch):
    reader = _FakeReader()
    global_df, indiv, _ = _one_tomogram(reader, "1@source.mrc")
    monkeypatch.setattr(srj, "_indiv_reader", lambda *a, **k: (indiv, reader))

    def fail_after_partial_write(_reader, out_path, **_):
        Path(out_path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(srj, "write_tiltseries_to_mrc", fail_after_partial_write)

    with pytest.raises(OSError, match="disk full"):
        srj._materialize_tiltseries(global_df, tmp_path, tmp_path / "stage", tmp_path / "shm", None)

    assert reader.closed
    assert not list((tmp_path / "shm").glob("zpt_*.mrc"))
    assert not any(str(tmp_path / "shm") in path for path in srj._ACTIVE_SHM)
