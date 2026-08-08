"""Unit tests for the data reader, staging fallback, and MRC streamer."""

import logging
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr

from zarr_particle_tools.core import data as data_module
from zarr_particle_tools.core.data import (
    DataReader,
    read_s3_mrc_metadata,
    resolve_staging_dir,
    write_tiltseries_to_mrc,
)


class _LocalFileS3:
    """Small s3fs stand-in backed by one local file."""

    def __init__(self, source):
        self.source = source

    def size(self, _):
        return self.source.stat().st_size

    def open(self, _, mode):
        return self.source.open(mode)


def test_non_square_stack_roundtrip(tmp_path):
    # ny != nx: guards axis-order bugs in the (section, y, x) -> MRC path
    rng = np.random.default_rng(1)
    stack = rng.standard_normal((5, 30, 48)).astype(np.float32)  # nz=5, ny=30, nx=48
    zpath = tmp_path / "ts.zarr"
    zarr.save_array(str(zpath), stack, chunks=(2, 16, 16))
    reader = DataReader(str(zpath))

    full = reader.read_full_stack()
    assert full.shape == (5, 30, 48) and full.dtype == np.float32 and full.flags["C_CONTIGUOUS"]
    assert np.array_equal(full, stack)

    mrc_path = tmp_path / "ts.mrc"
    write_tiltseries_to_mrc(reader, mrc_path, voxel_size=2.0)
    with mrcfile.open(str(mrc_path)) as m:
        assert m.data.shape == (5, 30, 48)  # header nz,ny,nx correct
        assert np.array_equal(m.data, stack)


def test_float16_source_casts_and_warns(tmp_path, caplog):
    rng = np.random.default_rng(2)
    stack16 = rng.standard_normal((4, 20, 20)).astype(np.float16)
    zpath = tmp_path / "ts16.zarr"
    zarr.save_array(str(zpath), stack16, chunks=(2, 10, 10))
    reader = DataReader(str(zpath))

    with caplog.at_level(logging.WARNING):
        full = reader.read_full_stack()
    assert full.dtype == np.float32
    assert np.array_equal(full, stack16.astype(np.float32))
    assert any("float16" in r.getMessage() for r in caplog.records)
    caplog.clear()  # expected warning; clear so the autouse warning-guard fixture doesn't fail


def test_s3_mrc_is_staged_mapped_and_cleaned_up(tmp_path, monkeypatch):
    stack = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
    source = tmp_path / "source.mrc"
    with mrcfile.new(source, overwrite=True) as mrc:
        mrc.set_data(stack)

    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    fake_s3 = _LocalFileS3(source)
    selected = {}
    monkeypatch.setattr(DataReader, "_get_s3fs", lambda _: fake_s3)
    monkeypatch.setattr(
        data_module,
        "resolve_staging_dir",
        lambda preferred, required_bytes: selected.update(preferred=preferred, required=required_bytes) or stage_root,
    )

    reader = DataReader("s3://example/tilt-series.mrc", staging_dir=stage_root)
    assert np.array_equal(reader.data, stack)
    assert selected == {"preferred": stage_root, "required": source.stat().st_size}
    staged_dirs = list(stage_root.glob("zpt-s3-mrc-*"))
    assert len(staged_dirs) == 1
    assert (staged_dirs[0] / "tilt-series.mrc").exists()

    reader.close()
    assert not staged_dirs[0].exists()


def test_s3_mrc_metadata_reads_header_without_pixels(tmp_path, monkeypatch):
    stack = np.zeros((3, 5, 7), dtype=np.float32)
    source = tmp_path / "source.mrc"
    with mrcfile.new(source, overwrite=True) as mrc:
        mrc.set_data(stack)
    fake_s3 = _LocalFileS3(source)
    monkeypatch.setattr(data_module, "global_fs", fake_s3)

    shape, object_bytes = read_s3_mrc_metadata("s3://example/tilt-series.mrc")

    assert shape == stack.shape
    assert object_bytes == source.stat().st_size


def test_s3_mrc_metadata_reports_authenticated_access_failure(monkeypatch):
    denied = type("DeniedS3", (), {"open": lambda *a, **k: (_ for _ in ()).throw(PermissionError("403"))})()
    monkeypatch.setattr(data_module, "global_fs", denied)
    monkeypatch.setattr(data_module, "_s3_anon", False)

    with pytest.raises(RuntimeError, match="authenticated access"):
        read_s3_mrc_metadata("s3://private/tilt-series.mrc")


def test_staging_dir_falls_back_when_preferred_is_not_writable(tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    preferred = blocker / "shm"
    fallback = tmp_path / "system-temp"
    monkeypatch.setattr(data_module.tempfile, "gettempdir", lambda: str(fallback))

    assert resolve_staging_dir(preferred, required_bytes=1) == fallback
    assert fallback.is_dir()
    assert not list(fallback.glob(".zpt-write-test-*"))


def test_staging_dir_falls_back_when_preferred_lacks_space(tmp_path, monkeypatch):
    preferred = tmp_path / "shm"
    fallback = tmp_path / "system-temp"
    real_disk_usage = data_module.shutil.disk_usage

    def disk_usage(path):
        if Path(path) == preferred:
            return real_disk_usage(path)._replace(free=1)
        return real_disk_usage(path)

    monkeypatch.setattr(data_module.tempfile, "gettempdir", lambda: str(fallback))
    monkeypatch.setattr(data_module.shutil, "disk_usage", disk_usage)

    assert resolve_staging_dir(preferred, required_bytes=2) == fallback


def test_staging_dir_fails_when_preferred_and_fallback_are_unusable(tmp_path, monkeypatch):
    preferred_blocker = tmp_path / "preferred-file"
    fallback_blocker = tmp_path / "fallback-file"
    preferred_blocker.write_text("x")
    fallback_blocker.write_text("x")
    monkeypatch.setattr(data_module.tempfile, "gettempdir", lambda: str(fallback_blocker / "child"))

    with pytest.raises(RuntimeError, match="No usable staging directory"):
        resolve_staging_dir(preferred_blocker / "child", required_bytes=1)
