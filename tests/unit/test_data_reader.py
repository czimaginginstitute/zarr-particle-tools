"""Unit tests for the zarr full-stack accessor + MRC streamer (non-square, float16 sources)."""

import logging

import mrcfile
import numpy as np
import zarr

from zarr_particle_tools.core.data import DataReader, write_tiltseries_to_mrc


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
