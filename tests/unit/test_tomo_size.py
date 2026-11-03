import pytest

from zarr_particle_tools.core.helpers import get_tomo_size_at_tiltseries_pixel_size


@pytest.mark.parametrize(
    "binned_size, voxel_spacing, tilt_pixel_size, is_z, expected",
    [
        # stock RELION reconstruct (--w 4000 --h 4000 --d 2000 --binned_angpix 10): 540 x 540 x 270 at 10 A, tilt pixel 1.35 A
        (540, 10.0, 1.35, False, 4000),
        (270, 10.0, 1.35, True, 2000),
        # same, tilt pixel 1.54 A: 616 x 616 x 308
        (616, 10.0, 1.54, False, 4000),
        (308, 10.0, 1.54, True, 2000),
        # 10440 / run 16463: 630 x 630 x 184 at 10.012 A, tilt pixel 1.54 A
        (630, 10.012, 1.54, False, 4096),
        (184, 10.012, 1.54, True, 1198),
        # 10426 / run 16848: 1022 x 1440 x 400 at 8.66 A, tilt pixel 2.165 A
        (1022, 8.66, 2.165, False, 4088),
        (1440, 8.66, 2.165, False, 5760),
        (400, 8.66, 2.165, True, 1600),
        # already bin 1
        (630, 10.0, 10.0, False, 630),
        (200, 10.0, 10.0, True, 200),
    ],
)
def test_tomo_size_at_tiltseries_pixel_size(binned_size, voxel_spacing, tilt_pixel_size, is_z, expected):
    assert get_tomo_size_at_tiltseries_pixel_size(binned_size, voxel_spacing / tilt_pixel_size, is_z=is_z) == expected


@pytest.mark.parametrize("binning", [2.0, 2.5, 3.0, 4.0, 10.012 / 1.54, 10.0 / 1.35, 8.0, 15.3])
def test_tomo_size_at_tiltseries_pixel_size_relion_round_trip(binning):
    """Even sizes that RELION's reconstruct (reconstruct_tomogram.cpp) bins back to the tomogram dims."""
    for n in range(1, 2000):
        w = get_tomo_size_at_tiltseries_pixel_size(n, binning)
        d = get_tomo_size_at_tiltseries_pixel_size(n, binning, is_z=True)
        assert w % 2 == 0 and d % 2 == 0
        assert int(w / binning + 0.5) == n  # w1 = w / spacing + 0.5
        assert int(d / binning) == n  # t1 = d / spacing
