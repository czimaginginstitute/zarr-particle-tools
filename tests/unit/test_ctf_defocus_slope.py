"""
Unit tests for the per-tomogram defocus slope (rlnTomoDefocusSlope).

RELION's `Tomogram::getCtf` offsets the central defocus by
`dz = handedness * pixelSize * defocusSlope * depthOffset`, and `TomogramSet::read` defaults
`defocusSlope` to 1.0 when the column is absent. Our `calculate_ctf` applies
`handedness * defocus_slope * depth_offset` (the pixel-size scaling is already carried by the
projection matrix, which is why extraction matches RELION at float32-ULP level).
"""

import numpy as np

from zarr_particle_tools.core.ctf import calculate_ctf, get_depth_offset
from zarr_particle_tools.core.forwardprojection import calculate_projection_matrix

BOX = 8
PIXEL = 2.0
BIN = 1
DEFOCUS = 20000.0
HANDEDNESS = 1
# off-centre and tilted, so the depth offset is non-zero (a centred particle at 0 tilt has none)
COORDINATE = np.array([12.0, -7.0, 25.0])
TILT = 30.0


def _pm() -> np.ndarray:
    return calculate_projection_matrix(rot=0.0, gmag=1.0, tx=0.0, ty=0.0, tilt=TILT)


def _ctf(defocus_slope=None, defocus_u=DEFOCUS, defocus_v=DEFOCUS) -> np.ndarray:
    kwargs = dict(
        coordinate=COORDINATE,
        tilt_projection_matrix=_pm(),
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        handedness=HANDEDNESS,
        tiltseries_pixel_size=PIXEL,
        phase_shift=0.0,
        defocus_u=defocus_u,
        defocus_v=defocus_v,
        defocus_angle=0.0,
        dose=0.0,
        ctf_scalefactor=1.0,
        bfactor=0.0,
        box_size=BOX,
        bin=BIN,
    )
    if defocus_slope is not None:
        kwargs["defocus_slope"] = defocus_slope
    return calculate_ctf(**kwargs)


def test_depth_offset_is_nonzero_for_this_geometry():
    """Guard the fixture: a zero depth offset would make every assertion below vacuous."""
    assert abs(get_depth_offset(_pm(), COORDINATE)) > 1e-6


def test_default_slope_is_one_and_bit_identical():
    np.testing.assert_array_equal(_ctf(), _ctf(defocus_slope=1.0))


def test_slope_scales_depth_offset_like_relion():
    # slope s is equivalent to slope 1 with DefocusU/V shifted by the extra (s-1) * handedness * dz
    slope = 2.5
    dz = get_depth_offset(_pm(), COORDINATE)
    extra = HANDEDNESS * (slope - 1.0) * dz
    np.testing.assert_allclose(
        _ctf(defocus_slope=slope),
        _ctf(defocus_u=DEFOCUS + extra, defocus_v=DEFOCUS + extra),
        atol=1e-12,
        rtol=1e-10,
    )


def test_slope_actually_changes_the_ctf():
    assert not np.allclose(_ctf(defocus_slope=1.0), _ctf(defocus_slope=3.0))


def test_zero_slope_removes_the_depth_dependence():
    # slope 0 kills the depth term, so an off-centre tilted particle matches the central CTF
    np.testing.assert_allclose(_ctf(defocus_slope=0.0), _ctf_central(), atol=1e-12, rtol=1e-10)


def _ctf_central() -> np.ndarray:
    """The same CTF with no depth offset at all (particle at the projection centre)."""
    return calculate_ctf(
        coordinate=np.array([0.0, 0.0, 0.0]),
        tilt_projection_matrix=_pm(),
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        handedness=HANDEDNESS,
        tiltseries_pixel_size=PIXEL,
        phase_shift=0.0,
        defocus_u=DEFOCUS,
        defocus_v=DEFOCUS,
        defocus_angle=0.0,
        dose=0.0,
        ctf_scalefactor=1.0,
        bfactor=0.0,
        box_size=BOX,
        bin=BIN,
    )
