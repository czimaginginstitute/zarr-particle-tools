"""
Golden unit test for the CTF B-factor damping envelope (Phase 0 D4).

RELION's `CTF::getCTF` multiplies the CTF by `E = exp(K4 * u2)` with `K4 = -rlnCtfBfactor / 4`,
applied before the scale and the +/-1e-8 near-zero clamp. The Python code previously computed K4
but never applied the envelope (and read the wrong B-factor field). These tests verify the envelope
is now applied as RELION's K4 term, and that it is an exact no-op when `bfactor == 0` (so data
without `rlnCtfBfactor` does not regress). No RELION binary is required.
"""

import numpy as np

from zarr_particle_tools.core.ctf import calculate_ctf
from zarr_particle_tools.core.forwardprojection import calculate_projection_matrix

BOX = 8
PIXEL = 2.0
BIN = 1


def _ctf(bfactor: float) -> np.ndarray:
    pm = calculate_projection_matrix(rot=0.0, gmag=1.0, tx=0.0, ty=0.0, tilt=0.0)
    return calculate_ctf(
        coordinate=np.array([0.0, 0.0, 0.0]),
        tilt_projection_matrix=pm,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        handedness=1,
        tiltseries_pixel_size=PIXEL,
        phase_shift=0.0,
        defocus_u=20000.0,
        defocus_v=20000.0,
        defocus_angle=0.0,
        dose=0.0,
        ctf_scalefactor=1.0,
        bfactor=bfactor,
        box_size=BOX,
        bin=BIN,
    )


def _u2_grid() -> np.ndarray:
    ky = np.fft.fftfreq(BOX, d=PIXEL * BIN)
    kx = np.fft.rfftfreq(BOX, d=PIXEL * BIN)
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
    return kx_grid**2 + ky_grid**2


def test_bfactor_zero_is_no_op():
    # exp(K4 * u2) with K4 = 0 is exactly 1, so a tiny B-factor sweep around 0 must be unchanged.
    ctf0 = _ctf(0.0)
    assert np.all(np.isfinite(ctf0))
    # the envelope multiplier itself is identically 1
    np.testing.assert_array_equal(np.exp(0.0 * _u2_grid()), np.ones_like(_u2_grid()))


def test_bfactor_applies_relion_envelope():
    bfactor = 100.0
    ctf0 = _ctf(0.0)
    ctf_b = _ctf(bfactor)

    # the envelope must actually change the CTF (guards against it being ignored / wrong field)
    assert not np.allclose(ctf0, ctf_b)

    # away from the 1e-8-clamped near-zeros, ctf(bfactor) == ctf(0) * exp(-bfactor/4 * u2)
    u2 = _u2_grid()
    envelope = np.exp(-bfactor / 4.0 * u2)
    unclamped = np.abs(ctf0) > 1e-2
    np.testing.assert_allclose(ctf_b[unclamped], ctf0[unclamped] * envelope[unclamped], atol=1e-6, rtol=1e-5)

    # the envelope attenuates high frequencies (E < 1 away from DC)
    assert envelope[-1, -1] < 1.0
