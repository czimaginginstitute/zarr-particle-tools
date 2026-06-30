"""
Golden unit tests locking the CTF phase-shift behavior.

The phase-shift plumbing bug (reading `rlnPhaseShift` from the optics table instead of the
per-tilt tilt-series table) silently forced the phase shift to 0. These tests verify that the
phase shift, once supplied, enters the CTF exactly as RELION's K5 term (gamma -= deg2rad(phase)),
so that the corrected per-tilt value is used correctly downstream. No RELION binary is required.
"""

import numpy as np

from zarr_particle_tools.core.ctf import calculate_ctf
from zarr_particle_tools.core.forwardprojection import calculate_projection_matrix


def _ctf(phase_shift_deg: float) -> np.ndarray:
    # tilt=0 projection -> zero depth offset -> uncorrected defocus; isotropic 2 um defocus.
    pm = calculate_projection_matrix(rot=0.0, gmag=1.0, tx=0.0, ty=0.0, tilt=0.0)
    return calculate_ctf(
        coordinate=np.array([0.0, 0.0, 0.0]),
        tilt_projection_matrix=pm,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        handedness=1,
        tiltseries_pixel_size=1.0,
        phase_shift=phase_shift_deg,
        defocus_u=20000.0,
        defocus_v=20000.0,
        defocus_angle=0.0,
        dose=0.0,
        ctf_scalefactor=1.0,
        bfactor=0.0,
        box_size=64,
        bin=1,
    )


def test_phase_shift_changes_the_ctf():
    # Guards against phase shift being ignored (the symptom of the wrong-table read).
    assert not np.allclose(_ctf(0.0), _ctf(90.0))


def test_phase_shift_90_matches_relion_k5_term():
    # RELION: ctf = -sin(gamma) with gamma including -K5, K5 = deg2rad(phase_shift).
    # A 90 deg shift maps -sin(gamma) -> -sin(gamma - pi/2) = cos(gamma), so for every pixel
    # ctf(0)^2 + ctf(90)^2 == 1, except at the 1e-8-clamped near-zeros (negligible squared).
    ctf0 = _ctf(0.0)
    ctf90 = _ctf(90.0)
    np.testing.assert_allclose(ctf0**2 + ctf90**2, 1.0, atol=1e-6)
