"""
Unit tests for the Wiener CTF correction used by reconstruct's --snr.

RELION's Reconstruction::ctfCorrect3D_Wiener FFTs the volume, divides by (weights + 1/SNR), and
inverse-FFTs (both with FFT::Both == numpy norm="ortho"). Unlike ctfCorrect3D_heuristic it applies no
radial-average floor and no Nyquist cutoff. No RELION binary required.
"""

import numpy as np

from zarr_particle_tools.core.backprojection import ctf_correct_3d_heuristic, ctf_correct_3d_wiener

BOX = 8


def _volume_and_weights(seed: int = 0):
    rng = np.random.default_rng(seed)
    volume = rng.normal(size=(BOX, BOX, BOX))
    # reconstruct accumulates complex128 weight volumes, and the heuristic requires that dtype
    weights = (np.abs(rng.normal(size=(BOX, BOX, BOX // 2 + 1))) + 0.5).astype(np.complex128)
    return volume, weights


def test_wiener_matches_the_relion_formula():
    volume, weights = _volume_and_weights()
    offset = 1.0 / 10.0
    expected = np.fft.irfftn(np.fft.rfftn(volume, norm="ortho") / (weights + offset), norm="ortho")
    np.testing.assert_allclose(ctf_correct_3d_wiener(volume, weights, offset), expected, atol=1e-12)


def test_wiener_offset_changes_the_result():
    volume, weights = _volume_and_weights()
    assert not np.allclose(ctf_correct_3d_wiener(volume, weights, 0.1), ctf_correct_3d_wiener(volume, weights, 1.0))


def test_wiener_differs_from_the_heuristic():
    """Guards against --snr silently falling through to the heuristic branch."""
    volume, weights = _volume_and_weights()
    assert not np.allclose(ctf_correct_3d_wiener(volume, weights, 0.1), ctf_correct_3d_heuristic(volume, weights))


def test_large_snr_approaches_plain_division():
    # offset -> 0 means dividing by the weights alone
    volume, weights = _volume_and_weights()
    plain = np.fft.irfftn(np.fft.rfftn(volume, norm="ortho") / weights, norm="ortho")
    np.testing.assert_allclose(ctf_correct_3d_wiener(volume, weights, 1e-12), plain, atol=1e-9)
