"""
Golden unit tests for the dose-based frequency cutoff used in particle reconstruction.

These verify `compute_dose_frequency_cutoff` reproduces RELION's `Tomogram::findDoseXRanges`
(evaluated at the y=0 row, the value passed as `maxFreq` to backprojection) and guard against
the previous broken `argmax` implementation. No RELION binary is required: the reference is an
independent re-expression of the same RELION rule.
"""

import numpy as np

from zarr_particle_tools.core.dose import calculate_dose_weight_image, compute_dose_frequency_cutoff


def _relion_find_dose_xranges_row0(dose_weights: np.ndarray, cutoff_fraction: float) -> np.ndarray:
    """Independent reference for RELION findDoseXRanges at row y=0.

    Dose weight is monotonically decreasing in frequency, so the count of x with dose weight
    strictly above cutoff equals RELION's (last kept x) + 1. This is a different numpy
    expression (sum of a mask) than the implementation under test (argmax of the complement).
    """
    row0 = np.asarray(dose_weights)[:, 0, :].real
    return (row0 > cutoff_fraction).sum(axis=1).astype(int)


def _old_buggy_idx(dose_weights: np.ndarray, cutoff_fraction: float) -> np.ndarray:
    """The pre-fix implementation (subtomo_reconstruct.py), kept here only as a regression guard."""
    freq_cutoff = np.asarray(dose_weights)[:, 0, :].real < cutoff_fraction
    return freq_cutoff.shape[1] - np.argmax(freq_cutoff[:, ::-1], axis=1)


def _dose_weight_stack(doses, box_size=64, pixel_size=4.0):
    return np.stack(
        [calculate_dose_weight_image(d, pixel_size, box_size, 0.0, 0.0) for d in doses]
    ).astype(np.complex128)


def test_matches_relion_find_dose_xranges():
    box_size = 64
    cutoff = 0.01
    doses = [10.0, 50.0, 100.0, 200.0, 1000.0]
    dose_weights = _dose_weight_stack(doses, box_size=box_size)

    got = compute_dose_frequency_cutoff(dose_weights, cutoff)
    ref = _relion_find_dose_xranges_row0(dose_weights, cutoff)

    np.testing.assert_array_equal(got, ref)
    # valid bounds
    assert np.all(got >= 0) and np.all(got <= box_size // 2 + 1)
    # higher cumulative dose -> more high-frequency attenuation -> non-increasing cutoff
    assert np.all(np.diff(got) <= 0)


def test_low_dose_keeps_full_nyquist():
    box_size = 64
    # all dose weights = 1 (> any sane cutoff) -> cutoff disabled (full Nyquist half-width)
    dose_weights = np.ones((1, box_size, box_size // 2 + 1), dtype=np.complex128)
    got = compute_dose_frequency_cutoff(dose_weights, 0.01)
    assert got[0] == box_size // 2 + 1


def test_all_below_cutoff_returns_zero():
    box_size = 32
    dose_weights = np.full((1, box_size, box_size // 2 + 1), 1e-6, dtype=np.complex128)
    got = compute_dose_frequency_cutoff(dose_weights, 0.01)
    assert got[0] == 0


def test_regression_old_argmax_disabled_cutoff_at_high_dose():
    box_size = 64
    cutoff = 0.01
    doses = [100.0, 200.0, 1000.0]
    dose_weights = _dose_weight_stack(doses, box_size=box_size)

    fixed = compute_dose_frequency_cutoff(dose_weights, cutoff)
    buggy = _old_buggy_idx(dose_weights, cutoff)

    # The old formula returned full Nyquist (cutoff disabled) for these high-dose tilts ...
    assert np.all(buggy == box_size // 2 + 1)
    # ... while the fix correctly truncates below Nyquist.
    assert np.all(fixed < box_size // 2 + 1)
