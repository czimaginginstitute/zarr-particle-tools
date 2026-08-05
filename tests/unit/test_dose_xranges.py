"""
Golden unit tests for the per-row dose frequency cutoff (Phase 1 D2).

`compute_dose_xranges` must reproduce RELION's `Tomogram::findDoseXRanges` (tomogram.cpp:442), the
per-row cutoff used to zero source-slice columns `x >= xRanges(y, f)` before backprojection
(reconstruct_particle.cpp:379-394). The reference here is a direct line-by-line transcription of the
RELION C++ loop; no RELION binary is required.
"""

import numpy as np

from zarr_particle_tools.core.dose import (
    calculate_dose_weight_image,
    compute_dose_frequency_cutoff,
    compute_dose_xranges,
)


def _relion_find_dose_xranges(dose_weights: np.ndarray, cutoff: float) -> np.ndarray:
    """Direct transcription of RELION Tomogram::findDoseXRanges (tomogram.cpp:442-472)."""
    dw = np.asarray(dose_weights).real
    fc, s, sh = dw.shape
    out = np.zeros((fc, s), dtype=int)
    for f in range(fc):
        for y in range(s):
            yy = y if y < s / 2 else y - s
            xmax = np.sqrt(s * s / 4 - yy * yy)
            o = 0
            for x in range(sh):
                if x > xmax:
                    break
                if dw[f, y, x] > cutoff:
                    o = x + 1
            out[f, y] = o
    return out


def _dose_weight_stack(doses, box_size=64, pixel_size=4.0):
    return np.stack([calculate_dose_weight_image(d, pixel_size, box_size, 0.0, 0.0) for d in doses]).astype(
        np.complex128
    )


def test_matches_relion_find_dose_xranges_per_row():
    box_size = 64
    cutoff = 0.01
    dw = _dose_weight_stack([10.0, 50.0, 100.0, 200.0, 1000.0], box_size=box_size)
    np.testing.assert_array_equal(compute_dose_xranges(dw, cutoff), _relion_find_dose_xranges(dw, cutoff))


def test_row0_equals_scalar_frequency_cutoff():
    # the D1 scalar cutoff is exactly row y=0 of the per-row matrix (RELION maxFreq = xRanges(0, f))
    box_size = 64
    cutoff = 0.01
    dw = _dose_weight_stack([10.0, 100.0, 1000.0], box_size=box_size)
    np.testing.assert_array_equal(compute_dose_xranges(dw, cutoff)[:, 0], compute_dose_frequency_cutoff(dw, cutoff))


def test_cutoff_is_anisotropic_and_within_nyquist():
    box_size = 64
    sh = box_size // 2 + 1
    cutoff = 0.01
    xr = compute_dose_xranges(_dose_weight_stack([200.0], box_size=box_size), cutoff)[0]  # (box_size,)
    # never exceeds the rfft half-width
    assert np.all(xr <= sh)
    # the per-row cutoff is NOT a constant broadcast of row 0 -> it removes an anisotropic wedge
    assert not np.all(xr == xr[0])
    # the highest-|y| row (Nyquist circle edge) is cut at least as hard as row 0
    assert xr[box_size // 2] < xr[0]
