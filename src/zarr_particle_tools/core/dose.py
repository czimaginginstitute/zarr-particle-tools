import numpy as np


def calculate_dose_weights(k2: np.ndarray, dose: float, bfactor: float, cutoff_fraction: float = 0) -> np.ndarray:
    """
    Calculates the dose-weighting filter in Fourier space for a single image (either B-factor or Grant & Grigorieff model).

    Args:
        k2 (np.ndarray): Squared spatial frequencies (k² = u²).
        dose (float): Electron dose.
        bfactor (float): If > 0, use B-factor model; otherwise use Grant & Grigorieff model.
        cutoff_fraction (float, optional): Set weights below this dose weight fraction to zero. Defaults to 0 (i.e., no cutoff).

    Returns:
        np.ndarray of weights.
    """
    if bfactor > 0.0:
        weights = np.exp(-bfactor * dose * k2 / 4.0)
    else:
        a = 0.245
        b = -1.665
        c = 2.81
        k = np.sqrt(k2)
        k[k == 0] = 1e-9
        d0 = a * (k**b) + c
        weights = np.exp(-0.5 * dose / d0)

    if cutoff_fraction > 0:
        weights[weights < cutoff_fraction] = 0

    return weights


def calculate_dose_weight_image(
    dose: float,
    tiltseries_pixel_size: float,
    box_size: int,
    bfactor_per_electron_dose: float,
    cutoff_fraction: float = 0,
) -> np.ndarray:
    """
    Calculates a 2D dose-weighting filter in Fourier space for a single image. Based on the RELION implementation in Damage::weightImage.

    Args:
        dose (float): The cumulative electron dose in e/A².
        tiltseries_pixel_size (float): The pixel size in Angstroms.
        box_size (int): The dimension of the image box in pixels.
        bfactor_per_electron_dose (float): The B-factor in A².
                                           If > 0, the B-factor model is used.
                                           Otherwise, the Grant & Grigorieff model is used.
        cutoff_fraction (float, optional): Set weights below this dose weight fraction to zero. Defaults to 0 (i.e., no cutoff).

    Returns:
        np.ndarray: A 2D array (box_size // 2 + 1, box_size) representing the dose-weighting filter in Fourier space.
    """
    s = box_size

    # fourier space coordinates
    ky = np.fft.fftfreq(s, d=tiltseries_pixel_size)
    kx = np.fft.rfftfreq(s, d=tiltseries_pixel_size)
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
    # squared spatial frequency
    k2 = kx_grid**2 + ky_grid**2

    return calculate_dose_weights(k2, dose, bfactor_per_electron_dose, cutoff_fraction)


def compute_dose_xranges(dose_weights: np.ndarray, cutoff_fraction: float) -> np.ndarray:
    """
    Per-row, per-tilt frequency cutoff matching RELION's ``Tomogram::findDoseXRanges``
    (``tomogram.cpp:442``).

    For each tilt ``f`` and each row ``y``, returns the exclusive upper bound on the usable rfft
    x-frequency: the largest ``x`` inside the Nyquist circle (``x <= sqrt(s^2/4 - yy^2)`` with
    ``yy = y < s/2 ? y : y - s``) whose dose weight is strictly greater than ``cutoff_fraction``,
    plus 1; or 0 if none qualifies. RELION zeroes source-slice columns ``x >= xRanges(y, f)``
    (both data and weight) before backprojection (``reconstruct_particle.cpp:379-394``), which
    removes a row-dependent (anisotropic) high-frequency wedge that a single spherical cutoff
    would keep.

    Args:
        dose_weights (np.ndarray): Stack of 2D dose-weight images of shape
            ``(n_tilts, box_size, box_size // 2 + 1)``, indexed ``[f, y, x]``. May be complex
            (the imaginary part is ignored).
        cutoff_fraction (float): Dose-weight threshold; columns at/below it are cut.

    Returns:
        np.ndarray: Integer cutoff index per ``(tilt, row)``, shape ``(n_tilts, box_size)``.
    """
    dw = np.asarray(dose_weights).real
    _, s, sh = dw.shape
    y = np.arange(s)
    yy = np.where(y < s / 2, y, y - s).astype(float)
    # Nyquist-circle bound per row; RELION computes s*s/4 in integer arithmetic (exact for even s).
    xmax = np.sqrt(np.maximum((s * s) / 4.0 - yy * yy, 0.0))  # (s,)
    within_circle = np.arange(sh)[None, :] <= xmax[:, None]  # (s, sh)
    qualifies = (dw > cutoff_fraction) & within_circle[None, :, :]  # (n_tilts, s, sh)
    has_any = qualifies.any(axis=2)  # (n_tilts, s)
    # index of the largest qualifying x per (f, y): first True scanning x from the high end.
    last_idx = sh - 1 - np.argmax(qualifies[:, :, ::-1], axis=2)
    return np.where(has_any, last_idx + 1, 0).astype(int)


def compute_dose_frequency_cutoff(dose_weights: np.ndarray, cutoff_fraction: float) -> np.ndarray:
    """
    Per-tilt scalar frequency cutoff = RELION ``xRanges(0, f)`` (the ``y = 0`` row), i.e. the value
    passed as the spherical ``maxFreq`` to ``backprojectSlice_backward``. This is exactly the
    ``y = 0`` slice of :func:`compute_dose_xranges`.

    Args:
        dose_weights (np.ndarray): Stack of 2D dose-weight images of shape
            ``(n_tilts, box_size, box_size // 2 + 1)``. May be complex (imag part ignored).
        cutoff_fraction (float): Dose-weight threshold below which frequencies are cut.

    Returns:
        np.ndarray: Integer cutoff index per tilt, shape ``(n_tilts,)``, in
        ``[0, box_size // 2 + 1]``.
    """
    return compute_dose_xranges(dose_weights, cutoff_fraction)[:, 0]
