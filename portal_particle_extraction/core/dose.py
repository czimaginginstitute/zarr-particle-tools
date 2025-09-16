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
