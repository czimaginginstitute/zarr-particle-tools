import numpy as np

def circular_mask(box_size: int) -> np.ndarray:
    """Return a centered circular mask within a square of given box size (in pixels)."""
    y, x = np.ogrid[-box_size // 2 : box_size // 2, -box_size // 2 : box_size // 2]
    mask = (x * x + y * y) <= (box_size // 2) ** 2
    return mask


def circular_soft_mask(box_size: int, falloff: float) -> np.ndarray:
    """Return a centered circular soft mask within a square of given box size (in pixels) (based on RELION soft mask)."""
    y, x = np.ogrid[-box_size // 2 : box_size // 2, -box_size // 2 : box_size // 2]
    mask = np.zeros((box_size, box_size))
    r = np.sqrt(x * x + y * y)
    mask[r < box_size / 2.0 - falloff] = 1.0
    falloff_zone = (r >= box_size / 2.0 - falloff) & (r < box_size / 2.0)
    mask[falloff_zone] = 0.5 - 0.5 * np.cos(np.pi * ((box_size / 2.0) - r[falloff_zone]) / falloff)
    return mask


def nyquist_filter_mask(box_size):
    """
    Return a circular low-pass mask based on a physical resolution cutoff (in Fourier space).

    Args:
        box_size (int): The dimension of the square image in pixels.

    Returns:
        numpy.ndarray: A 2D array representing the circular filter mask.
    """
    ky, kx = np.fft.fftfreq(box_size), np.fft.fftfreq(box_size)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    freq_radius_pix = np.sqrt(kx_grid**2 + ky_grid**2)
    # nyquist is at 2 * pixel_size in Angstroms, in pixels it's pixel_size / (2 * pixel_size) = 0.5
    freq_cutoff_pix = 0.5
    mask = (freq_radius_pix < freq_cutoff_pix)
    
    return mask