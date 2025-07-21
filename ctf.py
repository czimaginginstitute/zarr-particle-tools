"""
Helper functions for calculating the CTF and dose-weighting filters in Fourier space.
"""
import numpy as np
from projection import project_3d_point_to_2d


def calculate_dose_weights(k2: np.ndarray, dose: float, bfactor: float) -> np.ndarray:
    """
    Calculates the dose-weighting filter in Fourier space for a single image (either B-factor or Grant & Grigorieff model).

    Args:
        k2 (np.ndarray): Squared spatial frequencies (k² = u²).
        dose (float): Electron dose.
        bfactor (float): If > 0, use B-factor model; otherwise use Grant & Grigorieff model.

    Returns:
        np.ndarray of weights.
    """
    if bfactor > 0.0:
        return np.exp(-bfactor * dose * k2 / 4.0)
    else:
        a = 0.245
        b = -1.665
        c = 2.81
        k = np.sqrt(k2)
        k[k == 0] = 1e-9
        d0 = a * (k**b) + c
        return np.exp(-0.5 * dose / d0)


def calculate_dose_weight_image(dose: float, tiltseries_pixel_size: float, box_size: int, bfactor_per_electron_dose: float) -> np.ndarray:
    """
    Calculates a 2D dose-weighting filter in Fourier space for a single image. Based on the RELION implementation in Damage::weightImage.

    Args:
        dose (float): The cumulative electron dose in e/A².
        tiltseries_pixel_size (float): The pixel size in Angstroms.
        box_size (int): The dimension of the image box in pixels.
        bfactor_per_electron_dose (float): The B-factor in A².
                                           If > 0, the B-factor model is used.
                                           Otherwise, the Grant & Grigorieff model is used.

    Returns:
        np.ndarray: A 2D array (box_size, box_size) representing the dose-weighting filter in Fourier space.
    """
    s = box_size

    # fourier space coordinates
    ky = np.fft.fftfreq(s, d=tiltseries_pixel_size)
    kx = np.fft.fftfreq(s, d=tiltseries_pixel_size)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    # squared spatial frequency
    k2 = kx_grid**2 + ky_grid**2

    return calculate_dose_weights(k2, dose, bfactor_per_electron_dose)


def get_depth_offset(tilt_projection_matrix: np.ndarray, coordinate: np.ndarray) -> float:
    projected_point = project_3d_point_to_2d(coordinate, tilt_projection_matrix)
    projected_origin = project_3d_point_to_2d(np.array([0, 0, 0]), tilt_projection_matrix)
    return projected_point[2] - projected_origin[2]  # z coordinate in the projected space


# TODO: cache re-used values (origin_projection, etc.) and even the projected point itself - only should be computed once per particle & tilt
# TODO: cache / reuse across different particles?
def calculate_ctf(
    coordinate: np.ndarray,
    tilt_projection_matrix: np.ndarray,
    voltage: float,
    spherical_aberration: float,
    amplitude_contrast: float,
    handedness: int,
    tiltseries_pixel_size: float,
    phase_shift: float,
    defocus_u: float,
    defocus_v: float,
    defocus_angle: float,
    dose: float,
    bfactor: float,
    box_size: int,
) -> np.ndarray:
    """
    Calculates the CTF for a given particle and tilt in a tomogram.
    Based on the RELION implementation in Tomogram::getCtf, Tomogram::getDepthOffset, CTF::initialise, CTF::draw, CTF::getCtf.

    Args:
        coordinate (np.ndarray): The 3D coordinates of the particle in Angstroms, as a numpy array of shape (3,).
        tilt_projection_matrix (np.ndarray): The projection matrix for the tilt, a 4x4 numpy array (3D affine transformation matrix).
        voltage (float): The accelerating voltage in kV.
        spherical_aberration (float): The spherical aberration in mm.
        amplitude_contrast (float): The amplitude contrast in percent.
        handedness (int): The handedness of the tomogram, either 1 or -1.
        tiltseries_pixel_size (float): The tiltseries pixel size in Angstroms.
        phase_shift (float): The phase shift in degrees.
        defocus_u (float): The defocus in the u direction in Angstroms.
        defocus_v (float): The defocus in the v direction in Angstroms.
        defocus_angle (float): The defocus (azimuthal) angle in degrees.
        dose (float): The cumulative electron dose in e/A².
        bfactor (float): The B-factor in A². See calculate_dose_weight_image for more details on this parameter.
        box_size (int): The size of the (crop) box in pixels.

    Returns:
        np.ndarray: A 2D array representing the CTF in Fourier space.

    """
    voltage *= 1000  # kV to V
    spherical_aberration *= 1e7  # mm to Angstroms
    defocus_angle = np.deg2rad(defocus_angle)

    depth_offset = get_depth_offset(tilt_projection_matrix, coordinate)
    # TODO: implement defocus slope (rlnTomoDefocusSlope), for now just assume 1
    defocus_offset = handedness * depth_offset

    defocus_u_corrected = defocus_u + defocus_offset
    defocus_v_corrected = defocus_v + defocus_offset

    # no longer used by latest RELION version, but kept for reference
    # defocus_average = -1 * (defocus_u_corrected + defocus_v_corrected) / 2.0
    # defocus_difference = -1 * (defocus_u_corrected - defocus_v_corrected) / 2.0

    wavelength = 12.2643247 / np.sqrt(voltage * (1 + voltage * 0.978466e-6))

    # constants, based on RELION's CTF::initialise
    # K1 and K2: https://en.wikipedia.org/wiki/High-resolution_transmission_electron_microscopy#:~:text=transfer%20function.-,The%20phase%20contrast%20transfer%20function,-%5Bedit%5D
    K1 = np.pi * wavelength
    K2 = np.pi / 2 * spherical_aberration * wavelength**3
    # TODO: figure out how RELION determined these constants
    K3 = np.arctan(amplitude_contrast / np.sqrt(1 - amplitude_contrast**2))
    K4 = -1 * bfactor / 4.0
    K5 = np.deg2rad(phase_shift)

    if amplitude_contrast < 0.0 or amplitude_contrast > 1.0:
        raise ValueError("Amplitude contrast must be between 0 and 1.")
    if handedness != 1 and handedness != -1:
        raise ValueError("Handedness must be either 1 or -1.")
    if abs(defocus_u) < 1e-6 and abs(defocus_v) < 1e-6 and abs(amplitude_contrast) < 1e-6 and abs(spherical_aberration) < 1e-6:
        raise ValueError("CTF parameters are 0, please check your inputs.")

    # for astigmatism correction
    Q = np.array([
        [np.cos(defocus_angle), np.sin(defocus_angle)],
        [-np.sin(defocus_angle), np.cos(defocus_angle)]
    ])
    Q_t = np.array([
        [np.cos(defocus_angle), -np.sin(defocus_angle)],
        [np.sin(defocus_angle), np.cos(defocus_angle)]
    ])
    D = np.array([
        [-defocus_u_corrected, 0],
        [0, -defocus_v_corrected]
    ])
    A = Q_t @ D @ Q
    Axx = A[0, 0]
    Axy = A[0, 1]
    Ayy = A[1, 1]

    # TODO: support gamma offset here
    s = box_size

    # fourier space coordinates
    ky = np.fft.fftfreq(s, d=tiltseries_pixel_size)
    kx = np.fft.fftfreq(s, d=tiltseries_pixel_size)
    kx_grid, ky_grid = np.meshgrid(kx, ky)

    u2 = kx_grid**2 + ky_grid**2
    u4 = u2**2

    # phase shift (gamma)
    gamma = K1 * (Axx * kx_grid**2 + 2.0 * Axy * kx_grid * ky_grid + Ayy * ky_grid**2) + K2 * u4 - K5 - K3
    ctf = -1 * np.sin(gamma)

    # dose weighting
    ctf *= calculate_dose_weights(u2, dose, bfactor)

    mask = np.abs(ctf) < 1e-8
    ctf[mask] = np.sign(ctf[mask]) * 1e-8

    return ctf
