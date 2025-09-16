"""
Helper functions for calculating the CTF and dose-weighting filters in Fourier space.
"""

import logging
from functools import cache

import numpy as np

from portal_particle_extraction.core.forwardprojection import project_3d_point_to_2d

logger = logging.getLogger(__name__)


def get_depth_offset(tilt_projection_matrix: np.ndarray, coordinate: np.ndarray) -> float:
    projected_point = project_3d_point_to_2d(coordinate, tilt_projection_matrix)
    projected_origin = project_3d_point_to_2d(np.array([0, 0, 0]), tilt_projection_matrix)
    return projected_point[2] - projected_origin[2]  # z coordinate in the projected space


@cache
def _ctf_template(
    voltage: float,
    spherical_aberration: float,
    amplitude_contrast: float,
    handedness: int,
    tiltseries_pixel_size: float,
    phase_shift: float,
    defocus_angle: float,
    bfactor: float,
    box_size: int,
    bin: int,
):
    if amplitude_contrast < 0.0 or amplitude_contrast > 1.0:
        raise ValueError("Amplitude contrast must be between 0 and 1.")
    if handedness not in (1, -1):
        raise ValueError("Handedness must be either 1 or -1.")

    voltage *= 1000  # kV to V
    spherical_aberration *= 1e7  # mm to Angstroms
    defocus_angle = np.deg2rad(defocus_angle)

    # no longer used by latest RELION version, but kept for reference
    # defocus_average = -1 * (defocus_u_corrected + defocus_v_corrected) / 2.0
    # defocus_difference = -1 * (defocus_u_corrected - defocus_v_corrected) / 2.0

    wavelength = 12.2643247 / np.sqrt(voltage * (1 + voltage * 0.978466e-6))

    # constants, based on RELION's CTF::initialise
    # K1 and K2: https://en.wikipedia.org/wiki/High-resolution_transmission_electron_microscopy#:~:text=transfer%20function.-,The%20phase%20contrast%20transfer%20function,-%5Bedit%5D
    K1 = np.pi * wavelength
    K2 = np.pi / 2 * spherical_aberration * wavelength**3
    K3 = np.arctan(amplitude_contrast / np.sqrt(1 - amplitude_contrast**2))
    K4 = -1 * bfactor / 4.0  # noqa: F841
    K5 = np.deg2rad(phase_shift)

    # for astigmatism correction
    Q = np.array([[np.cos(defocus_angle), np.sin(defocus_angle)], [-np.sin(defocus_angle), np.cos(defocus_angle)]])
    Q_t = np.array([[np.cos(defocus_angle), -np.sin(defocus_angle)], [np.sin(defocus_angle), np.cos(defocus_angle)]])

    # fourier space coordinates
    ky = np.fft.fftfreq(box_size, d=tiltseries_pixel_size * bin)
    kx = np.fft.rfftfreq(box_size, d=tiltseries_pixel_size * bin)
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")

    u2 = kx_grid**2 + ky_grid**2
    u4 = u2**2

    return K1, K2, K3, K5, ky_grid, kx_grid, u2, u4, Q, Q_t


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
    ctf_scalefactor: float,
    bfactor: float,
    box_size: int,
    bin: int,
    debug: bool = False,
) -> np.ndarray:
    """
    Calculates the CTF for a given particle and tilt in a tomogram.
    Based on the RELION implementation in Tomogram::getCtf, Tomogram::getDepthOffset, CTF::initialise, CTF::draw, CTF::getCtf.

    Args:
        coordinate (np.ndarray): The 3D coordinates of the particle in Angstroms, as a numpy array of shape (3,).
        tilt_projection_matrix (np.ndarray): The projection matrix for the tilt, a 4x4 numpy array (3D affine transformation matrix).
        voltage (float): The accelerating voltage in kV.
        spherical_aberration (float): The spherical aberration in mm.
        amplitude_contrast (float): The amplitude contrast [0, 1].
        handedness (int): The handedness of the tomogram, either 1 or -1.
        tiltseries_pixel_size (float): The tiltseries pixel size in Angstroms.
        phase_shift (float): The phase shift in degrees.
        defocus_u (float): The defocus in the u direction in Angstroms.
        defocus_v (float): The defocus in the v direction in Angstroms.
        defocus_angle (float): The defocus (azimuthal) angle in degrees.
        dose (float): The cumulative electron dose in e/A².
        bfactor (float): The B-factor in A². See calculate_dose_weight_image for more details on this parameter.
        box_size (int): The size of the (crop) box in pixels.
        bin (int): The binning factor.

    Returns:
        np.ndarray: A 2D array representing the CTF in Fourier space of shape (box_size, box_size // 2 + 1).

    """
    if (
        abs(defocus_u) < 1e-6
        and abs(defocus_v) < 1e-6
        and abs(amplitude_contrast) < 1e-6
        and abs(spherical_aberration) < 1e-6
    ):
        raise ValueError("CTF parameters are 0, please check your inputs.")

    depth_offset = get_depth_offset(tilt_projection_matrix, coordinate)
    # TODO: implement defocus slope (rlnTomoDefocusSlope), for now just assume 1
    defocus_offset = handedness * depth_offset
    defocus_u_corrected = defocus_u + defocus_offset
    defocus_v_corrected = defocus_v + defocus_offset
    if debug:
        logger.debug(
            f"Original Defocus U: {defocus_u}, Original Defocus V: {defocus_v}, Corrected Defocus U: {defocus_u_corrected}, Corrected Defocus V: {defocus_v_corrected}, Depth offset: {depth_offset}"
        )

    K1, K2, K3, K5, ky_grid, kx_grid, u2, u4, Q, Q_t = _ctf_template(
        voltage,
        spherical_aberration,
        amplitude_contrast,
        handedness,
        tiltseries_pixel_size,
        phase_shift,
        defocus_angle,
        bfactor,
        box_size,
        bin,
    )

    # astigmatism correction
    D = np.array([[-defocus_u_corrected, 0], [0, -defocus_v_corrected]])
    A = Q_t @ D @ Q
    Axx = A[0, 0]
    Axy = A[0, 1]
    Ayy = A[1, 1]
    if debug:
        logger.debug(f"CTF astigmatism: {Axx}, {Axy}, {Ayy}")

    # TODO: support gamma offset here
    # phase shift (gamma)
    gamma = K1 * (Axx * kx_grid**2 + 2.0 * Axy * kx_grid * ky_grid + Ayy * ky_grid**2) + K2 * u4 - K5 - K3
    if debug:
        logger.debug(f"K1: {K1}, K2: {K2}, K3: {K3}, K5: {K5}")
        logger.debug(f"Gamma: {gamma}")
    ctf = -1 * np.sin(gamma)

    # dose weighting, which doesn't seem to be done on the CTF?
    # ctf *= calculate_dose_weights(u2, dose, bfactor)

    ctf *= ctf_scalefactor

    mask = np.abs(ctf) < 1e-8
    ctf[mask] = np.sign(ctf[mask]) * 1e-8

    return ctf
