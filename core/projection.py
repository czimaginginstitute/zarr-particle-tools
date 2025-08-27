"""
Helper functions for calculating projection matrices, projecting 3D points to 2D coordinates, and creating masks.
"""

import json

import numpy as np
import pandas as pd
from cryoet_alignment.io.aretomo3 import AreTomo3ALN

from core.data import DataReader


def in_plane_rotation_to_tilt_axis_rotation(rotation_matrix: list[list[float]]) -> float:
    np_matrix = np.array(rotation_matrix)
    return np.degrees(np.arctan2(np_matrix[1, 0], np_matrix[0, 0]))


def calculate_projection_matrix(
    rot: float, gmag: float, tx: float, ty: float, tilt: float, x_tilt: float = 0.0, radians: bool = False
) -> np.ndarray:
    """
    Calculates a 4x4 projection matrix based on the given rotation, translation, and tilt parameters (based on AreTomo .aln file).
    Calculations are based on affine projections in 3D space.

    Args:
        rot (float): Tilt axis rotation in radians or degrees (around the z-axis).
        gmag (float): Magnification factor.
        tx (float): Translation in the x direction (Angstroms).
        ty (float): Translation in the y direction (Angstroms).
        tilt (float): (Stage) tilt angle in radians or degrees (around the y-axis).
        x_tilt (float): Rotation around the x-axis in radians or degrees (rlnTomoXTilt in RELION, volume_x_rotation on the CryoET Data Portal). Default is 0.0.
        radians (bool): If input angles are in radians. Defaults to False (degrees).
    Returns:
        np.ndarray: A 4x4 projection matrix.
    """
    if not radians:
        rot = np.radians(rot)
        tilt = np.radians(tilt)
        x_tilt = np.radians(x_tilt)

    # fmt: off
    M_2d_translation = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    M_magnification = np.array([
        [gmag, 0, 0, 0],
        [0, gmag, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    M_axis_rot = np.array([
        [np.cos(rot), -np.sin(rot), 0, 0],
        [np.sin(rot), np.cos(rot), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    M_stage_tilt = np.array([
        [np.cos(tilt), 0, np.sin(tilt), 0],
        [0, 1, 0, 0],
        [-np.sin(tilt), 0, np.cos(tilt), 0],
        [0, 0, 0, 1]
    ])

    M_x_tilt = np.array([
        [1, 0, 0, 0],
        [0, np.cos(x_tilt), -np.sin(x_tilt), 0],
        [0, np.sin(x_tilt), np.cos(x_tilt), 0],
        [0, 0, 0, 1]
    ])
    # fmt: on

    return M_2d_translation @ M_magnification @ M_axis_rot @ M_stage_tilt @ M_x_tilt


def project_3d_point_to_2d(point_3d: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    """
    Projects a 3D point to a 2D point using the provided projection matrix.

    Args:
        point_3d (np.ndarray): A 3D point as a numpy array of shape (3,).
        projection_matrix (np.ndarray): A 4x4 projection matrix.

    Returns:
        np.ndarray: The projected 2D point as a numpy array of shape (2,).
    """
    if point_3d.shape != (3,):
        raise ValueError("point_3d must be a 1D array with 3 elements.")

    point_3d_homogeneous = np.append(point_3d, 1.0)
    projected_point = projection_matrix @ point_3d_homogeneous

    if projected_point[3] == 0:
        raise ValueError("Projection resulted in a point at infinity.")

    projected_point /= projected_point[3]
    return projected_point


# NOTE: Not currently used, but may be useful in the future.
def calculate_projection_matrix_from_aretomo_aln(
    aln: AreTomo3ALN, tiltseries_pixel_size: float = 1.0
) -> list[np.ndarray]:
    """
    Calculates the projection matrices for each section in the given AreTomo3ALN object.

    Args:
        aln (AreTomo3ALN): An AreTomo3ALN object containing alignment parameters.

    Returns:
        list[np.ndarray]: A list of 4x4 projection matrices for each section.
    """
    projection_matrices = []
    for section in aln.GlobalAlignments:
        rot = section.rot
        gmag = section.gmag
        tx = section.tx * tiltseries_pixel_size
        ty = section.ty * tiltseries_pixel_size
        tilt = section.tilt

        # AreTomo3 has no x_tilt
        projection_matrix = calculate_projection_matrix(rot, gmag, tx, ty, tilt, x_tilt=0.0, radians=False)
        projection_matrices.append(projection_matrix)

    return projection_matrices


def calculate_projection_matrix_from_starfile_df(tiltseries_df: pd.DataFrame) -> list[np.ndarray]:
    """
    Calculates the projection matrices for each section in the given tiltseries DataFrame (with section information).
    Args:
        tiltseries_df (pd.DataFrame): A DataFrame containing tiltseries information with alignment parameters.

    Returns:
        list[np.ndarray]: A list of 4x4 projection matrices for each section.
    """
    projection_matrices = []
    for _, tilt in tiltseries_df.iterrows():
        if {"_rlnTomoProjX", "_rlnTomoProjY", "_rlnTomoProjZ", "_rlnTomoProjW"}.issubset(tilt.index):
            projection_matrices.append(
                np.array(
                    [
                        json.loads(tilt["_rlnTomoProjX"]),
                        json.loads(tilt["_rlnTomoProjY"]),
                        json.loads(tilt["_rlnTomoProjZ"]),
                        json.loads(tilt["_rlnTomoProjW"]),
                    ]
                )
            )
            continue

        rot = tilt["rlnTomoZRot"]
        gmag = 1.0  # Assuming no magnification change
        tx = tilt["rlnTomoXShiftAngst"]
        ty = tilt["rlnTomoYShiftAngst"]
        tilt_angle = tilt["rlnTomoYTilt"]
        x_tilt = tilt["rlnTomoXTilt"]

        projection_matrix = calculate_projection_matrix(rot, gmag, tx, ty, tilt_angle, x_tilt=x_tilt, radians=False)
        projection_matrices.append(projection_matrix)

    return projection_matrices


# can likely be parallelized
def get_particles_to_tiltseries_coordinates(
    filtered_particles_df: pd.DataFrame,
    filtered_trajectories_dict: dict[int, pd.DataFrame] | None,
    tiltseries_df: pd.DataFrame,
    projection_matrices: list[np.ndarray],
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """
    Maps particle indices to their 2D coordinates in each of the tilts (projected from their 3D coordinates via the projection matrices).
    The output is a dictionary where the keys are particle indices and the values are another dictionary with tilt section indices as keys and tuples of (3D coordinate, projected 2D coordinate) as values.
    """
    particles_to_tiltseries_coordinates = {}
    for i, tilt in tiltseries_df.iterrows():
        section = int(tilt["rlnMicrographName"].split("@")[0])
        projection_matrix = projection_matrices[i]

        # match 1-indexing of RELION
        for default_particle_id, particle in enumerate(filtered_particles_df.itertuples(), start=1):
            # rlnOriginXAngst/YAngst/ZAngst are already included in the coordinate
            coordinate = np.array(
                [
                    particle.rlnCenteredCoordinateXAngst,
                    particle.rlnCenteredCoordinateYAngst,
                    particle.rlnCenteredCoordinateZAngst,
                ]
            )
            particle_id = (
                int(particle.rlnTomoParticleName.split("/")[-1])
                if "rlnTomoParticleName" in filtered_particles_df.columns
                else default_particle_id
            )
            # add motion correction if available
            if filtered_trajectories_dict is not None:
                if particle.rlnTomoParticleName not in filtered_trajectories_dict:
                    raise ValueError(f"Particle {particle.rlnTomoParticleName} not found in trajectories.")
                else:
                    trajectory = filtered_trajectories_dict[particle.rlnTomoParticleName]
                    tilt_trajectory = trajectory.iloc[i]
                    coordinate[0] += tilt_trajectory["rlnOriginXAngst"]
                    coordinate[1] += tilt_trajectory["rlnOriginYAngst"]
                    coordinate[2] += tilt_trajectory["rlnOriginZAngst"]

            projected_point = project_3d_point_to_2d(coordinate, projection_matrix)[:2]

            if particle_id not in particles_to_tiltseries_coordinates:
                particles_to_tiltseries_coordinates[particle_id] = {}

            particles_to_tiltseries_coordinates[particle_id][section] = (coordinate, projected_point)

    return particles_to_tiltseries_coordinates


def get_particle_crop_and_visibility(
    tiltseries_data: DataReader,
    particle_id: int,
    sections: dict,
    tiltseries_x: int,
    tiltseries_y: int,
    tiltseries_pixel_size: float,
    pre_bin_box_size: int,
    pre_bin_crop_size: float,
) -> tuple[list[dict], list[int]]:
    """
    Process the calculated (Angstrom) 2D coordinate of the particle to pixel coordinates and perform cropping.

    Returns a tuple of (particle_data, visible_sections), where particle_data is a list of dictionaries with the particle's 2D coordinates and cropping information,
    and visible_sections is a list indicating whether or not the particle is visible in each section (1 for visible, 0 for not visible).
    """
    particle_data = []
    visible_sections = []

    for section, coords in sections.items():
        coordinate, projected_point = coords
        x, y = projected_point
        # convert physical angstroms to floating-point pixel coordinates
        x_px_float = (x + tiltseries_x * tiltseries_pixel_size / 2.0) / tiltseries_pixel_size
        y_px_float = (y + tiltseries_y * tiltseries_pixel_size / 2.0) / tiltseries_pixel_size
        x_start_px_float = x_px_float - pre_bin_box_size / 2.0
        y_start_px_float = y_px_float - pre_bin_box_size / 2.0
        x_start_px = int(round(x_start_px_float))
        y_start_px = int(round(y_start_px_float))
        x_end_px = x_start_px + pre_bin_box_size
        y_end_px = y_start_px + pre_bin_box_size
        # for checking visibility and cropping later
        x_crop_start_px_float = x_px_float - pre_bin_crop_size / 2.0
        y_crop_start_px_float = y_px_float - pre_bin_crop_size / 2.0
        x_crop_end_px_float = x_crop_start_px_float + pre_bin_crop_size
        y_crop_end_px_float = y_crop_start_px_float + pre_bin_crop_size

        if (
            x_crop_start_px_float < 0
            or x_crop_end_px_float > tiltseries_x
            or y_crop_start_px_float < 0
            or y_crop_end_px_float > tiltseries_y
        ):
            visible_sections.append(0)
            continue

        # subpixel shift
        shift_x = x_start_px - x_start_px_float
        shift_y = y_start_px - y_start_px_float

        slice_key = (
            section - 1,
            slice(max(y_start_px, 0), min(y_end_px, tiltseries_y)),
            slice(max(x_start_px, 0), min(x_end_px, tiltseries_x)),
        )
        # add it to the DataReader cache to be computed later
        tiltseries_data.slice_data(slice_key)

        visible_sections.append(1)
        particle_data.append(
            {
                "particle_id": particle_id,
                "coordinate": coordinate,
                "section": section,
                "tiltseries_slice_key": slice_key,
                "subpixel_shift": (shift_y, shift_x),
                "x_pre_padding": max(0, -x_start_px),
                "y_pre_padding": max(0, -y_start_px),
                "x_post_padding": max(0, x_end_px - tiltseries_x),
                "y_post_padding": max(0, y_end_px - tiltseries_y),
            }
        )

    return particle_data, visible_sections


def fourier_crop(fft_image_stack: np.ndarray, factor: float) -> np.ndarray:
    """
    Performs binning by cropping a Fourier-space image stack. Based on the RELION Resampling::FourierCrop_fftwHalfStack function.
    """
    h0, wh0 = fft_image_stack.shape
    w0 = (wh0 - 1) * 2

    h1 = int(round(h0 / factor))
    w1 = int(round(w0 / factor))

    # Ensure the new width is even for RFFT compatibility
    if w1 % 2 != 0:
        w1 += 1

    wh1 = w1 // 2 + 1

    h1_top = (h1 + 1) // 2
    h1_bottom = h1 // 2

    cropped_w = fft_image_stack[:, :wh1]
    top_slice = cropped_w[:h1_top, :]
    bottom_slice = cropped_w[h0 - h1_bottom :, :]

    binned_fft_stack = np.concatenate((top_slice, bottom_slice), axis=0)
    return binned_fft_stack
