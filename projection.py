import numpy as np
from cryoet_alignment.io.aretomo3 import AreTomo3ALN


def calculate_projection_matrix(rot: float, gmag: float, tx: float, ty: float, tilt: float, radians: bool = False) -> np.ndarray:
    """
    Calculates a 4x4 projection matrix based on the given rotation, translation, and tilt parameters (based on AreTomo .aln file).
    Calculations are based on affine projections in 3D space.

    Args:
        rot (float): Tilt axis rotation in radians or degrees (around the z-axis).
        gmag (float): Magnification factor.
        tx (float): Translation in the x direction (Angstroms).
        ty (float): Translation in the y direction (Angstroms).
        tilt (float): (Stage) tilt angle in radians or degrees (around the y-axis).
        radians (bool): If input angles are in radians. Defaults to False (degrees).
    Returns:
        np.ndarray: A 4x4 projection matrix.
    """
    if not radians:
        rot = np.radians(rot)
        tilt = np.radians(tilt)

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

    return M_2d_translation @ M_magnification @ M_axis_rot @ M_stage_tilt

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


def calculate_projection_matrix_from_aretomo_aln(aln: AreTomo3ALN, tiltseries_pixel_size: float = 1.0) -> list[np.ndarray]:
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

        projection_matrix = calculate_projection_matrix(rot, gmag, tx, ty, tilt)
        projection_matrices.append(projection_matrix)

    return projection_matrices


def circular_mask(box_size: int) -> np.ndarray:
    """Return a centered circular mask within a square of given box size (in pixels)."""
    y, x = np.ogrid[-box_size // 2:box_size // 2, -box_size // 2:box_size // 2]
    mask = (x * x + y * y) <= (box_size // 2) ** 2
    return mask.astype(np.float32)


def circular_soft_mask(box_size: int, falloff: float) -> np.ndarray:
    """Return a centered circular soft mask within a square of given box size (in pixels) (based on RELION soft mask)."""
    y, x = np.ogrid[-box_size // 2:box_size // 2, -box_size // 2:box_size // 2]
    mask = np.zeros((box_size, box_size), dtype=np.float32)
    r = np.sqrt(x * x + y * y)
    mask[r < box_size / 2.0 - falloff] = 1.0
    falloff_zone = (r >= box_size / 2.0 - falloff) & (r < box_size / 2.0)
    mask[falloff_zone] = 0.5 - 0.5 * np.cos(np.pi * (r[falloff_zone] - (box_size / 2.0)) / falloff)
    return mask
