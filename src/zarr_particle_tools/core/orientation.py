"""
A particle's orientation as RELION 5 composes it.

RELION keeps two rotations per particle (``src/jaz/tomography/particle_set.cpp``):

- ``A_sub``, from ``rlnTomoSubtomogram{Rot,Tilt,Psi}`` (``ParticleSet::getSubtomogramMatrix``): the frame the
  particle is expressed in, e.g. the filament frame of a helical pick;
- ``A_part``, from ``rlnAngle{Rot,Tilt,Psi}`` (``ParticleSet::getParticleMatrix``): the pose within that frame.

Each is ``Euler::anglesToMatrix3`` of its angles (``src/jaz/math/Euler_angles_relion.h``), and the identity when its
columns are absent. The particle's full orientation, the map from particle to tomogram coordinates, is
``A = A_sub · A_part`` (``ParticleSet::getMatrix3x3``). ``A_sub`` also takes the origin offsets into tomogram
coordinates: the particle sits at ``coordinate - A_sub · origin`` (``ParticleSet::getPosition``).
"""

import numpy as np
import pandas as pd

from zarr_particle_tools.core.backprojection import get_rotation_matrix_from_euler

SUBTOMOGRAM_ANGLES = ("rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt", "rlnTomoSubtomogramPsi")
PARTICLE_ANGLES = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")


def euler_matrices(angles: np.ndarray) -> np.ndarray:
    """RELION's ``Euler::anglesToMatrix3`` of ``(N, 3)`` (rot, tilt, psi) angles in degrees, as ``(N, 3, 3)``."""
    angles = np.asarray(angles, dtype=float).reshape(-1, 3)
    return get_rotation_matrix_from_euler(angles).reshape(-1, 3, 3)


def _matrices(particles_df: pd.DataFrame, columns: tuple[str, str, str]) -> np.ndarray:
    """One rotation per row from three angle columns; a missing column reads as 0, all three missing as identity."""
    n = len(particles_df)
    if not any(column in particles_df.columns for column in columns):
        return np.tile(np.eye(3), (n, 1, 1))
    angles = np.stack(
        [
            particles_df[column].to_numpy(dtype=float) if column in particles_df.columns else np.zeros(n)
            for column in columns
        ],
        axis=1,
    )
    return euler_matrices(angles)


def has_subtomogram_orientation(particles_df: pd.DataFrame) -> bool:
    """Whether the particles carry a subtomogram frame (any of ``rlnTomoSubtomogram{Rot,Tilt,Psi}``)."""
    return any(column in particles_df.columns for column in SUBTOMOGRAM_ANGLES)


def subtomogram_matrices(particles_df: pd.DataFrame) -> np.ndarray:
    """``A_sub`` per particle, ``(N, 3, 3)`` (``ParticleSet::getSubtomogramMatrix``)."""
    return _matrices(particles_df, SUBTOMOGRAM_ANGLES)


def particle_matrices(particles_df: pd.DataFrame) -> np.ndarray:
    """``A_part`` per particle, ``(N, 3, 3)`` (``ParticleSet::getParticleMatrix``)."""
    return _matrices(particles_df, PARTICLE_ANGLES)


def particle_to_tomogram_matrices(particles_df: pd.DataFrame) -> np.ndarray:
    """``A = A_sub · A_part`` per particle, ``(N, 3, 3)`` (``ParticleSet::getMatrix3x3``)."""
    return subtomogram_matrices(particles_df) @ particle_matrices(particles_df)
