"""
A particle's orientation as RELION 5 composes it.

RELION keeps two rotations per particle (``src/jaz/tomography/particle_set.cpp``):

- ``A_sub``, from ``rlnTomoSubtomogram{Rot,Tilt,Psi}`` (``ParticleSet::getSubtomogramMatrix``): the frame the
  particle is expressed in, e.g. the filament frame of a helical pick;
- ``A_part``, from ``rlnAngle{Rot,Tilt,Psi}`` (``ParticleSet::getParticleMatrix``): the pose within that frame.

Each is ``Euler::anglesToMatrix3`` of its angles (``src/jaz/math/Euler_angles_relion.h``), and the identity when its
Rot column is absent; with Rot present, a missing Tilt or Psi reads as 0. The particle's full orientation, the map from particle to tomogram coordinates, is
``A = A_sub · A_part`` (``ParticleSet::getMatrix3x3``). ``A_sub`` also takes the origin offsets into tomogram
coordinates: the particle sits at ``coordinate - A_sub · origin`` (``ParticleSet::getPosition``).
"""

import numpy as np
import pandas as pd

from zarr_particle_tools.core.backprojection import get_rotation_matrix_from_euler
from zarr_particle_tools.core.constants import PARTICLE_ANGLES, SUBTOMOGRAM_ANGLES


def _matrices(particles_df: pd.DataFrame, columns: tuple[str, str, str]) -> np.ndarray:
    """One rotation per row from (rot, tilt, psi) columns: the identity without the rot column, as RELION gates it
    (ParticleSet::getSubtomogramMatrix, getParticleMatrix); a missing tilt or psi reads as 0."""
    n = len(particles_df)
    if columns[0] not in particles_df.columns:
        return np.tile(np.eye(3), (n, 1, 1))
    angles = np.stack(
        [
            particles_df[column].to_numpy(dtype=float) if column in particles_df.columns else np.zeros(n)
            for column in columns
        ],
        axis=1,
    )
    return get_rotation_matrix_from_euler(angles)


def subtomogram_matrices(particles_df: pd.DataFrame) -> np.ndarray:
    """``A_sub`` per particle, ``(N, 3, 3)`` (``ParticleSet::getSubtomogramMatrix``)."""
    return _matrices(particles_df, SUBTOMOGRAM_ANGLES)


def particle_matrices(particles_df: pd.DataFrame) -> np.ndarray:
    """``A_part`` per particle, ``(N, 3, 3)`` (``ParticleSet::getParticleMatrix``)."""
    return _matrices(particles_df, PARTICLE_ANGLES)


def particle_to_tomogram_matrices(particles_df: pd.DataFrame) -> np.ndarray:
    """``A = A_sub · A_part`` per particle, ``(N, 3, 3)`` (``ParticleSet::getMatrix3x3``)."""
    return subtomogram_matrices(particles_df) @ particle_matrices(particles_df)
