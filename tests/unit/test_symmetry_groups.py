"""
Structural checks on the symmetry operators (core/symmetry.py).

tests/test_reconstruct.py exercises symmetry indirectly against RELION references, but only for the
groups it has references for -- which let a broken IH slip through: i_transforms returns an ndarray, so
`I_matrices + MI_matrices` broadcast into an element-wise sum instead of concatenating, yielding 60
singular matrices instead of 120 valid ones. Every affected reconstruction was silently wrong.

These assert the group order and that every operator is a real orthogonal transform, which is cheap
and catches that whole class of bug for all groups at once. No RELION binary required.
"""

import numpy as np
import pytest

from zarr_particle_tools.core.symmetry import get_transforms_from_symmetry

# order of each point group; the mirror variants are exactly twice their rotation-only parent
EXPECTED_ORDERS = {
    "C1": 1,
    "C2": 2,
    "C7": 7,
    "D2": 4,
    "D7": 14,
    "T": 12,
    "TH": 24,
    "O": 24,
    "OH": 48,
    "I": 60,
    "I1": 60,
    "I2": 60,
    "I3": 60,
    "I4": 60,
    "IH": 120,
    "I1H": 120,
    "I2H": 120,
    "I3H": 120,
    "I4H": 120,
}
MIRROR_PARENTS = [("TH", "T"), ("OH", "O"), ("IH", "I"), ("I2H", "I2"), ("I4H", "I4")]


def _rot(matrix) -> np.ndarray:
    return np.asarray(matrix)[:3, :3]


@pytest.mark.parametrize("symmetry, order", sorted(EXPECTED_ORDERS.items()))
def test_group_order(symmetry, order):
    assert len(get_transforms_from_symmetry(symmetry)) == order


# The icosahedral/tetrahedral entries are truncated 9-digit literals taken from RELION's tables, so
# R @ R.T departs from the identity by ~1e-7 at worst. That is loose enough to accommodate them and
# still far tighter than the failure being guarded: the broken IH had determinant 0.0.
ORTHOGONALITY_ATOL = 1e-6


@pytest.mark.parametrize("symmetry", sorted(EXPECTED_ORDERS))
def test_every_operator_is_orthogonal_with_unit_determinant(symmetry):
    """A summed-instead-of-concatenated matrix shows up here as a non-orthogonal, singular operator."""
    for i, transform in enumerate(get_transforms_from_symmetry(symmetry)):
        r = _rot(transform)
        np.testing.assert_allclose(
            r @ r.T, np.eye(3), atol=ORTHOGONALITY_ATOL, err_msg=f"{symmetry}[{i}] not orthogonal"
        )
        assert abs(abs(np.linalg.det(r)) - 1.0) < ORTHOGONALITY_ATOL, f"{symmetry}[{i}] determinant is not +/-1"


@pytest.mark.parametrize("symmetry", sorted(EXPECTED_ORDERS))
def test_operators_are_distinct(symmetry):
    """Concatenation bugs can also duplicate operators; a point group has no repeats."""
    seen = {np.asarray(t).round(6).tobytes() for t in get_transforms_from_symmetry(symmetry)}
    assert len(seen) == len(get_transforms_from_symmetry(symmetry))


@pytest.mark.parametrize("mirror, parent", MIRROR_PARENTS)
def test_mirror_groups_double_their_parent_and_are_half_reflections(mirror, parent):
    transforms = get_transforms_from_symmetry(mirror)
    assert len(transforms) == 2 * len(get_transforms_from_symmetry(parent))
    dets = [np.linalg.det(_rot(t)) for t in transforms]
    assert sum(1 for d in dets if d < 0) == len(transforms) // 2, "expected half proper, half improper"


def test_identity_is_present():
    for symmetry in ("C1", "T", "OH", "IH"):
        assert any(
            np.allclose(_rot(t), np.eye(3), atol=1e-9) for t in get_transforms_from_symmetry(symmetry)
        ), f"{symmetry} is missing the identity"


def test_unsupported_icosahedral_variant_is_rejected():
    with pytest.raises(NotImplementedError):
        get_transforms_from_symmetry("I5H")
