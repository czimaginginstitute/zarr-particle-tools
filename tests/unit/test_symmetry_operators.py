"""
Golden unit tests for the point-group symmetry operators (Phase 1 D3).

After replacing RELION's truncated 6-figure axis literals / Euler tables with exact algebraic
generators + group closure, every supported group must form a proper finite group: correct order,
orthonormal operators, and closure to < 1e-12 (RELION's own operators only close to ~1e-6 for T and
~1e-7 for I3/I4 because of XMIPP_EQUAL_ACCURACY=1e-6). See docs/audit/relion_symmetry_source.md.
"""

import numpy as np
import pytest

from zarr_particle_tools.core.symmetry import get_transforms_from_symmetry

# group -> expected order
ORDERS = {
    "C1": 1,
    "C2": 2,
    "C7": 7,
    "D2": 4,
    "D7": 14,
    "T": 12,
    "TD": 24,
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
    "I3H": 120,
    "I4H": 120,
}

PROPER = {"C1", "C2", "C7", "D2", "D7", "T", "O", "I", "I1", "I2", "I3", "I4"}
IMPROPER = {"TD", "TH", "OH", "IH", "I1H", "I3H", "I4H"}


def _rot_blocks(sym: str) -> np.ndarray:
    return np.array([T[:3, :3] for T in get_transforms_from_symmetry(sym)])


@pytest.mark.parametrize("sym,order", ORDERS.items())
def test_group_order(sym, order):
    assert len(get_transforms_from_symmetry(sym)) == order


@pytest.mark.parametrize("sym", ORDERS)
def test_operators_orthonormal(sym):
    rs = _rot_blocks(sym)
    err = np.max(np.abs(np.einsum("nij,nik->njk", rs, rs) - np.eye(3)))
    assert err < 1e-12, f"{sym}: max |R^T R - I| = {err}"


@pytest.mark.parametrize("sym", ORDERS)
def test_group_closure_under_1e_12(sym):
    rs = _rot_blocks(sym)
    n = len(rs)
    worst = 0.0
    for a in range(n):
        prods = rs[a] @ rs  # (n, 3, 3): R_a . R_b for all b
        # nearest member distance for each product
        d = np.abs(prods[:, None, :, :] - rs[None, :, :, :]).reshape(n, n, 9).max(axis=2)
        worst = max(worst, float(d.min(axis=1).max()))
    assert worst < 1e-12, f"{sym}: closure defect {worst}"


@pytest.mark.parametrize("sym", sorted(PROPER))
def test_proper_groups_all_det_plus1(sym):
    dets = np.linalg.det(_rot_blocks(sym))
    np.testing.assert_allclose(dets, 1.0, atol=1e-10)


@pytest.mark.parametrize("sym", sorted(IMPROPER))
def test_improper_groups_mix_det_signs(sym):
    dets = np.linalg.det(_rot_blocks(sym))
    assert np.isclose(dets.min(), -1.0, atol=1e-10)
    assert np.isclose(dets.max(), 1.0, atol=1e-10)
    # a mirror/inversion group has exactly half proper, half improper
    assert np.isclose((dets > 0).sum(), (dets < 0).sum())
