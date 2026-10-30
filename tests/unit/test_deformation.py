"""RELION-5 2D deformation models (core/deformation.py) against closed forms and RELION's conventions."""

import numpy as np
import pandas as pd
import pytest

from zarr_particle_tools.core.deformation import (
    COEFFICIENTS,
    GRID_X,
    GRID_Y,
    TYPE,
    FourierDeformation,
    ImageDeformation,
    LinearDeformation,
    SplineDeformation,
    build_deformations,
    declaration_problems,
    parse_coefficients,
)

IMAGE = (4096, 4096)
GRID = (3, 3)


def _spline_nodes(gx=3, gy=3):
    """zeros laid out like RawImage<DataPoint>(gx, gy, 2): [dim, y, x, (value, slope_x, slope_y, twist)]"""
    return np.zeros((2, gy, gx, 4))


def test_parse_coefficients_relion_vector_syntax():
    v = parse_coefficients("[4.46478389507,8.67595989683,-0.5238]")
    np.testing.assert_allclose(v, [4.46478389507, 8.67595989683, -0.5238])
    np.testing.assert_allclose(parse_coefficients([1, 2]), [1.0, 2.0])


def test_spline_zero_coefficients_is_identity():
    m = SplineDeformation(IMAGE, GRID, _spline_nodes().ravel())
    pts = np.array([[0.0, 0.0], [100.5, 2000.25], [4095.0, 4095.0], [-50.0, 5000.0]])
    np.testing.assert_array_equal(m.shift(pts), 0.0)


def test_spline_constant_value_is_uniform_shift():
    nodes = _spline_nodes()
    nodes[0, :, :, 0] = 7.5  # x value at every node
    nodes[1, :, :, 0] = -3.25  # y value at every node
    m = SplineDeformation(IMAGE, GRID, nodes.ravel())
    pts = np.array([[0.0, 0.0], [100.5, 2000.25], [4095.0, 4095.0], [1234.0, 56.0]])
    # Hermite value basis sums to 1, so a constant value field with zero slopes is exact everywhere
    np.testing.assert_allclose(m.shift(pts), np.tile([7.5, -3.25], (len(pts), 1)), atol=1e-12)


def test_spline_interpolates_node_values_and_slopes():
    gx, gy = 3, 3
    nodes = _spline_nodes(gx, gy)
    rng = np.random.default_rng(0)
    nodes[0, :, :, 0] = rng.normal(0, 10, (gy, gx))
    m = SplineDeformation(IMAGE, (gx, gy), nodes.ravel())
    spacing = np.array([IMAGE[0] / (gx - 1), IMAGE[1] / (gy - 1)])
    # at the nodes the value is reproduced exactly (clamped eps at the far edge)
    for iy in range(gy):
        for ix in range(gx):
            p = np.array([[ix * spacing[0], iy * spacing[1]]])
            assert abs(m.shift(p)[0, 0] - nodes[0, iy, ix, 0]) < 1e-6
    # a pure slope_x at one node: shift at that node is 0 and grows linearly in x at the node's column
    nodes = _spline_nodes(gx, gy)
    nodes[0, 1, 1, 1] = 2.0  # slope_x (per cell, RELION's units) at the centre node, x dim
    m = SplineDeformation(IMAGE, (gx, gy), nodes.ravel())
    centre = np.array([spacing[0], spacing[1]])
    assert abs(m.shift(centre[None])[0, 0]) < 1e-9
    # Hermite: d/dt of the value at t=0 along x equals slope_x (in cell units) -> dx/dt * spacing
    dt = 1e-4
    s = m.shift((centre + [dt * spacing[0], 0.0])[None])[0, 0]
    assert abs(s / dt - 2.0) < 1e-3


def test_spline_clamps_outside_the_image():
    nodes = _spline_nodes()
    nodes[0, :, :, 0] = np.arange(9).reshape(3, 3)  # x value grows with the node index
    m = SplineDeformation(IMAGE, GRID, nodes.ravel())
    inside_corner = m.shift(np.array([[IMAGE[0] - 1e-6, IMAGE[1] - 1e-6]]))
    far_outside = m.shift(np.array([[10 * IMAGE[0], 10 * IMAGE[1]]]))
    np.testing.assert_allclose(far_outside, inside_corner, atol=1e-6)
    np.testing.assert_allclose(m.shift(np.array([[-1e6, -1e6]])), m.shift(np.array([[0.0, 0.0]])))


def test_spline_rejects_wrong_count_and_grid():
    with pytest.raises(ValueError):
        SplineDeformation(IMAGE, GRID, np.zeros(71))
    with pytest.raises(ValueError):
        SplineDeformation(IMAGE, (1, 3), np.zeros(24))


def test_linear_closed_form():
    m = LinearDeformation((4000, 3000), [0.01, -0.02, 0.03])
    p = np.array([[2000.0 + 100.0, 1500.0 + 10.0]])  # r = (100, 10) about the float centre
    np.testing.assert_allclose(m.shift(p), [[1.0, -2.0 + 0.3]])


def test_fourier_zero_is_identity_and_frequency_count():
    # grid 4x4: y=0 -> x in 1..2 (2), y=1 -> x in 0..2 (3), y=2,3 -> x in 1..2 (2 each) = 9 frequencies
    with pytest.raises(ValueError):
        FourierDeformation(IMAGE, (4, 4), np.zeros(4 * 9 - 1))
    m = FourierDeformation(IMAGE, (4, 4), np.zeros(4 * 9))
    assert len(m.freqs) == 9
    np.testing.assert_array_equal(m.shift(np.array([[1.0, 2.0], [300.0, 400.0]])), 0.0)


def test_image_deformation_uses_relion_pixel_frame():
    # a spline whose x value grows linearly across the image: shift(px) = px.x / W * 10
    nodes = _spline_nodes()
    nodes[0, :, :, 0] = np.array([0.0, 5.0, 10.0])[None, :]
    nodes[0, :, :, 1] = 5.0  # matching slope_x per cell so the field is exactly linear
    model = SplineDeformation(IMAGE, GRID, nodes.ravel())
    px = 1.54
    d = ImageDeformation(model, IMAGE, px)
    # centred-Angstrom point (0, 0) is RELION pixel (W//2, H//2) -> shift 5 px in x
    out = d(np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [5.0 * px, 0.0], atol=1e-9)
    # a point one quarter image to the left: pixel W/4 -> shift 2.5 px
    out = d(np.array([-IMAGE[0] / 4 * px, 0.0]))
    np.testing.assert_allclose(out, [-IMAGE[0] / 4 * px + 2.5 * px, 0.0], atol=1e-9)


def test_build_deformations_follows_relion_rules():
    ts = pd.DataFrame({"rlnMicrographName": ["1@a.mrcs", "2@a.mrcs"]})
    row_none = pd.Series({"rlnTomoName": "t"})
    assert build_deformations(row_none, ts, IMAGE, 1.0) is None
    row = pd.Series({"rlnTomoName": "t", GRID_X: 3, GRID_Y: 3, TYPE: "spline"})
    # tomogram_set.cpp:452: without a coefficients column RELION applies none
    assert build_deformations(row, ts, IMAGE, 1.0) is None
    coeffs = "[" + ",".join(["0"] * 72) + "]"
    ts[COEFFICIENTS] = [coeffs, coeffs]
    defs = build_deformations(row, ts, IMAGE, 1.0)
    assert len(defs) == 2 and all(isinstance(d, ImageDeformation) for d in defs)
    bad = pd.Series({"rlnTomoName": "t", GRID_X: 3, GRID_Y: 3, TYPE: "bspline"})
    with pytest.raises(ValueError, match="illegal"):
        build_deformations(bad, ts, IMAGE, 1.0)
    # NaN grid size = column absent for this row (pandas fills missing cells with NaN)
    nan_row = pd.Series({"rlnTomoName": "t", GRID_X: np.nan, GRID_Y: np.nan, TYPE: "spline"})
    assert build_deformations(nan_row, ts, IMAGE, 1.0) is None


def test_declaration_problems_are_what_build_deformations_refuses():
    ts = pd.DataFrame({"rlnMicrographName": ["1@a.mrcs", "2@a.mrcs"]})
    row = pd.Series({"rlnTomoName": "t", GRID_X: 3, GRID_Y: 3, TYPE: "spline"})
    assert declaration_problems(row, ts) == []
    assert declaration_problems(pd.Series({"rlnTomoName": "t"}), ts) == []
    bad_type = pd.Series({"rlnTomoName": "t", GRID_X: 3, GRID_Y: 3})
    assert [column for column, _ in declaration_problems(bad_type, ts)] == [TYPE]
    ts[COEFFICIENTS] = ["[" + ",".join(["0"] * 72) + "]", "[0,0]"]
    assert [column for column, _ in declaration_problems(row, ts)] == [COEFFICIENTS]
    with pytest.raises(ValueError, match="72 coefficients"):
        build_deformations(row, ts, IMAGE, 1.0)
