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
    validate_deformations,
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
    # centred-Angstrom point (0, 0) is pixel (W/2, H/2) -> shift 5 px in x
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


def test_linear_float_centre_on_odd_image():
    # RELION's Linear2DDeformationModel centres on 0.5 * imageSize (a float), also for odd sizes
    m = LinearDeformation((401, 201), [0.5, 0.25, -1.0])
    np.testing.assert_allclose(m.shift(np.array([[200.5, 100.5]])), [[0.0, 0.0]])
    # r = (10, 4): (0.5 * 10, 0.25 * 10 - 1.0 * 4)
    np.testing.assert_allclose(m.shift(np.array([[210.5, 104.5]])), [[5.0, -1.5]])


def test_fourier_frequency_list_matches_relion_order():
    # Fourier2DDeformationModel's constructor for a 4 x 4 grid, written out by hand in units of (pi/W, pi/H):
    # y=0: x=1..2; y=1: x=0..2; y=2 -> (y - 4) = -2: x=1..2; y=3 -> -1: x=1..2
    w, h = 400, 200
    m = FourierDeformation((w, h), (4, 4), np.zeros(4 * 9))
    expected = [(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (1, -2), (2, -2), (1, -1), (2, -1)]
    np.testing.assert_allclose(m.freqs, np.array(expected) * [np.pi / w, np.pi / h])


def test_fourier_nonzero_coefficients_closed_form():
    # 2 x 2 grid on a 400 x 200 image: frequencies f0 = (pi/400, 0), f1 = (pi/400, -pi/200), n_freq = 2.
    # RELION reads the coefficients as RawImage<dComplex>(n_freq, 2): dim 0 = c[0:4], dim 1 = c[4:8], and adds
    # real * cos(f.p) + imag * sin(f.p).  The trailing values pad RELION's (gx/2 + 1) * gy * 2 complex vector and
    # must not be read (a (gx/2 + 1) * gy stride would put dim 1 at c[8:]).
    c = np.arange(1.0, 17.0)  # 1..8 used, 9..16 padding
    m = FourierDeformation((400, 200), (2, 2), c)
    p = np.array([[100.0, 50.0]])  # t0 = pi/4, t1 = pi/4 - pi/4 = 0
    r = np.sqrt(0.5)
    expected = [[1 * r + 2 * r + 3, 5 * r + 6 * r + 7]]
    np.testing.assert_allclose(m.shift(p), expected, rtol=1e-12)


def test_image_deformation_float_centre_and_batched_points():
    # odd-sized stack: (0, 0) centred Angstrom sits at pixel (W/2, H/2) = (200.5, 100.5), the crop step's centre
    px = 2.0
    d = ImageDeformation(LinearDeformation((401, 201), [0.5, 0.25, -1.0]), (401, 201), px)
    np.testing.assert_allclose(d(np.array([0.0, 0.0])), [0.0, 0.0], atol=1e-12)
    pts = np.array([[0.0, 0.0], [20.0, 8.0], [-300.0, 44.0]])  # Angstrom
    batched = d(pts)
    assert batched.shape == pts.shape
    np.testing.assert_allclose(batched, np.array([d(q) for q in pts]))
    # (20, 8) A = (10, 4) px off centre -> shift (5, -1.5) px = (10, -3) A
    np.testing.assert_allclose(batched[1], [20.0 + 10.0, 8.0 - 3.0])


def test_validate_deformations_matches_build_checks():
    ts = pd.DataFrame({"rlnMicrographName": ["1@a.mrcs", "2@a.mrcs"]})
    row = pd.Series({"rlnTomoName": "t", GRID_X: 3, GRID_Y: 3, TYPE: "spline"})
    ts[COEFFICIENTS] = ["[" + ",".join(["0"] * 72) + "]", "[" + ",".join(["0"] * 71) + "]"]
    with pytest.raises(ValueError, match="72 coefficients"):
        validate_deformations(row, ts)
    ts[COEFFICIENTS] = ["[" + ",".join(["0"] * 72) + "]"] * 2
    validate_deformations(row, ts)
    validate_deformations(pd.Series({"rlnTomoName": "t"}), ts.drop(columns=COEFFICIENTS))  # no deformations


def test_spline_single_node_closed_form():
    # Spline2DDeformationModel::computeShift with only node (0, 0) of dim 0 non-zero, written out from RELION's
    # F matrix (vx^T F vy): value*b0(x)b0(y) + slope_x*b2(x)b0(y) + slope_y*b0(x)b2(y) + twist*b2(x)b2(y), where
    # b0(t) = 1 - 3t^2 + 2t^3 and b2(t) = t - 2t^2 + t^3 and (x, y) are the fractional cell coordinates.
    # Pins the DataPoint order (value, slope_x, slope_y, twist) and which Hermite basis each one multiplies.
    value, slope_x, slope_y, twist = 7.0, 3.0, -2.0, 1.5
    nodes = _spline_nodes()
    nodes[0, 0, 0] = [value, slope_x, slope_y, twist]
    m = SplineDeformation(IMAGE, GRID, nodes.ravel())
    spacing = np.array([IMAGE[0] / (GRID[0] - 1), IMAGE[1] / (GRID[1] - 1)])
    fx, fy = 0.3, 0.8
    b0 = lambda t: 1 - 3 * t**2 + 2 * t**3  # noqa: E731
    b2 = lambda t: t - 2 * t**2 + t**3  # noqa: E731
    expected = value * b0(fx) * b0(fy) + slope_x * b2(fx) * b0(fy) + slope_y * b0(fx) * b2(fy) + twist * b2(fx) * b2(fy)
    shift = m.shift(np.array([[fx * spacing[0], fy * spacing[1]]]))
    np.testing.assert_allclose(shift, [[expected, 0.0]], rtol=1e-12, atol=1e-12)
    # swapping slope_x and slope_y, or dropping the twist, gives a different number at this point
    assert not np.isclose(
        expected,
        value * b0(fx) * b0(fy) + slope_y * b2(fx) * b0(fy) + slope_x * b0(fx) * b2(fy) + twist * b2(fx) * b2(fy),
    )
    assert not np.isclose(expected, expected - twist * b2(fx) * b2(fy))


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
