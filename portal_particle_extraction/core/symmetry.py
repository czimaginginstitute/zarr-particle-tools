"""
Functions to generate symmetry operators as 4x4 affine matrices.
Based on RELION symmetry definitions in src/symmetries.cpp.
"""

# TODO: test these functions
# TODO: improve variable names for clarity
from functools import cache

import numpy as np


def _embed(R) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    return T


def _rotz(theta) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotx(theta) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_axis(axis, theta) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return c * np.eye(3) + s * K + (1 - c) * np.outer(axis, axis)  # Rodrigues' rotation formula


def _mirror_from_normal(n) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    return np.eye(3) - 2.0 * np.outer(n, n)


@cache
def cn_affines(n: int) -> list[np.ndarray]:
    """
    RELION CN (rot_axis n 0 0 1): rotations about +Z by k*2π/n, k=0..n-1.
    Returns list of n 4x4 matrices.
    """
    return [_embed(_rotz(2 * np.pi * k / n)) for k in range(n)]


@cache
def ci_affines() -> list[np.ndarray]:
    """
    RELION CI (inversion): { I, -I } in 3D, embedded in 4x4.
    """
    identity = np.eye(4)
    inv = np.eye(4)
    inv[:3, :3] = -np.eye(3)
    return [identity, inv]


@cache
def cs_affines() -> list[np.ndarray]:
    """
    RELION CS (mirror_plane 0 0 1): reflect in the XY-plane (normal +Z).
    Returns [I4, M_h].
    """
    Mh = _mirror_from_normal([0, 0, 1.0])
    return [_embed(np.eye(3)), _embed(Mh)]


@cache
def cnv_affines(n: int) -> list[np.ndarray]:
    """
    RELION CNV: CN around +Z plus a vertical mirror (normal in XZ plane).
    Returns list of 2n 4x4 matrices.
    """
    rots = cn_affines(n)
    Mv = _mirror_from_normal([0, 1.0, 0])
    mirrors = [_embed(Mv @ T[:3, :3]) for T in rots]
    return rots + mirrors


@cache
def cnh_affines(n: int) -> list[np.ndarray]:
    """
    RELION CNH: CN around +Z plus a horizontal mirror (normal +Z).
    Returns list of 2n 4x4 matrices.
    """
    rots = cn_affines(n)
    Mh = _mirror_from_normal([0, 0, 1.0])
    mirrors = [_embed(Mh @ T[:3, :3]) for T in rots]
    return rots + mirrors


@cache
def sn_affines(n: int) -> list[np.ndarray]:
    """
    RELION SN: n-fold improper rotation about +Z (rotation by 2π/n followed by reflection in XY-plane).
    Returns list of n 4x4 matrices.
    """
    if n % 2 != 0:
        raise ValueError("SN is only defined for even n.")
    m = n // 2
    rots = cn_affines(m)
    inv = np.eye(4)
    inv[:3, :3] = -np.eye(3)
    impropers = [inv @ T for T in rots]
    return rots + impropers


@cache
def dn_affines(n: int) -> list[np.ndarray]:
    """
    RELION DN: CN around +Z plus CN after rotation by π about +X.
    Returns list of 2n 4x4 matrices.
    """
    rots = cn_affines(n)
    rx = _rotx(np.pi)
    half_rots = [_embed(rx @ T[:3, :3]) for T in rots]
    return rots + half_rots


@cache
def dnv_affines(n: int) -> list[np.ndarray]:
    """
    RELION DNV: DN plus vertical mirrors (normals in YZ planes).
    Returns list of 4n 4x4 matrices.
    """
    dns = dn_affines(n)
    Mv = _mirror_from_normal([1.0, 0, 0])
    mirrors = [_embed(Mv @ T[:3, :3]) for T in dns]
    return dns + mirrors


@cache
def dnh_affines(n: int) -> list[np.ndarray]:
    """
    RELION DNH: DN plus a horizontal mirrors (normals in XY planes).
    Returns list of 4n 4x4 matrices.
    """
    dns = dn_affines(n)
    Mh = _mirror_from_normal([0, 0, 1.0])
    mirrors = [_embed(Mh @ T[:3, :3]) for T in dns]
    return dns + mirrors


@cache
def t_affines() -> list[np.ndarray]:
    """
    RELION tetrahedral T symmetry.
    Returns list of 12 4x4 matrices.
    """
    c3 = cn_affines(3)  # I, Rz(120), Rz(240) as 4×4
    Rz120 = c3[1][:3, :3]  # 3×3
    z = np.array([0.0, 0.0, 1.0])

    # RELION C2 axis and its two C3-rotated images
    v0 = np.array([0.0, 0.816496, 0.577350])
    C2_axes = [v0, Rz120 @ v0, Rz120 @ (Rz120 @ v0)]
    C2s = [_embed(_rot_axis(a, np.pi)) for a in C2_axes]  # 3 elements

    # 4 C3 axes = z and C2_i * z
    c3_axes = [z] + [T[:3, :3] @ z for T in C2s]
    R3s = []
    for a in c3_axes:
        R3s += [_embed(_rot_axis(a, 2 * np.pi / 3)), _embed(_rot_axis(a, -2 * np.pi / 3))]  # 8 elements

    mats = [np.eye(4)] + C2s + R3s  # 1 + 3 + 8 = 12

    return mats


@cache
def td_affines() -> list[np.ndarray]:
    """
    RELION tetrahedral + inversion TD symmetry.
    Returns list of 24 4x4 matrices.
    """
    T = t_affines()
    M = _mirror_from_normal([1.4142136, 2.4494897, 0.0])  # RELION's plane
    MT = [_embed(M @ A[:3, :3]) for A in T]
    return T + MT


@cache
def o_affines() -> list[np.ndarray]:
    """
    RELION octahedral O symmetry.
    Returns list of 24 4x4 matrices.
    """
    mats3 = [np.eye(3)]

    # C4 about coordinate axes (add 90°, 180°, 270°)
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        for th in (np.pi / 2, np.pi, 3 * np.pi / 2):
            mats3.append(_rot_axis(axis, th))

    # C2 about the 6 edge axes
    edge_axes = [
        [1, 1, 0],
        [1, -1, 0],
        [1, 0, 1],
        [1, 0, -1],
        [0, 1, 1],
        [0, 1, -1],
    ]
    for a in edge_axes:
        mats3.append(_rot_axis(a, np.pi))

    # C3 about the 4 body diagonals (z>0 reps): ±120°
    diag_axes = [[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]]
    for a in diag_axes:
        mats3.append(_rot_axis(a, 2 * np.pi / 3))
        mats3.append(_rot_axis(a, -2 * np.pi / 3))

    return [_embed(M) for M in mats3]


@cache
def oh_affines() -> list[np.ndarray]:
    """
    RELION octahedral + inversion OH symmetry.
    Returns list of 48 4x4 matrices.
    """
    O_matrices = o_affines()
    M = _mirror_from_normal([0.0, 1.0, 1.0])
    MO_matrices = [_embed(M @ T[:3, :3]) for T in O_matrices]
    return O_matrices + MO_matrices


def _trilinear_fftw_half_complex(volume: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Trilinear interpolation for a half Fourier volume. Based on RELION's Interpolation::linearXYZ_FftwHalf_complex.
    Returns the interpolated values at the (x,y,z) coordinates.
    """
    d, h, wh = volume.shape

    # sign handling (mirror and conjugate if x < 0)
    conj_mask = x <= 0.0
    xd = np.where(conj_mask, -x, x)
    yd = np.where(conj_mask, -y, y)
    zd = np.where(conj_mask, -z, z)

    # wrap negatives in y/z
    yd = np.where(yd < 0.0, yd + h, yd)
    zd = np.where(zd < 0.0, zd + d, zd)

    # integer bases computed BEFORE clamping (to match C++)
    x0_raw = np.floor(xd).astype(np.int64)
    y0_raw = np.floor(yd).astype(np.int64)
    z0_raw = np.floor(zd).astype(np.int64)

    xf = xd - x0_raw
    yf = yd - y0_raw
    zf = zd - z0_raw

    # clamp / wrap indices like in the C++ code
    x0 = np.minimum(x0_raw, wh - 1)
    x1 = x0 + 1
    x1 = np.where(x1 >= wh, wh - 2, x1)

    y0 = np.clip(y0_raw, 0, h - 1)
    y1 = (y0 + 1) % h

    z0 = np.clip(z0_raw, 0, d - 1)
    z1 = (z0 + 1) % d

    # gather the eight corners (note: ndarray order is [z, y, x])
    vx00 = (1 - xf) * volume[z0, y0, x0] + xf * volume[z0, y0, x1]
    vx10 = (1 - xf) * volume[z0, y1, x0] + xf * volume[z0, y1, x1]
    vx01 = (1 - xf) * volume[z1, y0, x0] + xf * volume[z1, y0, x1]
    vx11 = (1 - xf) * volume[z1, y1, x0] + xf * volume[z1, y1, x1]

    vxy0 = (1 - yf) * vx00 + yf * vx10
    vxy1 = (1 - yf) * vx01 + yf * vx11
    vxyz = (1 - zf) * vxy0 + zf * vxy1

    # conjugate if we mirrored across x
    out = np.where(conj_mask, np.conjugate(vxyz), vxyz)
    return out


def symmetrise_fs_complex(volume: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    """
    Symmetrise a half Fourier volume given a set of 4x4 affine transformation matrices. Based on RELION's Symmetry::symmetrise_FS_complex.

    Returns the symmetrised volume.
    """
    box_size = volume.shape[0]
    half_box_size = box_size // 2 + 1

    # precompute (y,z)-centered coordinates for all indices
    yy_all = np.arange(box_size, dtype=np.float64)
    yy_all[yy_all >= box_size // 2] -= box_size
    zz_all = np.arange(box_size, dtype=np.float64)
    zz_all[zz_all >= box_size // 2] -= box_size
    x_all = np.arange(half_box_size, dtype=np.float64)

    # start with the original volume
    accum = volume.astype(np.complex128, copy=True)

    # normalized coordinates for phase factor
    nx = x_all / half_box_size
    ny = yy_all / half_box_size
    nz = zz_all / half_box_size

    # skip identity at index 0
    for transform in transforms[1:]:
        # apply linear part of transform to coordinates; translation excluded here (w=0)
        px = transform[0, 0] * x_all + transform[0, 1] * yy_all + transform[0, 2] * zz_all
        py = transform[1, 0] * x_all + transform[1, 1] * yy_all + transform[1, 2] * zz_all
        pz = transform[2, 0] * x_all + transform[2, 1] * yy_all + transform[2, 2] * zz_all

        # trilinear sampling in the FFTW-half volume with the x-sign conjugation logic
        val = _trilinear_fftw_half_complex(volume, px, py, pz)

        # translation enters ONLY as a complex phase (exactly like the C++ dotp)
        tx, ty, tz = transform[0, 3], transform[1, 3], transform[2, 3]
        dotp = 2 * np.pi * (nx * tx + ny * ty + nz * tz)
        phase = np.cos(dotp) + 1j * np.sin(dotp)

        accum += val * phase

    return (accum / len(transforms)).astype(volume.dtype)
