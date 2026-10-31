"""RELION-5 per-tilt 2D image deformations (tomo Bayesian polishing with ``--deformation``).

RELION's ``Tomogram::projectPoint`` (src/jaz/tomography/tomogram.cpp) projects a 3D particle position into a
tilt image and, when the tomogram set declares deformations, displaces the resulting 2D point by a smooth field
evaluated at that point.  Every crop RELION takes from a tilt image goes through it, so an extraction that skips
the field is mis-centred by the local field value (tens of Angstrom on polished data).

Where the model lives in the star files (``TomogramSet::loadTomogram``, tomogram_set.cpp):

* tomograms star, ``data_global``: ``rlnTomoDeformationGridSizeX``, ``rlnTomoDeformationGridSizeY`` and
  ``rlnTomoDeformationType`` (``spline``, ``linear`` or ``Fourier``).  Deformations exist iff both grid-size
  columns are present.
* tilt-series star, one row per tilt: ``rlnTomoDeformationCoefficients``, a bracketed comma-separated list.

The 2D point is in RELION's image frame: unbinned tilt-image pixels with the origin at the image corner, i.e.
``p_px = p_centred_angst / pixel_size + (int(width / 2), int(height / 2))``.  The returned shift is in the same
pixels.  ``image_size`` is the tilt-series stack size (``stackSize.xy()``), never ``rlnTomoSizeX/Y``.

Coefficient layouts (RELION ``RawImage`` is x-fastest, then y, then the third index):

spline   ``8 * gx * gy`` numbers: ``RawImage<DataPoint>(gx, gy, 2)``, ``DataPoint = (value, slope_x, slope_y,
         twist)``; bicubic Hermite patches on a node grid spanning the image with spacing
         ``image_size / (grid - 1)``; points outside the image are clamped to the border cell
         (``Spline2DDeformationModel::projectPoint``).
linear   3 numbers ``(axx, axy, ayy)``: ``shift = (axx * rx, axy * rx + ayy * ry)`` with ``r = p - image_size / 2``.
Fourier  2 complex numbers per spatial frequency and dimension, ``RawImage<dComplex>(n_freq, 2)``; the frequency
         list is the one ``Fourier2DDeformationModel`` builds from the grid size.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GRID_X = "rlnTomoDeformationGridSizeX"
GRID_Y = "rlnTomoDeformationGridSizeY"
TYPE = "rlnTomoDeformationType"
COEFFICIENTS = "rlnTomoDeformationCoefficients"
TYPES = ("spline", "linear", "Fourier")


def parse_coefficients(value) -> np.ndarray:
    """``"[1.0,2.0,...]"`` (RELION's vector syntax), a list, or an array -> float array."""
    if isinstance(value, str):
        body = value.strip()
        if body.startswith("[") and body.endswith("]"):
            body = body[1:-1]
        return np.array([float(v) for v in body.split(",") if v.strip()], dtype=float)
    return np.asarray(value, dtype=float).ravel()


class SplineDeformation:
    """``Spline2DDeformationModel::computeShift``, vectorised over points."""

    def __init__(self, image_size: tuple[int, int], grid_size: tuple[int, int], coefficients):
        self.gx, self.gy = int(grid_size[0]), int(grid_size[1])
        if self.gx < 2 or self.gy < 2:
            raise ValueError(f"spline deformation grid must be at least 2 x 2, got {self.gx} x {self.gy}")
        c = parse_coefficients(coefficients)
        if c.size != 8 * self.gx * self.gy:
            raise ValueError(
                f"spline deformation needs 8 * {self.gx} * {self.gy} = {8 * self.gx * self.gy} coefficients, "
                f"got {c.size}"
            )
        # RawImage<DataPoint>(gx, gy, 2): x fastest, then y, then dim; 4 doubles per node -> [dim, y, x, field]
        self.nodes = c.reshape(2, self.gy, self.gx, 4)
        self.spacing = np.array([image_size[0] / (self.gx - 1), image_size[1] / (self.gy - 1)], dtype=float)

    def shift(self, points_px: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points_px, dtype=float)) / self.spacing
        eps = 1e-10
        p = np.minimum(np.maximum(p, 0.0), np.array([self.gx, self.gy]) - 1 - eps)
        cell = p.astype(int)
        fx, fy = p[:, 0] - cell[:, 0], p[:, 1] - cell[:, 1]
        vx = _hermite_basis(fx)  # (N, 4)
        vy = _hermite_basis(fy)
        out = np.empty((len(p), 2), dtype=float)
        for dim in range(2):
            n = self.nodes[dim]  # [y, x, field]
            d00 = n[cell[:, 1], cell[:, 0]]
            d01 = n[cell[:, 1] + 1, cell[:, 0]]
            d10 = n[cell[:, 1], cell[:, 0] + 1]
            d11 = n[cell[:, 1] + 1, cell[:, 0] + 1]
            value, slope_x, slope_y, twist = 0, 1, 2, 3
            # F as in computeShift, rows index vx, columns index vy
            F = np.stack(
                [
                    np.stack([d00[:, value], d01[:, value], d00[:, slope_y], d01[:, slope_y]], axis=-1),
                    np.stack([d10[:, value], d11[:, value], d10[:, slope_y], d11[:, slope_y]], axis=-1),
                    np.stack([d00[:, slope_x], d01[:, slope_x], d00[:, twist], d01[:, twist]], axis=-1),
                    np.stack([d10[:, slope_x], d11[:, slope_x], d10[:, twist], d11[:, twist]], axis=-1),
                ],
                axis=1,
            )  # (N, 4, 4)
            out[:, dim] = np.einsum("ni,nij,nj->n", vx, F, vy)
        return out


def _hermite_basis(t: np.ndarray) -> np.ndarray:
    t2 = t * t
    t3 = t2 * t
    return np.stack([1.0 - 3.0 * t2 + 2.0 * t3, 3.0 * t2 - 2.0 * t3, t - 2.0 * t2 + t3, -t2 + t3], axis=-1)


class LinearDeformation:
    """``Linear2DDeformationModel::computeShift``."""

    def __init__(self, image_size: tuple[int, int], coefficients):
        c = parse_coefficients(coefficients)
        if c.size != 3:
            raise ValueError(f"linear deformation needs 3 coefficients (axx, axy, ayy), got {c.size}")
        self.axx, self.axy, self.ayy = c
        self.centre = np.array([image_size[0] / 2.0, image_size[1] / 2.0], dtype=float)

    def shift(self, points_px: np.ndarray) -> np.ndarray:
        r = np.atleast_2d(np.asarray(points_px, dtype=float)) - self.centre
        return np.stack([self.axx * r[:, 0], self.axy * r[:, 0] + self.ayy * r[:, 1]], axis=-1)


class FourierDeformation:
    """``Fourier2DDeformationModel::computeShift`` with the frequency list of its constructor."""

    def __init__(self, image_size: tuple[int, int], grid_size: tuple[int, int], coefficients):
        gx, gy = int(grid_size[0]), int(grid_size[1])
        freqs = []
        for y in range(gy):
            if y < gy // 2:
                for x in range(1 if y == 0 else 0, gx // 2 + 1):
                    freqs.append((x * np.pi / image_size[0], y * np.pi / image_size[1]))
            else:
                for x in range(1, gx // 2 + 1):
                    freqs.append((x * np.pi / image_size[0], (y - gy) * np.pi / image_size[1]))
        self.freqs = np.array(freqs, dtype=float).reshape(-1, 2)
        n = len(self.freqs)
        c = parse_coefficients(coefficients)
        if c.size < 4 * n:
            raise ValueError(f"Fourier deformation needs at least 4 * {n} coefficients, got {c.size}")
        z = c[: 2 * 2 * n].reshape(2, n, 2)  # [dim, freq, (real, imag)] with stride n_freq per dim
        self.real, self.imag = z[..., 0], z[..., 1]

    def shift(self, points_px: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points_px, dtype=float))
        t = p @ self.freqs.T  # (N, n_freq)
        cos_t, sin_t = np.cos(t), np.sin(t)
        return np.stack([cos_t @ self.real[d] + sin_t @ self.imag[d] for d in range(2)], axis=-1)


class ImageDeformation:
    """One tilt's deformation, callable on centred-Angstrom 2D points (what the projection code produces)."""

    def __init__(self, model, image_size: tuple[int, int], pixel_size: float):
        self.model = model
        self.pixel_size = float(pixel_size)
        # RELION's tilt_image_center: integer division (tomogram.cpp setProjectionMatrix)
        self.origin_px = np.array([int(image_size[0] // 2), int(image_size[1] // 2)], dtype=float)

    def shift_px(self, points_px: np.ndarray) -> np.ndarray:
        return self.model.shift(points_px)

    def __call__(self, point_angst: np.ndarray) -> np.ndarray:
        point_angst = np.asarray(point_angst, dtype=float)
        p_px = point_angst / self.pixel_size + self.origin_px
        return point_angst + self.model.shift(p_px)[0] * self.pixel_size


def make_model(deformation_type: str, image_size, grid_size, coefficients):
    if deformation_type == "spline":
        return SplineDeformation(image_size, grid_size, coefficients)
    if deformation_type == "linear":
        return LinearDeformation(image_size, coefficients)
    if deformation_type == "Fourier":
        return FourierDeformation(image_size, grid_size, coefficients)
    raise ValueError(f"illegal deformation type '{deformation_type}' (RELION knows {', '.join(TYPES)})")


def has_deformations(tomogram_row: pd.Series) -> bool:
    """RELION: deformations exist iff both grid-size columns are present (NaN here means the column was absent)."""
    return all(c in tomogram_row.index and pd.notna(tomogram_row[c]) for c in (GRID_X, GRID_Y))


def declaration_problems(tomogram_row: pd.Series, tiltseries_df: pd.DataFrame) -> list[tuple[str, str]]:
    """``(column, message)`` for each reason a tomogram's declared deformations cannot be built.

    Empty if they can or none are declared; as in RELION, a declaration without coefficients applies none.
    """
    if not has_deformations(tomogram_row):
        return []
    name = tomogram_row.get("rlnTomoName", "?")
    deformation_type = str(tomogram_row[TYPE]) if TYPE in tomogram_row.index else ""
    if deformation_type not in TYPES:
        return [
            (TYPE, f"tomogram {name}: illegal deformation type '{deformation_type}' (RELION knows {', '.join(TYPES)})")
        ]
    if COEFFICIENTS not in tiltseries_df.columns:
        return []
    grid = (int(tomogram_row[GRID_X]), int(tomogram_row[GRID_Y]))
    for coefficients in tiltseries_df[COEFFICIENTS].tolist():
        try:  # the image size scales the field, not the number of coefficients
            make_model(deformation_type, (1, 1), grid, coefficients)
        except ValueError as exc:
            return [(COEFFICIENTS, f"tomogram {name}: {exc}")]
    return []


def build_deformations(
    tomogram_row: pd.Series,
    tiltseries_df: pd.DataFrame,
    image_size: tuple[int, int],
    pixel_size: float,
) -> list[ImageDeformation] | None:
    """Per-tilt deformations for one tilt series, in tilt-series-star row order; ``None`` if none are applied.

    ``image_size`` is ``(width, height)`` of the tilt-series stack in unbinned pixels.
    """
    problems = declaration_problems(tomogram_row, tiltseries_df)
    if problems:
        raise ValueError("; ".join(f"{column}: {message}" for column, message in problems))
    if not has_deformations(tomogram_row) or COEFFICIENTS not in tiltseries_df.columns:
        return None
    deformation_type = str(tomogram_row[TYPE])
    grid = (int(tomogram_row[GRID_X]), int(tomogram_row[GRID_Y]))
    return [
        ImageDeformation(make_model(deformation_type, image_size, grid, coeffs), image_size, pixel_size)
        for coeffs in tiltseries_df[COEFFICIENTS].tolist()
    ]
