"""Particles are cut out where RELION 5 projects them, odd tomogram and tilt-image sizes included.

RELION decenters a particle about ``size / 2.0`` (tomogram_set.cpp:313) but builds its projection about
``int(size / 2)``, both for the specimen and for the tilt image (Tomogram::setProjectionMatrix,
tomogram.cpp:17-62). On 10521 (tomograms 4089 x 5760 x 2387) ignoring that put every crop half a pixel off.
"""

import numpy as np
import pandas as pd
import pytest

from zarr_particle_tools.core.forwardprojection import (
    calculate_projection_matrix_from_starfile_df,
    get_particle_crop_and_visibility,
    get_particles_to_tiltseries_coordinates,
    specimen_center_offset,
)

PIXEL = 1.341
BOX = 64


def _rotation(axis: str, degrees: float) -> np.ndarray:
    """gravis t3Matrix::rotation(axis, angle) for a coordinate axis: right-handed, in degrees (t3Matrix.h:478-496)."""
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    r = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]], "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]]}
    r["z"] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    out = np.eye(4)
    out[:3, :3] = r[axis]
    return out


def _translation(v) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = v
    return out


def relion_project(centered_angst, tilt: pd.Series, tomo_size, image_size) -> np.ndarray:
    """Tomogram::projectPoint of ParticleSet::getPosition: the tilt-image pixel RELION extracts around."""
    w0, h0, d0 = tomo_size
    nx, ny = image_size
    s0 = _translation(-np.array([int(w0 / 2), int(h0 / 2), int(d0 / 2)], dtype=float))
    s1 = _translation([tilt.rlnTomoXShiftAngst / PIXEL, tilt.rlnTomoYShiftAngst / PIXEL, 0.0])
    s2 = _translation([int(nx / 2), int(ny / 2), 0.0])
    rotations = _rotation("z", tilt.rlnTomoZRot) @ _rotation("y", tilt.rlnTomoYTilt) @ _rotation("x", tilt.rlnTomoXTilt)
    projection = s1 @ s2 @ rotations @ s0
    decentered = np.asarray(centered_angst) / PIXEL + np.array([w0 / 2.0, h0 / 2.0, d0 / 2.0])
    return (projection @ np.append(decentered, 1.0))[:2]


class _Reader:
    def slice_data(self, key):
        pass


TILTS = pd.DataFrame(
    {
        "rlnMicrographName": ["1@ts.mrcs", "2@ts.mrcs", "3@ts.mrcs"],
        "rlnTomoXTilt": [0.0, 1.5, -2.0],
        "rlnTomoYTilt": [-45.01, 0.0, 30.0],
        "rlnTomoZRot": [-95.2774, -95.2774, 84.0],
        "rlnTomoXShiftAngst": [-143.390448, 12.0, 300.0],
        "rlnTomoYShiftAngst": [292.131486, -5.0, 10.0],
    }
)
PARTICLES = pd.DataFrame(
    {
        "rlnTomoParticleName": ["t/1", "t/2", "t/3"],
        "rlnCenteredCoordinateXAngst": [-104.994397, 428.885616, 0.0],
        "rlnCenteredCoordinateYAngst": [240.277624, -1320.460524, 0.0],
        "rlnCenteredCoordinateZAngst": [0.639891, 193.34534, 0.0],
    }
)


@pytest.mark.parametrize("tomo_size", [(4089, 5760, 2387), (4092, 5760, 2388), (511, 513, 201)])
@pytest.mark.parametrize("image_size", [(5760, 4092), (5759, 4091)])
def test_crops_are_centered_where_relion_projects(tomo_size, image_size):
    row = pd.Series(dict(zip(("rlnTomoSizeX", "rlnTomoSizeY", "rlnTomoSizeZ"), tomo_size, strict=True)))
    coordinates = get_particles_to_tiltseries_coordinates(
        PARTICLES,
        None,
        TILTS,
        calculate_projection_matrix_from_starfile_df(TILTS),
        projection_offset=specimen_center_offset(row, PIXEL),
    )
    checked = 0
    for (name, sections), (_, particle) in zip(coordinates.items(), PARTICLES.iterrows(), strict=True):
        centered = particle[
            ["rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst", "rlnCenteredCoordinateZAngst"]
        ]
        data, visible = get_particle_crop_and_visibility(_Reader(), name, sections, *image_size, PIXEL, BOX, BOX)
        assert visible == [1, 1, 1]
        for tilt, crop in zip(TILTS.itertuples(), data, strict=True):
            shift_y, shift_x = crop["subpixel_shift"]
            got = np.array([crop["tiltseries_key"][3] - shift_x, crop["tiltseries_key"][1] - shift_y]) + BOX / 2
            np.testing.assert_allclose(
                got, relion_project(centered.to_numpy(float), tilt, tomo_size, image_size), atol=1e-6
            )
            # the CTF depth is measured from the size / 2.0 center (Tomogram::getDepthOffset): no offset there
            np.testing.assert_array_equal(crop["coordinate"], centered.to_numpy(float))
            checked += 1
    assert checked == len(PARTICLES) * len(TILTS)


def test_even_or_unknown_sizes_move_nothing():
    even = pd.Series({"rlnTomoSizeX": 4092, "rlnTomoSizeY": 5760, "rlnTomoSizeZ": 2388})
    assert not specimen_center_offset(even, PIXEL).any()
    assert not specimen_center_offset(pd.Series({"rlnTomoName": "t"}), PIXEL).any()
    odd = pd.Series({"rlnTomoSizeX": 4089, "rlnTomoSizeY": 5760, "rlnTomoSizeZ": 2387.0})
    np.testing.assert_array_equal(specimen_center_offset(odd, PIXEL), [PIXEL / 2, 0.0, PIXEL / 2])
