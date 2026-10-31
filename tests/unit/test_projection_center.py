"""Particles are cut out about the tilt image's centre ``(W / 2, H / 2)``, odd sizes included.

RELION builds its projection about ``int(size / 2)`` (Tomogram::setProjectionMatrix, tomogram.cpp:17-62) while it
decentres about ``size / 2.0`` (tomogram_set.cpp:313); zarr-particle-tools uses ``size / 2.0`` throughout, so along an
odd dimension its crops sit half a pixel from RELION's, and the tomogram size does not enter.
"""

import numpy as np
import pandas as pd
import pytest

from zarr_particle_tools.core.forwardprojection import (
    calculate_projection_matrix_from_starfile_df,
    get_particle_crop_and_visibility,
    get_particles_to_tiltseries_coordinates,
)

PIXEL = 1.341
BOX = 64


def _rotation(axis: str, degrees: float) -> np.ndarray:
    """gravis t3Matrix::rotation(axis, angle) for a coordinate axis: right-handed, in degrees (t3Matrix.h:478-496)."""
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    r = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]], "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]]}
    r["z"] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return np.array(r[axis])


def expected_centre(centered_angst, tilt, image_size) -> np.ndarray:
    """RELION's projection of a centred coordinate (rotations, then the tilt shift), about the image centre W / 2."""
    nx, ny = image_size
    rotations = _rotation("z", tilt.rlnTomoZRot) @ _rotation("y", tilt.rlnTomoYTilt) @ _rotation("x", tilt.rlnTomoXTilt)
    projected = (rotations @ (np.asarray(centered_angst) / PIXEL))[:2]
    return projected + [tilt.rlnTomoXShiftAngst / PIXEL, tilt.rlnTomoYShiftAngst / PIXEL] + [nx / 2.0, ny / 2.0]


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


@pytest.mark.parametrize("image_size", [(5760, 4092), (5759, 4091)])
def test_crops_are_centered_about_half_the_image_size(image_size):
    coordinates = get_particles_to_tiltseries_coordinates(
        PARTICLES, None, TILTS, calculate_projection_matrix_from_starfile_df(TILTS)
    )
    checked = 0
    for (name, sections), (_, particle) in zip(coordinates.items(), PARTICLES.iterrows(), strict=True):
        centered = particle[
            ["rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst", "rlnCenteredCoordinateZAngst"]
        ].to_numpy(float)
        data, visible = get_particle_crop_and_visibility(_Reader(), name, sections, *image_size, PIXEL, BOX, BOX)
        assert visible == [1, 1, 1]
        for tilt, crop in zip(TILTS.itertuples(), data, strict=True):
            shift_y, shift_x = crop["subpixel_shift"]
            got = np.array([crop["tiltseries_key"][3] - shift_x, crop["tiltseries_key"][1] - shift_y]) + BOX / 2
            np.testing.assert_allclose(got, expected_centre(centered, tilt, image_size), atol=1e-6)
            np.testing.assert_array_equal(crop["coordinate"], centered)  # the CTF depth uses the coordinate as is
            checked += 1
    assert checked == len(PARTICLES) * len(TILTS)
