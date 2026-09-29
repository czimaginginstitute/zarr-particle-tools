"""A particle STAR without optics (relion_tomo_import_coordinates) is read as RELION 5 reads it."""

import pandas as pd

from zarr_particle_tools.core.helpers import particles_and_optics

TOMOGRAMS = pd.DataFrame(
    {
        "rlnTomoName": [16848, 16849],
        "rlnVoltage": [300.0, 300.0],
        "rlnSphericalAberration": [2.7, 2.7],
        "rlnAmplitudeContrast": [0.07, 0.07],
        "rlnTomoTiltSeriesPixelSize": [2.165, 2.165],
        "rlnOpticsGroup": [1, 2],
        "rlnOpticsGroupName": ["run_16848", "run_16849"],
    }
)
PARTICLES = pd.DataFrame({"rlnTomoName": [16848, 16849, 16849], "rlnOpticsGroup": [1, 1, 1]})


def test_one_optics_group_per_tomogram_and_each_particle_gets_its_tomograms():
    particles, optics = particles_and_optics(PARTICLES, TOMOGRAMS)
    assert list(particles["rlnOpticsGroup"]) == [1, 2, 2]
    assert list(optics["rlnOpticsGroupName"]) == ["run_16848", "run_16849"]
    assert list(PARTICLES["rlnOpticsGroup"]) == [1, 1, 1]  # the input is not modified


def test_a_two_block_star_is_returned_as_is():
    optics = pd.DataFrame({"rlnOpticsGroup": [7]})
    particles, got = particles_and_optics({"optics": optics, "particles": PARTICLES}, TOMOGRAMS)
    assert got is optics and particles is PARTICLES


def test_a_particle_on_an_unknown_tomogram_keeps_its_group_for_the_caller_to_skip():
    particles, _ = particles_and_optics(pd.DataFrame({"rlnTomoName": [16848, 1], "rlnOpticsGroup": [1, 1]}), TOMOGRAMS)
    assert list(particles["rlnOpticsGroup"]) == [1, 1]
