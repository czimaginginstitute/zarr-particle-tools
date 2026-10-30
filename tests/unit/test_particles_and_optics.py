"""A particle STAR without optics (relion_tomo_import_coordinates) gets them as RELION's ParticleSet builds them."""

from pathlib import Path

import pandas as pd
import pytest

from zarr_particle_tools.core.helpers import particles_and_optics, read_tomograms_starfile

TOMOGRAMS = pd.DataFrame(
    {
        "rlnTomoName": ["16848", "16849", "16850", "16851"],
        "rlnVoltage": [300.0, 300.0, 300.0, 200.0],
        "rlnSphericalAberration": [2.7, 2.71, 2.7, 2.7],
        "rlnAmplitudeContrast": [0.07, 0.07, 0.07, 0.07],
        "rlnTomoTiltSeriesPixelSize": [2.165, 2.165, 2.165, 1.5],
        "rlnOpticsGroupName": ["a", "a", "b", "c"],
    }
)


def test_groups_follow_relion_particle_set():
    # particle_set.cpp:41-121: group by rlnOpticsGroupName, numbered by first appearance in the particles, only
    # groups with particles, values from the group's first tomogram in particle order; no rlnOpticsGroup needed
    particles = pd.DataFrame({"rlnTomoName": ["16850", "16849", "16848", "16850"], "rlnOpticsGroup": [1, 1, 1, 1]})
    got, optics = particles_and_optics(particles, TOMOGRAMS)
    assert list(got["rlnOpticsGroup"]) == [1, 2, 2, 1]
    assert list(optics["rlnOpticsGroup"]) == [1, 2]
    assert list(optics["rlnOpticsGroupName"]) == ["b", "a"]
    assert list(optics["rlnSphericalAberration"]) == [2.7, 2.71]  # group a's first tomogram here is 16849
    assert "rlnTomoName" not in optics.columns
    assert list(particles["rlnOpticsGroup"]) == [1, 1, 1, 1]  # the input is not modified


def test_without_group_names_each_tomogram_is_a_group():
    particles = pd.DataFrame({"rlnTomoName": ["16851", "16848"]})
    got, optics = particles_and_optics({"particles": particles}, TOMOGRAMS.drop(columns="rlnOpticsGroupName"))
    assert list(got["rlnOpticsGroup"]) == [1, 2]
    assert list(optics["rlnOpticsGroupName"]) == ["16851", "16848"]
    assert list(optics["rlnVoltage"]) == [200.0, 300.0]


def test_a_two_block_star_is_returned_as_is():
    optics = pd.DataFrame({"rlnOpticsGroup": [7]})
    particles = pd.DataFrame({"rlnTomoName": ["16848"]})
    got, got_optics = particles_and_optics({"optics": optics, "particles": particles}, TOMOGRAMS)
    assert got_optics is optics and got is particles


def test_a_particle_on_an_unknown_tomogram_gets_no_group_for_the_caller_to_skip():
    got, optics = particles_and_optics(pd.DataFrame({"rlnTomoName": ["1", "16848"]}), TOMOGRAMS)
    assert list(got["rlnOpticsGroup"]) == [0, 1]
    assert list(optics["rlnOpticsGroupName"]) == ["a"]


def test_missing_optics_values_are_named():
    with pytest.raises(ValueError, match="rlnVoltage"):
        particles_and_optics(pd.DataFrame({"rlnTomoName": ["16848"]}), TOMOGRAMS.drop(columns="rlnVoltage"))


SYNTHETIC = Path(__file__).parents[1] / "data" / "relion_project_synthetic" / "tomograms.star"


@pytest.mark.skipif(not SYNTHETIC.exists(), reason="test data not downloaded")
def test_relion_tomograms_star_without_optics_group_numbers():
    _, tomograms_df = read_tomograms_starfile(SYNTHETIC)
    assert "rlnOpticsGroup" not in tomograms_df.columns
    got, optics = particles_and_optics(pd.DataFrame({"rlnTomoName": ["session1_TS_1"] * 2}), tomograms_df)
    assert list(got["rlnOpticsGroup"]) == [1, 1]
    assert optics[["rlnOpticsGroup", "rlnOpticsGroupName"]].values.tolist() == [[1, "polnet"]]
