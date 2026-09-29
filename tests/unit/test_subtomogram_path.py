from pathlib import Path

import pytest

from zarr_particle_tools.subtomo_extract import subtomogram_path


@pytest.mark.parametrize(
    "tomo_name, particle_name, expected",
    [
        ("tomo_1", "tomo_1/12", "Subtomograms/tomo_1/12_stack2d.mrcs"),
        ("tomo_1", "tomo_1/007", "Subtomograms/tomo_1/007_stack2d.mrcs"),
        ("tomo_1", "tomo_1/ext1_-3", "Subtomograms/tomo_1/ext1_-3_stack2d.mrcs"),
        ("tomo_1_bin4", "tomo_1/12", "Subtomograms/tomo_1/12_stack2d.mrcs"),
        ("tomo_1", "tomo_1/run2/12", "Subtomograms/tomo_1/run2/12_stack2d.mrcs"),
        ("tomo_1", "p12", "Subtomograms/tomo_1/p12_stack2d.mrcs"),
        ("tomo_1", 12, "Subtomograms/tomo_1/12_stack2d.mrcs"),
    ],
)
def test_subtomogram_path_matches_relion(tomo_name, particle_name, expected):
    assert subtomogram_path(Path("out"), tomo_name, particle_name) == Path("out") / expected
