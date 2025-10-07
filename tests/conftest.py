import logging
import os
from pathlib import Path

import pytest
import starfile

from tests.helpers.compare import df_equal, mrc_equal


@pytest.fixture(autouse=True)
def fail_on_log_warning_or_error(caplog):
    caplog.set_level(logging.WARNING)
    yield
    bad_logs = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    if bad_logs:
        msgs = "\n".join(f"{rec.levelname}: {rec.getMessage()}" for rec in bad_logs)
        pytest.fail(f"Unexpected warning/error logs were emitted:\n{msgs}")


@pytest.fixture
def validate_optimisation_set_starfile():
    def _validate_starfile(star_file: Path):
        optimisation_dict = starfile.read(star_file)
        assert len(optimisation_dict) == 2, f"Expected exactly two rows in {star_file}, found {len(optimisation_dict)}"
        assert "rlnTomoTomogramsFile" in optimisation_dict
        assert "rlnTomoParticlesFile" in optimisation_dict
        assert os.path.exists(optimisation_dict["rlnTomoTomogramsFile"])
        assert os.path.exists(optimisation_dict["rlnTomoParticlesFile"])

    return _validate_starfile


@pytest.fixture
def validate_particles_starfile():
    def _validate_starfile(star_file: Path, expected_starfile: Path):
        star_file_data = starfile.read(star_file)
        expected_data = starfile.read(expected_starfile)
        assert star_file_data["general"] == expected_data["general"]
        star_file_data["optics"].drop(columns=["BoxSize"], inplace=True, errors="ignore")
        assert df_equal(star_file_data["optics"], expected_data["optics"])
        particles_simplified = star_file_data["particles"].drop(columns=["rlnImageName"])
        expected_particles_simplified = expected_data["particles"].drop(columns=["rlnImageName"])
        assert df_equal(particles_simplified, expected_particles_simplified)

    return _validate_starfile


@pytest.fixture
def validate_starfile():
    def _validate_starfile(star_file: Path, expected_starfile: Path):
        star_file_data = starfile.read(star_file)
        expected_data = starfile.read(expected_starfile)
        assert type(star_file_data) is type(
            expected_data
        ), f"Type mismatch: {type(star_file_data)} vs {type(expected_data)}"
        if type(star_file_data) is dict:
            for key in expected_data:
                assert key in star_file_data
                assert df_equal(star_file_data[key], expected_data[key])
        else:
            assert df_equal(star_file_data, expected_data)

    return _validate_starfile


@pytest.fixture
def compare_mrcs_dirs():
    def _compare_dirs(dir1: str, dir2: str, tol: float):
        dir1 = Path(dir1)
        dir2 = Path(dir2)

        for file1 in dir1.rglob("*.mrcs"):
            relative_path = file1.relative_to(dir1)
            file2 = dir2 / relative_path

            assert file2.exists(), f"Expected file missing: {file2}"
            assert mrc_equal(file1, file2, tol=tol), f"{file1} and {file2} differ beyond tol={tol}"

    return _compare_dirs
