import pytest
import starfile
import os
from pathlib import Path
from tests.helpers.compare import mrcs_equal, df_equal


@pytest.fixture
def validate_optimisation_set_starfile():
    def _validate_starfile(star_file: Path):
        df = starfile.read(star_file)
        assert len(df) == 1, f"Expected exactly one row in {star_file}, found {len(df)}"
        assert len(df.columns) == 2, f"Expected exactly two columns in {star_file}, found {len(df.columns)}"
        assert "rlnTomoTomogramsFile" in df.columns
        assert "rlnTomoParticlesFile" in df.columns
        assert os.path.exists(df["rlnTomoTomogramsFile"].iloc[0])
        assert os.path.exists(df["rlnTomoParticlesFile"].iloc[0])

    return _validate_starfile


@pytest.fixture
def validate_particles_starfile():
    def _validate_starfile(star_file: Path, expected_starfile: Path):
        star_file_data = starfile.read(star_file)
        expected_data = starfile.read(expected_starfile)
        assert star_file_data["general"] == expected_data["general"]
        assert df_equal(star_file_data["optics"], expected_data["optics"])
        particles_simplified = star_file_data["particles"].drop(columns=["rlnImageName"])
        expected_particles_simplified = expected_data["particles"].drop(columns=["rlnImageName"])
        assert df_equal(particles_simplified, expected_particles_simplified)

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
            assert mrcs_equal(file1, file2, tol=tol), f"{file1} and {file2} differ beyond tol={tol}"

    return _compare_dirs
