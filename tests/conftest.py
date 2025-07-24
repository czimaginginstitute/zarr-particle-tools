import pytest
import starfile
import os
from pathlib import Path
from tests.helpers.compare import mrcs_equal, df_equal


@pytest.fixture
def synthetic_data_root() -> Path:
    return Path("tests/data/relion_project_synthetic")


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
    def _compare_dirs(dir1: str, dir2: str, tol: float = 1e-6):
        for fname in os.listdir(dir1):
            if not fname.endswith(".mrcs"):
                continue
            f1 = os.path.join(dir1, fname)
            f2 = os.path.join(dir2, fname)
            assert os.path.exists(f2), f"Expected file missing: {f2}"
            assert mrcs_equal(f1, f2, tol), f"{f1} and {f2} differ beyond tolerance {tol}"

    return _compare_dirs
