import pandas as pd
import pytest
import starfile

from zarr_particle_tools.core.helpers import read_tomograms_starfile
from zarr_particle_tools.orchestrate import read_tomograms_df
from zarr_particle_tools.subtomo_relion_job import read_global_tomograms


@pytest.mark.parametrize("read", [read_tomograms_starfile, read_global_tomograms, read_tomograms_df])
def test_duplicate_tomograms_rejected_with_one_message(tmp_path, read):
    tomograms = pd.DataFrame({"rlnTomoName": ["007", "008", "007"], "rlnTomoTiltSeriesStarFile": ["a", "b", "a"]})
    starfile.write({"global": tomograms}, tmp_path / "tomograms.star")
    with pytest.raises(ValueError, match=r"Tomograms listed more than once in .*tomograms\.star: 007$"):
        read(tmp_path / "tomograms.star")


@pytest.mark.parametrize("read", [read_tomograms_starfile, read_global_tomograms])
def test_missing_tomogram_name_column(tmp_path, read):
    starfile.write({"global": pd.DataFrame({"rlnTomoTiltSeriesStarFile": ["a"]})}, tmp_path / "tomograms.star")
    with pytest.raises(ValueError, match="missing required column\\(s\\): rlnTomoName"):
        read(tmp_path / "tomograms.star")
