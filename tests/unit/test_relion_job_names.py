import pandas as pd
import starfile

from zarr_particle_tools.core.constants import TILTSERIES_URI_RELION_COLUMN
from zarr_particle_tools.core.helpers import STAR_NAME_COLUMNS
from zarr_particle_tools.subtomo_relion_job import _restore_zarr_source, _stage_per_tomogram, read_global_tomograms


def test_per_tomogram_staging_keeps_zero_padded_names(tmp_path):
    # RELION gets the staged per-tomogram particles.star; "007" / "001" must not become 7 / 1
    starfile.write(
        {"global": pd.DataFrame({"rlnTomoName": ["007", "008"], "rlnTomoTiltSeriesStarFile": ["a.star", "b.star"]})},
        tmp_path / "tomograms.star",
    )
    particles = pd.DataFrame({"rlnTomoName": ["007", "008", "007"], "rlnTomoParticleName": ["001", "002", "003"]})
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnOpticsGroupName": ["007"]})
    starfile.write({"optics": optics, "particles": particles}, tmp_path / "particles.star")

    global_df, _ = read_global_tomograms(tmp_path / "tomograms.star")
    staged = _stage_per_tomogram(global_df, tmp_path / "particles.star", tmp_path / "_phase1")

    assert [name for name, _, _ in staged] == ["007", "008"]
    name, particles_star, tomo_df = staged[0]
    assert tomo_df["rlnTomoName"].tolist() == ["007"]
    written = starfile.read(particles_star, parse_as_string=STAR_NAME_COLUMNS)
    assert written["particles"]["rlnTomoParticleName"].tolist() == ["001", "003"]
    assert written["particles"]["rlnTomoName"].eq("007").all()
    assert written["optics"]["rlnOpticsGroupName"].tolist() == ["007"]


def test_restore_zarr_source_keeps_zero_padded_names(tmp_path):
    # the job rewrites RELION's output tomograms.star; that rewrite must not rename "007" to 7
    starfile.write(
        {"global": pd.DataFrame({"rlnTomoName": ["007"], "rlnTomoFrameCount": [3]})}, tmp_path / "tomograms.star"
    )
    global_df = pd.DataFrame({"rlnTomoName": ["007"], TILTSERIES_URI_RELION_COLUMN: ["s3://bucket/ts.zarr"]})

    _restore_zarr_source(tmp_path, global_df, tmp_path, None)

    restored = starfile.read(tmp_path / "tomograms.star", parse_as_string=STAR_NAME_COLUMNS)
    assert restored["rlnTomoName"].tolist() == ["007"]
    assert restored[TILTSERIES_URI_RELION_COLUMN].tolist() == ["s3://bucket/ts.zarr"]
