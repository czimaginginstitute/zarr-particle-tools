"""Copick runs named by portal run id map onto portal-named tomograms (rlnTomoName = get_tomo_name)."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import starfile

import zarr_particle_tools.generate.cdp_import_starfiles as cdp_import
import zarr_particle_tools.generate.copick_generate_starfiles as copick_generate
import zarr_particle_tools.orchestrate as orch
from tests.unit.test_portal_import import run
from zarr_particle_tools.core.constants import OPTICS_DF_COLUMNS, PARTICLES_DF_COLUMNS
from zarr_particle_tools.core.helpers import STAR_NAME_COLUMNS
from zarr_particle_tools.subtomo_extract import parse_extract_copick_local_subtomograms


class Pick:
    def __init__(self, run_name):
        self.run = SimpleNamespace(name=run_name)

    def df(self, format):
        return pd.DataFrame({c: [1.0] for c in PARTICLES_DF_COLUMNS if c not in ("rlnTomoName", "rlnOpticsGroup")})


@pytest.mark.parametrize("naming", ["run id", "rlnTomoName"])
def test_pipeline_and_extract_map_the_same_copick_runs(tmp_path, monkeypatch, naming):
    runs = [run(16848), run(16849)]
    names = ["16848", "16849"] if naming == "run id" else [r.tomo_name for r in runs]
    tomograms = cdp_import.get_tomograms_df(runs, tmp_path / "Import")
    star = tmp_path / "tomograms.star"
    starfile.write({"global": tomograms}, star)
    optics = tomograms[OPTICS_DF_COLUMNS]
    monkeypatch.setattr(copick_generate, "get_copick_picks", lambda *args: [Pick(n) for n in names])

    # zarr-particle-pipeline copick-local
    orch._check_copick_run_mapping(names, optics)
    particles = copick_generate.copick_picks_to_starfile(None, "p", "1", "u", names, optics)
    assert particles["rlnTomoName"].tolist() == [r.tomo_name for r in runs]
    assert particles["rlnOpticsGroup"].tolist() == [1, 2]

    # zarr-particle-extract copick-local (a dry run does not create its output dir, but checks the tilt series)
    (tmp_path / "extract").mkdir()
    for _, row in tomograms.iterrows():
        Path(row["rlnTomoTiltSeriesStarFile"]).parent.mkdir(parents=True, exist_ok=True)
        starfile.write(
            {row["rlnTomoName"]: pd.DataFrame({"rlnMicrographName": ["1@x.mrcs"]})}, row["rlnTomoTiltSeriesStarFile"]
        )
    parse_extract_copick_local_subtomograms(
        box_size=None,
        output_dir=tmp_path / "extract",
        copick_config=None,
        copick_name="p",
        copick_session_id="1",
        copick_user_id="u",
        copick_run_names=names,
        tomograms_starfile=star,
        dry_run=True,
    )
    written = starfile.read(tmp_path / "extract" / "particles.star", parse_as_string=STAR_NAME_COLUMNS)
    assert written["particles"]["rlnTomoName"].tolist() == [r.tomo_name for r in runs]


def test_a_run_id_does_not_match_another_run_by_substring():
    optics = cdp_import.get_optics_df([run(16848), run(1684, 99)])
    assert copick_generate.copick_run_optics_rows(optics, "1684")["rlnTomoName"].tolist() == [run(1684, 99).tomo_name]
