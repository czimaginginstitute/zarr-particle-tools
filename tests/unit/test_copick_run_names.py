"""Copick runs named by portal run id map onto portal-named tomograms (rlnTomoName = get_tomo_name)."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import starfile

import zarr_particle_tools.generate.cdp_import_starfiles as cdp_import
import zarr_particle_tools.generate.copick_generate_starfiles as copick_generate
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

    # zarr-particle-pipeline copick-local maps through the same function
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


def _map(monkeypatch, runs, optics):
    monkeypatch.setattr(copick_generate, "get_copick_picks", lambda *args: [Pick(r) for r in runs])
    return copick_generate.copick_picks_to_starfile(None, "p", "1", "u", runs, optics)


def _optics(group_names, **columns):
    return pd.DataFrame(
        {"rlnOpticsGroup": range(1, len(group_names) + 1), "rlnOpticsGroupName": list(group_names), **columns}
    )


def test_portal_run_ids_map_by_anchored_prefix(monkeypatch):
    optics = _optics(["run_33379_tiltseries_1", "run_44444_tiltseries_2"])
    assert _map(monkeypatch, ["33379", "44444"], optics)["rlnOpticsGroup"].tolist() == [1, 2]


def test_exact_names_map(monkeypatch):
    assert _map(monkeypatch, ["tomo1", "tomo2"], _optics(["tomo1", "tomo2"]))["rlnOpticsGroup"].tolist() == [1, 2]


def test_two_runs_mapping_to_one_row_are_rejected(monkeypatch):
    """Extraction and the pipeline both refuse it: they map through copick_picks_to_starfile."""
    optics = _optics(["tomo1", "tomo2"], rlnTomoName=["run_1", "x"])
    with pytest.raises(ValueError, match="both match optics group tomo1"):
        _map(monkeypatch, ["tomo1", "run_1"], optics)


def test_run_id_colliding_with_another_tiltseries_id_is_rejected(monkeypatch):
    """A bare containment test matched run 16106 against run_33379_tiltseries_16106, writing its particles under the
    wrong tomogram; portal run and tilt-series IDs share an ID space."""
    with pytest.raises(ValueError, match="No matching optics group found for 16106"):
        _map(monkeypatch, ["33379", "16106"], _optics(["run_33379_tiltseries_16106", "run_44444_tiltseries_5"]))


def test_ambiguous_runs_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="No matching optics group found for 1"):
        _map(monkeypatch, ["1"], _optics(["run_10_tiltseries_1", "run_11_tiltseries_2"]))
