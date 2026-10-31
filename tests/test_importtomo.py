"""
Portal selection and the S3 import (``zarr-particle-importtomo`` / ``zarrparticletools.importtomo``).

The resolver and import tests query the live CryoET Data Portal (dataset 10426, runs 16848 = tomo153 and
16849 = tomo154), like ``test_generate.py``.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import mrcfile
import pandas as pd
import pytest
import starfile
from click.testing import CliRunner

from zarr_particle_tools.core.helpers import STAR_NAME_COLUMNS
from zarr_particle_tools.generate import portal_selection
from zarr_particle_tools.importtomo import cli

RUN = 16848
TOMO = "run_16848_tiltseries_16582_alignment_17772_spacing_17051"
BLOCK_EXTRACTION_STACK = "import sys\nsys.modules['dask'] = sys.modules['s3fs'] = sys.modules['zarr'] = None\n"


def _imports_without_the_extraction_stack(module: str) -> None:
    subprocess.run([sys.executable, "-c", f"{BLOCK_EXTRACTION_STACK}import {module}"], check=True)


def test_resolver_imports_without_the_extraction_stack():
    """A process that only plans jobs resolves selections without dask, s3fs or zarr."""
    _imports_without_the_extraction_stack("zarr_particle_tools.generate.portal_selection")


def test_job_imports_without_the_extraction_stack():
    pytest.importorskip("pipeliner", reason="ccpem-pipeliner is not installed")
    _imports_without_the_extraction_stack("zarr_particle_tools.pipeliner.importtomo_pipeliner_job")


@pytest.fixture
def job():
    pytest.importorskip("pipeliner", reason="ccpem-pipeliner is not installed")
    from zarr_particle_tools.pipeliner.importtomo_pipeliner_job import PythonPortalImportTomoJob

    job = PythonPortalImportTomoJob()
    job.output_dir = "Import/job001/"
    return job


def test_job_command(job):
    job.joboptions["dataset_ids"].value = "10426"
    job.joboptions["run_ids"].value = "16848,16849"
    cmd = [str(x) for x in job.get_commands()[0].cmd]
    assert cmd[:4] == ["zarr-particle-importtomo", "data-portal", "--output-dir", "Import/job001/"]
    assert cmd[cmd.index("--dataset-ids") + 1] == "10426"
    assert cmd[cmd.index("--hand") + 1] == "-1"
    assert "--selection" not in cmd and "--staging" not in cmd

    job.joboptions["staging"].value = True
    assert "--staging" in [str(x) for x in job.get_commands()[0].cmd]


def test_job_takes_a_stored_selection(job):
    job.joboptions["in_selection"].value = "ApexAgent/portal_selection.json"
    assert job.additional_joboption_validation() == []
    cmd = [str(x) for x in job.get_commands()[0].cmd]
    assert cmd[cmd.index("--selection") + 1] == "ApexAgent/portal_selection.json"
    assert "--dataset-ids" not in cmd and "--tomogram-type" not in cmd


def test_job_refuses_a_stored_selection_with_ids_or_filters(job):
    job.joboptions["in_selection"].value = "ApexAgent/portal_selection.json"
    job.joboptions["run_ids"].value = "16848"
    job.joboptions["tomogram_type"].value = "denoised"
    (result,) = job.additional_joboption_validation()
    assert result.type == "error"
    assert {o.label for o in result.raised_by} == {
        job.joboptions[name].label for name in ("in_selection", "run_ids", "tomogram_type")
    }
    with pytest.raises(ValueError, match="excludes the ID and filter options"):
        job.get_commands()


def test_resolve_one_run():
    selection = portal_selection.resolve(run_ids=[RUN])
    assert selection.status == "eligible"
    (run,) = selection.runs
    assert (run.dataset_id, run.tomogram_id, run.alignment_id, run.voxel_spacing_id) == (10426, 21114, 17772, 17051)
    # RELION's unit: unbinned tilt pixels (1022x1440x400 voxels at 8.66 A, tilt pixel 2.165 A)
    assert run.rln_tomo_size == (4088, 5760, 1600)
    assert run.tomogram_size == (1022, 1440, 400)
    assert run.tomo_name == TOMO
    assert run.fractional_dose == pytest.approx(2.3529, abs=1e-4)
    assert len(run.written_z_indices) == 51
    assert run.tiltseries_uri.startswith("s3://") and run.tiltseries_uri.endswith(".zarr")
    assert selection.to_dict()["runs"][0]["tomo_name"] == TOMO


def test_an_ambiguous_type_lists_its_candidates_and_a_method_settles_it():
    ambiguous = portal_selection.resolve(run_ids=[RUN], tomogram_type="raw")
    assert ambiguous.status == "ambiguous" and not ambiguous.runs
    assert {c.tomogram_id for c in ambiguous.problems[0].candidates} == {21115, 21116}  # WBP raw, SART raw

    settled = portal_selection.resolve(run_ids=[RUN], tomogram_type="raw", reconstruction_method="SART")
    assert settled.status == "eligible" and settled.runs[0].tomogram_id == 21116


def test_local_alignment_is_imported_with_a_warning(caplog):
    """Run 16463's alignment is LOCAL: its global per-tilt geometry is written, with one warning."""
    with caplog.at_level(logging.WARNING, logger="zarr_particle_tools.generate.portal_selection"):
        selection = portal_selection.resolve(run_ids=[16463])
    assert selection.status == "eligible"
    assert sum("LOCAL" in m for m in caplog.messages) == 1
    assert selection.to_dict()["runs"][0]["alignment_type"] == "LOCAL"


def test_resolve_and_build_query_each_per_section_record_once(tmp_path, monkeypatch):
    import cryoet_data_portal as cdp

    import zarr_particle_tools.generate.cdp_import_starfiles as cdp_import

    calls = []
    for model in (cdp.PerSectionParameters, cdp.PerSectionAlignmentParameters):
        monkeypatch.setattr(model, "find", lambda *a, _f=model.find, _m=model, **k: calls.append(_m) or _f(*a, **k))
    cdp_import.build(portal_selection.resolve(run_ids=[RUN]).runs, tmp_path)
    assert sorted(m.__name__ for m in calls) == ["PerSectionAlignmentParameters", "PerSectionParameters"]


def test_resolve_refetches_a_cached_tilt_series():
    import zarr_particle_tools.generate.cdp_cache as cdp_cache

    stale = portal_selection.resolve(run_ids=[RUN])
    cdp_cache.get_tiltseries([stale.runs[0].tiltseries_id])[0].pixel_spacing = 9.99  # the cached record, edited
    assert portal_selection.resolve(run_ids=[RUN]).runs[0].tiltseries_pixel_size == pytest.approx(2.165)


def test_selection_round_trips(tmp_path):
    selection = portal_selection.resolve(run_ids=[RUN])
    path = selection.write(tmp_path / portal_selection.SELECTION_FILENAME)
    assert portal_selection.Selection.read(path).comparable() == selection.comparable()


def _tilts(path) -> pd.DataFrame:
    data = starfile.read(path, always_dict=True, parse_as_string=STAR_NAME_COLUMNS)
    return next(iter(data.values()))


def test_import_writes_a_project_relative_s3_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = ["data-portal", "--run-id", str(RUN), "--run-ids", "16849", "--output-dir", "Import/job001"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output

    tomograms = _tilts("Import/job001/tomograms.star")
    assert list(tomograms["rlnTomoName"]) == [TOMO, "run_16849_tiltseries_16583_alignment_17773_spacing_17052"]
    assert list(tomograms["rlnOpticsGroupName"]) == ["run_16848_tiltseries_16582", "run_16849_tiltseries_16583"]
    row = tomograms.iloc[0]
    assert (row["rlnTomoSizeX"], row["rlnTomoSizeY"], row["rlnTomoSizeZ"]) == (4088, 5760, 1600)
    assert row["rlnTomoTiltSeriesStarFile"] == f"Import/job001/tiltseries/{TOMO}.star"
    assert row["tomoTiltSeriesURI"].endswith("tomo153/TiltSeries/100/tomo153.zarr")
    assert row["rlnTomoHand"] == -1
    assert row["rlnTomoTomogramBinning"] == pytest.approx(4.0)
    assert row["rlnTomoImportFractionalDose"] == pytest.approx(2.3529, abs=1e-4)

    tilts = _tilts(f"Import/job001/tiltseries/{TOMO}.star")
    assert len(tilts) == 51
    assert tilts["rlnMicrographName"].iloc[0] == "1@Import/job001/tiltseries/tiltseries_placeholder.mrcs"

    placeholder = Path("Import/job001/tiltseries/tiltseries_placeholder.mrcs")
    with mrcfile.open(placeholder, header_only=True) as mrc:
        assert (int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz)) == (4092, 5760, 51)
    assert os.stat(placeholder).st_blocks * 512 < 1024**2, "the placeholder must stay sparse"

    record = json.loads(Path("Import/job001/portal_selection.json").read_text())
    assert record["import"]["runs"][TOMO]["sections_written"] == 51
    assert record["import"]["runs"][TOMO]["sections_excluded"] == []
    assert record["runs"][0]["tomo_name"] == TOMO


def test_a_stored_selection_imports_exactly_or_names_the_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    selection = portal_selection.resolve(run_ids=[RUN])
    stored = selection.write(tmp_path / "stored.json")
    ok = CliRunner().invoke(cli, ["data-portal", "--selection", str(stored), "--output-dir", "a"])
    assert ok.exit_code == 0, ok.output

    for key, value in (("rln_tomo_size", [1022, 1440, 400]), ("sections_digest", "0" * 64)):
        tampered = json.loads(stored.read_text())
        tampered["runs"][0][key] = value
        (tmp_path / "tampered.json").write_text(json.dumps(tampered))
        changed = CliRunner().invoke(
            cli, ["data-portal", "--selection", str(tmp_path / "tampered.json"), "--output-dir", "b"]
        )
        assert changed.exit_code != 0
        assert "changed since the selection was resolved" in changed.output and key in changed.output
        assert not (tmp_path / "b").exists(), "a refused import writes nothing"


def test_a_stored_selection_from_an_earlier_version_imports(tmp_path, monkeypatch):
    """A selection written before tomo_name, the per-section digest and alignment_type existed is unchanged on those
    keys."""
    monkeypatch.chdir(tmp_path)
    record = portal_selection.resolve(run_ids=[RUN]).to_dict()
    for key in ("tomo_name", "fractional_dose", "sections_digest", "alignment_type"):
        record["runs"][0].pop(key)
    record["criteria"].pop("smallest_spacing")
    (tmp_path / "old.json").write_text(json.dumps(record))
    result = CliRunner().invoke(cli, ["data-portal", "--selection", "old.json", "--output-dir", "a"])
    assert result.exit_code == 0, result.output
    assert list(_tilts("a/tomograms.star")["rlnTomoName"]) == [TOMO]


def test_a_stored_selection_is_imported_from_the_portal_it_was_resolved_on(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    record = portal_selection.resolve(run_ids=[RUN]).to_dict()
    record["api_url"] = "https://graphql.cryoet.staging.si.czi.technology/graphql"
    (tmp_path / "staging.json").write_text(json.dumps(record))
    result = CliRunner().invoke(cli, ["data-portal", "--selection", "staging.json", "--output-dir", "a"])
    assert result.exit_code != 0 and "--staging" in result.output
    assert not (tmp_path / "a").exists()


def test_an_ambiguous_selection_is_refused_with_its_candidates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli, ["data-portal", "--run-ids", str(RUN), "--tomogram-type", "raw", "--output-dir", "x"]
    )
    assert result.exit_code != 0
    assert "ambiguous" in result.output and "21115" in result.output and "21116" in result.output
    assert not (tmp_path / "x").exists()
