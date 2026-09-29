"""
Portal selection and the S3 import (``zarr-particle-importtomo`` / ``zarrparticletools.importtomo``).

The resolver and import tests query the live CryoET Data Portal (dataset 10426, runs 16848 = tomo153 and
16849 = tomo154), like ``test_generate.py``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import mrcfile
import pandas as pd
import pytest
import starfile
from click.testing import CliRunner

from zarr_particle_tools import portal_selection
from zarr_particle_tools.importtomo import cli

RUN = 16848


def test_resolver_and_job_import_without_the_extraction_stack():
    """A process that only plans jobs must be able to resolve a selection and build the job."""
    # pipeliner itself imports scipy, so the job module is held only to what zpt could add
    code = (
        "import sys\n"
        "import zarr_particle_tools.portal_selection\n"
        "heavy = [m for m in ('dask', 'zarr', 's3fs', 'scipy') if m in sys.modules]\n"
        "import zarr_particle_tools.pipeliner.importtomo_pipeliner_job\n"
        "heavy += [m for m in ('dask', 'zarr', 's3fs') if m in sys.modules and m not in heavy]\n"
        "print(','.join(sorted(heavy)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "", f"imported the extraction stack: {out}"


def test_job_command():
    pytest.importorskip("pipeliner", reason="ccpem-pipeliner is not installed")
    from zarr_particle_tools.pipeliner.importtomo_pipeliner_job import PythonPortalImportTomoJob

    job = PythonPortalImportTomoJob()
    job.output_dir = "Import/job001/"
    job.joboptions["dataset_ids"].value = "10426"
    job.joboptions["run_ids"].value = "16848,16849"
    cmd = [str(x) for x in job.get_commands()[0].cmd]
    assert cmd[:3] == ["zarr-particle-importtomo", "--output-dir", "Import/job001/"]
    assert cmd[cmd.index("--dataset-ids") + 1] == "10426"
    assert cmd[cmd.index("--hand") + 1] == "-1"
    assert "--selection" not in cmd

    job.joboptions["in_selection"].value = "ApexAgent/portal_selection.json"
    cmd = [str(x) for x in job.get_commands()[0].cmd]
    assert cmd[cmd.index("--selection") + 1] == "ApexAgent/portal_selection.json"
    assert "--dataset-ids" not in cmd and "--tomogram-type" not in cmd


def test_resolve_one_run():
    selection = portal_selection.resolve(run_ids=[RUN])
    assert selection.status == "eligible"
    (run,) = selection.runs
    assert (run.dataset_id, run.tomogram_id, run.alignment_id, run.voxel_spacing_id) == (10426, 21114, 17772, 17051)
    # RELION's unit: unbinned tilt pixels (the alignment's volume / 2.165 A), not tomogram voxels (1022x1440x400)
    assert run.rln_tomo_size == (4088, 5760, 1600)
    assert run.tomogram_size == (1022, 1440, 400)
    assert run.tomo_name == str(RUN)
    assert len(run.written_z_indices) == 51
    assert run.tiltseries_uri.startswith("s3://") and run.tiltseries_uri.endswith(".zarr")


def test_an_ambiguous_type_lists_its_candidates_and_a_method_settles_it():
    ambiguous = portal_selection.resolve(run_ids=[RUN], tomogram_type="raw")
    assert ambiguous.status == "ambiguous" and not ambiguous.runs
    assert {c.tomogram_id for c in ambiguous.problems[0].candidates} == {21115, 21116}  # WBP raw, SART raw

    settled = portal_selection.resolve(run_ids=[RUN], tomogram_type="raw", reconstruction_method="SART")
    assert settled.status == "eligible" and settled.runs[0].tomogram_id == 21116


def test_selection_round_trips(tmp_path):
    selection = portal_selection.resolve(run_ids=[RUN])
    path = selection.write(tmp_path / portal_selection.SELECTION_FILENAME)
    assert portal_selection.Selection.read(path).comparable() == selection.comparable()


def _tilts(path) -> pd.DataFrame:
    data = starfile.read(path, always_dict=True)
    return next(iter(data.values()))


def test_import_writes_a_project_relative_s3_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["--run-ids", f"{RUN},16849", "--output-dir", "Import/job001"])
    assert result.exit_code == 0, result.output

    tomograms = _tilts("Import/job001/tomograms.star")
    assert list(tomograms["rlnTomoName"].astype(str)) == [str(RUN), "16849"]
    row = tomograms.iloc[0]
    assert (row["rlnTomoSizeX"], row["rlnTomoSizeY"], row["rlnTomoSizeZ"]) == (4088, 5760, 1600)
    assert row["rlnTomoTiltSeriesStarFile"] == f"Import/job001/tiltseries/{RUN}.star"
    assert row["tomoTiltSeriesURI"].endswith("tomo153/TiltSeries/100/tomo153.zarr")
    assert row["rlnTomoHand"] == -1

    tilts = _tilts(f"Import/job001/tiltseries/{RUN}.star")
    assert len(tilts) == 51
    assert tilts["rlnMicrographName"].iloc[0] == "1@Import/job001/tiltseries/tiltseries_placeholder.mrcs"

    placeholder = Path("Import/job001/tiltseries/tiltseries_placeholder.mrcs")
    with mrcfile.open(placeholder, header_only=True) as mrc:
        assert int(mrc.header.nz) >= 51
    assert os.stat(placeholder).st_blocks * 512 < 1024**2, "the placeholder must stay sparse"

    record = json.loads(Path("Import/job001/portal_selection.json").read_text())
    assert record["import"]["runs"][str(RUN)]["sections_written"] == 51
    assert record["import"]["runs"][str(RUN)]["sections_excluded"] == []


def test_import_rows_are_the_generator_rows(tmp_path, monkeypatch):
    """One converter: the per-tilt values equal the annotation-driven generator's for the same objects."""
    from zarr_particle_tools.generate.cdp_generate_starfiles import generate_individual_tomogram_starfile

    monkeypatch.chdir(tmp_path)
    assert CliRunner().invoke(cli, ["--run-ids", str(RUN), "--output-dir", "new"]).exit_code == 0
    (tmp_path / "old" / "tiltseries").mkdir(parents=True)
    old, _ = generate_individual_tomogram_starfile(17772, 17051, tmp_path / "old")
    new = _tilts(f"new/tiltseries/{RUN}.star")
    columns = [c for c in old.columns if c != "rlnMicrographName"]
    pd.testing.assert_frame_equal(
        new[columns].reset_index(drop=True), old[columns].reset_index(drop=True), check_dtype=False
    )


def test_a_stored_selection_imports_exactly_or_names_the_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    selection = portal_selection.resolve(run_ids=[RUN])
    stored = selection.write(tmp_path / "stored.json")
    ok = CliRunner().invoke(cli, ["--selection", str(stored), "--output-dir", "a"])
    assert ok.exit_code == 0, ok.output

    tampered = json.loads(stored.read_text())
    tampered["runs"][0]["rln_tomo_size"] = [1022, 1440, 400]
    (tmp_path / "tampered.json").write_text(json.dumps(tampered))
    changed = CliRunner().invoke(cli, ["--selection", str(tmp_path / "tampered.json"), "--output-dir", "b"])
    assert changed.exit_code != 0
    assert "changed since the selection was resolved" in changed.output and "rln_tomo_size" in changed.output
    assert not (tmp_path / "b").exists(), "a refused import writes nothing"


def test_an_ambiguous_selection_is_refused_with_its_candidates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["--run-ids", str(RUN), "--tomogram-type", "raw", "--output-dir", "x"])
    assert result.exit_code != 0
    assert "ambiguous" in result.output and "21115" in result.output and "21116" in result.output
    assert not (tmp_path / "x").exists()
