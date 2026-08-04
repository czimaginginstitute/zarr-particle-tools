"""
End-to-end smoke tests for the per-source subcommand variants, stopping short of GPUs, RELION and
SLURM: the ctf-refine / polish portal variants stop at --dry-run, the pipeline variants at
--prepare-only. Nothing here is mocked, which is the point - these cover the seam between star-file
generation and the consumer, where a mocked test cannot see a wrong path or a lost block name.

The --dry-run cases need only portal access (no RELION), so they run wherever tests/test_generate.py
does. The pipeline cases additionally need py2rely + the RELION binaries, and the copick cases need a
copick project, so those skip unless the environment provides them.
"""

import importlib.util
import os
import shutil
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import starfile
from click.testing import CliRunner

import zarr_particle_tools.orchestrate as orchestrate
import zarr_particle_tools.subtomo_ctfrefine as ctfrefine
import zarr_particle_tools.subtomo_polish as polish

SYNTHETIC = Path("tests/data/relion_project_synthetic")
# a run with ground-truth point annotations, as used by tests/test_generate.py
PORTAL_RUN_ID = "16463"
PORTAL_ANNOTATION = "cytosolic ribosome"

HAS_PY2RELY = shutil.which("py2rely") is not None and all(
    shutil.which(b) is not None for b in orchestrate.RELION_BINARIES
)
HAS_COPICK = importlib.util.find_spec("copick") is not None
# point this at a copick project (whose runs are Data Portal run IDs) to include the copick variants
COPICK_CONFIG = os.environ.get("ZPT_SMOKE_COPICK_CONFIG")
COPICK_NAME = os.environ.get("ZPT_SMOKE_COPICK_NAME", "ribosome")
COPICK_USER = os.environ.get("ZPT_SMOKE_COPICK_USER", "octopi")
COPICK_SESSION = os.environ.get("ZPT_SMOKE_COPICK_SESSION", "1")

needs_pipeline = pytest.mark.skipif(not HAS_PY2RELY, reason="py2rely / RELION binaries not on PATH")
needs_copick_project = pytest.mark.skipif(
    not (HAS_COPICK and COPICK_CONFIG), reason="copick or ZPT_SMOKE_COPICK_CONFIG unavailable"
)


def _reference_mrc(path: Path) -> Path:
    """A tiny valid .mrc, enough for click's exists=True and py2rely's parameter generation."""
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.zeros((8, 8, 8), dtype=np.float32))
    return path


def _assert_valid_tomograms_star(star: Path):
    """
    The generated star must be RELION-readable and its tilt stars must actually resolve.

    Both halves have failed in the past: the block name was dropped (emitting `data_`, which RELION
    reads as zero tomograms) and the tilt-star paths were project-root-relative (resolving to
    <output-dir>/input/input/tiltseries/...).
    """
    assert star.exists(), f"{star} was not generated"
    assert "data_global" in star.read_text(), "RELION only enters a block literally named `global`"
    data = starfile.read(star)
    df = data["global"] if isinstance(data, dict) else data
    assert len(df) > 0, "generated tomograms.star has no rows"
    for value in df["rlnTomoTiltSeriesStarFile"]:
        assert Path(value).exists(), f"tilt star does not resolve: {value}"
        assert "input/input" not in str(value), f"doubled input dir: {value}"


@pytest.mark.parametrize("module", [ctfrefine, polish], ids=["ctfrefine", "polish"])
def test_portal_dry_run_generates_a_usable_tomograms_star(module, tmp_path):
    out = tmp_path / "job"
    ref = _reference_mrc(tmp_path / "half1.mrc")
    result = CliRunner().invoke(
        module.cli,
        [
            "data-portal",
            "--run-ids",
            PORTAL_RUN_ID,
            "--annotation-names",
            PORTAL_ANNOTATION,
            "--ground-truth",
            "--particles-starfile",
            str(SYNTHETIC / "particles.star"),
            "--ref1",
            str(ref),
            "--ref2",
            str(ref),
            "--box-size",
            "64",
            "--output-dir",
            str(out),
            "--dry-run",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    _assert_valid_tomograms_star(out / "input" / "tomograms.star")


@pytest.mark.parametrize("module", [ctfrefine, polish], ids=["ctfrefine", "polish"])
@needs_copick_project
def test_copick_portal_dry_run_generates_a_usable_tomograms_star(module, tmp_path):
    out = tmp_path / "job"
    ref = _reference_mrc(tmp_path / "half1.mrc")
    result = CliRunner().invoke(
        module.cli,
        [
            "copick-data-portal",
            "--copick-config",
            COPICK_CONFIG,
            "--copick-name",
            COPICK_NAME,
            "--copick-user-id",
            COPICK_USER,
            "--copick-session-id",
            COPICK_SESSION,
            "--particles-starfile",
            str(SYNTHETIC / "particles.star"),
            "--ref1",
            str(ref),
            "--ref2",
            str(ref),
            "--box-size",
            "64",
            "--output-dir",
            str(out),
            "--dry-run",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    _assert_valid_tomograms_star(out / "input" / "tomograms.star")


def _stage_local_project(out: Path) -> Path:
    """py2rely resolves star paths relative to the project dir, so stage the inputs inside it."""
    (out / "input").mkdir(parents=True, exist_ok=True)
    for name in ("particles.star", "tomograms.star"):
        shutil.copy(SYNTHETIC / name, out / "input" / name)
    if (SYNTHETIC / "tiltseries").is_dir():
        shutil.copytree(SYNTHETIC / "tiltseries", out / "input" / "tiltseries", dirs_exist_ok=True)
    return out / "input"


@needs_pipeline
def test_pipeline_local_prepare_only(tmp_path):
    out = tmp_path / "sta"
    staged = _stage_local_project(out)
    result = CliRunner().invoke(
        orchestrate.cli,
        [
            "local",
            "--output-dir",
            str(out),
            "--particles-starfile",
            str(staged / "particles.star"),
            "--tomograms-starfile",
            str(staged / "tomograms.star"),
            "--protein-diameter",
            "330",
            "--reference-template",
            str(_reference_mrc(tmp_path / "ref.mrc")),
            "--prepare-only",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert (out / "pipeline.sh").exists(), "py2rely did not produce pipeline.sh"
    assert (out / "all_sta_parameters.json").exists()


@needs_pipeline
@needs_copick_project
def test_pipeline_copick_local_prepare_only(tmp_path):
    out = tmp_path / "sta"
    staged = _stage_local_project(out)
    result = CliRunner().invoke(
        orchestrate.cli,
        [
            "copick-local",
            "--output-dir",
            str(out),
            "--tomograms-starfile",
            str(staged / "tomograms.star"),
            "--copick-config",
            COPICK_CONFIG,
            "--copick-name",
            COPICK_NAME,
            "--copick-user-id",
            COPICK_USER,
            "--copick-session-id",
            COPICK_SESSION,
            "--protein-diameter",
            "330",
            "--reference-template",
            str(_reference_mrc(tmp_path / "ref.mrc")),
            "--prepare-only",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert (out / "pipeline.sh").exists(), "py2rely did not produce pipeline.sh"
