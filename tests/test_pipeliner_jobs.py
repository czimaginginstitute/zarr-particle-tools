"""
Tests for the ccpem-pipeliner job wrappers, which are what py2rely actually invokes.

Skipped unless pipeliner is importable; the dev extra installs it, so CI runs these. Worth having:
the reconstruct wrapper used to delete Wiener_SNR and point_group from its job options as
"not supported", so a pipeline asking for D2 with a Wiener SNR silently got C1 with the radial-average
heuristic instead -- no error, just a different reconstruction.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pipeliner", reason="ccpem-pipeliner is not installed")

from zarr_particle_tools.pipeliner.subtomo_reconstruct_pipeliner_job import (  # noqa: E402
    PythonRelionSubtomoReconstructJob,
)


def _command(job) -> list[str]:
    commands = job.get_commands()
    assert len(commands) == 1
    return [str(x) for x in commands[0].cmd]


@pytest.fixture
def job():
    j = PythonRelionSubtomoReconstructJob()
    j.output_dir = "Reconstruct/job001/"
    j.joboptions["in_optimisation"].value = "input/optimisation_set.star"
    return j


def test_snr_and_symmetry_options_are_exposed(job):
    """They are supported by zarr-particle-reconstruct, so the wrapper must not strip them."""
    assert "point_group" in job.joboptions
    assert "Wiener_SNR" in job.joboptions


def test_symmetry_is_forwarded(job):
    job.joboptions["point_group"].value = "D2"
    cmd = _command(job)
    assert "--symmetry" in cmd
    assert cmd[cmd.index("--symmetry") + 1] == "D2"


def test_positive_snr_is_forwarded(job):
    job.joboptions["Wiener_SNR"].value = 10.0
    cmd = _command(job)
    assert "--snr" in cmd
    assert float(cmd[cmd.index("--snr") + 1]) == 10.0


def test_non_positive_snr_is_omitted_so_the_heuristic_is_used(job):
    """RELION only takes the Wiener branch for SNR > 0; anything else must leave our default alone."""
    for value in (0.0, -1.0):
        job.joboptions["Wiener_SNR"].value = value
        assert "--snr" not in _command(job), f"SNR {value} should not be forwarded"


def test_default_command_is_well_formed(job):
    cmd = _command(job)
    assert cmd[:2] == ["zarr-particle-reconstruct", "local"]
    for flag in ("--output-dir", "--box-size", "--bin", "--optimisation-set-starfile"):
        assert flag in cmd
    # the job option default is C1, which is forwarded explicitly rather than left implicit
    assert cmd[cmd.index("--symmetry") + 1] == "C1"


def test_entry_point_is_registered_when_installed():
    """py2rely selects our jobs by entry-point name; a rename would silently fall back to RELION."""
    if importlib.util.find_spec("pipeliner") is None:  # pragma: no cover - guarded by importorskip
        pytest.skip("pipeliner unavailable")
    from importlib.metadata import entry_points

    names = {ep.name for ep in entry_points(group="ccpem_pipeliner.jobs")}
    assert "zarrparticletools.reconstruct" in names


# ----------------------------------------------------------------------------- refusals through pipeliner's hook
import subprocess  # noqa: E402
import sys  # noqa: E402

import pandas as pd  # noqa: E402
import starfile  # noqa: E402

from zarr_particle_tools.pipeliner.subtomo_ctfrefine_pipeliner_job import (  # noqa: E402
    PythonRelionSubtomoCtfRefineJob,
)
from zarr_particle_tools.pipeliner.subtomo_extract_pipeliner_job import PythonRelionPseudoSubtomoJob  # noqa: E402
from zarr_particle_tools.pipeliner.subtomo_polish_pipeliner_job import PythonRelionSubtomoPolishJob  # noqa: E402


def _errors(job) -> list[tuple[list[str], str]]:
    """(option names, message) of every error validate_joboptions reports, as a planner reads them."""
    names = {id(option): name for name, option in job.joboptions.items()}
    return [
        ([names[id(option)] for option in result.raised_by], result.message)
        for result in job.validate_joboptions()
        if result.type == "error"
    ]


def _zpt_errors(job) -> list[tuple[list[str], str]]:
    """Only the errors raised by zarr-particle-tools, not pipeliner's required-option checks."""
    return [error for error in _errors(job) if "required" not in error[1]]


def _project(root, deformation_type="linear"):
    """A one-tomogram project (cwd-relative paths) with an optimisation set, as a pipeline would hold it."""
    (root / "Import").mkdir()
    tilts = pd.DataFrame({"rlnMicrographName": ["1@x.mrc", "2@x.mrc"], "tomoTiltSeriesURI": ["s3://b/ts.zarr"] * 2})
    starfile.write({"TS_1": tilts}, root / "Import" / "TS_1.star")
    tomograms = pd.DataFrame(
        {
            "rlnTomoName": ["TS_1"],
            "rlnTomoTiltSeriesStarFile": ["Import/TS_1.star"],
            "rlnOpticsGroupName": ["optics1"],
            "rlnTomoDeformationGridSizeX": [1],
            "rlnTomoDeformationGridSizeY": [1],
            "rlnTomoDeformationType": [deformation_type],
        }
    )
    starfile.write({"global": tomograms}, root / "Import" / "tomograms.star")
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnOpticsGroupName": ["optics1"]})
    particles = pd.DataFrame({"rlnTomoName": ["TS_1"], "rlnOpticsGroup": [1]})
    starfile.write({"optics": optics, "particles": particles}, root / "Import" / "particles.star")
    starfile.write(
        {"rlnTomoParticlesFile": "Import/particles.star", "rlnTomoTomogramsFile": "Import/tomograms.star"},
        root / "Import" / "optimisation_set.star",
    )


def _relion_job(cls):
    job = cls()
    job.output_dir = "Job/job001/"
    job.joboptions["in_optimisation"].value = "Import/optimisation_set.star"
    job.joboptions["in_halfmaps"].value = "Refine3D/job001/run_half1_class001_unfil.mrc"
    job.joboptions["in_refmask"].value = "MaskCreate/job001/mask.mrc"
    job.joboptions["in_post"].value = "PostProcess/job001/postprocess.star"
    return job


@pytest.mark.parametrize(
    "option, value",
    [("max_dose", "30"), ("min_nr_frames", "2"), ("do_output_2dstacks", "No"), ("do_extract_reproject", "Yes")],
)
def test_extraction_options_it_does_not_implement_are_refused(option, value):
    job = PythonRelionPseudoSubtomoJob()
    job.output_dir = "Extract/job001/"
    job.joboptions["in_optimisation"].value = "Import/optimisation_set.star"
    job.joboptions[option].value = value
    assert [names for names, _ in _zpt_errors(job)] == [[option]]
    with pytest.raises(ValueError, match=option):
        job.get_commands()


def test_other_args_are_refused_where_the_program_is_not_relion(job):
    job.joboptions["other_args"].value = "--apply_orientations"
    assert [names for names, _ in _zpt_errors(job)] == [["other_args"]]
    with pytest.raises(ValueError, match="other_args"):
        job.get_commands()


@pytest.mark.parametrize("cls", [PythonRelionSubtomoCtfRefineJob, PythonRelionSubtomoPolishJob])
def test_other_args_reach_stock_relion(cls):
    job = _relion_job(cls)
    if cls is PythonRelionSubtomoPolishJob:
        job.joboptions["do_motion"].value = "Yes"
    job.joboptions["other_args"].value = "--some_relion_flag 2"
    cmd = _command(job)
    assert cmd[cmd.index("--relion-args") + 1] == "--some_relion_flag 2"
    assert "--require-ram-staging" not in cmd  # like the CLIs, the jobs fall back from /dev/shm unless asked not to


def test_set_and_direct_entries_together_are_refused(job):
    job.joboptions["in_particles"].value = "Import/particles.star"
    assert [names for names, _ in _zpt_errors(job)] == [["do_use_direct_entries", "in_optimisation"]]


@pytest.mark.parametrize(
    "options, refused",
    [
        ({"do_shift_align": "Yes", "do_motion": "Yes"}, ["do_shift_align", "do_motion"]),
        ({"do_shift_align": "No", "do_motion": "No"}, ["do_shift_align", "do_motion"]),
        ({"do_shift_align": "Yes", "shift_align_type": "Only particles"}, ["shift_align_type"]),
        ({"do_motion": "Yes", "do_sq_exp_ker": "Yes"}, ["do_sq_exp_ker"]),
        ({"do_motion": "Yes", "shift_align_type": "Only particles"}, []),
    ],
)
def test_polish_options_it_does_not_implement_are_refused(options, refused):
    job = _relion_job(PythonRelionSubtomoPolishJob)
    for name, value in options.items():
        job.joboptions[name].value = value
    assert [name for names, _ in _zpt_errors(job) for name in names] == refused


def test_inputs_are_checked_once_they_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    job = PythonRelionPseudoSubtomoJob()
    job.output_dir = "Extract/job001/"
    job.joboptions["in_optimisation"].value = "Import/optimisation_set.star"
    assert _zpt_errors(job) == []  # scheduled before its inputs exist
    _project(tmp_path, deformation_type="bspline")
    [(names, message)] = _zpt_errors(job)
    assert names == ["in_optimisation"]
    assert message.startswith("rlnTomoDeformationType in Import/tomograms.star: tomogram TS_1: illegal deformation")
    # ctf refinement runs stock RELION, which reads the declaration itself
    assert _zpt_errors(_relion_job(PythonRelionSubtomoCtfRefineJob)) == []
    # direct entries name the input the problem came from
    job.joboptions["in_optimisation"].value = ""
    job.joboptions["in_particles"].value = "Import/particles.star"
    job.joboptions["in_tomograms"].value = "Import/tomograms.star"
    assert [names for names, _ in _zpt_errors(job)] == [["in_tomograms"]]


def test_output_nodes_are_the_files_the_jobs_keep(job):
    job.create_output_nodes()
    assert {Path(n.name).name for n in job.output_nodes} == {"merged.mrc", "half1.mrc", "half2.mrc"}
    ctf = _relion_job(PythonRelionSubtomoCtfRefineJob)
    ctf.create_output_nodes()
    assert {Path(n.name).name for n in ctf.output_nodes} == {"tomograms.star", "optimisation_set.star"}
    polish = _relion_job(PythonRelionSubtomoPolishJob)
    polish.joboptions["do_shift_align"].value = "Yes"
    polish.create_output_nodes()
    assert {Path(n.name).name for n in polish.output_nodes} == {"tomograms.star", "optimisation_set.star"}


@pytest.mark.parametrize("flag", ["--do_even_aberrations", "--do_odd_aberrations"])
def test_ctf_refinement_of_aberrations_declares_its_particles(flag):
    # relion_tomo_refine_ctf writes particles.star and points the optimisation set at it (ctf_refinement.cpp:298)
    ctf = _relion_job(PythonRelionSubtomoCtfRefineJob)
    ctf.joboptions["other_args"].value = f"{flag} --ne 4"
    ctf.create_output_nodes()
    assert {Path(n.name).name for n in ctf.output_nodes} == {
        "particles.star",
        "tomograms.star",
        "optimisation_set.star",
    }


def test_wrappers_and_their_checks_import_without_the_extraction_stack(tmp_path):
    """A planner's venv holds pipeliner, pandas and starfile, but not dask, s3fs or zarr."""
    _project(tmp_path, deformation_type="bspline")
    code = (
        "import sys\n"
        "for m in ('dask', 's3fs', 'zarr'):\n"
        "    sys.modules[m] = None\n"
        "from zarr_particle_tools.pipeliner.subtomo_extract_pipeliner_job import PythonRelionPseudoSubtomoJob\n"
        "import zarr_particle_tools.pipeliner.subtomo_reconstruct_pipeliner_job\n"
        "import zarr_particle_tools.pipeliner.subtomo_ctfrefine_pipeliner_job\n"
        "import zarr_particle_tools.pipeliner.subtomo_polish_pipeliner_job\n"
        "job = PythonRelionPseudoSubtomoJob()\n"
        "job.joboptions['in_optimisation'].value = 'Import/optimisation_set.star'\n"
        "print(any('bspline' in r.message for r in job.validate_joboptions()))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "True"
