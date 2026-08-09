"""
Tests for the ccpem-pipeliner job wrappers, which are what py2rely actually invokes.

Skipped unless pipeliner is importable; the dev extra installs it, so CI runs these. Worth having:
the reconstruct wrapper used to delete Wiener_SNR and point_group from its job options as
"not supported", so a pipeline asking for D2 with a Wiener SNR silently got C1 with the radial-average
heuristic instead -- no error, just a different reconstruction.
"""

import importlib.util

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
