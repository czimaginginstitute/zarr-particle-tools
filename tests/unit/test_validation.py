"""validation: options the programs do not implement and input metadata they would ignore."""

import pandas as pd
import pytest
import starfile

from zarr_particle_tools import validation as v


def _project(tmp_path, monkeypatch, *, particles_extra=None, optics_extra=None, tilts_extra=None, uri=True):
    """A minimal RELION project: tomograms.star + one tilt star + particles.star, cwd = project."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Import").mkdir()
    tilts = pd.DataFrame({"rlnMicrographName": ["1@Import/placeholder.mrcs", "2@Import/placeholder.mrcs"]})
    if uri:
        tilts["tomoTiltSeriesURI"] = "s3://bucket/ts.zarr"
    for column, value in (tilts_extra or {}).items():
        tilts[column] = value
    starfile.write({"16848": tilts}, "Import/16848.star")
    starfile.write(
        {"global": pd.DataFrame({"rlnTomoName": ["16848"], "rlnTomoTiltSeriesStarFile": ["Import/16848.star"]})},
        "Import/tomograms.star",
    )
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnImagePixelSize": [8.66], "rlnImageSize": [64]})
    for column, value in (optics_extra or {}).items():
        optics[column] = value
    particles = pd.DataFrame({"rlnTomoName": ["16848"] * 2, "rlnOpticsGroup": [1, 1]})
    for column, value in (particles_extra or {}).items():
        particles[column] = value
    starfile.write({"optics": optics, "particles": particles}, "Import/particles.star")
    return "Import/particles.star", "Import/tomograms.star"


def test_relion_key_value_optimisation_set_with_empty_trajectories(tmp_path, monkeypatch):
    """What relion_refine writes: three keys, project-relative paths, an empty trajectories entry."""
    particles, tomograms = _project(tmp_path, monkeypatch)
    (tmp_path / "Refine3D").mkdir()
    (tmp_path / "Refine3D" / "run_optimisation_set.star").write_text(
        "data_\n"
        f"_rlnTomoParticlesFile {particles}\n"
        f"_rlnTomoTomogramsFile {tomograms}\n"
        '_rlnTomoTrajectoriesFile ""\n'
    )
    got = v.read_optimisation_set("Refine3D/run_optimisation_set.star")
    assert [str(p) if p else None for p in got] == [particles, tomograms, None]


def test_options_the_programs_do_not_implement():
    assert [p.subject for p in v.check_options(v.EXTRACT, {"max_dose": "30", "do_output_2dstacks": "No"})] == [
        "max_dose",
        "do_output_2dstacks",
    ]
    assert v.check_options(v.POLISH, {"do_shift_align": "Yes", "do_motion": "Yes"})
    assert v.check_options(v.POLISH, {"do_shift_align": "No", "do_motion": "No"})
    assert not v.check_options(v.POLISH, {"do_shift_align": "Yes", "do_motion": "No"})


def test_direct_entries_conflict_only_when_both_forms_are_named():
    """pipeliner's particle reconstruction defaults do_use_direct_entries to Yes; a set alone is unambiguous."""
    alone = {"do_use_direct_entries": "Yes", "in_optimisation": "Extract/job001/optimisation_set.star"}
    assert not v.check_options(v.RECONSTRUCT, alone)
    both = {**alone, "in_particles": "Import/particles.star"}
    assert [p.subject for p in v.check_options(v.RECONSTRUCT, both)] == ["do_use_direct_entries"]


def test_clean_inputs_pass(tmp_path, monkeypatch):
    particles, tomograms = _project(tmp_path, monkeypatch)
    for job_type in (v.EXTRACT, v.RECONSTRUCT, v.CTFREFINE, v.POLISH):
        assert v.check_inputs(job_type, particles, tomograms) == []


@pytest.mark.parametrize(
    "kwargs, subject",
    [
        ({"optics_extra": {"rlnOddZernike": ["[0.0,0.3]"]}}, "rlnOddZernike"),
        ({"optics_extra": {"rlnMagMat00": [1.01], "rlnMagMat11": [1.0]}}, "rlnMagMat00"),
        ({"tilts_extra": {"rlnTomoDeformationType": ["spline", "spline"]}}, "rlnTomoDeformationType"),
    ],
)
def test_metadata_the_python_implementation_would_ignore(tmp_path, monkeypatch, kwargs, subject):
    particles, tomograms = _project(tmp_path, monkeypatch, **kwargs)
    assert subject in [p.subject for p in v.check_inputs(v.EXTRACT, particles, tomograms)]
    # stock RELION applies it, so CTF refinement is not refused for it
    assert subject not in [p.subject for p in v.check_inputs(v.CTFREFINE, particles, tomograms)]


def test_a_series_with_no_readable_pixels_is_refused(tmp_path, monkeypatch):
    particles, tomograms = _project(tmp_path, monkeypatch, uri=False)
    assert [p.subject for p in v.check_inputs(v.CTFREFINE, particles, tomograms)] == ["rlnMicrographName"]
