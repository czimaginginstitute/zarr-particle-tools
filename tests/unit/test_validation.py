"""validation: options the programs do not implement and input metadata they would ignore."""

import pandas as pd
import pytest
import starfile

from zarr_particle_tools import validation as v


def _project(
    tmp_path, monkeypatch, *, particles_extra=None, optics_extra=None, tilts_extra=None, global_extra=None, uri=True
):
    """A minimal RELION project: tomograms.star + one tilt star + particles.star, cwd = project."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Import").mkdir()
    tilts = pd.DataFrame({"rlnMicrographName": ["1@Import/placeholder.mrcs", "2@Import/placeholder.mrcs"]})
    if uri:
        tilts["tomoTiltSeriesURI"] = "s3://bucket/ts.zarr"
    for column, value in (tilts_extra or {}).items():
        tilts[column] = value
    starfile.write({"16848": tilts}, "Import/16848.star")
    tomograms = pd.DataFrame({"rlnTomoName": ["16848"], "rlnTomoTiltSeriesStarFile": ["Import/16848.star"]})
    for column, value in (global_extra or {}).items():
        tomograms[column] = value
    starfile.write({"global": tomograms}, "Import/tomograms.star")
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


def test_other_args_are_refused_for_every_job():
    """No wrapper passes other_args on, so a RELION flag in it (e.g. --apply_orientations) would be dropped."""
    for job_type in v.JOB_TYPES:
        given = {"other_args": "--apply_orientations", "do_shift_align": "Yes"}
        assert [p.subject for p in v.check_options(job_type, given)] == ["other_args"]
        assert v.check_options(job_type, {**given, "other_args": ""}) == []


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
    ],
)
def test_metadata_the_python_implementation_would_ignore(tmp_path, monkeypatch, kwargs, subject):
    particles, tomograms = _project(tmp_path, monkeypatch, **kwargs)
    assert subject in [p.subject for p in v.check_inputs(v.EXTRACT, particles, tomograms)]
    # stock RELION applies it, so CTF refinement is not refused for it
    assert subject not in [p.subject for p in v.check_inputs(v.CTFREFINE, particles, tomograms)]


DEFORMED = {
    "rlnTomoDeformationGridSizeX": [2],
    "rlnTomoDeformationGridSizeY": [2],
    "rlnTomoDeformationType": ["linear"],
}
COEFFICIENTS = {"rlnTomoDeformationCoefficients": ["[0.001,0,0]", "[0,0,0.001]"]}


def test_2d_deformations_are_extracted_but_not_reconstructed(tmp_path, monkeypatch):
    particles, tomograms = _project(tmp_path, monkeypatch, global_extra=DEFORMED, tilts_extra=COEFFICIENTS)
    assert v.check_inputs(v.EXTRACT, particles, tomograms) == []
    assert v.check_inputs(v.CTFREFINE, particles, tomograms) == []
    assert [p.subject for p in v.check_inputs(v.RECONSTRUCT, particles, tomograms)] == ["rlnTomoDeformationType"]


def test_coefficients_without_grid_sizes_are_not_deformations(tmp_path, monkeypatch):
    """RELION reads deformations only when the tomogram row has both grid sizes, and ignores stray coefficients."""
    particles, tomograms = _project(tmp_path, monkeypatch, tilts_extra=COEFFICIENTS)
    assert v.check_inputs(v.RECONSTRUCT, particles, tomograms) == []


@pytest.mark.parametrize(
    "global_extra, tilts_extra, subject",
    [
        ({**DEFORMED, "rlnTomoDeformationType": ["bspline"]}, COEFFICIENTS, "rlnTomoDeformationType"),
        (DEFORMED, None, "rlnTomoDeformationCoefficients"),
    ],
)
def test_deformations_extraction_could_not_build_are_refused(tmp_path, monkeypatch, global_extra, tilts_extra, subject):
    particles, tomograms = _project(tmp_path, monkeypatch, global_extra=global_extra, tilts_extra=tilts_extra)
    assert [p.subject for p in v.check_inputs(v.EXTRACT, particles, tomograms)] == [subject]


@pytest.mark.parametrize(
    "column, readers", [("rlnTomoSubtomogramRot", (v.EXTRACT, v.RECONSTRUCT)), ("rlnAngleRot", (v.RECONSTRUCT,))]
)
def test_angles_left_blank_in_some_rows_are_refused_where_they_are_read(tmp_path, monkeypatch, column, readers):
    """A merged STAR leaves NaN where a source had no such column; that is not a rotation, so it is not guessed."""
    particles, tomograms = _project(tmp_path, monkeypatch, particles_extra={column: [10.0, float("nan")]})
    for job_type in v.JOB_TYPES:
        expected = [column] if job_type in readers else []
        assert [p.subject for p in v.check_inputs(job_type, particles, tomograms)] == expected
    (tmp_path / "filled").mkdir()
    particles, tomograms = _project(tmp_path / "filled", monkeypatch, particles_extra={column: [10.0, 0.0]})
    assert v.check_inputs(v.RECONSTRUCT, particles, tomograms) == []


def test_a_series_with_no_readable_pixels_is_refused(tmp_path, monkeypatch):
    particles, tomograms = _project(tmp_path, monkeypatch, uri=False)
    assert [p.subject for p in v.check_inputs(v.CTFREFINE, particles, tomograms)] == ["rlnMicrographName"]
