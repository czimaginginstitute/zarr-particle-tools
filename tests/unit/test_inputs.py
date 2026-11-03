"""Optimisation sets, tilt-series lookup and input metadata the programs would not honor (core.helpers)."""

import pandas as pd
import pytest
import starfile

from zarr_particle_tools.core.helpers import check_inputs, input_problems, read_optimisation_set


def _project(
    tmp_path,
    monkeypatch,
    *,
    particles_extra=None,
    optics_extra=None,
    tilts_extra=None,
    global_extra=None,
    uri=True,
    stack="Import/placeholder.mrcs",
):
    """A minimal RELION project: tomograms.star + one tilt star + particles.star, cwd = project."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Import").mkdir()
    tilts = pd.DataFrame({"rlnMicrographName": [f"1@{stack}", f"2@{stack}"]})
    if uri:
        tilts["tomoTiltSeriesURI"] = "s3://bucket/ts.zarr"
    for column, value in (tilts_extra or {}).items():
        tilts[column] = value
    starfile.write({"16848": tilts}, "Import/16848.star")
    tomograms = pd.DataFrame(
        {
            "rlnTomoName": ["16848"],
            "rlnTomoTiltSeriesStarFile": ["Import/16848.star"],
            "rlnOpticsGroupName": ["run_16848"],
        }
    )
    for column, value in (global_extra or {}).items():
        tomograms[column] = value
    starfile.write({"global": tomograms}, "Import/tomograms.star")
    optics = pd.DataFrame(
        {"rlnOpticsGroup": [1], "rlnOpticsGroupName": ["run_16848"], "rlnTomoTiltSeriesPixelSize": [2.165]}
    )
    for column, value in (optics_extra or {}).items():
        optics[column] = value
    particles = pd.DataFrame({"rlnTomoName": ["16848"] * 2, "rlnOpticsGroup": [1, 1]})
    for column, value in (particles_extra or {}).items():
        particles[column] = value
    starfile.write({"optics": optics, "particles": particles}, "Import/particles.star")
    return tmp_path / "Import/particles.star", tmp_path / "Import/tomograms.star"


def _columns(problems):
    return [message.split(" ")[0] for _, message in problems]


def test_relion_key_value_optimisation_set_with_empty_trajectories(tmp_path, monkeypatch):
    """What relion_refine writes: three keys, project-relative paths, an empty trajectories entry."""
    particles, tomograms = _project(tmp_path, monkeypatch)
    (tmp_path / "Refine3D").mkdir()
    (tmp_path / "Refine3D" / "run_optimisation_set.star").write_text(
        "data_\n"
        "_rlnTomoParticlesFile Import/particles.star\n"
        "_rlnTomoTomogramsFile Import/tomograms.star\n"
        '_rlnTomoTrajectoriesFile ""\n'
    )
    got = read_optimisation_set("Refine3D/run_optimisation_set.star")
    assert [str(p) if p else None for p in got] == ["Import/particles.star", "Import/tomograms.star", None]


def test_absolute_optimisation_set_reads_from_any_directory(tmp_path, monkeypatch):
    """zarr-particle-extract writes absolute entries, so a set named from $HOME resolves (review case)."""
    particles, tomograms = _project(tmp_path, monkeypatch)
    starfile.write(
        {"rlnTomoParticlesFile": str(particles), "rlnTomoTomogramsFile": str(tomograms)},
        tmp_path / "optimisation_set.star",
    )
    elsewhere = tmp_path / "home"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert read_optimisation_set(tmp_path / "optimisation_set.star") == (particles, tomograms, None)


def test_optimisation_set_entries_relative_to_the_set(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    starfile.write(
        {"rlnTomoParticlesFile": "particles.star", "rlnTomoTomogramsFile": "tomograms.star"},
        tmp_path / "Import" / "optimisation_set.star",
    )
    monkeypatch.chdir(tmp_path.parent)
    particles, tomograms, _ = read_optimisation_set(tmp_path / "Import" / "optimisation_set.star")
    assert (particles, tomograms) == (tmp_path / "Import/particles.star", tmp_path / "Import/tomograms.star")


def test_optimisation_set_with_a_missing_entry_is_refused(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    starfile.write(
        {"rlnTomoParticlesFile": "Import/absent.star", "rlnTomoTomogramsFile": "Import/tomograms.star"}, "o.star"
    )
    with pytest.raises(FileNotFoundError, match="rlnTomoParticlesFile in o.star"):
        read_optimisation_set("o.star")


def test_clean_inputs_pass(tmp_path, monkeypatch):
    particles, tomograms = _project(tmp_path, monkeypatch)
    for program in ("extract", "reconstruct", "relion"):
        assert input_problems(program, particles, tomograms) == []


def test_subtomogram_orientations_are_honored(tmp_path, monkeypatch):
    particles, tomograms = _project(tmp_path, monkeypatch)
    starfile.write(
        {
            "optics": pd.DataFrame({"rlnOpticsGroup": [1], "rlnOpticsGroupName": ["run_16848"]}),
            "particles": pd.DataFrame(
                {"rlnTomoName": ["16848"] * 2, "rlnTomoSubtomogramRot": [0.0, 12.0], "rlnOriginXAngst": [0.0, 3.0]}
            ),
        },
        particles,
    )
    for program in ("extract", "reconstruct", "relion"):
        assert input_problems(program, particles, tomograms) == []


NAN = float("nan")
OFFSETS = {"rlnOriginXAngst": [1.0, 2.0], "rlnOriginYAngst": [0.0, 0.0], "rlnOriginZAngst": [0.0, 0.0]}


@pytest.mark.parametrize(
    "extra, readers",
    [
        ({"rlnTomoSubtomogramRot": [10.0, NAN]}, ("reconstruct",)),  # extraction has no offsets to rotate
        ({"rlnTomoSubtomogramRot": [10.0, NAN], **OFFSETS}, ("extract", "reconstruct")),
        ({"rlnAngleRot": [10.0, NAN]}, ("reconstruct",)),
        ({"rlnAngleTilt": [10.0, NAN]}, ()),  # without rlnAngleRot, A_part is the identity
    ],
)
def test_angles_left_blank_in_some_rows_are_refused_where_they_are_read(tmp_path, monkeypatch, extra, readers):
    """A merged STAR leaves NaN where a source had no such column; that is not a rotation, so it is not guessed."""
    particles, tomograms = _project(tmp_path, monkeypatch, particles_extra=extra)
    blank = next(column for column, values in extra.items() if any(v != v for v in values))
    for program in ("extract", "reconstruct", "relion"):
        assert _columns(input_problems(program, particles, tomograms)) == ([blank] if program in readers else [])
    # --dont-apply-offsets: extraction does not read A_sub
    assert input_problems("extract", particles, tomograms, apply_offsets=False) == []
    (tmp_path / "filled").mkdir()
    filled = {column: [0.0 if v != v else v for v in values] for column, values in extra.items()}
    particles, tomograms = _project(tmp_path / "filled", monkeypatch, particles_extra=filled)
    assert input_problems("reconstruct", particles, tomograms) == []


@pytest.mark.parametrize(
    "kwargs, extract, reconstruct",
    [
        ({"optics_extra": {"rlnEvenZernike": ["[0.0,0.3]"]}}, ["rlnEvenZernike"], ["rlnEvenZernike"]),
        ({"optics_extra": {"rlnOddZernike": ["[0.0,0.3]"]}}, [], ["rlnOddZernike"]),
        ({"particles_extra": {"rlnEvenZernike": ["[0.1]", "[0]"]}}, [], ["rlnEvenZernike"]),
        ({"optics_extra": {"rlnEvenZernike": ["[0,0,0]"]}}, [], []),
        ({"optics_extra": {"rlnMagMat00": [1.01]}}, [], []),  # no tomography program reads it
    ],
)
def test_aberrations_as_relion_applies_them(tmp_path, monkeypatch, kwargs, extract, reconstruct):
    # relion_tomo_subtomo applies only even terms of the optics table, and only with CTF
    particles, tomograms = _project(tmp_path, monkeypatch, **kwargs)
    assert _columns(input_problems("extract", particles, tomograms)) == extract
    assert input_problems("extract", particles, tomograms, apply_ctf=False) == []
    assert _columns(input_problems("reconstruct", particles, tomograms)) == reconstruct
    assert input_problems("relion", particles, tomograms) == []


def test_reconstruction_needs_one_tilt_series_pixel_size(tmp_path, monkeypatch):
    particles, tomograms = _project(tmp_path, monkeypatch)
    optics = pd.DataFrame(
        {"rlnOpticsGroup": [1, 2], "rlnOpticsGroupName": ["a", "b"], "rlnTomoTiltSeriesPixelSize": [2.165, 1.0]}
    )
    starfile.write({"optics": optics, "particles": pd.DataFrame({"rlnTomoName": ["16848"]})}, particles)
    assert _columns(input_problems("reconstruct", particles, tomograms)) == ["rlnTomoTiltSeriesPixelSize"]
    assert input_problems("extract", particles, tomograms) == []


def test_reconstruction_takes_particles_of_mixed_extractions(tmp_path, monkeypatch):
    """Box, pixel size and binning are re-extracted at one --box-size / --bin; only the tilt pixel size must agree."""
    particles, tomograms = _project(tmp_path, monkeypatch)
    optics = pd.DataFrame(
        {
            "rlnOpticsGroup": [1, 2],
            "rlnOpticsGroupName": ["a", "b"],
            "rlnImageSize": [64, 128],
            "rlnImagePixelSize": [2.165, 4.33],
            "rlnTomoSubtomogramBinning": [1, 2],
            "rlnTomoTiltSeriesPixelSize": [2.165, 2.165],
        }
    )
    starfile.write({"optics": optics, "particles": pd.DataFrame({"rlnTomoName": ["16848"]})}, particles)
    assert input_problems("reconstruct", particles, tomograms) == []


DEFORMED = {
    "rlnTomoDeformationGridSizeX": [2],
    "rlnTomoDeformationGridSizeY": [2],
    "rlnTomoDeformationType": ["linear"],
}
COEFFICIENTS = {"rlnTomoDeformationCoefficients": ["[0.001,0,0]", "[0,0,0.001]"]}


def test_2d_deformations_are_honored(tmp_path, monkeypatch):
    particles, tomograms = _project(tmp_path, monkeypatch, global_extra=DEFORMED, tilts_extra=COEFFICIENTS)
    for program in ("extract", "reconstruct", "relion"):
        assert input_problems(program, particles, tomograms) == []


def test_a_declaration_without_coefficients_applies_none(tmp_path, monkeypatch):
    particles, tomograms = _project(tmp_path, monkeypatch, global_extra=DEFORMED)
    assert input_problems("extract", particles, tomograms) == []


def test_an_illegal_deformation_type_is_refused_unless_deformations_are_ignored(tmp_path, monkeypatch):
    bad = {**DEFORMED, "rlnTomoDeformationType": ["bspline"]}
    particles, tomograms = _project(tmp_path, monkeypatch, global_extra=bad, tilts_extra=COEFFICIENTS)
    problems = input_problems("extract", particles, tomograms)
    assert problems == [("tomograms", problems[0][1])]
    assert problems[0][1].startswith(f"rlnTomoDeformationType in {tomograms}: tomogram 16848: illegal deformation type")
    assert input_problems("extract", particles, tomograms, ignore_deformations=True) == []
    assert input_problems("relion", particles, tomograms) == []
    with pytest.raises(ValueError, match="bspline"):
        check_inputs("reconstruct", particles, tomograms)


def test_a_series_with_no_readable_pixels_is_refused(tmp_path, monkeypatch):
    particles, tomograms = _project(tmp_path, monkeypatch, uri=False)
    problems = input_problems("relion", particles, tomograms)
    assert _columns(problems) == ["rlnMicrographName"] and "Import/placeholder.mrcs does not exist" in problems[0][1]


def test_an_s3_stack_without_a_uri_column_is_read_from_s3(tmp_path, monkeypatch):
    particles, tomograms = _project(tmp_path, monkeypatch, uri=False, stack="s3://bucket/TS_1.mrc")
    for program in ("extract", "relion"):
        assert input_problems(program, particles, tomograms) == []


def test_stacks_are_found_under_the_tiltseries_relative_dir(tmp_path, monkeypatch):
    """ctf-refine with --tiltseries-relative-dir, started from another directory (review case)."""
    particles, tomograms = _project(tmp_path, monkeypatch, uri=False)
    (tmp_path / "Import" / "placeholder.mrcs").write_bytes(b"")
    global_df = starfile.read(tomograms)
    starfile.write(
        {"global": global_df.assign(rlnTomoTiltSeriesStarFile=str(tmp_path / "Import/16848.star"))}, tomograms
    )
    elsewhere = tmp_path / "home"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert input_problems("relion", particles, tomograms, tiltseries_relative_dir=tmp_path) == []
    assert _columns(input_problems("relion", particles, tomograms)) == ["rlnMicrographName"]


def test_only_tomograms_with_particles_are_checked_for_extraction(tmp_path, monkeypatch):
    particles, tomograms = _project(tmp_path, monkeypatch)
    global_df = starfile.read(tomograms)
    extra = global_df.iloc[[0]].assign(rlnTomoName="no_particles", rlnTomoTiltSeriesStarFile="Import/absent.star")
    starfile.write({"global": pd.concat([global_df, extra])}, tomograms)
    assert input_problems("extract", particles, tomograms) == []
    assert _columns(input_problems("relion", particles, tomograms)) == ["rlnTomoTiltSeriesStarFile"]  # staged: all
