"""Optimisation sets as RELION and zarr-particle-extract write them (core.helpers.read_optimisation_set)."""

import pandas as pd
import pytest
import starfile

from zarr_particle_tools.core.helpers import read_optimisation_set


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
