"""The shared portal builder without the portal: optics groups, placeholders, paths, stored selections."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import zarr_particle_tools.generate.cdp_cache as cdp_cache
import zarr_particle_tools.generate.cdp_import_starfiles as cdp_import
from zarr_particle_tools.core.helpers import get_tomo_name, get_unbinned_tomo_size, project_relative
from zarr_particle_tools.generate.portal_selection import (
    RunSelection,
    Section,
    Selection,
    _is_identity,
    _sections,
    _settle,
)


def run(run_id: int, tiltseries_id: int | None = None, spacing_id: int = 17051, pixel: float = 2.165) -> RunSelection:
    tiltseries_id = tiltseries_id or run_id - 266
    return RunSelection(
        dataset_id=10426,
        run_id=run_id,
        run_name=f"tomo{run_id}",
        tiltseries_id=tiltseries_id,
        alignment_id=run_id + 924,
        tomogram_id=run_id + 4266,
        voxel_spacing_id=spacing_id,
        voxel_spacing=8.66,
        tiltseries_pixel_size=pixel,
        voltage_kv=300.0,
        spherical_aberration_mm=2.7,
        tomogram_size=(1022, 1440, 400),
        rln_tomo_size=(4088, 5760, 1600),
        tiltseries_uri=f"s3://b/{run_id}/ts.zarr",
        tomogram_uri=f"s3://b/{run_id}/t.zarr",
        sections=[Section(0, True), Section(1, False, "no CTF parameters")],
        tomo_name=get_tomo_name(run_id, tiltseries_id, run_id + 924, spacing_id),
        fractional_dose=2.35,
    )


def test_one_optics_group_per_tilt_series():
    runs = [run(16848), run(16849), run(16848, spacing_id=17999)]  # the third: run 16848's tilt series, again
    optics = cdp_import.get_optics_df(runs)
    assert optics["rlnOpticsGroupName"].tolist() == [
        "run_16848_tiltseries_16582",
        "run_16849_tiltseries_16583",
        "run_16848_tiltseries_16582",
    ]
    assert optics["rlnOpticsGroup"].tolist() == [1, 2, 1]
    assert optics["rlnTomoName"].tolist() == [r.tomo_name for r in runs]


def test_tomograms_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tomograms = cdp_import.get_tomograms_df([run(16848)], Path("Import/job001"))
    row = tomograms.iloc[0]
    assert list(tomograms.columns) == cdp_import.TOMOGRAMS_STAR_COLUMNS
    assert row["rlnTomoTiltSeriesStarFile"] == f"Import/job001/tiltseries/{row['rlnTomoName']}.star"
    assert (row["rlnTomoSizeX"], row["rlnTomoSizeY"], row["rlnTomoSizeZ"]) == (4088, 5760, 1600)
    assert row["rlnTomoTomogramBinning"] == pytest.approx(4.0)
    assert row["rlnTomoImportFractionalDose"] == pytest.approx(2.35)
    assert row["tomoTomogramURI"] == "s3://b/16848/t.zarr"
    generator = cdp_import.get_tomograms_df([run(16848)], tmp_path / "proj" / "input", tmp_path / "proj")
    assert generator.iloc[0]["rlnTomoTiltSeriesStarFile"].startswith("input/tiltseries/")


@pytest.mark.parametrize(
    "binned, binning, is_z, expected",
    [
        (1022, 4.0, False, 4088),
        (400, 4.0, True, 1600),
        (630, 10.012 / 1.54, False, 4096),
        (184, 10.012 / 1.54, True, 1198),
    ],
)
def test_unbinned_tomo_size(binned, binning, is_z, expected):
    assert get_unbinned_tomo_size(binned, binning, is_z) == expected


def test_one_placeholder_per_image_size():
    tiltseries = {
        16582: SimpleNamespace(size_x=4092, size_y=5760, size_z=51),
        16583: SimpleNamespace(size_x=4092, size_y=5760, size_z=41),
        16106: SimpleNamespace(size_x=4096, size_y=4096, size_z=31),
    }
    same_of, same = cdp_import._placeholders([run(16848), run(16849)], tiltseries)
    assert set(same_of.values()) == {"tiltseries/tiltseries_placeholder.mrcs"}
    assert same == {"tiltseries/tiltseries_placeholder.mrcs": ((51, 5760, 4092), 2.165)}

    mixed_of, mixed = cdp_import._placeholders([run(16848), run(16463, 16106, pixel=1.54)], tiltseries)
    assert len(set(mixed_of.values())) == 2
    assert sorted(mixed.values()) == [((31, 4096, 4096), 1.54), ((51, 5760, 4092), 2.165)]


def _tomogram_set(names: list[str]) -> cdp_import.TomogramSet:
    tilts = pd.DataFrame({"rlnMicrographName": ["1@x.mrcs"]})
    return cdp_import.TomogramSet(
        pd.DataFrame({"rlnTomoName": names}),
        {name: tilts for name in names},
        {"tiltseries/tiltseries_placeholder.mrcs": ((2, 4, 4), 1.0)},
    )


def test_write_replaces_an_earlier_write(tmp_path):
    out = tmp_path / "Import" / "job001"
    cdp_import.write(_tomogram_set(["a", "b"]), out)
    (out / "run.out").write_text("pipeliner's own file")
    cdp_import.write(_tomogram_set(["a"]), out, {"portal_selection.json": "{}"})
    assert sorted(p.name for p in (out / "tiltseries").iterdir()) == ["a.star", "tiltseries_placeholder.mrcs"]
    assert (out / "run.out").exists() and (out / "portal_selection.json").exists()
    assert [p.name for p in out.iterdir() if p.name.startswith(".")] == []


def test_project_relative_keeps_a_symlinked_output_dir_relative(tmp_path, monkeypatch):
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", tmp_path / "proj_link")
    monkeypatch.chdir(tmp_path)
    assert project_relative("proj_link/tomograms.star") == "proj_link/tomograms.star"
    assert project_relative(tmp_path.parent / "x.star") == str(tmp_path.parent / "x.star")


def test_a_stored_selection_without_newer_keys_reads():
    record = Selection([], [run(16848)], []).to_dict()
    for key in ("tomo_name", "fractional_dose", "sections_digest"):
        record["runs"][0].pop(key)
    stored = Selection.from_dict(json.loads(json.dumps(record)))
    assert stored.runs[0].tomo_name == run(16848).tomo_name
    assert stored.runs[0].fractional_dose is None


def test_affine_identity():
    assert _is_identity(None)
    assert _is_identity("{{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}}")
    assert not _is_identity("{{1,0,0,10},{0,1,0,0},{0,0,1,0},{0,0,0,1}}")
    assert not _is_identity([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def test_sections_exclude_incomplete_records():
    def ctf(z, **kw):
        values = {"major_defocus": 1.0, "minor_defocus": 1.0, "astigmatic_angle": 0.0, "phase_shift": 0.0}
        return SimpleNamespace(z_index=z, frame_id=z, **{**values, "max_resolution": 10.0, **kw})

    def aln(z, **kw):
        values = {"x_offset": 0.0, "y_offset": 0.0, "volume_x_rotation": 0.0, "in_plane_rotation": [], "tilt_angle": 0}
        return SimpleNamespace(z_index=z, **{**values, **kw})

    frames = [SimpleNamespace(id=z, accumulated_dose=None if z == 3 else 1.0) for z in range(5)]
    sections, reason = _sections(
        SimpleNamespace(size_z=5),
        [ctf(0), ctf(1), ctf(2, major_defocus=None), ctf(3)],
        [aln(0), aln(1, x_offset=None), aln(2), aln(3), aln(4)],
        frames,
    )
    assert reason == ""
    assert [s.written for s in sections] == [True, False, False, False, False]
    assert "x_offset" in sections[1].reason and "major_defocus" in sections[2].reason
    assert "accumulated dose" in sections[3].reason and sections[4].reason == "no CTF parameters"


def _portal(monkeypatch, tomogram=None, alignment=None, tiltseries=None):
    """A run with one two-section tilt series, served by monkeypatched cdp_cache getters; the keyword dicts override
    record fields."""
    run_ = SimpleNamespace(id=16848, dataset_id=10426, name="tomo153")
    tomo = SimpleNamespace(
        id=21114,
        alignment_id=17772,
        tomogram_voxel_spacing_id=17051,
        voxel_spacing=8.66,
        size_x=1022,
        size_y=1440,
        size_z=400,
        s3_omezarr_dir="s3://b/t.zarr",
        offset_x=0,
        offset_y=0,
        offset_z=0,
        reconstruction_method="WBP",
        processing="raw",
        processing_software=None,
        is_visualization_default=True,
    )
    aln = SimpleNamespace(
        id=17772,
        alignment_type="GLOBAL",
        affine_transformation_matrix=None,
        tiltseries_id=16582,
        volume_x_offset=0,
        volume_y_offset=0,
        volume_z_offset=0,
    )
    ts = SimpleNamespace(
        id=16582,
        s3_omezarr_dir="s3://b/ts.zarr",
        acceleration_voltage=300000,
        spherical_aberration_constant=2.7,
        pixel_spacing=2.165,
        size_x=4092,
        size_y=5760,
        size_z=2,
    )
    for record, overrides in ((tomo, tomogram), (aln, alignment), (ts, tiltseries)):
        vars(record).update(overrides or {})
    ctf = [
        SimpleNamespace(
            z_index=z,
            frame_id=z,
            major_defocus=1.0,
            minor_defocus=1.0,
            astigmatic_angle=0.0,
            phase_shift=0.0,
            max_resolution=10.0,
        )
        for z in range(2)
    ]
    per_section = [
        SimpleNamespace(
            z_index=z, x_offset=0.0, y_offset=0.0, volume_x_rotation=0.0, in_plane_rotation=[], tilt_angle=0.0
        )
        for z in range(2)
    ]
    frames = [SimpleNamespace(id=z, accumulated_dose=2.0 * z, exposure_dose=2.0) for z in range(2)]
    monkeypatch.setattr(cdp_cache, "get_alignments", lambda ids: [aln])
    monkeypatch.setattr(cdp_cache, "get_tiltseries", lambda ids: [ts])
    monkeypatch.setattr(cdp_cache, "get_all_per_section_parameters_by_tiltseries_id", lambda ids: {ts.id: ctf})
    monkeypatch.setattr(cdp_cache, "get_per_section_alignments_by_alignment_id", lambda ids: {aln.id: per_section})
    monkeypatch.setattr(cdp_cache, "get_frames_by_run_id", lambda ids: {run_.id: frames})
    return [(run_, tomo)]


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"tiltseries": {"acceleration_voltage": None}}, "acceleration_voltage"),
        ({"tiltseries": {"spherical_aberration_constant": None}}, "spherical_aberration_constant"),
        ({"tiltseries": {"pixel_spacing": None}}, "pixel_spacing"),
        ({"tiltseries": {"s3_omezarr_dir": None}}, "s3_omezarr_dir"),
        ({"tomogram": {"voxel_spacing": None}}, "voxel_spacing"),
        ({"tomogram": {"s3_omezarr_dir": None}}, "s3_omezarr_dir"),
        ({"alignment": {"affine_transformation_matrix": "{{1,0,0,10},{0,1,0,0},{0,0,1,0},{0,0,0,1}}"}}, "affine"),
    ],
)
def test_settle_reports_null_portal_fields(monkeypatch, overrides, field):
    selected, problems = _settle(_portal(monkeypatch, **overrides))
    assert selected == []
    (problem,) = problems
    assert problem.status == "ineligible" and field in problem.reason


def test_settle_writes_unbinned_tomo_size(monkeypatch):
    """A 630 x 630 x 184 tomogram at 10.012 A over 1.54 A tilt pixels (RELION's unit: unbinned tilt pixels)."""
    chosen = _portal(
        monkeypatch,
        tomogram={"voxel_spacing": 10.012, "size_x": 630, "size_y": 630, "size_z": 184},
        tiltseries={"pixel_spacing": 1.54},
    )
    (selected,), problems = _settle(chosen)
    assert problems == []
    assert selected.rln_tomo_size == (4096, 4096, 1198)
    assert all(type(n) is int for n in selected.rln_tomo_size)


def test_write_leaves_output_untouched_on_failure(tmp_path, monkeypatch):
    out = tmp_path / "Import" / "job001"
    cdp_import.write(_tomogram_set(["a", "b"]), out)
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cdp_import, "write_tiltseries_placeholder", fail)
    with pytest.raises(OSError, match="disk full"):
        cdp_import.write(_tomogram_set(["c"]), out)
    assert {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()} == before
    assert [p for d in (out, out.parent) for p in d.iterdir() if p.name.startswith(".import-")] == []
