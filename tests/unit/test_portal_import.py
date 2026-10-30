"""The shared portal builder without the portal: optics groups, placeholders, paths, stored selections."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import zarr_particle_tools.generate.cdp_import_starfiles as cdp_import
from zarr_particle_tools.core.helpers import get_tomo_name, get_unbinned_tomo_size, project_relative
from zarr_particle_tools.generate.portal_selection import RunSelection, Section, Selection, _is_identity, _sections


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
