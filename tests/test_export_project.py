from types import SimpleNamespace

import click
import pytest
import starfile

import zarr_particle_tools.export_project as ep
from zarr_particle_tools.core.constants import TILTSERIES_MRCS_PLACEHOLDER, TILTSERIES_URI_RELION_COLUMN

TOMO = "run_16848_tiltseries_16582_alignment_17772_spacing_17051"


def _build_fake_project(tmp_path, mrc_present=True):
    """A minimal generated project: tomograms.star + one URI-based tilt star + the zarr placeholder."""
    ts_dir = tmp_path / "tiltseries"
    ts_dir.mkdir()
    starfile.write({"global": _df({"rlnTomoName": [TOMO]})}, tmp_path / "tomograms.star")
    tilt = _df(
        {
            "rlnMicrographName": [
                f"1@{tmp_path / TILTSERIES_MRCS_PLACEHOLDER}",
                f"2@{tmp_path / TILTSERIES_MRCS_PLACEHOLDER}",
            ],
            TILTSERIES_URI_RELION_COLUMN: ["s3://bucket/tomo.zarr", "s3://bucket/tomo.zarr"],
            "rlnTomoYTilt": [-50.0, -48.0],
        }
    )
    starfile.write({TOMO: tilt}, ts_dir / f"{TOMO}.star")
    (tmp_path / TILTSERIES_MRCS_PLACEHOLDER).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / TILTSERIES_MRCS_PLACEHOLDER).write_bytes(b"placeholder")
    return tmp_path


def _df(cols):
    import pandas as pd

    return pd.DataFrame(cols)


def test_materialize_project_to_disk(monkeypatch, tmp_path):
    _build_fake_project(tmp_path)
    monkeypatch.setattr(
        ep.cdp_cache, "get_tiltseries", lambda tid: [SimpleNamespace(id=tid, s3_mrc_file="s3://bucket/tomo.mrc")]
    )

    downloaded = {}

    def _fake_get(src, dest):
        downloaded["src"] = src
        downloaded["dest"] = dest
        # simulate a downloaded MRC on disk
        with open(dest, "wb") as f:
            f.write(b"MRCDATA")

    monkeypatch.setattr(ep.core_data.global_fs, "get", _fake_get)

    ep.materialize_project_to_disk(tmp_path)

    # tiltseries_id 16582 parsed from TOMO -> portal s3 mrc downloaded to tiltseries/<tomo>.mrc
    assert downloaded["src"] == "bucket/tomo.mrc"
    mrc_path = tmp_path / "tiltseries" / f"{TOMO}.mrc"
    assert mrc_path.exists()

    # tilt star repointed to the on-disk mrc, URI column dropped, tilt index preserved
    tilt = starfile.read(tmp_path / "tiltseries" / f"{TOMO}.star")
    if isinstance(tilt, dict):
        tilt = next(iter(tilt.values()))
    assert TILTSERIES_URI_RELION_COLUMN not in tilt.columns
    assert list(tilt["rlnMicrographName"]) == [f"1@{mrc_path}", f"2@{mrc_path}"]

    # zarr placeholder removed
    assert not (tmp_path / TILTSERIES_MRCS_PLACEHOLDER).exists()


def test_materialize_missing_mrc_fails(monkeypatch, tmp_path):
    _build_fake_project(tmp_path)
    monkeypatch.setattr(ep.cdp_cache, "get_tiltseries", lambda tid: [SimpleNamespace(id=tid, s3_mrc_file=None)])
    monkeypatch.setattr(ep.core_data.global_fs, "get", lambda *a: None)
    with pytest.raises(click.ClickException, match="no MRC file on the portal"):
        ep.materialize_project_to_disk(tmp_path)


def test_download_uses_current_shared_filesystem(monkeypatch, tmp_path):
    calls = {}
    current = SimpleNamespace(get=lambda source, dest: calls.update(source=source, dest=dest))
    monkeypatch.setattr(ep.core_data, "global_fs", current)

    destination = tmp_path / "tilt.mrc"
    assert ep._download_s3_file("s3://private/tilt.mrc", destination) == destination
    assert calls == {"source": "private/tilt.mrc", "dest": str(destination)}


def test_export_data_portal_project_wiring(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(ep.cdp_generate, "generate_starfiles", lambda **k: called.update(gen=k) or (None, None, None))
    monkeypatch.setattr(ep, "materialize_project_to_disk", lambda od: called.update(materialized=od))
    ep.export_data_portal_project(tmp_path, dataset_ids=[10426], annotation_names=["ribosome"])
    assert called["gen"]["dataset_ids"] == [10426]
    assert called["materialized"] == tmp_path
