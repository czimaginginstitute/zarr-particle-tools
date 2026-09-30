"""The portal import makes one optics group per dataset, split only where the optics differ."""

from zarr_particle_tools.importtomo import optics_group_names
from zarr_particle_tools.portal_selection import RunSelection


def run(run_id: int, dataset_id: int = 10521, pixel: float = 1.341) -> RunSelection:
    return RunSelection(
        dataset_id=dataset_id,
        run_id=run_id,
        run_name=f"ts_{run_id}",
        tiltseries_id=run_id,
        alignment_id=run_id,
        tomogram_id=run_id,
        voxel_spacing_id=run_id,
        voxel_spacing=10.005,
        tiltseries_pixel_size=pixel,
        voltage_kv=300.0,
        spherical_aberration_mm=2.7,
        tomogram_size=(548, 772, 320),
        rln_tomo_size=(4088, 5760, 2387),
        tiltseries_uri="s3://b/ts.zarr",
        tomogram_uri="s3://b/t.zarr",
    )


def test_one_optics_group_per_dataset():
    names = optics_group_names([run(1), run(2), run(3, dataset_id=10522)], 0.07)
    assert names == {1: "dataset_10521", 2: "dataset_10521", 3: "dataset_10522"}


def test_a_dataset_whose_runs_differ_in_optics_is_split():
    names = optics_group_names([run(1), run(2, pixel=1.5), run(3)], 0.07)
    assert names == {1: "dataset_10521_optics1", 2: "dataset_10521_optics2", 3: "dataset_10521_optics1"}
