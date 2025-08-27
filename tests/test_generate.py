import shutil
from pathlib import Path

import mrcfile
import numpy as np
import pytest
from click.testing import CliRunner

from generate.cdp_generate_starfiles import cli, resolve_annotation_files

# TODO: add more tests that actually check the values (after real testing)
# TODO: add tests - with all possible parameters and edge cases, including:
# - no annotations
# - no alignments
# - no tiltseries
# - no CTF parameters
# - across multiple tomograms
# - across multiple alignments
# - across multiple tiltseries
# - across multiple runs
# - across multiple datasets
# - across multiple deposition ids

RESOLVE_PARAM_CONFIGS = [
    {
        "test_id": "no_filter",
        "expected_annotation_file_min_count": 1000,
    },
    {
        "test_id": "run_16463_ribosome_simple",
        "expected_annotation_file_ids": [74186],
        "run_ids": [16463],
        "annotation_names": ["cytosolic ribosome"],
    },
    {
        "test_id": "run_16463_ribosome_simple_nomatch",
        "expected_fail": True,
        "run_ids": [16463],
        "annotation_names": ["nonexistent annotation"],
    },
    {
        "test_id": "run_16467_beta-gal_inexact",
        "expected_annotation_file_ids": [74209],
        "run_ids": [16467],
        "annotation_names": ["beta-gal"],
        "inexact_match": True,
    },
    {
        "test_id": "run_16468_ferritin_precise",
        "expected_annotation_file_ids": [74213],
        "deposition_ids": [10310],
        "deposition_titles": ["CZII - CryoET Object Identification Challenge"],
        "dataset_ids": [10440],
        "dataset_titles": ["CZII - CryoET Object Identification Challenge - Experimental Training Data"],
        "organism_names": ["not_reported"],
        "cell_names": ["not_reported"],
        "run_ids": [16468],
        "run_names": ["TS_86_3"],
        "tiltseries_ids": [16111],
        "alignment_ids": [17035],
        "tomogram_ids": [17035],
        "annotation_ids": [31995],
        "annotation_names": ["ferritin complex"],
        "inexact_match": False,
        "ground_truth": True,
    },
]


@pytest.mark.parametrize(
    "params",
    [pytest.param(config, id=config["test_id"]) for config in RESOLVE_PARAM_CONFIGS],
)
def test_resolve_annotation_files(params):
    del params["test_id"]
    expected_annotation_file_ids = params.pop("expected_annotation_file_ids", None)
    expected_annotation_file_min_count = params.pop("expected_annotation_file_min_count", None)
    expected_fail = params.pop("expected_fail", False)

    try:
        annotation_files = resolve_annotation_files(**params)
    except ValueError as e:
        if expected_fail:
            return  # Test passes if failure is expected
        raise e

    annotation_file_ids = [f.id for f in annotation_files]
    if expected_annotation_file_ids is not None:
        assert len(set(annotation_file_ids)) == len(annotation_file_ids), "Duplicate annotation file IDs found"
        assert len(annotation_file_ids) == len(expected_annotation_file_ids), "Mismatch in number of annotation files"
        assert set(annotation_file_ids) == set(expected_annotation_file_ids), "Mismatch in annotation file IDs"

    if expected_annotation_file_min_count is not None:
        assert len(annotation_file_ids) >= expected_annotation_file_min_count, "Fewer annotation files than expected"


def test_cli_baseline(validate_starfile):
    output_dir = Path("tests/output/data_portal_16363_ribosome")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "--run-ids",
        "16463",
        "--annotation-names",
        "cytosolic ribosome",
        "--output-dir",
        str(output_dir),
        "--debug",
    ]

    runner = CliRunner()
    runner.invoke(cli, args, catch_exceptions=False)

    particles_starfile = output_dir / "particles.star"
    tomograms_starfile = output_dir / "tomograms.star"
    tiltseries_starfile = output_dir / "tiltseries" / "run_16463_tiltseries_16106_alignment_17015_spacing_16666.star"
    placeholder_mrcs = output_dir / "tiltseries" / "tiltseries_placeholder.mrcs"

    assert particles_starfile.exists(), "Expected particles.star file not found"
    assert tomograms_starfile.exists(), "Expected tomograms.star file not found"
    assert tiltseries_starfile.exists(), "Expected tiltseries star file not found"
    assert placeholder_mrcs.exists(), "Expected tiltseries_placeholder.mrcs file not found"

    with mrcfile.open(placeholder_mrcs, mode="r", permissive=True) as mrc:
        assert np.min(mrc.data) == 0 and np.max(mrc.data) == 0, "Placeholder mrcs data is not all zeros"
        assert mrc.header.nx == 4096, "Unexpected header.nx value"
        assert mrc.header.ny == 4096, "Unexpected header.ny value"
        assert mrc.header.nz == 31, "Unexpected header.nz value"
        assert mrc.header.mx == 4096, "Unexpected header.mx value"
        assert mrc.header.my == 4096, "Unexpected header.my value"
        assert mrc.header.mz == 31, "Unexpected header.mz value"
        assert np.isclose(mrc.header.cella.x.item(), 4096 * 1.54), "Unexpected header.cella.x value"
        assert np.isclose(mrc.header.cella.y.item(), 4096 * 1.54), "Unexpected header.cella.y value"
        assert np.isclose(mrc.header.cella.z.item(), 31.0), "Unexpected header.cella.z value"
        assert np.allclose(
            (mrc.voxel_size.x.item(), mrc.voxel_size.y.item(), mrc.voxel_size.z.item()), (1.54, 1.54, 1.0)
        ), "Unexpected voxel size"

    reference_dir = Path("tests/data/data_portal_16363_ribosome")
    tiltseries_reference_starfile = (
        reference_dir / "tiltseries" / "run_16463_tiltseries_16106_alignment_17015_spacing_16666.star"
    )
    particles_reference_starfile = reference_dir / "particles.star"
    tomograms_reference_starfile = reference_dir / "tomograms.star"

    assert tiltseries_reference_starfile.exists(), "Expected tiltseries reference star file not found"
    assert particles_reference_starfile.exists(), "Expected particles reference star file not found"
    assert tomograms_reference_starfile.exists(), "Expected tomograms reference star file not found"

    validate_starfile(tomograms_starfile, tomograms_reference_starfile)
    validate_starfile(particles_starfile, particles_reference_starfile)
    validate_starfile(tiltseries_starfile, tiltseries_reference_starfile)
