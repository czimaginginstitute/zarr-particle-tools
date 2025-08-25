import pytest

from generate.generate_starfiles import resolve_annotation_files

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
