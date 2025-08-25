# import pytest

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

# RESOLVE_PARAM_CONFIGS = [
#     {
#         "test_id": "no_filter",
#         "deposition_ids": None,
#         "deposition_titles": None,
#         "dataset_ids": None,
#         "dataset_titles": None,
#         "organism_names": None,
#         "cell_names": None,
#         "run_ids": None,
#         "run_names": None,
#         "tiltseries_ids": None,
#         "alignment_ids": None,
#         "tomogram_ids": None,
#         "tomogram_names": None,
#         "annotation_ids": None,
#         "annotation_names": None,
#     },
# ]


# @pytest.mark.parametrize(
#     "params",
#     [pytest.param(config, id=config["test_id"]) for config in RESOLVE_PARAM_CONFIGS],
# )
# def test_resolve_annotation_files(params):
#     test_id = params.pop("test_id")
#     pass
