# TODO: add tests for crop size parameter
# TODO: create more focused tests for individual functions?
# TODO: create a highly sensitive cross-correlation test for MRCs? (better represents the testing we're trying to do)
# TODO: don't store tiltseries / RELION *.mrcs data in the repository, but host them on zenodo
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from subtomo_extract import cli, extract_subtomograms

DATASET_CONFIGS = {
    "synthetic": {
        "data_root": Path("tests/data/relion_project_synthetic"),
        "tol": 1e-7,
        "float_tol": 1e-4,
    },
    "unroofing": {
        "data_root": Path("tests/data/relion_project_unroofing"),
        "tol": 5e-5,  # relaxed requirements due to noisier data
        "float_tol": 1e-4,
    },
}

EXTRACTION_PARAMETERS = {
    "baseline": {"box_size": 64, "bin": 1},
    "float16": {"box_size": 64, "bin": 1, "float16": True},
    "box16_bin4": {"box_size": 16, "bin": 4},
    "box16_bin6": {"box_size": 16, "bin": 6},
    "box32_bin2": {"box_size": 32, "bin": 2},
    "box32_bin4": {"box_size": 32, "bin": 4},
    "noctf": {"box_size": 64, "bin": 1, "no_ctf": True},
    "nocirclecrop": {"box_size": 64, "bin": 1, "no_circle_crop": True},
    "noctf_nocirclecrop": {"box_size": 64, "bin": 1, "no_ctf": True, "no_circle_crop": True},
    "box16_bin4_noctf": {"box_size": 16, "bin": 4, "no_ctf": True},
    "box16_bin4_nocirclecrop": {"box_size": 16, "bin": 4, "no_circle_crop": True},
    "box16_bin4_noctf_nocirclecrop": {"box_size": 16, "bin": 4, "no_ctf": True, "no_circle_crop": True},
    "box128_crop64": {"box_size": 128, "bin": 1, "crop_size": 64},
    "box64_bin2_crop32": {"box_size": 64, "bin": 2, "crop_size": 32},
    "box32_bin6_crop16": {"box_size": 32, "bin": 6, "crop_size": 16},
}

PARAMS = [
    (dataset, dataset_config, extract_suffix, extract_arguments)
    for dataset, dataset_config in DATASET_CONFIGS.items()
    for extract_suffix, extract_arguments in EXTRACTION_PARAMETERS.items()
]


@pytest.mark.parametrize(
    "dataset, dataset_config, extract_suffix, extract_arguments",
    PARAMS,
    ids=[f"{dataset}_{extract_suffix}" for dataset, _, extract_suffix, _ in PARAMS],
)
def test_extract_local_subtomograms_parametrized(
    validate_optimisation_set_starfile,
    validate_particles_starfile,
    compare_mrcs_dirs,
    dataset,
    dataset_config,
    extract_suffix,
    extract_arguments,
):
    data_root = dataset_config["data_root"]
    tol = dataset_config["tol"]
    float_tol = dataset_config["float_tol"]
    float16 = extract_arguments.get("float16", False)

    output_dir = Path(f"tests/output/{dataset}_{extract_suffix}/")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    extract_subtomograms(
        box_size=extract_arguments.get("box_size"),
        crop_size=extract_arguments.get("crop_size"),
        bin=extract_arguments.get("bin"),
        float16=float16,
        no_ctf=extract_arguments.get("no_ctf", False),
        no_circle_crop=extract_arguments.get("no_circle_crop", False),
        output_dir=output_dir,
        particles_starfile=data_root / "particles.star",
        tiltseries_relative_dir=data_root,
        tomograms_starfile=data_root / "tomograms.star",
    )

    validate_optimisation_set_starfile(output_dir / "optimisation_set.star")
    validate_particles_starfile(
        output_dir / "particles.star",
        data_root / f"relion_output_{extract_suffix}/particles.star",
    )

    subtomo_dir = output_dir / "Subtomograms/"
    relion_dir = data_root / f"relion_output_{extract_suffix}/Subtomograms/"
    # extra tolerance for float16 data
    if float16:
        compare_mrcs_dirs(relion_dir, subtomo_dir, tol=float_tol)
    else:
        compare_mrcs_dirs(relion_dir, subtomo_dir, tol=tol)


@pytest.mark.parametrize(
    "dataset, extract_suffix",
    [
        ("unroofing", "baseline"),
        ("synthetic", "box16_bin4_noctf_nocirclecrop"),
    ],
    ids=["unroofing_baseline", "synthetic_box16_bin4_noctf_nocirclecrop"],
)
def test_cli_extract_local(tmp_path, compare_mrcs_dirs, dataset, extract_suffix):
    dataset_config = DATASET_CONFIGS[dataset]
    extract_arguments = EXTRACTION_PARAMETERS[extract_suffix]

    output_dir = tmp_path / f"{dataset}_{extract_suffix}"
    data_root = dataset_config["data_root"]

    args = [
        "local",
        "--particles-starfile",
        str(data_root / "particles.star"),
        "--tiltseries-relative-dir",
        str(data_root),
        "--tomograms-starfile",
        str(data_root / "tomograms.star"),
        "--box-size",
        str(extract_arguments["box_size"]),
        "--bin",
        str(extract_arguments.get("bin", 1)),
        "--output-dir",
        str(output_dir),
    ]

    if extract_arguments.get("float16"):
        args.append("--float16")
    if extract_arguments.get("no_ctf"):
        args.append("--no-ctf")
    if extract_arguments.get("no_circle_crop"):
        args.append("--no-circle-crop")

    runner = CliRunner()
    runner.invoke(cli, args, catch_exceptions=False)

    subtomo_dir = output_dir / "Subtomograms/"
    relion_dir = data_root / f"relion_output_{extract_suffix}/Subtomograms/"
    # extra tolerance for float16 data
    if extract_arguments.get("float16"):
        compare_mrcs_dirs(relion_dir, subtomo_dir, tol=dataset_config["tol"] * 100)
    else:
        compare_mrcs_dirs(relion_dir, subtomo_dir, tol=dataset_config["tol"])
