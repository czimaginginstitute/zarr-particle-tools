# TODO: create more focused tests for individual functions?
# TODO: create a highly sensitive cross-correlation test for MRCs? (better represents the testing we're trying to do)
# TODO: don't store tiltseries / RELION *.mrcs data in the repository, but host them on zenodo
import pytest
import shutil
from pathlib import Path
from data_portal_subtomo_extract import extract_local_subtomograms

DATASET_CONFIGS = {
    "synthetic": {
        "data_root": Path("tests/data/relion_project_synthetic"),
        "particles_tomo_name_prefix": "session1_",
        "tiltseries_x": 630,
        "tiltseries_y": 630,
        "tol": 1e-7,
    },
    "unroofing": {
        "data_root": Path("tests/data/relion_project_unroofing"),
        "particles_tomo_name_prefix": "session1_",
        "tiltseries_x": 4092,
        "tiltseries_y": 5760,
        "tol": 5e-5, # relaxed requirements due to noisier data
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
}

PARAMS = [(dataset, dataset_config, extract_suffix, extract_arguments) for dataset, dataset_config in DATASET_CONFIGS.items() for extract_suffix, extract_arguments in EXTRACTION_PARAMETERS.items()]


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
    float16 = extract_arguments.get("float16", False)
    
    output_dir = Path(f"tests/output/{dataset}_{extract_suffix}/")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    extract_local_subtomograms(
        particles_starfile=data_root / "particles.star",
        particles_tomo_name_prefix=dataset_config["particles_tomo_name_prefix"],
        box_size=extract_arguments.get("box_size"),
        bin=extract_arguments.get("bin"),
        tiltseries_dir=data_root / "tiltSeries/",
        tiltseries_starfile=data_root / "tiltSeries/tomograms.star",
        tiltseries_x=dataset_config["tiltseries_x"],
        tiltseries_y=dataset_config["tiltseries_y"],
        aln_dir=data_root / "aln/",
        float16=float16,
        no_ctf=extract_arguments.get("no_ctf", False),
        no_circle_crop=extract_arguments.get("no_circle_crop", False),
        output_dir=output_dir,
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
        compare_mrcs_dirs(relion_dir, subtomo_dir, tol=tol * 100)
    else:
        compare_mrcs_dirs(relion_dir, subtomo_dir, tol=tol)
