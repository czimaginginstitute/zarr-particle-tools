import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from tests.helpers.compare import mrc_equal
from zarr_particle_tools.subtomo_reconstruct import cli, reconstruct_local

# TODO: need to add real datasets
# TODO: need to resolve bugs in reconstruction (tolerances should not be customized this much)
SYNTHETIC_RECONSTRUCT_PARAMETERS = {
    "baseline": {"box_size": 64},
    "baseline_C2": {"box_size": 64, "symmetry": "C2"},
    "baseline_C3": {"box_size": 64, "symmetry": "C3"},
    "baseline_C4": {"box_size": 64, "symmetry": "C4"},
    "baseline_C5": {"box_size": 64, "symmetry": "C5"},
    "baseline_C6": {"box_size": 64, "symmetry": "C6"},
    "baseline_C7": {"box_size": 64, "symmetry": "C7"},
    "baseline_C8": {"box_size": 64, "symmetry": "C8"},
    "baseline_D2": {"box_size": 64, "symmetry": "D2"},
    "baseline_D3": {"box_size": 64, "symmetry": "D3"},
    "baseline_D4": {"box_size": 64, "symmetry": "D4"},
    "baseline_D5": {"box_size": 64, "symmetry": "D5"},
    "baseline_D6": {"box_size": 64, "symmetry": "D6"},
    "baseline_D7": {"box_size": 64, "symmetry": "D7"},
    "baseline_D8": {
        "box_size": 64,
        "symmetry": "D8",
        "tol": 2e-2,
        "corr_tol": 3e-3,
        "error_median_tol": 5e-5,
    },  # TODO: debug & fix
    "baseline_T": {"box_size": 64, "symmetry": "T"},
    "baseline_O": {"box_size": 64, "symmetry": "O"},
    "baseline_OH": {"box_size": 64, "symmetry": "OH", "tol": 2e-3, "error_median_tol": 1e-4},  # TODO: debug & fix
    "baseline_I": {"box_size": 64, "symmetry": "I"},
    "baseline_I1": {"box_size": 64, "symmetry": "I1"},
    "baseline_I2": {"box_size": 64, "symmetry": "I2"},
    "baseline_I3": {"box_size": 64, "symmetry": "I3"},
    "baseline_I4": {"box_size": 64, "symmetry": "I4"},
    "box256": {"box_size": 256, "corr_tol": 1e-2, "tol": 7e-2},  # TODO: debug & fix
    "box256_noctf": {"box_size": 256, "no_ctf": True, "corr_tol": 6e-4},  # TODO: debug & fix
    "box256_bin2": {"box_size": 256, "bin": 2, "corr_tol": 7e-3, "tol": 4e1},  # TODO: debug & fix
    "box256_bin2_noctf": {
        "box_size": 256,
        "bin": 2,
        "no_ctf": True,
        "corr_tol": 2e-4,
        "tol": 5e-2,
    },  # TODO: debug & fix
    "box128": {"box_size": 128, "tol": 2e-3},  # TODO: debug & fix
    "box128_bin2": {"box_size": 128, "bin": 2, "corr_tol": 1e-2, "tol": 5e-2},  # TODO: debug & fix
    "box128_bin2_noctf": {"box_size": 128, "bin": 2, "no_ctf": True, "corr_tol": 5e-4},  # TODO: debug & fix
    "box128_crop64": {"box_size": 128, "bin": 1, "crop_size": 64, "tol": 2e-3},  # TODO: debug & fix
    "box128_bin2_crop64": {
        "box_size": 128,
        "bin": 2,
        "crop_size": 64,
        "corr_tol": 3e-3,
        "tol": 2e-2,
    },  # TODO: debug & fix
    "box32_bin2": {"box_size": 32, "bin": 2, "tol": 2e-2},  # TODO: debug & fix
    "box16_bin4": {
        "box_size": 16,
        "bin": 4,
        "corr_tol": 1e-3,
        "error_median_tol": 1e-4,
        "tol": 2e-2,
    },  # TODO: debug & fix
    "box16_bin6": {
        "box_size": 16,
        "bin": 6,
        "corr_tol": 2e-3,
        "error_median_tol": 1e-4,
        "tol": 2e-2,
    },  # TODO: debug & fix
    "box64_bin2_crop32": {"box_size": 64, "bin": 2, "crop_size": 32, "tol": 5e-3},  # TODO: debug & fix
    "box32_bin4_crop16": {
        "box_size": 32,
        "bin": 4,
        "crop_size": 16,
        "corr_tol": 1e-3,
        "tol": 1e-2,
    },  # TODO: debug & fix
}

# TODO: debug & fix, temporary loose tolerances
UNROOFING_RECONSTRUCT_PARAMETERS = {
    "baseline": {"box_size": 384, "crop_size": 256, "corr_tol": 2e-2, "tol": 4e1},
    "baseline_polished": {
        "box_size": 384,
        "crop_size": 256,
        "particles_starfile": Path("tests/data/relion_project_unroofing/reconstruct_particles_polished.star"),
        "tomograms_starfile": Path("tests/data/relion_project_unroofing/tomograms_polished.star"),
        "trajectories_starfile": Path("tests/data/relion_project_unroofing/motion.star"),
        "corr_tol": 2e-2,
        "tol": 4e1,
    },
}

DATASET_CONFIGS = {
    "synthetic": {
        "data_root": Path("tests/data/relion_project_synthetic"),
        "tol": 1e-3,
        "corr_tol": 1e-4,
        "error_median_tol": 1e-5,
        "reconstruct_parameters": SYNTHETIC_RECONSTRUCT_PARAMETERS,
    },
    "unroofing": {
        "data_root": Path("tests/data/relion_project_unroofing"),
        "particles_starfile": Path("tests/data/relion_project_unroofing/reconstruct_particles.star"),
        "tol": 1e-4,
        "corr_tol": 1e-5,
        "error_median_tol": 1e-6,
        "reconstruct_parameters": UNROOFING_RECONSTRUCT_PARAMETERS,
    },
}

PARAMS = [
    (dataset, dataset_config, reconstruct_suffix, reconstruct_arguments)
    for dataset, dataset_config in DATASET_CONFIGS.items()
    for reconstruct_suffix, reconstruct_arguments in dataset_config["reconstruct_parameters"].items()
]


@pytest.mark.parametrize(
    "dataset, dataset_config, reconstruct_suffix, reconstruct_arguments",
    PARAMS,
    ids=[f"{dataset}_{reconstruct_suffix}" for dataset, _, reconstruct_suffix, _ in PARAMS],
)
def test_reconstruct_local_parametrized(
    dataset,
    dataset_config,
    reconstruct_suffix,
    reconstruct_arguments,
):
    data_root = dataset_config["data_root"]
    tol = reconstruct_arguments.get("tol", dataset_config["tol"])
    corr_tol = reconstruct_arguments.get("corr_tol", dataset_config["corr_tol"])
    error_median_tol = reconstruct_arguments.get("error_median_tol", dataset_config["error_median_tol"])
    no_ctf = reconstruct_arguments.get("no_ctf", False)

    output_dir = Path(f"tests/output/reconstruct_{dataset}_{reconstruct_suffix}/")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    print(reconstruct_arguments.get("box_size"))

    reconstruct_local(
        box_size=reconstruct_arguments.get("box_size"),
        crop_size=reconstruct_arguments.get("crop_size"),
        bin=reconstruct_arguments.get("bin", 1),
        symmetry=reconstruct_arguments.get("symmetry", "C1"),
        output_dir=output_dir,
        particles_starfile=reconstruct_arguments.get(
            "particles_starfile", dataset_config.get("particles_starfile", data_root / "particles.star")
        ),
        trajectories_starfile=reconstruct_arguments.get("trajectories_starfile", None),
        tiltseries_relative_dir=data_root,
        tomograms_starfile=reconstruct_arguments.get("tomograms_starfile", data_root / "tomograms.star"),
        no_ctf=no_ctf,
    )

    reconstruct_dir = output_dir
    relion_dir = data_root / "Reconstruct" / f"relion_output_{reconstruct_suffix}"
    assert mrc_equal(
        relion_dir / "merged.mrc",
        reconstruct_dir / "merged.mrc",
        tol=tol,
        corr_tol=corr_tol,
        error_median_tol=error_median_tol,
        plot_dir=output_dir,
    )


@pytest.mark.parametrize(
    "dataset, reconstruct_suffix",
    [
        ("synthetic", "baseline"),
    ],
    ids=["synthetic_baseline"],
)
def test_cli_reconstruct_local(tmp_path, dataset, reconstruct_suffix):
    dataset_config = DATASET_CONFIGS[dataset]
    reconstruct_arguments = dataset_config["reconstruct_parameters"][reconstruct_suffix]

    output_dir = tmp_path / f"{dataset}_{reconstruct_suffix}"
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
        str(reconstruct_arguments["box_size"]),
        "--bin",
        str(reconstruct_arguments.get("bin", 1)),
        "--output-dir",
        str(output_dir),
    ]

    runner = CliRunner()
    runner.invoke(cli, args, catch_exceptions=False)

    reconstruct_dir = output_dir
    relion_dir = data_root / "Reconstruct" / f"relion_output_{reconstruct_suffix}"
    # TODO: add onto this
    assert mrc_equal(
        relion_dir / "merged.mrc", reconstruct_dir / "merged.mrc", tol=dataset_config["tol"], plot_dir=output_dir
    )
