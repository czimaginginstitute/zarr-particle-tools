import shutil
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import starfile
from click.testing import CliRunner

from tests.helpers.compare import mrc_close_unmasked, mrc_headers_match
from zarr_particle_tools.subtomo_reconstruct import cli, reconstruct_local

# All cases pass at the float32 storage floor via the magnitude-aware unmasked comparator; measured
# worst voxels run ~1-36x ULP (C8/D8 highest from 45deg interpolation), so ulp_factor=64 gives margin.
# baseline_OH is the sole exception: RELION's improper-group symmetrization is non-Hermitian on kx=0
# (a RELION bug; Python is correct), so it keeps a loose ulp_factor.
DEFAULT_ULP_FACTOR = 64.0
# data_half* (pre-gridding-correction) runs highest (~72x ULP); final/full halves ~3-4x.
HALF_ULP_FACTOR = 128.0

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
    "baseline_D8": {"box_size": 64, "symmetry": "D8"},
    "baseline_T": {"box_size": 64, "symmetry": "T"},
    "baseline_O": {"box_size": 64, "symmetry": "O"},
    "baseline_OH": {"box_size": 64, "symmetry": "OH", "ulp_factor": 40000.0},  # RELION kx=0 non-Hermitian bug
    "baseline_I": {"box_size": 64, "symmetry": "I"},
    "baseline_I1": {"box_size": 64, "symmetry": "I1"},
    "baseline_I2": {"box_size": 64, "symmetry": "I2"},  # same symmetry as baseline_I (I == I2)
    "baseline_I3": {"box_size": 64, "symmetry": "I3"},
    "baseline_I4": {"box_size": 64, "symmetry": "I4"},
    "box256": {"box_size": 256},
    "box256_noctf": {"box_size": 256, "no_ctf": True},
    "box256_bin2": {"box_size": 256, "bin": 2},
    "box256_bin2_noctf": {"box_size": 256, "bin": 2, "no_ctf": True},
    "box128": {"box_size": 128},
    "box128_bin2": {"box_size": 128, "bin": 2},
    "box128_bin2_noctf": {"box_size": 128, "bin": 2, "no_ctf": True},
    "box128_crop64": {"box_size": 128, "bin": 1, "crop_size": 64},
    "box128_bin2_crop64": {"box_size": 128, "bin": 2, "crop_size": 64},
    "box32_bin2": {"box_size": 32, "bin": 2},
    "box16_bin4": {"box_size": 16, "bin": 4},
    "box16_bin6": {"box_size": 16, "bin": 6},
    "box64_bin2_crop32": {"box_size": 64, "bin": 2, "crop_size": 32},
    "box32_bin4_crop16": {"box_size": 32, "bin": 4, "crop_size": 16},
}

UNROOFING_RECONSTRUCT_PARAMETERS = {
    "baseline": {"box_size": 384, "crop_size": 256},
    "baseline_polished": {
        "box_size": 384,
        "crop_size": 256,
        "particles_starfile": Path("tests/data/relion_project_unroofing/reconstruct_particles_polished.star"),
        "tomograms_starfile": Path("tests/data/relion_project_unroofing/tomograms_polished.star"),
        "trajectories_starfile": Path("tests/data/relion_project_unroofing/motion.star"),
    },
}

DATASET_CONFIGS = {
    "synthetic": {
        "data_root": Path("tests/data/relion_project_synthetic"),
        "reconstruct_parameters": SYNTHETIC_RECONSTRUCT_PARAMETERS,
    },
    "unroofing": {
        "data_root": Path("tests/data/relion_project_unroofing"),
        "particles_starfile": Path("tests/data/relion_project_unroofing/reconstruct_particles.star"),
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
    ulp_factor = reconstruct_arguments.get("ulp_factor", DEFAULT_ULP_FACTOR)
    no_ctf = reconstruct_arguments.get("no_ctf", False)

    output_dir = Path(f"tests/output/reconstruct_{dataset}_{reconstruct_suffix}/")
    if output_dir.exists():
        shutil.rmtree(output_dir)

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
    assert mrc_close_unmasked(relion_dir / "merged.mrc", reconstruct_dir / "merged.mrc", ulp_factor=ulp_factor)
    assert mrc_headers_match(relion_dir / "merged.mrc", reconstruct_dir / "merged.mrc")

    # Half-map parity vs RELION when the particles carry random subsets (unroofing).
    if (relion_dir / "half1.mrc").exists():
        half_files = [f"data_{t}.mrc" for t in ("half1", "half2")]
        half_files += [f"{t}_full.mrc" for t in ("half1", "half2")]
        half_files += [f"{t}.mrc" for t in ("half1", "half2")]
        for name in half_files:
            assert mrc_close_unmasked(relion_dir / name, reconstruct_dir / name, ulp_factor=HALF_ULP_FACTOR)
            assert mrc_headers_match(relion_dir / name, reconstruct_dir / name)


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
    assert mrc_close_unmasked(relion_dir / "merged.mrc", reconstruct_dir / "merged.mrc", ulp_factor=DEFAULT_ULP_FACTOR)


def _read_mrc(path):
    with mrcfile.open(path, mode="r") as mrc:
        return np.asarray(mrc.data, dtype=np.float64)


def test_reconstruct_half_maps_selfconsistency(tmp_path):
    """Sanity-check the half-set code path (synthetic has no RELION half refs): inject random subsets,
    reconstruct, and assert the half artifacts are written, well-formed, finite, and distinct."""
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    box_size = 64

    metadata = starfile.read(data_root / "particles.star")
    particles = metadata["particles"].copy()
    particles["rlnRandomSubset"] = [1 if i % 2 == 0 else 2 for i in range(len(particles))]
    subset_starfile = tmp_path / "particles_subsets.star"
    starfile.write({"optics": metadata["optics"], "particles": particles}, subset_starfile)

    output_dir = tmp_path / "reconstruct_half"
    reconstruct_local(
        box_size=box_size,
        output_dir=output_dir,
        particles_starfile=subset_starfile,
        tiltseries_relative_dir=data_root,
        tomograms_starfile=data_root / "tomograms.star",
    )

    merged = _read_mrc(output_dir / "merged.mrc")
    for name in ["data_half1.mrc", "data_half2.mrc", "half1_full.mrc", "half2_full.mrc", "half1.mrc", "half2.mrc"]:
        path = output_dir / name
        assert path.exists(), f"missing half-map artifact: {name}"
        data = _read_mrc(path)
        assert data.shape == merged.shape
        assert np.isfinite(data).all()
        assert mrc_headers_match(output_dir / "merged.mrc", path)

    assert not np.allclose(_read_mrc(output_dir / "half1.mrc"), _read_mrc(output_dir / "half2.mrc"))
