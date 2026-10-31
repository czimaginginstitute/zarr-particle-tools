import shutil
import subprocess
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile
from click.testing import CliRunner

import zarr_particle_tools.generate.copick_generate_starfiles as copick_generate
from tests.helpers.compare import mrc_equal
from tests.helpers.deformation import deformed_project, synthetic_names, write_deformed_project
from zarr_particle_tools.core.helpers import STAR_NAME_COLUMNS, read_optimisation_set
from zarr_particle_tools.subtomo_extract import (
    cli,
    extract_subtomograms,
    parse_extract_copick_local_subtomograms,
    parse_extract_local_subtomograms,
    subtomogram_path,
)
from zarr_particle_tools.subtomo_reconstruct import reconstruct_copick_local

DATASET_CONFIGS = {
    "synthetic": {
        "data_root": Path("tests/data/relion_project_synthetic"),
        "tol": 5e-8,
        "float_tol": 1e-4,
    },
    "unroofing": {
        "data_root": Path("tests/data/relion_project_unroofing"),
        # 5e-5 accommodates RELION's float32-before-cropCircle mean-subtraction ordering, which
        # leaves a ~3e-5 DC residual on the no-CTF cases; test_extract_strict.py pins that
        # explicitly via extra_atol, and checks the rest at ~10x the float32 ULP.
        "tol": 5e-5,
        "float_tol": 1e-6,
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
        data_root / f"Extract/relion_output_{extract_suffix}/particles.star",
    )

    subtomo_dir = output_dir / "Subtomograms/"
    relion_dir = data_root / f"Extract/relion_output_{extract_suffix}/Subtomograms/"
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

    tol = dataset_config["tol"]
    float_tol = dataset_config["float_tol"]
    float16 = extract_arguments.get("float16", False)

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
    relion_dir = data_root / f"Extract/relion_output_{extract_suffix}/Subtomograms/"
    if float16:
        compare_mrcs_dirs(relion_dir, subtomo_dir, tol=float_tol)
    else:
        compare_mrcs_dirs(relion_dir, subtomo_dir, tol=tol)


@pytest.mark.parametrize("dataset, extract_suffix", [("unroofing", "baseline"), ("unroofing", "box64_bin2_crop32")])
def test_cli_extract_data_portal(tmp_path, dataset, extract_suffix):
    """
    A test to ensure that the data portal CLI can run without error and produces the expected output files.
    This test does not compare the output files to any reference data, as of now.
    """
    extract_arguments = EXTRACTION_PARAMETERS[extract_suffix]
    output_dir = tmp_path / f"{dataset}_{extract_suffix}_data_portal"

    args = [
        "data-portal",
        "--run-id",
        "16848,16851",
        "--annotation-names",
        "ribosome",
        "--inexact-match",
        "--ground-truth",
        "--box-size",
        str(extract_arguments["box_size"]),
        "--crop-size",
        str(extract_arguments.get("crop_size", extract_arguments["box_size"])),
        "--output-dir",
        str(output_dir),
    ]

    if extract_arguments.get("bin", 1) != 1:
        args.append("--bin")
        args.append(str(extract_arguments["bin"]))
    if extract_arguments.get("float16"):
        args.append("--float16")
    if extract_arguments.get("no_ctf"):
        args.append("--no-ctf")
    if extract_arguments.get("no_circle_crop"):
        args.append("--no-circle-crop")

    runner = CliRunner()
    runner.invoke(cli, args, catch_exceptions=False)

    assert (output_dir / "particles.star").exists()
    assert (output_dir / "tomograms.star").exists()

    assert (output_dir / "tiltseries/tiltseries_placeholder.mrcs").exists()
    assert (output_dir / "tiltseries/run_16848_tiltseries_16582_alignment_17772_spacing_17051.star").exists()
    assert (output_dir / "tiltseries/run_16851_tiltseries_16585_alignment_17775_spacing_17054.star").exists()

    assert (
        output_dir / "Subtomograms/run_16848_tiltseries_16582_alignment_17772_spacing_17051/1_stack2d.mrcs"
    ).exists()
    assert (
        output_dir / "Subtomograms/run_16848_tiltseries_16582_alignment_17772_spacing_17051/438_stack2d.mrcs"
    ).exists()
    assert not (
        output_dir / "Subtomograms/run_16848_tiltseries_16582_alignment_17772_spacing_17051/439_stack2d.mrcs"
    ).exists()
    assert (
        len(list((output_dir / "Subtomograms/run_16848_tiltseries_16582_alignment_17772_spacing_17051").glob("*.mrcs")))
        == 438
    )
    assert (
        output_dir / "Subtomograms/run_16851_tiltseries_16585_alignment_17775_spacing_17054/10_stack2d.mrcs"
    ).exists()


def test_extract_arbitrary_particle_names(tmp_path):
    # RELION treats rlnTomoParticleName as an opaque string: output paths come from the full name, rows follow
    # tomograms.star order then input order. session1_TS_0 is a copy of session1_TS_1 listed after it, and every
    # particle is a copy of RELION's session1_TS_1/{i + 1} in the reference output.
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    (tmp_path / "tiltseries").mkdir(parents=True)
    for f in ["TS_1.mrcs", "TS_1.star"]:
        (tmp_path / "tiltseries" / f).symlink_to((data_root / "tiltseries" / f).resolve())
    ts_star = (data_root / "tiltseries/TS_1.star").read_text()
    (tmp_path / "tiltseries/TS_0.star").write_text(ts_star.replace("data_session1_TS_1", "data_session1_TS_0"))
    tomograms = starfile.read(data_root / "tomograms.star")
    ts0 = tomograms.assign(rlnTomoName="session1_TS_0", rlnTomoTiltSeriesStarFile="tiltseries/TS_0.star")
    starfile.write({"global": pd.concat([tomograms, ts0])}, tmp_path / "tomograms.star")

    particles_data = starfile.read(data_root / "particles.star")
    ts1_particles = particles_data["particles"]
    ts1_names = ["session1_TS_1/007", "session1_TS_1/ext1_-3", "p3", *[f"session1_TS_1/{29 - i}" for i in range(3, 25)]]
    ts0_names = ["session1_TS_0/12", "session1_TS_0/ext1_-3", "session1_TS_0/1"]
    ts0_particles = ts1_particles.iloc[: len(ts0_names)].assign(rlnTomoName="session1_TS_0")
    particles_data["particles"] = pd.concat(
        [ts0_particles.assign(rlnTomoParticleName=ts0_names), ts1_particles.assign(rlnTomoParticleName=ts1_names)]
    )
    starfile.write(particles_data, tmp_path / "particles.star")

    output_dir = tmp_path / "output"
    extract_subtomograms(
        box_size=64,
        output_dir=output_dir,
        particles_starfile=tmp_path / "particles.star",
        tiltseries_relative_dir=tmp_path,
        tomograms_starfile=tmp_path / "tomograms.star",
    )

    particles = starfile.read(output_dir / "particles.star")["particles"]
    assert particles["rlnImageName"][1] == str(
        (output_dir / "Subtomograms/session1_TS_1/ext1_-3_stack2d.mrcs").resolve()
    )
    expected = [("session1_TS_1", i, name) for i, name in enumerate(ts1_names)]
    expected += [("session1_TS_0", i, name) for i, name in enumerate(ts0_names)]
    assert particles["rlnTomoParticleName"].tolist() == [name for _, _, name in expected]
    relion_dir = data_root / "Extract/relion_output_baseline/Subtomograms/session1_TS_1"
    for (tomo_name, i, name), image_name in zip(expected, particles["rlnImageName"], strict=True):
        path = subtomogram_path(output_dir, tomo_name, name).resolve()
        assert image_name == str(path)
        assert mrc_equal(relion_dir / f"{i + 1}_stack2d.mrcs", path, tol=DATASET_CONFIGS["synthetic"]["tol"])


def test_extract_keeps_numeric_names(tmp_path):
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    tomograms = starfile.read(data_root / "tomograms.star").assign(rlnTomoName="007", rlnOpticsGroupName="007")
    starfile.write({"global": tomograms}, tmp_path / "tomograms.star")
    particles_data = starfile.read(data_root / "particles.star")
    names = [f"{i:03d}" for i in range(1, len(particles_data["particles"]) + 1)]
    particles_data["optics"] = particles_data["optics"].assign(rlnOpticsGroupName="007")
    particles_data["particles"] = particles_data["particles"].assign(rlnTomoName="007", rlnTomoParticleName=names)
    starfile.write(particles_data, tmp_path / "particles.star")

    output_dir = tmp_path / "output"
    extract_subtomograms(
        box_size=64,
        output_dir=output_dir,
        particles_starfile=tmp_path / "particles.star",
        tiltseries_relative_dir=data_root,
        tomograms_starfile=tmp_path / "tomograms.star",
    )

    output = starfile.read(output_dir / "particles.star", parse_as_string=STAR_NAME_COLUMNS)
    assert output["optics"]["rlnOpticsGroupName"].eq("007").all()
    particles = output["particles"]
    assert particles["rlnTomoName"].eq("007").all()
    assert particles["rlnTomoParticleName"].tolist() == names
    assert (output_dir / "Subtomograms/007/001_stack2d.mrcs").exists()


def test_extract_unnamed_particles_with_trajectories(tmp_path):
    # unnamed particles are named before trajectories are looked up by name; zero shifts must reproduce RELION's baseline
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    n_particles = len(starfile.read(data_root / "particles.star")["particles"])
    n_tilts = len(starfile.read(data_root / "tiltseries/TS_1.star"))
    zero_shifts = pd.DataFrame(
        0.0, index=range(n_tilts), columns=["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
    )
    trajectories = {"general": {"rlnParticleNumber": n_particles}}
    trajectories |= {f"session1_TS_1/{i}": zero_shifts for i in range(1, n_particles + 1)}
    starfile.write(trajectories, tmp_path / "motion.star")

    output_dir = tmp_path / "output"
    extract_subtomograms(
        box_size=64,
        output_dir=output_dir,
        particles_starfile=data_root / "particles.star",
        tiltseries_relative_dir=data_root,
        tomograms_starfile=str(data_root / "tomograms.star"),
        trajectories_starfile=str(tmp_path / "motion.star"),
    )

    relion_dir = data_root / "Extract/relion_output_baseline/Subtomograms/session1_TS_1"
    for i in range(1, n_particles + 1):
        assert mrc_equal(
            relion_dir / f"{i}_stack2d.mrcs",
            output_dir / f"Subtomograms/session1_TS_1/{i}_stack2d.mrcs",
            tol=DATASET_CONFIGS["synthetic"]["tol"],
        )


def test_extract_rejects_particles_sharing_a_subtomogram_path(tmp_path):
    # the same slash-containing names in two tomograms map to the same RELION output path
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    tomograms = starfile.read(data_root / "tomograms.star")
    starfile.write(
        {"global": pd.concat([tomograms, tomograms.assign(rlnTomoName="session1_TS_0")])}, tmp_path / "tomograms.star"
    )
    particles_data = starfile.read(data_root / "particles.star")
    ts1_particles = particles_data["particles"]
    particles_data["particles"] = pd.concat([ts1_particles, ts1_particles.assign(rlnTomoName="session1_TS_0")])
    particles_data["particles"]["rlnTomoParticleName"] = "session1_TS_1/" + (
        particles_data["particles"].groupby("rlnTomoName").cumcount() + 1
    ).astype(str)
    starfile.write(particles_data, tmp_path / "particles.star")

    with pytest.raises(ValueError, match="more than one particle"):
        extract_subtomograms(
            box_size=64,
            output_dir=tmp_path / "output",
            particles_starfile=tmp_path / "particles.star",
            tiltseries_relative_dir=data_root,
            tomograms_starfile=tmp_path / "tomograms.star",
        )
    assert not (tmp_path / "output" / "Subtomograms").exists()


def test_extract_warns_about_undefined_tomograms(tmp_path, caplog):
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    particles_data = starfile.read(data_root / "particles.star")
    particles_data["particles"]["rlnTomoName"] = "unlisted"
    starfile.write(particles_data, tmp_path / "particles.star")

    with pytest.raises(ValueError, match="No particles found that belong to any of the defined tomograms"):
        extract_subtomograms(
            box_size=64,
            output_dir=tmp_path / "output",
            particles_starfile=tmp_path / "particles.star",
            tiltseries_relative_dir=data_root,
            tomograms_starfile=data_root / "tomograms.star",
        )
    assert "Particles were found belonging to the following undefined tomograms: unlisted" in caplog.text


def test_extract_copick_local_keeps_zero_padded_names(tmp_path, monkeypatch):
    # copick runs are matched to optics groups by name; zero-padded names must not be parsed as ints
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    tomograms = starfile.read(data_root / "tomograms.star")
    tomograms = pd.concat(
        [
            tomograms.assign(rlnTomoName=name, rlnOpticsGroupName=name, rlnOpticsGroup=i)
            for i, name in enumerate(["007", "008"], start=1)
        ]
    )
    starfile.write({"global": tomograms}, tmp_path / "tomograms.star")
    picked = starfile.read(data_root / "particles.star")["particles"].iloc[:3]

    class Pick:
        def __init__(self, run_name):
            self.run = type("Run", (), {"name": run_name})

        def df(self, format):
            return picked.drop(columns=["rlnTomoName", "rlnOpticsGroup"])

    monkeypatch.setattr(copick_generate, "get_copick_picks", lambda *args: [Pick("007"), Pick("008")])

    output_dir = tmp_path / "output"
    parse_extract_copick_local_subtomograms(
        box_size=64,
        output_dir=output_dir,
        copick_config=None,
        copick_name="particle",
        copick_session_id="0",
        copick_user_id="user",
        copick_run_names=["007", "008"],
        tiltseries_relative_dir=data_root,
        tomograms_starfile=tmp_path / "tomograms.star",
    )

    particles = starfile.read(output_dir / "particles.star", parse_as_string=["rlnTomoName", "rlnTomoParticleName"])
    assert particles["particles"]["rlnTomoParticleName"].tolist() == [
        f"{t}/{i}" for t in ["007", "008"] for i in (1, 2, 3)
    ]
    relion_dir = data_root / "Extract/relion_output_baseline/Subtomograms/session1_TS_1"
    for tomo_name in ["007", "008"]:
        for i in (1, 2, 3):
            assert mrc_equal(
                relion_dir / f"{i}_stack2d.mrcs",
                output_dir / f"Subtomograms/{tomo_name}/{i}_stack2d.mrcs",
                tol=DATASET_CONFIGS["synthetic"]["tol"],
            )


def test_extract_rejects_duplicate_tomograms(tmp_path):
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    tomograms = starfile.read(data_root / "tomograms.star")
    starfile.write({"global": pd.concat([tomograms, tomograms])}, tmp_path / "tomograms.star")

    with pytest.raises(ValueError, match="listed more than once.*session1_TS_1"):
        extract_subtomograms(
            box_size=64,
            output_dir=tmp_path / "output",
            particles_starfile=data_root / "particles.star",
            tiltseries_relative_dir=data_root,
            tomograms_starfile=tmp_path / "tomograms.star",
        )
    assert not (tmp_path / "output" / "Subtomograms").exists()


def _write_trajectories(path, particle_names, shifts):
    """motion.star: one block per particle with a per-tilt rlnOrigin{X,Y,Z}Angst shift."""
    columns = ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
    blocks = {name: pd.DataFrame(shift, columns=columns) for name, shift in zip(particle_names, shifts, strict=True)}
    starfile.write({"general": {"rlnParticleNumber": len(blocks)}, **blocks}, path)


def test_extract_applies_trajectories(tmp_path):
    # without CTF, shifting every tilt by d is the same as moving the particle by d
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    particles_data = starfile.read(data_root / "particles.star")
    particles = particles_data["particles"]
    names = [f"session1_TS_1/{i}" for i in range(1, len(particles) + 1)]
    n_tilts = len(starfile.read(data_root / "tiltseries/TS_1.star"))
    shift = np.array([12.0, -7.5, 4.0])
    _write_trajectories(tmp_path / "motion.star", names, [np.tile(shift, (n_tilts, 1))] * len(names))
    moved = particles_data | {
        "particles": particles.assign(
            rlnCenteredCoordinateXAngst=particles["rlnCenteredCoordinateXAngst"] + shift[0],
            rlnCenteredCoordinateYAngst=particles["rlnCenteredCoordinateYAngst"] + shift[1],
            rlnCenteredCoordinateZAngst=particles["rlnCenteredCoordinateZAngst"] + shift[2],
        )
    }
    starfile.write(moved, tmp_path / "moved.star")

    for name, particles_starfile, trajectories_starfile in [
        ("trajectories", data_root / "particles.star", tmp_path / "motion.star"),
        ("moved", tmp_path / "moved.star", None),
    ]:
        extract_subtomograms(
            box_size=64,
            no_ctf=True,
            output_dir=tmp_path / name,
            particles_starfile=particles_starfile,
            tiltseries_relative_dir=data_root,
            tomograms_starfile=data_root / "tomograms.star",
            trajectories_starfile=trajectories_starfile,
        )
    for i in range(1, len(particles) + 1):
        np.testing.assert_allclose(
            mrcfile.read(tmp_path / f"trajectories/Subtomograms/session1_TS_1/{i}_stack2d.mrcs"),
            mrcfile.read(tmp_path / f"moved/Subtomograms/session1_TS_1/{i}_stack2d.mrcs"),
            atol=1e-5,
        )


@pytest.mark.skipif(shutil.which("relion_tomo_subtomo") is None, reason="relion_tomo_subtomo not on PATH")
def test_extract_with_trajectories_matches_relion(tmp_path):
    # per-tilt trajectories, extracted by RELION and by us from the same inputs
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    (tmp_path / "tiltseries").symlink_to((data_root / "tiltseries").resolve())
    shutil.copy(data_root / "tomograms.star", tmp_path)
    particles_data = starfile.read(data_root / "particles.star")
    names = [f"session1_TS_1/{i}" for i in range(1, len(particles_data["particles"]) + 1)]
    particles_data["particles"]["rlnTomoParticleName"] = names
    starfile.write(particles_data, tmp_path / "particles.star")
    n_tilts = len(starfile.read(data_root / "tiltseries/TS_1.star"))
    rng = np.random.default_rng(0)
    _write_trajectories(tmp_path / "motion.star", names, rng.uniform(-8, 8, (len(names), n_tilts, 3)))

    relion_args = ["--p", "particles.star", "--t", "tomograms.star", "--mot", "motion.star", "--o", "relion/"]
    relion_args += ["--b", "64", "--crop", "64", "--bin", "1", "--stack2d", "--j", "4"]
    subprocess.run(["relion_tomo_subtomo", *relion_args], cwd=tmp_path, check=True, capture_output=True)
    extract_subtomograms(
        box_size=64,
        output_dir=tmp_path / "ours",
        particles_starfile=tmp_path / "particles.star",
        tiltseries_relative_dir=tmp_path,
        tomograms_starfile=tmp_path / "tomograms.star",
        trajectories_starfile=tmp_path / "motion.star",
    )

    for i in range(1, len(names) + 1):
        stack = f"Subtomograms/session1_TS_1/{i}_stack2d.mrcs"
        assert mrc_equal(
            tmp_path / "relion" / stack, tmp_path / "ours" / stack, tol=DATASET_CONFIGS["synthetic"]["tol"]
        )


def test_extract_particles_without_optics_as_relion_does(tmp_path, validate_particles_starfile, compare_mrcs_dirs):
    """relion_tomo_import_coordinates writes no optics block, and RELION's tomograms.star has no rlnOpticsGroup."""
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    particles = starfile.read(data_root / "particles.star")["particles"]
    starfile.write({"particles": particles.drop(columns="rlnOpticsGroup")}, tmp_path / "particles.star")
    parse_extract_local_subtomograms(
        box_size=64,
        output_dir=tmp_path / "out",
        particles_starfile=tmp_path / "particles.star",
        tiltseries_relative_dir=data_root,
        tomograms_starfile=data_root / "tomograms.star",
    )
    relion = data_root / "Extract" / "relion_output_baseline"
    validate_particles_starfile(tmp_path / "out" / "particles.star", relion / "particles.star")
    compare_mrcs_dirs(
        relion / "Subtomograms", tmp_path / "out" / "Subtomograms", tol=DATASET_CONFIGS["synthetic"]["tol"]
    )


# ----------------------------------------------------------------------------- RELION-5 2D deformations
def test_extract_deformation_equals_trajectory(tmp_path):
    # a spline with a constant value field shifts every crop by s pixels in every tilt; the same crop positions
    # follow from a per-tilt 3D trajectory v_t with P_t[:2, :3] v_t = s * pixel_size (least-norm solution)
    from zarr_particle_tools.core.forwardprojection import calculate_projection_matrix_from_starfile_df

    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    names = synthetic_names(data_root)
    tilt_df = starfile.read(data_root / "tiltseries" / "TS_1.star")
    pixel_size = float(starfile.read(data_root / "tomograms.star")["rlnTomoTiltSeriesPixelSize"].iloc[0])
    shift_px = np.array([6.3, -4.1])
    nodes = np.zeros((2, 3, 3, 4))
    nodes[0, :, :, 0], nodes[1, :, :, 0] = shift_px
    write_deformed_project(tmp_path / "deformed", data_root, "spline", (3, 3), [nodes.ravel()] * len(tilt_df), names)

    trajectory = []
    for matrix in calculate_projection_matrix_from_starfile_df(tilt_df):
        a = matrix[:2, :3]
        trajectory.append(a.T @ np.linalg.solve(a @ a.T, shift_px * pixel_size))
    _write_trajectories(tmp_path / "motion.star", names, [np.array(trajectory)] * len(names))
    particles_data = starfile.read(data_root / "particles.star")
    particles_data["particles"]["rlnTomoParticleName"] = names
    starfile.write(particles_data, tmp_path / "particles.star")

    extract_subtomograms(
        box_size=64,
        output_dir=tmp_path / "deformation",
        particles_starfile=tmp_path / "deformed" / "particles.star",
        tiltseries_relative_dir=tmp_path / "deformed",
        tomograms_starfile=tmp_path / "deformed" / "tomograms.star",
    )
    extract_subtomograms(
        box_size=64,
        output_dir=tmp_path / "trajectory",
        particles_starfile=tmp_path / "particles.star",
        tiltseries_relative_dir=data_root,
        tomograms_starfile=data_root / "tomograms.star",
        trajectories_starfile=tmp_path / "motion.star",
    )
    extract_subtomograms(
        box_size=64,
        output_dir=tmp_path / "ignored",
        particles_starfile=tmp_path / "deformed" / "particles.star",
        tiltseries_relative_dir=tmp_path / "deformed",
        tomograms_starfile=tmp_path / "deformed" / "tomograms.star",
        ignore_deformations=True,
    )
    extract_subtomograms(
        box_size=64,
        output_dir=tmp_path / "plain",
        particles_starfile=tmp_path / "particles.star",
        tiltseries_relative_dir=data_root,
        tomograms_starfile=data_root / "tomograms.star",
    )
    n_differ = 0
    for i in range(1, len(names) + 1):
        stack = f"Subtomograms/session1_TS_1/{i}_stack2d.mrcs"
        deformed = mrcfile.read(tmp_path / "deformation" / stack)
        np.testing.assert_allclose(deformed, mrcfile.read(tmp_path / "trajectory" / stack), atol=1e-5)
        # --ignore-deformations is exactly the undeformed extraction, and differs from the deformed one
        plain = mrcfile.read(tmp_path / "plain" / stack)
        np.testing.assert_array_equal(mrcfile.read(tmp_path / "ignored" / stack), plain)
        # the shift can also change which tilts are visible, so a shape difference counts as a difference
        n_differ += deformed.shape != plain.shape or not np.allclose(deformed, plain, atol=1e-5)
    assert n_differ == len(names)


@pytest.mark.skipif(shutil.which("relion_tomo_subtomo") is None, reason="relion_tomo_subtomo not on PATH")
@pytest.mark.parametrize("deformation_type", ["spline", "linear", "Fourier"])
def test_extract_with_deformations_matches_relion(tmp_path, deformation_type):
    # per-tilt 2D deformations, extracted by RELION and by us from the same inputs
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    names = deformed_project(tmp_path, data_root, deformation_type)

    relion_args = ["--p", "particles.star", "--t", "tomograms.star", "--o", "relion/"]
    relion_args += ["--b", "64", "--crop", "64", "--bin", "1", "--stack2d", "--j", "4"]
    subprocess.run(["relion_tomo_subtomo", *relion_args], cwd=tmp_path, check=True, capture_output=True)
    for name, ignore in (("ours", False), ("ignored", True)):
        extract_subtomograms(
            box_size=64,
            output_dir=tmp_path / name,
            particles_starfile=tmp_path / "particles.star",
            tiltseries_relative_dir=tmp_path,
            tomograms_starfile=tmp_path / "tomograms.star",
            ignore_deformations=ignore,
        )

    tol = DATASET_CONFIGS["synthetic"]["tol"]
    n_ignored_differ = 0
    for i in range(1, len(names) + 1):
        stack = f"Subtomograms/session1_TS_1/{i}_stack2d.mrcs"
        assert mrc_equal(tmp_path / "relion" / stack, tmp_path / "ours" / stack, tol=tol)
        relion, ignored = mrcfile.read(tmp_path / "relion" / stack), mrcfile.read(tmp_path / "ignored" / stack)
        n_ignored_differ += relion.shape != ignored.shape or not np.allclose(relion, ignored, atol=tol)
    # the comparison is sensitive: without the deformation the same crops do not match RELION
    assert n_ignored_differ == len(names)


class _Pick:
    """A copick pick on run ``name`` holding the given synthetic particles."""

    def __init__(self, name, particles):
        self.run = type("Run", (), {"name": name})
        self.particles = particles

    def df(self, format):
        return self.particles.drop(columns=["rlnTomoName", "rlnOpticsGroup"])


def _pick_on_ts1(monkeypatch, particles):
    monkeypatch.setattr(copick_generate, "get_copick_picks", lambda *args: [_Pick("session1_TS_1", particles)])


COPICK = {"copick_config": None, "copick_name": "particle", "copick_session_id": "0", "copick_user_id": "user"}


def _extract(kind, project, output_dir, monkeypatch, **kwargs):
    """Extract ``project`` (its particles, or the same particles as copick picks on run session1_TS_1)."""
    inputs = {"tiltseries_relative_dir": project, "tomograms_starfile": project / "tomograms.star"}
    if kind == "local":
        return parse_extract_local_subtomograms(
            box_size=64, output_dir=output_dir, particles_starfile=project / "particles.star", **inputs, **kwargs
        )
    _pick_on_ts1(monkeypatch, starfile.read(project / "particles.star")["particles"])
    return parse_extract_copick_local_subtomograms(
        box_size=64, output_dir=output_dir, copick_run_names=["session1_TS_1"], **COPICK, **inputs, **kwargs
    )


def _bspline_project(project, data_root):
    """The synthetic project declaring a deformation type RELION does not know."""
    deformed_project(project, data_root, "linear")
    tomograms = starfile.read(project / "tomograms.star", always_dict=True)
    tomograms["global"]["rlnTomoDeformationType"] = "bspline"
    starfile.write(tomograms, project / "tomograms.star")


def _files(directory):
    return {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("damage", ["unknown deformation type", "missing tilt-series stack"])
def test_copick_extraction_and_reconstruction_refuse_bad_inputs_before_writing(tmp_path, monkeypatch, damage):
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    deformed_project(tmp_path, data_root, "linear")
    if damage == "unknown deformation type":
        tomograms = starfile.read(tmp_path / "tomograms.star", always_dict=True)
        tomograms["global"]["rlnTomoDeformationType"] = "bspline"
        starfile.write(tomograms, tmp_path / "tomograms.star")
        column = "rlnTomoDeformationType"
    else:
        (tmp_path / "tiltseries" / "TS_1.mrcs").unlink()
        column = "rlnMicrographName"
    _pick_on_ts1(monkeypatch, starfile.read(tmp_path / "particles.star")["particles"])
    copick = {
        **COPICK,
        "copick_run_names": ["session1_TS_1"],
        "tiltseries_relative_dir": tmp_path,
        "tomograms_starfile": tmp_path / "tomograms.star",
    }
    with pytest.raises(ValueError, match=column):
        parse_extract_copick_local_subtomograms(box_size=64, output_dir=tmp_path / "extract", **copick)
    with pytest.raises(ValueError, match=column):
        reconstruct_copick_local(box_size=64, output_dir=tmp_path / "reconstruct", **copick)
    for output_dir in (tmp_path / "extract", tmp_path / "reconstruct"):
        assert not (output_dir / "particles.star").exists()
        assert not (output_dir / "Subtomograms").exists()


@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("kind", ["local", "copick"])
def test_refused_extraction_leaves_output_untouched(tmp_path, monkeypatch, kind, overwrite):
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    _bspline_project(tmp_path / "bad", data_root)
    output_dir = tmp_path / "out"
    if overwrite:
        _extract(kind, data_root, output_dir, monkeypatch)
        before = _files(output_dir)
        assert sum(path.suffix == ".mrcs" for path in before) == 25
        with pytest.raises(ValueError, match="rlnTomoDeformationType"):
            _extract(kind, tmp_path / "bad", output_dir, monkeypatch, overwrite=True)
        assert _files(output_dir) == before
    else:
        with pytest.raises(ValueError, match="rlnTomoDeformationType"):
            _extract(kind, tmp_path / "bad", output_dir, monkeypatch)
        assert not (output_dir / "Subtomograms").exists()
        _extract(kind, data_root, output_dir, monkeypatch)  # the fixed rerun needs no --overwrite


@pytest.mark.parametrize("damage", ["missing stack", "unknown deformation type"])
def test_copick_extraction_ignores_tomograms_without_picks(tmp_path, monkeypatch, damage):
    # RELION skips a tomogram without particles before reading it, so nothing about session1_TS_2 is checked
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    project = tmp_path / "project"
    (project / "tiltseries").mkdir(parents=True)
    stack = (data_root / "tiltseries" / "TS_1.mrcs").resolve()
    (project / "tiltseries" / "TS_1.mrcs").symlink_to(stack)
    tilts = starfile.read(data_root / "tiltseries" / "TS_1.star")
    starfile.write({"TS_1": tilts}, project / "tiltseries" / "TS_1.star")
    ts2_tilts = tilts.assign(rlnMicrographName=tilts["rlnMicrographName"].str.replace("TS_1", "TS_2"))
    starfile.write({"TS_2": ts2_tilts}, project / "tiltseries" / "TS_2.star")
    tomograms = starfile.read(data_root / "tomograms.star")
    ts2 = tomograms.assign(rlnTomoName="session1_TS_2", rlnTomoTiltSeriesStarFile="tiltseries/TS_2.star")
    if damage == "unknown deformation type":
        (project / "tiltseries" / "TS_2.mrcs").symlink_to(stack)
        ts2 = ts2.assign(rlnTomoDeformationGridSizeX=2, rlnTomoDeformationGridSizeY=2, rlnTomoDeformationType="bspline")
    starfile.write({"global": pd.concat([tomograms, ts2], ignore_index=True)}, project / "tomograms.star")
    _pick_on_ts1(monkeypatch, starfile.read(data_root / "particles.star")["particles"].iloc[:3])

    parse_extract_copick_local_subtomograms(
        box_size=64,
        output_dir=tmp_path / "out",
        copick_run_names=["session1_TS_1"],
        tiltseries_relative_dir=project,
        tomograms_starfile=project / "tomograms.star",
        **COPICK,
    )
    assert len(list((tmp_path / "out" / "Subtomograms").rglob("*.mrcs"))) == 3


def test_copick_dry_run_refuses_what_the_real_run_refuses(tmp_path, monkeypatch):
    _bspline_project(tmp_path / "bad", DATASET_CONFIGS["synthetic"]["data_root"])
    with pytest.raises(ValueError, match="rlnTomoDeformationType"):
        _extract("copick", tmp_path / "bad", tmp_path / "out", monkeypatch, dry_run=True)
    assert not (tmp_path / "out" / "particles.star").exists()


def test_ignore_deformations_extracts_a_declaration_it_could_not_build_as_undeformed(tmp_path, monkeypatch):
    data_root = DATASET_CONFIGS["synthetic"]["data_root"]
    _bspline_project(tmp_path / "bad", data_root)
    _extract("local", tmp_path / "bad", tmp_path / "ignored", monkeypatch, ignore_deformations=True)
    _extract("local", data_root, tmp_path / "plain", monkeypatch)
    ignored, plain = (
        {p.relative_to(d): p for d in [tmp_path / name / "Subtomograms"] for p in d.rglob("*.mrcs")}
        for name in ("ignored", "plain")
    )
    assert sorted(ignored) == sorted(plain) and len(plain) == 25
    for name, path in plain.items():
        np.testing.assert_array_equal(mrcfile.read(ignored[name]), mrcfile.read(path))


def test_optimisation_set_names_absolute_paths(tmp_path, monkeypatch):
    # extract into a project-relative job directory, then name the set from elsewhere
    data_root = DATASET_CONFIGS["synthetic"]["data_root"].resolve()
    monkeypatch.chdir(tmp_path)
    parse_extract_local_subtomograms(
        box_size=64,
        output_dir=Path("Extract/job001"),
        particles_starfile=data_root / "particles.star",
        tiltseries_relative_dir=data_root,
        tomograms_starfile=data_root / "tomograms.star",
    )
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path / "home")
    particles, tomograms, _ = read_optimisation_set(tmp_path / "Extract/job001/optimisation_set.star")
    assert particles.is_absolute() and particles.samefile(tmp_path / "Extract/job001/particles.star")
    assert tomograms.is_absolute() and tomograms.samefile(data_root / "tomograms.star")
