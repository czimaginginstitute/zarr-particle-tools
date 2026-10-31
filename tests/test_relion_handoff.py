"""
What the ctf-refine / polish wrapper hands RELION, checked without RELION: a stand-in binary records the
optimisation set, tomograms, tilt series, stacks and particles it is given, then writes a RELION-like
output. Mirrors a real per-tomogram run: zarr tilt series, portal-style and zero-padded names, one
tomogram with trimmed tilts, trajectories, --mask / --fsc.
"""

import json
import shutil
import sys

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile
import zarr

from tests.test_extract import DATASET_CONFIGS
from zarr_particle_tools.core.constants import TILTSERIES_URI_RELION_COLUMN
from zarr_particle_tools.core.data import _remove_staging_dirs
from zarr_particle_tools.core.helpers import STAR_NAME_COLUMNS
from zarr_particle_tools.subtomo_ctfrefine import run_ctf_refine
from zarr_particle_tools.subtomo_polish import run_polish

DATA = DATASET_CONFIGS["synthetic"]["data_root"]
TOMOGRAMS = {"007": slice(None), "run_16848_tiltseries_16582": slice(2, None)}  # name -> tilts kept

FAKE_RELION = """#!{python}
import json, os, sys
from pathlib import Path
import mrcfile, numpy as np, starfile

argv = sys.argv[1:]
cwd = Path.cwd()
log = Path(os.environ["FAKE_RELION_LOG"])
call = log / f"call_{{len(list(log.glob('call_*')))}}"
call.mkdir()
names = {names}
opt = starfile.read(argv[argv.index("--i") + 1]).iloc[0].to_dict()
tomograms = starfile.read(opt["rlnTomoTomogramsFile"], parse_as_string=names)
tomograms = tomograms["global"] if isinstance(tomograms, dict) else tomograms
record = {{"argv": argv, "optimisation_set": opt, "tomograms": []}}
for i, row in tomograms.reset_index(drop=True).iterrows():
    tilts = starfile.read(cwd / row["rlnTomoTiltSeriesStarFile"])
    tilts = next(iter(tilts.values())) if isinstance(tilts, dict) else tilts
    stack = Path(row["rlnTomoTiltSeriesName"])
    with mrcfile.open(stack, header_only=True) as m:
        nz = int(m.header.nz)
    if stack.stat().st_size > 1024:  # a materialized stack rather than a header-only stub
        np.save(call / f"stack_{{i}}.npy", mrcfile.read(stack))
    record["tomograms"].append({{
        "name": row["rlnTomoName"],
        "frame_count": int(row["rlnTomoFrameCount"]),
        "micrographs": tilts["rlnMicrographName"].tolist(),
        "nz": nz,
        "stub": stack.stat().st_size <= 1024,
    }})
particles = starfile.read(opt["rlnTomoParticlesFile"], parse_as_string=names)["particles"]
record["particles"] = particles["rlnTomoParticleName"].tolist()
(call / "record.json").write_text(json.dumps(record))

out = Path(argv[argv.index("--o") + 1])
(out / "temp").mkdir(parents=True, exist_ok=True)
(out / "temp" / f"{{call.name}}.txt").write_text("per-tomogram evidence")
starfile.write({{"global": tomograms}}, out / "tomograms.star", overwrite=True)
"""


def _build_project(root):
    """Two zarr-backed tomograms from the synthetic tilt series, their particles, and non-zero trajectories."""
    (root / "tiltseries").mkdir(parents=True)
    stack = mrcfile.read(DATA / "tiltseries/TS_1.mrcs").astype(np.float32)
    tilts = starfile.read(DATA / "tiltseries/TS_1.star")
    base = starfile.read(DATA / "tomograms.star")
    particles_data = starfile.read(DATA / "particles.star")
    rows, particles, trajectories = [], [], {"general": {"rlnParticleNumber": 0}}
    for i, (name, kept) in enumerate(TOMOGRAMS.items()):
        zarr.save_array(str(root / f"{i}.zarr"), stack)
        tomo_tilts = tilts.iloc[kept].assign(**{TILTSERIES_URI_RELION_COLUMN: str((root / f"{i}.zarr").resolve())})
        starfile.write({name: tomo_tilts}, root / f"tiltseries/{name}.star")
        rows.append(base.assign(rlnTomoName=name, rlnTomoTiltSeriesStarFile=f"tiltseries/{name}.star"))
        tomo_particles = particles_data["particles"].iloc[i * 10 : i * 10 + 10]
        tomo_particles = tomo_particles.assign(
            rlnTomoName=name, rlnTomoParticleName=[f"{name}/{j:03d}" for j in range(1, 11)], rlnRandomSubset=[1, 2] * 5
        )
        particles.append(tomo_particles)
        for particle_name in tomo_particles["rlnTomoParticleName"]:
            shifts = np.full((len(tomo_tilts), 3), 1.5)
            trajectories[particle_name] = pd.DataFrame(
                shifts, columns=["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
            )
    trajectories["general"]["rlnParticleNumber"] = len(trajectories) - 1
    starfile.write({"global": pd.concat(rows)}, root / "tomograms.star")
    starfile.write({"optics": particles_data["optics"], "particles": pd.concat(particles)}, root / "particles.star")
    starfile.write(trajectories, root / "motion.star")
    for f in ["ref1.mrc", "ref2.mrc", "mask.mrc", "fsc.star"]:
        (root / f).touch()
    return stack


@pytest.mark.parametrize(
    "run_job, binary", [(run_ctf_refine, "relion_tomo_refine_ctf"), (run_polish, "relion_tomo_align")]
)
def test_relion_handoff(tmp_path, monkeypatch, run_job, binary):
    project, log, shm = tmp_path / "project", tmp_path / "calls", tmp_path / "shm"
    stack = _build_project(project)
    log.mkdir()
    shm.mkdir()
    fake = tmp_path / binary
    fake.write_text(FAKE_RELION.format(python=sys.executable, names=STAR_NAME_COLUMNS))
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_RELION_LOG", str(log))

    output_dir = tmp_path / "output"
    run_job(
        output_dir=output_dir,
        box_size=64,
        ref1=project / "ref1.mrc",
        ref2=project / "ref2.mrc",
        mask=project / "mask.mrc",
        fsc=project / "fsc.star",
        particles_starfile=project / "particles.star",
        tomograms_starfile=project / "tomograms.star",
        trajectories_starfile=project / "motion.star",
        tiltseries_relative_dir=project,
        relion_bin=str(fake),
        shm_dir=shm,
        n_workers=1,
    )

    calls = [json.loads((log / f"call_{i}" / "record.json").read_text()) for i in range(len(list(log.iterdir())))]
    assert len(calls) == len(TOMOGRAMS) + 1  # one per tomogram, then one collect
    for call in calls:
        argv = call["argv"]
        for flag, value in [("--mask", "mask.mrc"), ("--fsc", "fsc.star"), ("--b", "64")]:
            assert argv[argv.index(flag) + 1].endswith(value)
        assert call["optimisation_set"]["rlnTomoTrajectoriesFile"] == str((project / "motion.star").resolve())

    # phase 1: each tomogram alone, with its own particles and a materialized stack of exactly its tilts
    for i, ((name, kept), call) in enumerate(zip(TOMOGRAMS.items(), calls[:-1], strict=True)):
        assert "--only_do_unfinished" not in call["argv"]
        (tomogram,) = call["tomograms"]
        n_tilts = len(stack[kept])
        assert tomogram["name"] == name and not tomogram["stub"]
        assert tomogram["frame_count"] == tomogram["nz"] == len(tomogram["micrographs"]) == n_tilts
        assert [m.split("@")[0] for m in tomogram["micrographs"]] == [str(j) for j in range(1, n_tilts + 1)]
        np.testing.assert_array_equal(np.load(log / f"call_{i}" / "stack_0.npy"), stack[kept])
        assert call["particles"] == [f"{name}/{j:03d}" for j in range(1, 11)]

    # phase 2: every tomogram, header-only stubs sized to each tomogram's tilts
    collect = calls[-1]
    assert "--only_do_unfinished" in collect["argv"]
    assert [t["name"] for t in collect["tomograms"]] == list(TOMOGRAMS)
    for tomogram, kept in zip(collect["tomograms"], TOMOGRAMS.values(), strict=True):
        assert tomogram["stub"] and tomogram["frame_count"] == tomogram["nz"] == len(stack[kept])
    assert len(collect["particles"]) == 10 * len(TOMOGRAMS)

    # RELION's output is pointed back at the zarr tilt series, and nothing is left staged
    restored = starfile.read(output_dir / "tomograms.star", parse_as_string=STAR_NAME_COLUMNS)
    assert restored["rlnTomoName"].tolist() == list(TOMOGRAMS)
    assert restored[TILTSERIES_URI_RELION_COLUMN].tolist() == [
        str((project / f"{i}.zarr").resolve()) for i in range(len(TOMOGRAMS))
    ]
    assert not {"rlnTomoTiltSeriesName", "rlnTomoFrameCount"} & set(restored.columns)
    assert len(list((output_dir / "temp").iterdir())) == len(TOMOGRAMS) + 1
    assert not list(shm.iterdir())


def _fake_relion(tmp_path, monkeypatch, binary="relion_tomo_refine_ctf"):
    log = tmp_path / "calls"
    log.mkdir()
    fake = tmp_path / binary
    fake.write_text(FAKE_RELION.format(python=sys.executable, names=STAR_NAME_COLUMNS))
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_RELION_LOG", str(log))
    return fake, log


def _calls(log):
    return [json.loads((log / f"call_{i}" / "record.json").read_text()) for i in range(len(list(log.iterdir())))]


def _mrc_project(root):
    """The synthetic project as RELION holds it: relative ``N@tiltseries/TS_1.mrcs`` stacks, no tilt-series URI."""
    (root / "tiltseries").mkdir(parents=True)
    (root / "tiltseries" / "TS_1.mrcs").symlink_to((DATA / "tiltseries/TS_1.mrcs").resolve())
    shutil.copy(DATA / "tiltseries/TS_1.star", root / "tiltseries")
    shutil.copy(DATA / "tomograms.star", root)
    particles_data = starfile.read(DATA / "particles.star")
    particles = particles_data["particles"].iloc[:10]
    particles = particles.assign(rlnTomoParticleName=[f"session1_TS_1/{i}" for i in range(1, 11)])
    starfile.write({"optics": particles_data["optics"], "particles": particles}, root / "particles.star")
    for f in ["ref1.mrc", "ref2.mrc"]:
        (root / f).touch()


def _ctf_refine(project, output_dir, fake, shm, **kwargs):
    return run_ctf_refine(
        output_dir=output_dir,
        box_size=64,
        ref1=project / "ref1.mrc",
        ref2=project / "ref2.mrc",
        particles_starfile=project / "particles.star",
        tomograms_starfile=project / "tomograms.star",
        tiltseries_relative_dir=project,
        relion_bin=str(fake),
        shm_dir=shm,
        n_workers=1,
        **kwargs,
    )


def test_stacks_under_the_relative_dir_reach_relion_from_another_directory(tmp_path, monkeypatch):
    project, shm, elsewhere = tmp_path / "project", tmp_path / "shm", tmp_path / "elsewhere"
    _mrc_project(project)
    shm.mkdir()
    elsewhere.mkdir()
    fake, log = _fake_relion(tmp_path, monkeypatch)
    monkeypatch.chdir(elsewhere)
    _ctf_refine(project, tmp_path / "output", fake, shm)
    (call,) = _calls(log)
    assert call["particles"] == [f"session1_TS_1/{i}" for i in range(1, 11)]


def test_relion_args_reach_every_relion_call(tmp_path, monkeypatch):
    project, shm = tmp_path / "project", tmp_path / "shm"
    _build_project(project)
    shm.mkdir()
    fake, log = _fake_relion(tmp_path, monkeypatch)
    _ctf_refine(project, tmp_path / "output", fake, shm, relion_args="--extra_flag 3")
    calls = _calls(log)
    assert len(calls) == len(TOMOGRAMS) + 1
    for call in calls:
        assert call["argv"][call["argv"].index("--extra_flag") + 1] == "3"
    assert "--only_do_unfinished" in calls[-1]["argv"]


@pytest.mark.parametrize("keep_shm", [True, False])
def test_keep_shm_leaves_the_staged_tilt_series_after_exit(tmp_path, monkeypatch, keep_shm):
    project, shm = tmp_path / "project", tmp_path / "shm"
    _mrc_project(project)
    shm.mkdir()
    fake, _ = _fake_relion(tmp_path, monkeypatch)
    _ctf_refine(project, tmp_path / "output", fake, shm, keep_shm=keep_shm)
    _remove_staging_dirs()  # what exit and SIGTERM run
    kept = list(shm.glob("zpt-*"))
    if keep_shm:
        assert len(kept) == 1 and list(kept[0].rglob("*.mrc*"))
    else:
        assert kept == []
