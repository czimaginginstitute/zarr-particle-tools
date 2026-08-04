"""
Phase 4 (Bayesian polish / frame alignment, relion_tomo_align) verification.

RELION-gated equivalence tests (skipped when relion_tomo_align not on PATH):
- zarr -> /dev/shm path == stock RELION reading the real MRC (identical pixels);
- two-phase per-tomogram == all-at-once (align is per-tomogram independent).
Uses --no-motion (standard alignment) for robustness on the tiny synthetic data; equivalence holds
regardless of convergence because both paths feed identical inputs to the same RELION binary.
"""

import shutil
import subprocess
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile
import zarr

from zarr_particle_tools.subtomo_ctfrefine import run_ctf_refine
from zarr_particle_tools.subtomo_polish import run_polish

DATA = Path(__file__).parent / "data" / "relion_project_synthetic"
RELION_BIN = "relion_tomo_align"
# these jobs materialize tilt series into a RAM-backed /dev/shm, which does not exist on macOS
HAS_SHM = Path("/dev/shm").exists()
PROJ_COLS = ["rlnTomoXTilt", "rlnTomoYTilt", "rlnTomoZRot", "rlnTomoXShiftAngst", "rlnTomoYShiftAngst"]


def _proj(tiltseries_star: Path) -> np.ndarray:
    d = starfile.read(str(tiltseries_star))
    df = d if isinstance(d, pd.DataFrame) else next(iter(d.values()))
    return df[PROJ_COLS].astype(float).to_numpy()


def _particles_with_subset():
    parts = starfile.read(str(DATA / "particles.star"))
    parts["particles"]["rlnRandomSubset"] = [(i % 2) + 1 for i in range(len(parts["particles"]))]
    return parts


def _build_multi_tomo_zarr_case(dest: Path, n: int):
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "tiltseries").mkdir(exist_ok=True)
    with mrcfile.open(str(DATA / "tiltseries" / "TS_1.mrcs"), permissive=True) as m:
        stack = np.asarray(m.data, dtype=np.float32)
    parts = _particles_with_subset()
    indiv0 = starfile.read(str(DATA / "tiltseries" / "TS_1.star"))
    indiv0 = indiv0 if isinstance(indiv0, pd.DataFrame) else next(iter(indiv0.values()))
    rows, all_p = [], []
    for i in range(n):
        name = f"session1_TS_{i}"
        zpath = dest / f"TS_{i}.zarr"
        zarr.save_array(str(zpath), stack, chunks=(16, 256, 256))
        indiv = indiv0.copy()
        indiv["tomoTiltSeriesURI"] = str(zpath.resolve())
        starfile.write({name: indiv}, str(dest / "tiltseries" / f"{name}.star"))
        rows.append(
            {
                "rlnTomoName": name,
                "rlnVoltage": 300.0,
                "rlnSphericalAberration": 2.7,
                "rlnAmplitudeContrast": 0.07,
                "rlnMicrographOriginalPixelSize": 10.0,
                "rlnTomoHand": -1,
                "rlnOpticsGroupName": "polnet",
                "rlnTomoTiltSeriesPixelSize": 10.0,
                "rlnTomoTiltSeriesStarFile": f"tiltseries/{name}.star",
                "rlnTomoSizeX": 630,
                "rlnTomoSizeY": 630,
                "rlnTomoSizeZ": 200,
            }
        )
        p = parts["particles"].copy()
        p["rlnTomoName"] = name
        all_p.append(p)
    starfile.write({"global": pd.DataFrame(rows)}, str(dest / "tomograms.star"))
    starfile.write(
        {"optics": parts["optics"], "particles": pd.concat(all_p, ignore_index=True)}, str(dest / "particles.star")
    )
    return dest / "particles.star", dest / "tomograms.star"


@pytest.mark.skipif(
    shutil.which(RELION_BIN) is None or not HAS_SHM, reason="relion_tomo_align not on PATH or /dev/shm unavailable"
)
def test_polish_zarr_matches_stock_relion(tmp_path):
    ref = DATA / "Reconstruct" / "relion_output_baseline" / "merged.mrc"
    box = 64
    parts = _particles_with_subset()
    with mrcfile.open(str(DATA / "tiltseries" / "TS_1.mrcs"), permissive=True) as m:
        nz, stack = int(m.data.shape[0]), np.asarray(m.data, dtype=np.float32)
    grow = {
        "rlnTomoName": "session1_TS_1",
        "rlnVoltage": 300.0,
        "rlnSphericalAberration": 2.7,
        "rlnAmplitudeContrast": 0.07,
        "rlnMicrographOriginalPixelSize": 10.0,
        "rlnTomoHand": -1,
        "rlnOpticsGroupName": "polnet",
        "rlnTomoTiltSeriesPixelSize": 10.0,
        "rlnTomoSizeX": 630,
        "rlnTomoSizeY": 630,
        "rlnTomoSizeZ": 200,
    }

    # baseline: stock relion_tomo_align reading the real MRC (option A)
    base = tmp_path / "baseline"
    (base / "tiltseries").mkdir(parents=True)
    shutil.copy(DATA / "tiltseries" / "TS_1.mrcs", base / "tiltseries" / "TS_1.mrcs")
    shutil.copy(DATA / "tiltseries" / "TS_1.star", base / "tiltseries" / "TS_1.star")
    starfile.write({"optics": parts["optics"], "particles": parts["particles"]}, str(base / "particles.star"))
    starfile.write(
        {
            "global": pd.DataFrame(
                [
                    {
                        **grow,
                        "rlnTomoTiltSeriesName": "tiltseries/TS_1.mrcs",
                        "rlnTomoFrameCount": nz,
                        "rlnTomoTiltSeriesStarFile": "tiltseries/TS_1.star",
                    }
                ]
            )
        },
        str(base / "tomograms.star"),
    )
    starfile.write(
        {
            "optimisation_set": pd.DataFrame(
                [{"rlnTomoParticlesFile": "particles.star", "rlnTomoTomogramsFile": "tomograms.star"}]
            )
        },
        str(base / "optimisation_set.star"),
    )
    subprocess.run(
        [
            RELION_BIN,
            "--i",
            "optimisation_set.star",
            "--ref1",
            str(ref.resolve()),
            "--ref2",
            str(ref.resolve()),
            "--b",
            str(box),
            "--r",
            "8",
            "--o",
            "out/",
            "--j",
            "4",
        ],
        check=True,
        cwd=str(base),
    )

    # ours: zarr -> /dev/shm
    ours = tmp_path / "ours"
    (ours / "tiltseries").mkdir(parents=True)
    zpath = ours / "TS_1.zarr"
    zarr.save_array(str(zpath), stack, chunks=(16, 256, 256))
    indiv = starfile.read(str(DATA / "tiltseries" / "TS_1.star"))
    indiv = (indiv if isinstance(indiv, pd.DataFrame) else next(iter(indiv.values()))).copy()
    indiv["tomoTiltSeriesURI"] = str(zpath.resolve())
    starfile.write({"session1_TS_1": indiv}, str(ours / "tiltseries" / "TS_1.star"))
    starfile.write({"optics": parts["optics"], "particles": parts["particles"]}, str(ours / "particles.star"))
    starfile.write(
        {"global": pd.DataFrame([{**grow, "rlnTomoTiltSeriesStarFile": "tiltseries/TS_1.star"}])},
        str(ours / "tomograms.star"),
    )
    run_polish(
        output_dir=ours / "out",
        box_size=box,
        ref1=ref,
        ref2=ref,
        particles_starfile=ours / "particles.star",
        tomograms_starfile=ours / "tomograms.star",
        do_motion=False,
        align_range=8,
        threads=4,
    )

    b = _proj(base / "out" / "tiltseries" / "TS_1.star")
    o = _proj(ours / "out" / "tiltseries" / "session1_TS_1.star")
    assert np.array_equal(b, o), "zarr-fed polish differs from stock RELION"


@pytest.mark.skipif(
    shutil.which(RELION_BIN) is None or not HAS_SHM, reason="relion_tomo_align not on PATH or /dev/shm unavailable"
)
def test_polish_two_phase_matches_all_at_once(tmp_path):
    ref = DATA / "Reconstruct" / "relion_output_baseline" / "merged.mrc"
    parts, tomos = _build_multi_tomo_zarr_case(tmp_path / "proj", n=2)
    common = dict(
        box_size=64,
        ref1=ref,
        ref2=ref,
        particles_starfile=parts,
        tomograms_starfile=tomos,
        do_motion=False,
        align_range=8,
        threads=4,
    )
    run_polish(output_dir=tmp_path / "pt", per_tomogram=True, **common)
    run_polish(output_dir=tmp_path / "aa", per_tomogram=False, **common)
    for name in ("session1_TS_0", "session1_TS_1"):
        pt = _proj(tmp_path / "pt" / "tiltseries" / f"{name}.star")
        aa = _proj(tmp_path / "aa" / "tiltseries" / f"{name}.star")
        assert np.array_equal(pt, aa), f"per-tomogram != all-at-once for {name}"


@pytest.mark.skipif(
    shutil.which(RELION_BIN) is None or not HAS_SHM, reason="relion_tomo_align not on PATH or /dev/shm unavailable"
)
def test_polish_two_phase_motion_matches_all_at_once(tmp_path):
    """Two-phase == all-at-once with --motion (the Bayesian-polish path); motion.star is produced."""
    ref = DATA / "Reconstruct" / "relion_output_baseline" / "merged.mrc"
    parts, tomos = _build_multi_tomo_zarr_case(tmp_path / "proj", n=2)
    common = dict(
        box_size=64,
        ref1=ref,
        ref2=ref,
        particles_starfile=parts,
        tomograms_starfile=tomos,
        do_motion=True,
        s_vel=0.2,
        s_div=5000.0,
        align_range=8,
        threads=4,
    )
    run_polish(output_dir=tmp_path / "pt", per_tomogram=True, **common)
    run_polish(output_dir=tmp_path / "aa", per_tomogram=False, **common)
    assert (tmp_path / "pt" / "motion.star").exists() and (tmp_path / "aa" / "motion.star").exists()
    for name in ("session1_TS_0", "session1_TS_1"):
        pt = _proj(tmp_path / "pt" / "tiltseries" / f"{name}.star")
        aa = _proj(tmp_path / "aa" / "tiltseries" / f"{name}.star")
        assert np.array_equal(pt, aa), f"motion per-tomogram != all-at-once for {name}"


@pytest.mark.skipif(
    shutil.which("relion_tomo_refine_ctf") is None or not HAS_SHM,
    reason="relion binaries not on PATH or /dev/shm unavailable",
)
def test_polish_output_chains_into_ctfrefine_on_zarr(tmp_path):
    """finding-1 fix: polish output tomograms.star keeps the zarr locator, so a following CTF-refine
    re-materializes from zarr (rather than the deleted shm / dropped tomoTiltSeriesURI)."""
    ref = DATA / "Reconstruct" / "relion_output_baseline" / "merged.mrc"
    parts, tomos = _build_multi_tomo_zarr_case(tmp_path / "proj", n=2)
    run_polish(
        output_dir=tmp_path / "polish",
        box_size=64,
        ref1=ref,
        ref2=ref,
        particles_starfile=parts,
        tomograms_starfile=tomos,
        do_motion=False,
        align_range=8,
        threads=4,
    )
    # chain: CTF-refine consuming the polish output (its tomograms.star carries tomoTiltSeriesURI)
    run_ctf_refine(
        output_dir=tmp_path / "ctf",
        box_size=64,
        ref1=ref,
        ref2=ref,
        particles_starfile=tmp_path / "polish" / "particles.star",
        tomograms_starfile=tmp_path / "polish" / "tomograms.star",
        do_defocus=True,
        threads=4,
    )
    for name in ("session1_TS_0", "session1_TS_1"):
        assert (tmp_path / "ctf" / "tiltseries" / f"{name}.star").exists(), f"chaining lost {name}"
