"""
Phase 3 (CTF refinement) verification.

Two tiers:
- Always-on unit test: the zarr full-stack accessor + the zarr->MRC streamer round-trip exactly
  (float32), so the pixels we hand RELION equal the source.
- RELION-gated equivalence test: run stock relion_tomo_refine_ctf on the real MRC tilt series vs.
  our zarr->/dev/shm orchestration and assert the refined per-tilt CTF is identical. Skipped when
  relion_tomo_refine_ctf is not on PATH.
"""

import shutil
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile
import zarr

from zarr_particle_tools.core.data import DataReader, write_tiltseries_to_mrc
from zarr_particle_tools.subtomo_ctfrefine import run_ctf_refine

DATA = Path(__file__).parent / "data" / "relion_project_synthetic"
RELION_BIN = "relion_tomo_refine_ctf"


def test_read_full_stack_and_shm_writer_roundtrip(tmp_path):
    """read_full_stack and write_tiltseries_to_mrc preserve float32 pixels exactly."""
    rng = np.random.default_rng(0)
    stack = rng.standard_normal((7, 40, 48)).astype(np.float32)

    zpath = tmp_path / "ts.zarr"
    zarr.save_array(str(zpath), stack, chunks=(4, 16, 16))
    reader = DataReader(str(zpath))

    full = reader.read_full_stack()
    assert full.dtype == np.float32 and full.flags["C_CONTIGUOUS"]
    assert np.array_equal(full, stack)

    mrc_path = tmp_path / "ts.mrc"
    write_tiltseries_to_mrc(reader, mrc_path, voxel_size=1.5)
    with mrcfile.open(str(mrc_path)) as m:
        assert np.array_equal(m.data, stack)
        assert int(m.header.ispg) == 0 and int(m.header.mz) == 1  # image stack
        assert float(m.voxel_size.x) == pytest.approx(1.5)

    # MRC source round-trips identically too
    assert np.array_equal(DataReader(str(mrc_path)).read_full_stack(), stack)


def _refined_defocus(tiltseries_star: Path) -> pd.DataFrame:
    d = starfile.read(str(tiltseries_star))
    df = d if isinstance(d, pd.DataFrame) else next(iter(d.values()))
    return df[["rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle"]].astype(float)


@pytest.mark.skipif(shutil.which(RELION_BIN) is None, reason="relion_tomo_refine_ctf not on PATH")
def test_ctfrefine_zarr_matches_stock_relion(tmp_path):
    """zarr->/dev/shm CTF refinement == stock RELION on the real MRC (identical pixels)."""
    ref = DATA / "Reconstruct" / "relion_output_baseline" / "merged.mrc"
    box = 64

    # shared particles.star with half-sets (refine_ctf needs rlnRandomSubset)
    parts = starfile.read(str(DATA / "particles.star"))
    parts["particles"]["rlnRandomSubset"] = [(i % 2) + 1 for i in range(len(parts["particles"]))]

    with mrcfile.open(str(DATA / "tiltseries" / "TS_1.mrcs"), permissive=True) as m:
        nz = int(m.data.shape[0])
        stack = np.asarray(m.data, dtype=np.float32)

    global_row = {
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

    # --- baseline: stock RELION reading the real MRC (option A) ---
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
                        **global_row,
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
    import subprocess

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
            "--do_defocus",
            "--o",
            "ctfout/",
            "--j",
            "4",
        ],
        check=True,
        cwd=str(base),
    )

    # --- ours: zarr -> /dev/shm via run_ctf_refine ---
    ours = tmp_path / "ours"
    (ours / "tiltseries").mkdir(parents=True)
    zpath = ours / "TS_1.zarr"
    zarr.save_array(str(zpath), stack, chunks=(16, 256, 256))
    indiv = starfile.read(str(DATA / "tiltseries" / "TS_1.star"))
    indiv = indiv if isinstance(indiv, pd.DataFrame) else next(iter(indiv.values()))
    indiv = indiv.copy()
    indiv["tomoTiltSeriesURI"] = str(zpath.resolve())
    starfile.write({"session1_TS_1": indiv}, str(ours / "tiltseries" / "TS_1.star"))
    starfile.write({"optics": parts["optics"], "particles": parts["particles"]}, str(ours / "particles.star"))
    starfile.write(
        {"global": pd.DataFrame([{**global_row, "rlnTomoTiltSeriesStarFile": "tiltseries/TS_1.star"}])},
        str(ours / "tomograms.star"),
    )
    run_ctf_refine(
        output_dir=ours / "ctfout",
        box_size=box,
        ref1=ref,
        ref2=ref,
        particles_starfile=ours / "particles.star",
        tomograms_starfile=ours / "tomograms.star",
        do_defocus=True,
        threads=4,
    )

    b = _refined_defocus(base / "ctfout" / "tiltseries" / "TS_1.star")
    o = _refined_defocus(ours / "ctfout" / "tiltseries" / "session1_TS_1.star")
    assert np.array_equal(b.to_numpy(), o.to_numpy()), "zarr-fed CTF refinement differs from stock RELION"


def test_ctfrefine_errors_on_tomo_name_mismatch(tmp_path):
    """Hard error (before RELION) if particles reference no tomogram in the tomograms.star — the
    'runs right after tomograms.star generation' safety for the portal/copick helper."""
    ref = DATA / "Reconstruct" / "relion_output_baseline" / "merged.mrc"
    parts, tomos = _build_multi_tomo_zarr_case(tmp_path / "proj", n=2)
    p = starfile.read(str(parts))
    p["particles"]["rlnTomoName"] = "OTHER_" + p["particles"]["rlnTomoName"].astype(str)  # no overlap
    starfile.write(p, str(parts))
    with pytest.raises(ValueError, match="No rlnTomoName overlap"):
        run_ctf_refine(
            output_dir=tmp_path / "o",
            box_size=64,
            ref1=ref,
            ref2=ref,
            particles_starfile=parts,
            tomograms_starfile=tomos,
            do_defocus=True,
            threads=4,
        )


def _build_multi_tomo_zarr_case(dest: Path, n: int):
    """Build an n-tomogram synthetic project (TS_1 duplicated) with zarr tilt series."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "tiltseries").mkdir(exist_ok=True)
    with mrcfile.open(str(DATA / "tiltseries" / "TS_1.mrcs"), permissive=True) as m:
        stack = np.asarray(m.data, dtype=np.float32)
    parts = starfile.read(str(DATA / "particles.star"))
    base_p = parts["particles"].copy()
    base_p["rlnRandomSubset"] = [(i % 2) + 1 for i in range(len(base_p))]
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
        p = base_p.copy()
        p["rlnTomoName"] = name
        all_p.append(p)
    starfile.write({"global": pd.DataFrame(rows)}, str(dest / "tomograms.star"))
    starfile.write(
        {"optics": parts["optics"], "particles": pd.concat(all_p, ignore_index=True)}, str(dest / "particles.star")
    )
    return dest / "particles.star", dest / "tomograms.star"


@pytest.mark.skipif(shutil.which(RELION_BIN) is None, reason="relion_tomo_refine_ctf not on PATH")
def test_ctfrefine_per_tomogram_matches_all_at_once(tmp_path):
    """Per-tomogram mode (1 tilt series in RAM + merge) == all-at-once, for defocus (exact split)."""
    ref = DATA / "Reconstruct" / "relion_output_baseline" / "merged.mrc"
    parts, tomos = _build_multi_tomo_zarr_case(tmp_path / "proj", n=2)
    common = dict(
        box_size=64, ref1=ref, ref2=ref, particles_starfile=parts, tomograms_starfile=tomos, do_defocus=True, threads=4
    )

    # n_workers=2 exercises the spawn multiprocessing.Pool path deterministically (2 tomograms)
    run_ctf_refine(output_dir=tmp_path / "pt", per_tomogram=True, n_workers=2, **common)
    run_ctf_refine(output_dir=tmp_path / "aa", per_tomogram=False, **common)

    for name in ("session1_TS_0", "session1_TS_1"):
        pt = _refined_defocus(tmp_path / "pt" / "tiltseries" / f"{name}.star")
        aa = _refined_defocus(tmp_path / "aa" / "tiltseries" / f"{name}.star")
        assert np.array_equal(pt.to_numpy(), aa.to_numpy()), f"per-tomogram != all-at-once for {name}"


@pytest.mark.skipif(shutil.which(RELION_BIN) is None, reason="relion_tomo_refine_ctf not on PATH")
@pytest.mark.parametrize(
    "flags",
    [
        {"do_defocus": True, "do_scale": True, "per_frame_scale": True},  # per-frame scale (independent)
        {"do_defocus": True, "do_even_aberrations": True, "do_odd_aberrations": True},  # joint aberrations
    ],
    ids=["per_frame_scale", "aberrations"],
)
def test_ctfrefine_two_phase_flag_variants(tmp_path, flags):
    """Two-phase == all-at-once for per-frame scale and for joint even/odd aberrations."""
    ref = DATA / "Reconstruct" / "relion_output_baseline" / "merged.mrc"
    parts, tomos = _build_multi_tomo_zarr_case(tmp_path / "proj", n=2)
    common = dict(
        box_size=64, ref1=ref, ref2=ref, particles_starfile=parts, tomograms_starfile=tomos, threads=4, **flags
    )
    run_ctf_refine(output_dir=tmp_path / "pt", per_tomogram=True, **common)
    run_ctf_refine(output_dir=tmp_path / "aa", per_tomogram=False, **common)

    for name in ("session1_TS_0", "session1_TS_1"):
        pt = _refined_defocus(tmp_path / "pt" / "tiltseries" / f"{name}.star")
        aa = _refined_defocus(tmp_path / "aa" / "tiltseries" / f"{name}.star")
        assert np.array_equal(pt.to_numpy(), aa.to_numpy()), f"defocus differs for {name}"
    # aberrations land in particles.star (per optics group) -> compare those columns if present
    if flags.get("do_even_aberrations"):
        pp = starfile.read(str(tmp_path / "pt" / "particles.star"))
        ap = starfile.read(str(tmp_path / "aa" / "particles.star"))
        for tbl in ("optics", "particles"):
            for c in pp.get(tbl, pd.DataFrame()).columns:
                if "Zernike" in c:
                    assert pp[tbl][c].astype(str).tolist() == ap[tbl][c].astype(str).tolist(), f"{tbl}:{c} differs"


@pytest.mark.skipif(shutil.which(RELION_BIN) is None, reason="relion_tomo_refine_ctf not on PATH")
def test_ctfrefine_zero_particle_tomogram(tmp_path):
    """A tomogram with no particles is skipped gracefully (RELION continues; two-phase must too)."""
    ref = DATA / "Reconstruct" / "relion_output_baseline" / "merged.mrc"
    parts, tomos = _build_multi_tomo_zarr_case(tmp_path / "proj", n=2)
    # drop all particles for TS_1 -> 0-particle tomogram
    p = starfile.read(str(parts))
    p["particles"] = p["particles"][p["particles"]["rlnTomoName"] == "session1_TS_0"]
    starfile.write(p, str(parts))
    run_ctf_refine(
        output_dir=tmp_path / "pt",
        per_tomogram=True,
        box_size=64,
        ref1=ref,
        ref2=ref,
        particles_starfile=parts,
        tomograms_starfile=tomos,
        do_defocus=True,
        threads=4,
    )
    assert (tmp_path / "pt" / "tiltseries" / "session1_TS_0.star").exists()  # the populated one refined


@pytest.mark.skipif(shutil.which(RELION_BIN) is None, reason="relion_tomo_refine_ctf not on PATH")
def test_ctfrefine_two_phase_joint_scale_matches_all_at_once(tmp_path):
    """Two-phase collect (header stubs + --only_do_unfinished) == all-at-once for a global-scale
    (joint) fit -- exercises RELION's fitGlobalScale finalise without any stack in the collect pass."""
    ref = DATA / "Reconstruct" / "relion_output_baseline" / "merged.mrc"
    parts, tomos = _build_multi_tomo_zarr_case(tmp_path / "proj", n=2)
    common = dict(
        box_size=64,
        ref1=ref,
        ref2=ref,
        particles_starfile=parts,
        tomograms_starfile=tomos,
        do_defocus=True,
        do_scale=True,
        threads=4,
    )  # global Lambert

    run_ctf_refine(output_dir=tmp_path / "pt", per_tomogram=True, **common)
    run_ctf_refine(output_dir=tmp_path / "aa", per_tomogram=False, **common)

    for name in ("session1_TS_0", "session1_TS_1"):
        pt = _refined_defocus(tmp_path / "pt" / "tiltseries" / f"{name}.star")
        aa = _refined_defocus(tmp_path / "aa" / "tiltseries" / f"{name}.star")
        assert np.array_equal(pt.to_numpy(), aa.to_numpy()), f"two-phase joint != all-at-once for {name}"
