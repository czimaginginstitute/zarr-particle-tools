"""
Shared harness for running RELION tomography jobs (CTF refinement, Bayesian polish) against tilt
series stored as OME-Zarr, without modifying RELION.

Strategy (docs/audit/phase3_4_ctfrefine_polish_design.md): reuse the stock RELION binary and replace
only the pixel source. Per tomogram we stream the zarr tilt series into a RAM-backed MRC (/dev/shm),
point rlnTomoTiltSeriesName at it, and let RELION read its own format. A two-phase mode keeps at most
n_workers tilt series in RAM: phase 1 processes each tomogram alone (parallel pool, writes RELION temp
evidence); phase 2 runs RELION's own joint finalise via --only_do_unfinished + 1 KB header stubs, so
results match all-at-once for every fit type.

Job-specific behaviour is injected as a `build_cmd(relion_bin, opt_set, output_dir, box_size, ref1,
ref2, mask, fsc, threads, opts) -> list[str]` callable plus the binary name; everything else here is
shared between the CTF-refine and polish jobs.
"""

import atexit
import hashlib
import logging
import multiprocessing as mp
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional, Union

import mrcfile.dtypes as mrc_dtypes
import numpy as np
import pandas as pd
import starfile

from zarr_particle_tools.core.constants import TILTSERIES_URI_RELION_COLUMN
from zarr_particle_tools.core.data import get_tiltseries_datareader, write_tiltseries_to_mrc

logger = logging.getLogger(__name__)


def _indiv_reader(row, src_base, tiltseries_relative_dir):
    """Load a tomogram's individual tilt star + a DataReader, propagating a global tomoTiltSeriesURI
    (present on our chained outputs) into the individual df so the zarr locator resolves."""
    ip = Path(str(row["rlnTomoTiltSeriesStarFile"]))
    ip = ip if ip.is_absolute() else src_base / ip
    indiv = read_single_table(ip)
    if TILTSERIES_URI_RELION_COLUMN in row and TILTSERIES_URI_RELION_COLUMN not in indiv.columns:
        indiv = indiv.copy()
        indiv[TILTSERIES_URI_RELION_COLUMN] = row[TILTSERIES_URI_RELION_COLUMN]
    return indiv, get_tiltseries_datareader(indiv, tiltseries_relative_dir or Path("./"))


# Track materialized shm MRCs so a graceful exit/SIGTERM cleans them (SIGKILL is uncatchable).
_ACTIVE_SHM: set = set()


def _cleanup_shm(*_):
    for p in list(_ACTIVE_SHM):
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass
        _ACTIVE_SHM.discard(p)


atexit.register(_cleanup_shm)
if threading.current_thread() is threading.main_thread():  # signal handlers only in the main thread
    try:
        _prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _on_sigterm(signum, frame):
            _cleanup_shm()
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signum)  # re-raise default termination

        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):
        pass


def _fs_type(path: Path) -> str:
    """Filesystem type of the mount containing `path` (longest matching /proc/mounts entry)."""
    rp = os.path.realpath(str(path))
    best_mp, best_fs = "", "?"
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mp_, fs = parts[1], parts[2]
                if (rp == mp_ or rp.startswith(mp_.rstrip("/") + "/")) and len(mp_) >= len(best_mp):
                    best_mp, best_fs = mp_, fs
    except OSError:
        pass
    return best_fs


def _peek_stack_bytes(global_df, src_base, tiltseries_relative_dir) -> list:
    """Per-tomogram tilt-series size in bytes (float32), from lazy zarr/MRC metadata (no download)."""
    sizes = []
    for _, row in global_df.iterrows():
        _, reader = _indiv_reader(row, src_base, tiltseries_relative_dir)
        sizes.append(int(np.prod(reader.data.shape)) * 4)
    return sizes


def _preflight_budget(shm_dir: Path, stack_bytes: list, concurrency: int, all_at_once: bool) -> None:
    """
    Upfront RAM check for the whole run: warn if shm_dir isn't tmpfs; estimate PEAK shm usage
    (sum of all stacks for all-at-once, else `concurrency` largest concurrent stacks) and error if it
    won't fit / warn if it exceeds 50% of free. Avoids the per-worker race where each checks one stack.
    """
    fs = _fs_type(shm_dir)
    if fs not in ("tmpfs", "ramfs"):
        logger.warning(
            "shm-dir %s is on '%s', not tmpfs/ramfs: the tilt series will hit PHYSICAL DISK "
            "(slow, and violates the RAM-only guarantee). Use /dev/shm or another tmpfs.",
            shm_dir,
            fs,
        )
    if not stack_bytes:
        return
    peak = sum(stack_bytes) if all_at_once else min(concurrency, len(stack_bytes)) * max(stack_bytes)
    free = shutil.disk_usage(shm_dir).free
    mode = "all-at-once (sum of all tilt series)" if all_at_once else f"{concurrency} concurrent worker(s)"
    if peak > free:
        raise RuntimeError(
            f"Not enough space on {shm_dir}: {mode} needs ~{peak/1e9:.1f} GB, {free/1e9:.1f} GB free. "
            f"Lower --n-workers, use per-tomogram mode, or a larger tmpfs."
        )
    if peak > 0.5 * free:
        logger.warning(
            "Estimated peak shm usage (%.1f GB, %s) exceeds 50%% of free space on %s (%.1f GB free).",
            peak / 1e9,
            mode,
            shm_dir,
            free / 1e9,
        )


def _slug(name: str, max_len: int = 200) -> str:
    """Filesystem-safe form of a tomogram name; hash-suffix only if pathologically long."""
    s = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if len(s) > max_len:
        s = s[:max_len] + "_" + hashlib.sha1(name.encode()).hexdigest()[:8]
    return s


def read_single_table(path: Union[str, Path]) -> pd.DataFrame:
    """Read a star file that holds a single (optionally named) data block as a DataFrame."""
    data = starfile.read(str(path))
    if isinstance(data, dict):
        if len(data) != 1:
            raise ValueError(f"Expected a single data block in {path}, found {list(data)}.")
        return next(iter(data.values()))
    return data


def resolve_optimisation_set(optimisation_set_starfile: Path) -> tuple[Path, Path, Optional[Path]]:
    """Return (particles, tomograms, trajectories) paths from an optimisation_set.star."""
    opt = read_single_table(optimisation_set_starfile)
    base = Path(optimisation_set_starfile).parent
    row = opt.iloc[0] if isinstance(opt, pd.DataFrame) else opt

    def _resolve(val):
        if val is None or isinstance(val, float):
            return None
        p = Path(str(val))
        return p if p.is_absolute() else (base / p)

    particles = _resolve(row.get("rlnTomoParticlesFile"))
    tomograms = _resolve(row.get("rlnTomoTomogramsFile"))
    trajectories = _resolve(row.get("rlnTomoTrajectoriesFile")) if "rlnTomoTrajectoriesFile" in row else None
    if particles is None or tomograms is None:
        raise ValueError(f"{optimisation_set_starfile} is missing rlnTomoParticlesFile/rlnTomoTomogramsFile.")
    return particles, tomograms, trajectories


def read_global_tomograms(tomograms_starfile: Path) -> tuple[pd.DataFrame, Path]:
    """Return (global tomograms DataFrame, base dir for resolving rlnTomoTiltSeriesStarFile)."""
    data = starfile.read(str(tomograms_starfile))
    global_df = data["global"] if isinstance(data, dict) else data
    if "rlnTomoTiltSeriesStarFile" not in global_df.columns:
        raise ValueError(f"{tomograms_starfile} has no rlnTomoTiltSeriesStarFile column.")
    return global_df.copy(), Path(tomograms_starfile).parent


def _materialize_tiltseries(global_df, src_base, stage_dir, shm_dir, tiltseries_relative_dir):
    """
    Write a tomograms.star into stage_dir with rlnTomoTiltSeriesName repointed at a /dev/shm MRC
    streamed from each tomogram's zarr tilt series (RELION option-A stack load). stage_dir must be
    separate from the RELION output dir so its tiltseries/*.star don't collide with RELION's outputs.
    Pass a 1-row global_df for a single tomogram. Returns (patched_tomograms_star, [shm_paths]).
    """
    global_df = global_df.copy()
    ts_out_dir = stage_dir / "tiltseries"
    ts_out_dir.mkdir(parents=True, exist_ok=True)
    shm_dir.mkdir(parents=True, exist_ok=True)
    run_tag = uuid.uuid4().hex[:8]

    shm_paths, names, frame_counts, indiv_rel = [], [], [], []
    for _, row in global_df.iterrows():
        tomo_name = str(row["rlnTomoName"])
        indiv_df, reader = _indiv_reader(row, src_base, tiltseries_relative_dir)
        indiv_out = ts_out_dir / f"{tomo_name}.star"
        starfile.write({tomo_name: indiv_df}, str(indiv_out), overwrite=True)

        if len(reader.data.shape) != 3:
            raise ValueError(f"Tomogram {tomo_name}: tilt series must be 3-D (section,y,x), got {reader.data.shape}.")
        nz, ny, nx = (int(x) for x in reader.data.shape)
        if nz != len(indiv_df):
            raise ValueError(f"Tomogram {tomo_name}: stack sections ({nz}) != tilt rows ({len(indiv_df)}).")
        voxel = float(row["rlnTomoTiltSeriesPixelSize"]) if "rlnTomoTiltSeriesPixelSize" in row else None
        shm_path = shm_dir / f"zpt_{run_tag}_{_slug(tomo_name)}.mrc"
        write_tiltseries_to_mrc(reader, shm_path, voxel_size=voxel)
        _ACTIVE_SHM.add(str(shm_path))
        shm_paths.append(shm_path)
        names.append(str(shm_path))
        frame_counts.append(nz)
        indiv_rel.append(f"tiltseries/{tomo_name}.star")  # relative to CWD (=stage_dir); RELION prepends --o on write
        logger.info(f"Materialized {tomo_name}: {reader.locator} -> {shm_path} ({nz} tilts).")

    global_df["rlnTomoTiltSeriesName"] = names
    global_df["rlnTomoFrameCount"] = frame_counts
    global_df["rlnTomoTiltSeriesStarFile"] = indiv_rel
    patched = stage_dir / "tomograms.star"
    starfile.write({"global": global_df}, str(patched), overwrite=True)
    return patched, shm_paths


def _write_optimisation_set(stage_dir, particles, tomograms, trajectories):
    """Write an optimisation_set.star with absolute particle/tomogram (+ trajectory) paths."""
    row = {
        "rlnTomoParticlesFile": str(Path(particles).resolve()),
        "rlnTomoTomogramsFile": str(Path(tomograms).resolve()),
    }
    if trajectories is not None:
        row["rlnTomoTrajectoriesFile"] = str(Path(trajectories).resolve())
    path = stage_dir / "optimisation_set.star"
    starfile.write({"optimisation_set": pd.DataFrame([row])}, str(path), overwrite=True)
    return path


def _write_header_stub(path, nx, ny, nz, pixel=1.0):
    """1 KB MRC header (mode 2), correct dims, no data body -- satisfies RELION's header-only getSize()."""
    h = np.zeros(1, dtype=mrc_dtypes.HEADER_DTYPE)
    h["nx"], h["ny"], h["nz"] = nx, ny, nz
    h["mode"] = 2
    h["mx"], h["my"], h["mz"] = nx, ny, 1
    h["cella"]["x"], h["cella"]["y"], h["cella"]["z"] = nx * pixel, ny * pixel, nz * pixel
    h["mapc"], h["mapr"], h["maps"] = 1, 2, 3
    h["map"] = b"MAP "
    h["machst"] = np.frombuffer(b"\x44\x44\x00\x00", dtype="u1")
    h["nsymbt"] = 0
    Path(path).write_bytes(h.tobytes())


def _run_relion_job(
    build_cmd,
    relion_bin,
    global_df,
    src_base,
    particles_star,
    trajectories,
    output_dir,
    shm_dir,
    tiltseries_relative_dir,
    box_size,
    ref1,
    ref2,
    mask,
    fsc,
    threads,
    keep_shm,
    opts,
    extra_args=(),
) -> Path:
    """One RELION invocation over the given tomograms into output_dir (materialize + stage + run)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = output_dir / "relion_input"
    stage_dir.mkdir(parents=True, exist_ok=True)
    patched, shm_paths = _materialize_tiltseries(global_df, src_base, stage_dir, Path(shm_dir), tiltseries_relative_dir)
    opt_set = _write_optimisation_set(stage_dir, Path(particles_star), patched, trajectories)
    cmd = build_cmd(relion_bin, opt_set, output_dir, box_size, ref1, ref2, mask, fsc, threads, opts) + list(extra_args)
    logger.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(stage_dir))  # CWD=stage_dir: relative tilt stars resolve
    finally:
        if not keep_shm:
            for p in shm_paths:
                p.unlink(missing_ok=True)
                _ACTIVE_SHM.discard(str(p))
    return output_dir


def _phase1_worker(task) -> str:
    """Run one-tomogram processing (writes RELION temp evidence). Module-level for mp.Pool."""
    build_cmd, relion_bin, row_df, particles_star, trajectories, output_dir, common = task
    _run_relion_job(
        build_cmd,
        relion_bin,
        row_df,
        particles_star=particles_star,
        trajectories=trajectories,
        output_dir=output_dir,
        **common,
    )
    return str(output_dir)


def _two_phase(
    build_cmd,
    relion_bin,
    global_df,
    src_base,
    particles_star,
    trajectories,
    output_dir,
    n_workers,
    box_size,
    ref1,
    ref2,
    mask,
    fsc,
    threads,
    opts,
    common,
) -> Path:
    """
    Memory-bounded per-tomogram run (<= n_workers tilt series in RAM): phase 1 processes each
    tomogram alone (parallel pool, writes temp); phase 2 gathers temp and runs one --only_do_unfinished
    collect with 1 KB header stubs, so RELION's own finalise (defocus/scale/aberrations/motion) runs
    with no stack loaded. Matches all-at-once for every fit type.
    """
    names = [str(r["rlnTomoName"]) for _, r in global_df.iterrows()]
    parts = starfile.read(str(particles_star))
    parts = parts if isinstance(parts, dict) else {"particles": parts}
    pdf = parts["particles"]

    p1_root = output_dir / "_phase1"
    tasks, processed = [], []
    for name in names:
        sub_particles = pdf[pdf["rlnTomoName"] == name]
        if len(sub_particles) == 0:  # RELION skips 0-particle tomograms; don't launch a run for one
            logger.info("Skipping tomogram %s in phase 1: no particles.", name)
            continue
        d = p1_root / _slug(name)  # slug: our internal dir must be filesystem-safe
        d.mkdir(parents=True, exist_ok=True)
        ps = d / "particles.star"
        sub = {k: (sub_particles if k == "particles" else v) for k, v in parts.items()}
        starfile.write(sub, str(ps), overwrite=True)
        tasks.append(
            (build_cmd, relion_bin, global_df[global_df["rlnTomoName"] == name].copy(), ps, trajectories, d, common)
        )
        processed.append(name)
    if n_workers and n_workers > 1:
        with mp.get_context("spawn").Pool(min(n_workers, len(tasks))) as pool:
            pool.map(_phase1_worker, tasks)
    else:
        for t in tasks:
            _phase1_worker(t)

    # gather per-tomogram temp evidence into the shared collect dir (output_dir/temp)
    temp_root = output_dir / "temp"
    for name in processed:
        src_temp = p1_root / _slug(name) / "temp"
        if not src_temp.exists():
            continue
        for src in src_temp.rglob("*"):
            if src.is_file():
                dst = temp_root / src.relative_to(src_temp)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)

    # phase 2 staging: full tomograms.star with header stubs + individual tilt stars + opt set
    stage = output_dir / "relion_input"
    (stage / "tiltseries").mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in global_df.iterrows():
        name = str(row["rlnTomoName"])
        indiv, reader = _indiv_reader(row, src_base, common.get("tiltseries_relative_dir"))
        starfile.write({name: indiv}, str(stage / "tiltseries" / f"{name}.star"), overwrite=True)
        nz, ny, nx = (int(x) for x in reader.data.shape)  # lazy: zarr metadata only
        stub = stage / f"{name}_stub.mrc"
        _write_header_stub(
            stub, nx, ny, nz, float(row["rlnTomoTiltSeriesPixelSize"]) if "rlnTomoTiltSeriesPixelSize" in row else 1.0
        )
        r = row.to_dict()
        r["rlnTomoTiltSeriesName"] = str(stub.resolve())
        r["rlnTomoFrameCount"] = nz
        r["rlnTomoTiltSeriesStarFile"] = f"tiltseries/{name}.star"
        rows.append(r)
    starfile.write({"global": pd.DataFrame(rows)}, str(stage / "tomograms.star"), overwrite=True)
    opt_set = _write_optimisation_set(stage, particles_star, stage / "tomograms.star", trajectories)

    cmd = build_cmd(relion_bin, opt_set, output_dir, box_size, ref1, ref2, mask, fsc, threads, opts) + [
        "--only_do_unfinished"
    ]
    logger.info("Phase 2 (collect): %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(stage))
    return output_dir


def _restore_zarr_source(output_dir: Path, global_df, src_base, tiltseries_relative_dir) -> None:
    """
    Make the job output re-consumable by our own tools without touching RELION's refined per-tilt
    stars (re-writing those via starfile would lossily reformat the refined values). Instead, only the
    output GLOBAL tomograms.star (which holds no refined per-tilt numerics) is edited: add a per-tomogram
    tomoTiltSeriesURI (the original zarr locator) and drop the stale rlnTomoTiltSeriesName/FrameCount that
    point at the deleted shm MRC / header stub. On a chained run, _materialize_tiltseries propagates that
    global URI into the individual tilt star for get_tiltseries_datareader.
    """
    from zarr_particle_tools.core.constants import TILTSERIES_URI_RELION_COLUMN

    out_tomo = output_dir / "tomograms.star"
    if not out_tomo.exists():  # e.g. an aberrations-only run doesn't rewrite tomograms.star
        return

    # original zarr locator per tomogram name (URI column, else the rlnMicrographName path)
    locator = {}
    for _, row in global_df.iterrows():
        ip = Path(str(row["rlnTomoTiltSeriesStarFile"]))
        ip = ip if ip.is_absolute() else src_base / ip
        indiv = read_single_table(ip)
        if TILTSERIES_URI_RELION_COLUMN in indiv.columns:
            locator[str(row["rlnTomoName"])] = str(indiv[TILTSERIES_URI_RELION_COLUMN].iloc[0])
        elif "rlnMicrographName" in indiv.columns:
            locator[str(row["rlnTomoName"])] = str(indiv["rlnMicrographName"].iloc[0]).split("@")[-1]

    gdf = read_single_table(out_tomo)
    gdf = gdf.drop(columns=[c for c in ("rlnTomoTiltSeriesName", "rlnTomoFrameCount") if c in gdf.columns])
    gdf[TILTSERIES_URI_RELION_COLUMN] = gdf["rlnTomoName"].astype(str).map(locator)
    starfile.write({"global": gdf}, str(out_tomo), overwrite=True)


def run_relion_tomo_job(
    build_cmd: Callable,
    relion_bin: str,
    output_dir: Union[str, Path],
    box_size: int,
    ref1: Union[str, Path],
    ref2: Union[str, Path],
    opts: dict,
    particles_starfile: Optional[Path] = None,
    tomograms_starfile: Optional[Path] = None,
    trajectories_starfile: Optional[Path] = None,
    optimisation_set_starfile: Optional[Path] = None,
    tiltseries_relative_dir: Optional[Path] = None,
    mask: Optional[Path] = None,
    fsc: Optional[Path] = None,
    threads: int = 6,
    shm_dir: Union[str, Path] = "/dev/shm",
    keep_shm: bool = False,
    per_tomogram: bool = True,
    n_workers: int = 0,
) -> Path:
    """Generic driver: resolve inputs, then run per-tomogram two-phase (default) or all-at-once."""
    start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shm_dir = Path(shm_dir)

    if optimisation_set_starfile is not None:
        particles_starfile, tomograms_starfile, traj = resolve_optimisation_set(Path(optimisation_set_starfile))
        trajectories_starfile = trajectories_starfile or traj
    if particles_starfile is None or tomograms_starfile is None:
        raise ValueError("Provide either an optimisation set or both particles and tomograms star files.")

    global_df, src_base = read_global_tomograms(Path(tomograms_starfile))
    if not global_df["rlnTomoName"].is_unique:  # RELION's per-tomogram key; duplicates are malformed
        dups = global_df["rlnTomoName"][global_df["rlnTomoName"].duplicated()].unique().tolist()
        raise ValueError(f"Duplicate rlnTomoName in {tomograms_starfile}: {dups}. Names must be unique.")

    # Hard-error if the particles reference no tomogram in this set (e.g. a generated tomograms.star
    # whose rlnTomoName scheme doesn't match refined particles) — otherwise RELION silently does nothing.
    _parts = starfile.read(str(particles_starfile))
    _pdf = _parts["particles"] if isinstance(_parts, dict) else _parts
    if "rlnTomoName" in _pdf.columns:
        p_names, t_names = set(_pdf["rlnTomoName"].astype(str)), set(global_df["rlnTomoName"].astype(str))
        if not (p_names & t_names):
            raise ValueError(
                "No rlnTomoName overlap between particles and tomograms — the job would process zero "
                f"particles. particles e.g. {sorted(p_names)[:3]}; tomograms e.g. {sorted(t_names)[:3]}."
            )

    two_phase = per_tomogram and len(global_df) > 1
    if two_phase and (n_workers is None or n_workers <= 0):  # auto: ~1/4 cores (each worker also uses --j)
        n_workers = min(len(global_df), max(1, min(16, (os.cpu_count() or 4) // 4)))
    _preflight_budget(
        shm_dir,
        _peek_stack_bytes(global_df, src_base, tiltseries_relative_dir),
        n_workers or 1,
        all_at_once=not two_phase,
    )

    common = dict(
        src_base=src_base,
        shm_dir=shm_dir,
        tiltseries_relative_dir=tiltseries_relative_dir,
        box_size=box_size,
        ref1=ref1,
        ref2=ref2,
        mask=mask,
        fsc=fsc,
        threads=threads,
        keep_shm=keep_shm,
        opts=opts,
    )

    if two_phase:
        _two_phase(
            build_cmd,
            relion_bin,
            global_df,
            src_base,
            particles_starfile,
            trajectories_starfile,
            output_dir,
            n_workers,
            box_size,
            ref1,
            ref2,
            mask,
            fsc,
            threads,
            opts,
            common,
        )
        logger.info(
            "Two-phase %s (%d tomograms, %d worker(s)) -> %s",
            Path(relion_bin).name,
            len(global_df),
            n_workers,
            output_dir,
        )
    else:
        _run_relion_job(
            build_cmd,
            relion_bin,
            global_df,
            particles_star=particles_starfile,
            trajectories=trajectories_starfile,
            output_dir=output_dir,
            **common,
        )

    # make the output re-consumable by our tools (restore zarr locator, drop stale shm refs)
    _restore_zarr_source(output_dir, global_df, src_base, tiltseries_relative_dir)
    logger.info("%s finished in %.1fs. Output: %s", Path(relion_bin).name, time.time() - start, output_dir)
    return output_dir
