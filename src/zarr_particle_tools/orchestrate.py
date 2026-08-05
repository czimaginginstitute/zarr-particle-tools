"""
zarr-particle-pipeline: dataset-ID auto-orchestrator for the full STA pipeline.

Given a CryoET Data Portal dataset ID (+ optional run subset) and a point annotation, resolve inputs via
the portal, derive + verify the tilt-series pixel size, generate the zarr-native star files (whose
tomoTiltSeriesURI column makes py2rely auto-select our zarr jobs), then drive py2rely end to end:
prepare relion5-parameters -> prepare relion5-pipeline -> sbatch pipeline.sh.

Everything acquisition-related (voltage, Cs, defocus, dose, tilt angles, tomogram dims) already flows
through the generated star files; the only scalar auto-derived here is the pixel size, cross-checked
against each tilt-series MRC header.
"""

import importlib.util
import logging
import math
import shutil
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path

import click
import starfile

import zarr_particle_tools.cli.options as cli_options
import zarr_particle_tools.generate.cdp_cache as cdp_cache
import zarr_particle_tools.generate.cdp_generate_starfiles as cdp_generate
import zarr_particle_tools.generate.copick_generate_starfiles as copick_generate
from zarr_particle_tools.core.constants import OPTICS_DF_COLUMNS, THREAD_POOL_WORKER_COUNT
from zarr_particle_tools.core.data import global_fs
from zarr_particle_tools.core.helpers import setup_logging

logger = logging.getLogger(__name__)

PARAMETERS_FILE = "all_sta_parameters.json"
PIPELINE_FILE = "pipeline.sh"
DEFAULT_PIXEL_SIZE_TOL = 1e-3  # relative; loose enough for portal rounding, tight enough to catch a wrong tilt series

PY2RELY_URL = "https://github.com/chanzuckerberg/py2rely"
# stock RELION binaries the py2rely STA chain invokes (Refine3D/Class3D, MaskCreate, PostProcess) plus the
# ones our zarr ctf-refine / polish wrappers call
RELION_BINARIES = (
    "relion_refine",
    "relion_mask_create",
    "relion_postprocess",
    "relion_tomo_refine_ctf",
    "relion_tomo_align",
)
# our pipeliner jobs, registered as ccpem_pipeliner.jobs entry points by any pip install;
# py2rely selects them (over stock RELION) when it sees the tomoTiltSeriesURI column
REQUIRED_JOB_ENTRY_POINTS = (
    "zarrparticletools.pseudosubtomo",
    "zarrparticletools.reconstruct",
    "zarrparticletools.ctfrefine",
    "zarrparticletools.polish",
)


def preflight_problems(require_copick: bool = False) -> list[str]:
    """Return human-readable reasons the environment can't run the py2rely pipeline (empty == good to go)."""
    problems = []
    if shutil.which("py2rely") is None:
        problems.append(f"py2rely not on PATH (install py2rely: {PY2RELY_URL})")
    missing_bins = [b for b in RELION_BINARIES if shutil.which(b) is None]
    if missing_bins:
        problems.append(
            f"RELION binaries not on PATH: {', '.join(missing_bins)} "
            "(source RELION's setup-env.sh and add its build/bin to PATH)"
        )
    modules = [("pipeliner", "ccpem-pipeliner")]
    if require_copick:  # only needed for the copick-data-portal path
        modules.append(("copick", "copick"))
    for module, pkg in modules:
        if importlib.util.find_spec(module) is None:
            problems.append(f"Python module '{module}' not importable (install {pkg})")
    try:
        registered = {ep.name for ep in entry_points(group="ccpem_pipeliner.jobs")}
    except Exception:  # noqa: BLE001
        registered = set()
    missing_eps = [ep for ep in REQUIRED_JOB_ENTRY_POINTS if ep not in registered]
    if missing_eps:
        problems.append(
            f"pipeliner job entry points not registered: {', '.join(missing_eps)} "
            "(reinstall the package so py2rely selects the zarr jobs)"
        )
    return problems


def assert_preflight(require_copick: bool = False) -> None:
    """Fail fast, reporting every gap at once, if the environment can't run the py2rely pipeline."""
    problems = preflight_problems(require_copick=require_copick)
    if problems:
        raise click.ClickException("Preflight checks failed:\n  - " + "\n  - ".join(problems))


def resolve_selected_tiltseries(annotation_files) -> list:
    """Dedup the tilt series backing the resolved annotation files (via alignment_id -> tiltseries_id)."""
    tiltseries_ids = {
        cdp_cache.get_alignments(f.alignment_id)[0].tiltseries_id for f in annotation_files if f.alignment_id
    }
    tiltseries_ids.discard(None)
    if not tiltseries_ids:
        raise click.ClickException("No tilt series found for the resolved annotations.")
    return cdp_cache.get_tiltseries(sorted(tiltseries_ids))


def resolve_copick_tiltseries(run_ids: list[int]) -> list:
    """Dedup the tilt series backing the given Data Portal run IDs (one alignment per run, as enforced upstream)."""
    alignments_by_run = cdp_cache.get_alignments_by_run_id(sorted(run_ids))
    tiltseries_ids = {a.tiltseries_id for alignments in alignments_by_run.values() for a in alignments}
    tiltseries_ids.discard(None)
    if not tiltseries_ids:
        raise click.ClickException("No tilt series found for the copick runs.")
    return cdp_cache.get_tiltseries(sorted(tiltseries_ids))


def read_mrc_header_pixel_size(s3_mrc_file: str) -> float | None:
    """Range-read the 1024-byte MRC header from S3 and return voxel_x = cella.xlen / mx (None if unavailable)."""
    if not s3_mrc_file:
        return None
    path = s3_mrc_file.removeprefix("s3://")
    try:
        with global_fs.open(path, "rb") as f:
            hdr = f.read(1024)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not read MRC header %s: %s", s3_mrc_file, e)
        return None
    if len(hdr) < 52:
        return None
    mx = struct.unpack_from("<i", hdr, 28)[0]
    xlen = struct.unpack_from("<f", hdr, 40)[0]
    if mx == 0:
        return None
    return xlen / mx


def derive_and_verify_pixel_size(
    tiltseries: list, tol: float = DEFAULT_PIXEL_SIZE_TOL, override: float | None = None
) -> float:
    """
    Derive the tilt-series pixel size from portal metadata (asserting a single value across the selection),
    then cross-check every tilt-series MRC header and fail loudly on mismatch. Uses the portal value so it
    stays consistent with rlnTomoTiltSeriesPixelSize written into the star files.
    """
    portal_values = sorted({round(t.pixel_spacing, 6) for t in tiltseries})
    if len(portal_values) != 1:
        raise click.ClickException(
            f"Selected tilt series report multiple pixel sizes {portal_values} (mixed optics is not "
            "supported yet). Narrow the selection (e.g. --run-ids) to a single pixel size."
        )
    pixel_size = float(override) if override is not None else portal_values[0]
    if override is not None:
        logger.info("Using pixel size override %.6f A (portal reports %.6f A).", pixel_size, portal_values[0])
    else:
        logger.info("Derived pixel size %.6f A from %d tilt series.", pixel_size, len(tiltseries))

    with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKER_COUNT) as pool:
        header_sizes = list(pool.map(lambda t: (t.id, read_mrc_header_pixel_size(t.s3_mrc_file)), tiltseries))

    mismatches = []
    checked = 0
    for ts_id, hps in header_sizes:
        if hps is None:
            logger.warning("[TiltSeries %s] No MRC header available; skipping pixel-size cross-check.", ts_id)
            continue
        checked += 1
        if not math.isclose(hps, pixel_size, rel_tol=tol):
            mismatches.append((ts_id, hps))
    if mismatches:
        detail = ", ".join(f"ts {i}: header={h:.6f}" for i, h in mismatches)
        raise click.ClickException(
            f"Pixel size {pixel_size:.6f} A disagrees with MRC header(s) beyond rel tol {tol}: {detail}. "
            "Portal metadata may be wrong; verify or pass --pixel-size to override."
        )
    logger.info("Pixel-size cross-check passed against %d/%d tilt-series MRC headers.", checked, len(tiltseries))
    return pixel_size


def _run(cmd: list[str], cwd: Path) -> None:
    logger.info("Running: %s (cwd=%s)", " ".join(str(c) for c in cmd), cwd)
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=True)


def run_py2rely_parameters(
    output_dir: Path,
    tomograms_star: Path,
    particles_star: Path,
    pixel_size: float,
    protein_diameter: float,
    symmetry: str,
    low_pass: float,
    box_scaling: float,
    binning_list: str,
    nthreads: int,
    denovo_generation: bool,
    nclasses: int | None,
    ninit_models: int | None,
    max_dose: float | None,
) -> Path:
    cmd = [
        "py2rely",
        "prepare",
        "relion5-parameters",
        "--output",
        PARAMETERS_FILE,
        "--tilt-series",
        tomograms_star,
        "--particles",
        particles_star,
        "--tilt-series-pixel-size",
        pixel_size,
        "--protein-diameter",
        protein_diameter,
        "--symmetry",
        symmetry,
        "--low-pass",
        low_pass,
        "--box-scaling",
        box_scaling,
        "--binning-list",
        binning_list,
        "--nthreads",
        nthreads,
    ]
    if denovo_generation:
        cmd.append("--denovo-generation")
    if nclasses is not None:
        cmd += ["--nclasses", nclasses]
    if ninit_models is not None:
        cmd += ["--ninit-models", ninit_models]
    if max_dose is not None:
        cmd += ["--max-dose", max_dose]
    _run(cmd, output_dir)
    return output_dir / PARAMETERS_FILE


def run_py2rely_pipeline(
    output_dir: Path,
    reference_template: Path | None,
    run_denovo_generation: bool,
    run_class3d: bool,
    num_gpus: int,
    gpu_constraint: str | None,
    cpu_constraint: str,
    timeout: int,
    num_days: int,
    extract3d: bool,
    class_selection: str | None,
    manual_masking: bool,
) -> Path:
    cmd = [
        "py2rely",
        "prepare",
        "relion5-pipeline",
        "--parameter",
        PARAMETERS_FILE,
        "--run-denovo-generation",
        str(run_denovo_generation),
        "--run-class3D",
        str(run_class3d),
        "--extract3D",
        str(extract3d),
        "--manual-masking",
        str(manual_masking),
        "--num-gpus",
        num_gpus,
        "--cpu-constraint",
        cpu_constraint,
        "--timeout",
        timeout,
        "--num-days",
        num_days,
    ]
    if reference_template is not None:
        cmd += ["--reference-template", reference_template]
    if gpu_constraint:
        cmd += ["--gpu-constraint", gpu_constraint]
    if class_selection is not None:
        cmd += ["--class-selection", class_selection]
    _run(cmd, output_dir)
    return output_dir / PIPELINE_FILE


@dataclass
class Py2RelyConfig:
    """py2rely-facing options shared by both orchestrator sources (everything but input selection)."""

    protein_diameter: float
    reference_template: Path | None = None
    symmetry: str = "C1"
    low_pass: float = 50.0
    box_scaling: float = 2.0
    binning_list: str = "4,2,1"
    nthreads: int = 5
    denovo_generation: bool = False
    run_class3d: bool = False
    nclasses: int | None = None
    ninit_models: int | None = None
    max_dose: float | None = None
    extract3d: bool = False
    class_selection: str | None = None
    manual_masking: bool = False
    num_gpus: int = 2
    gpu_constraint: str | None = None
    cpu_constraint: str = "16,12"
    timeout: int = 24
    num_days: int = 14
    prepare_only: bool = False

    def __post_init__(self):
        if not self.denovo_generation and self.reference_template is None:
            raise click.ClickException("--reference-template is required unless --run-denovo-generation is set.")
        if self.max_dose is not None and self.max_dose > 0:
            # py2rely would accept this and the zarr Extract job would then raise NotImplementedError
            # mid-pipeline, after SLURM submission; fail here instead
            raise click.ClickException(
                "--max-dose is not supported: the zarr extract job has no max-dose handling, so the "
                "pipeline would fail once it reached Extract. Omit the flag."
            )


def read_tomograms_df(tomograms_star: Path):
    """
    Read a tomograms.star into a single DataFrame. starfile returns a bare DataFrame for a
    single-block file; a multi-block file must carry the RELION `global` block, matching how the
    rest of the codebase reads these (subtomo_relion_job.read_global_tomograms, subtomo_extract).
    """
    df = starfile.read(tomograms_star)
    if isinstance(df, dict):
        if "global" not in df:
            raise click.ClickException(
                f"{tomograms_star} has blocks {sorted(df)} but no `global` block; expected a RELION "
                "tomograms.star with a data_global block."
            )
        df = df["global"]
    if df is None or len(df.columns) == 0:
        raise click.ClickException(f"{tomograms_star} has no tabular data.")
    return df


def pixel_size_from_tomograms_star(tomograms_star: Path, override: float | None = None) -> float:
    """
    Read the tilt-series pixel size out of a tomograms.star (asserting a single value across rows).
    Unlike the portal path there is no metadata to cross-check against, so nothing is verified here.
    """
    if override is not None:
        logger.info("Using pixel size override %.6f A.", float(override))
        return float(override)
    df = read_tomograms_df(tomograms_star)
    if "rlnTomoTiltSeriesPixelSize" not in df.columns:
        raise click.ClickException(
            f"{tomograms_star} has no rlnTomoTiltSeriesPixelSize column; pass --pixel-size explicitly."
        )
    values = sorted({round(float(v), 6) for v in df["rlnTomoTiltSeriesPixelSize"]})
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise click.ClickException(
            f"{tomograms_star} has a non-positive or non-finite rlnTomoTiltSeriesPixelSize {values}; "
            "pass --pixel-size explicitly."
        )
    if len(values) != 1:
        raise click.ClickException(
            f"{tomograms_star} reports multiple pixel sizes {values} (mixed optics is not supported yet). "
            "Narrow the star file to a single pixel size, or pass --pixel-size."
        )
    logger.info("Derived pixel size %.6f A from %s.", values[0], tomograms_star.name)
    return values[0]


def _require_under_output_dir(star: Path, flag: str, output_dir: Path) -> Path:
    """
    py2rely resolves star paths relative to the project dir, so inputs must live inside it.

    Accepts the star if either its absolute or its symlink-resolved path sits under either form of
    output_dir, so symlinking a star into <output_dir>/input works (py2rely just opens the relative
    path) and a symlinked output_dir does not spuriously reject a real file.
    """
    star = Path(star).absolute()
    candidates = (star, star.resolve())
    for candidate in candidates:
        if candidate.is_relative_to(output_dir):
            return candidate
    # output_dir itself may be a symlink; re-express the hit in the caller's terms so that the
    # relative_to() in _prepare_and_submit still succeeds
    resolved_base = output_dir.resolve()
    for candidate in candidates:
        if candidate.is_relative_to(resolved_base):
            return output_dir / candidate.relative_to(resolved_base)
    raise click.ClickException(
        f"{flag} must be inside --output-dir ({output_dir}), because py2rely resolves star paths "
        f"relative to the project directory. Got {star}.\nGenerate or copy it under "
        f"{output_dir / 'input'} first, e.g.:\n"
        f"  zarr-particle-tomograms data-portal --output-dir {output_dir / 'input'} ..."
    )


def _optics_from_tomograms_star(tomograms_star: Path):
    """Pull the optics columns out of a tomograms.star, naming anything the file is missing."""
    df = read_tomograms_df(tomograms_star)
    missing = [c for c in OPTICS_DF_COLUMNS if c not in df.columns]
    if missing:
        raise click.ClickException(
            f"{tomograms_star} is missing the optics column(s) {missing}. A RELION-native "
            "tomograms.star keeps the integer rlnOpticsGroup in its particles' data_optics block "
            "rather than data_global; use a tomograms.star written by zarr-particle-tomograms, or "
            "add the missing column(s)."
        )
    return df[OPTICS_DF_COLUMNS].drop_duplicates().reset_index(drop=True)


def _uses_portal_run_naming(
    copick_run_names: list[str], optics_df, tomograms_label: str = "the tomograms.star"
) -> bool:
    """
    Decide how copick run names map onto rlnOpticsGroupName: exactly (a locally-written
    tomograms.star) or by containment (portal naming, e.g. run 33379 -> run_33379_tiltseries_16106).

    Deliberately strict: every run must resolve by exactly one of the two schemes, and portal naming
    additionally requires all-numeric run names, or we fail rather than guess.
    """
    group_names = list(optics_df["rlnOpticsGroupName"].astype(str))
    if len(optics_df) == 1:
        return False  # a single optics group is applied to everything, so naming is irrelevant
    runs = [str(r) for r in copick_run_names]
    if all(r in group_names for r in runs):
        logger.info("Matching %d copick run(s) to optics groups by exact name.", len(runs))
        return False

    # anchored on the run_<id>_ prefix the portal generator emits: a bare substring test would let a
    # run ID collide with another tomogram's tiltseries ID (both are 5-digit portal IDs)
    def portal_matches(run: str) -> list:
        return [g for g in group_names if g.startswith(f"run_{run}_")]

    portal_style = all(r.isdigit() for r in runs) and all(len(portal_matches(r)) == 1 for r in runs)
    injective = len({portal_matches(r)[0] for r in runs}) == len(runs) if portal_style else False
    if portal_style and injective:
        logger.info(
            "Matching %d copick run(s) to optics groups by Data Portal run-ID naming (e.g. %s -> %s).",
            len(runs),
            runs[0],
            portal_matches(runs[0])[0],
        )
        return True
    unresolved = [r for r in runs if r not in group_names and len(portal_matches(r)) != 1] or runs
    raise click.ClickException(
        f"Cannot map copick run name(s) {unresolved[:5]} onto the {len(group_names)} optics group(s) in "
        f"{tomograms_label}. Run names must either equal an rlnOpticsGroupName exactly, or be Data Portal "
        "run IDs each matching the run_<id>_ prefix of exactly one distinct optics group."
    )


def orchestrate_local(
    output_dir: Path,
    cfg: Py2RelyConfig,
    particles_starfile: Path,
    tomograms_starfile: Path,
    pixel_size: float | None = None,
) -> Path:
    """
    Drive the py2rely STA pipeline from RELION stars you already have (no portal query).

    py2rely only needs a tomograms.star + particles.star + pixel size, so this works for any source.
    The four zarr jobs are auto-selected only if the tomograms.star carries tomoTiltSeriesURI;
    otherwise py2rely runs stock RELION against whatever the tilt stars point at.
    """
    assert_preflight()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    particles_star = _require_under_output_dir(particles_starfile, "--particles-starfile", output_dir)
    tomograms_star = _require_under_output_dir(tomograms_starfile, "--tomograms-starfile", output_dir)
    pixel_size = pixel_size_from_tomograms_star(tomograms_star, override=pixel_size)
    return _prepare_and_submit(output_dir, particles_star, tomograms_star, pixel_size, cfg)


def orchestrate_copick_local(
    output_dir: Path,
    cfg: Py2RelyConfig,
    tomograms_starfile: Path,
    copick_config: Path,
    copick_name: str,
    copick_session_id: str,
    copick_user_id: str,
    copick_run_names: list[str] = None,
    pixel_size: float | None = None,
) -> Path:
    """Generate particles.star from copick picks against a local tomograms.star, then prepare + submit."""
    assert_preflight(require_copick=True)
    output_dir = Path(output_dir).resolve()
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    tomograms_star = _require_under_output_dir(tomograms_starfile, "--tomograms-starfile", output_dir)

    if not copick_run_names:
        picks = copick_generate.get_copick_picks(
            copick_config, copick_name, copick_session_id, copick_user_id, copick_run_names
        )
        copick_run_names = [p.run.name for p in picks]

    optics_df = _optics_from_tomograms_star(tomograms_star)
    particles_df = copick_generate.copick_picks_to_starfile(
        copick_config,
        copick_name,
        copick_session_id,
        copick_user_id,
        copick_run_names,
        optics_df,
        data_portal_runs=_uses_portal_run_naming(copick_run_names, optics_df, str(tomograms_star)),
    )
    particles_star = input_dir / "particles.star"
    starfile.write({"optics": optics_df, "particles": particles_df}, particles_star)
    logger.info("Generated %s with %d particles.", particles_star, len(particles_df))

    pixel_size = pixel_size_from_tomograms_star(tomograms_star, override=pixel_size)
    return _prepare_and_submit(output_dir, particles_star, tomograms_star, pixel_size, cfg)


def _prepare_and_submit(
    output_dir: Path, particles_star: Path, tomograms_star: Path, pixel_size: float, cfg: Py2RelyConfig
) -> Path:
    """Common tail: py2rely prepare relion5-parameters -> prepare relion5-pipeline -> (optionally) sbatch."""
    # py2rely runs from output_dir; pass the star paths relative to it
    run_py2rely_parameters(
        output_dir=output_dir,
        tomograms_star=tomograms_star.relative_to(output_dir),
        particles_star=particles_star.relative_to(output_dir),
        pixel_size=pixel_size,
        protein_diameter=cfg.protein_diameter,
        symmetry=cfg.symmetry,
        low_pass=cfg.low_pass,
        box_scaling=cfg.box_scaling,
        binning_list=cfg.binning_list,
        nthreads=cfg.nthreads,
        denovo_generation=cfg.denovo_generation,
        nclasses=cfg.nclasses,
        ninit_models=cfg.ninit_models,
        max_dose=cfg.max_dose,
    )
    pipeline_sh = run_py2rely_pipeline(
        output_dir=output_dir,
        reference_template=cfg.reference_template,
        run_denovo_generation=cfg.denovo_generation,
        run_class3d=cfg.run_class3d,
        num_gpus=cfg.num_gpus,
        gpu_constraint=cfg.gpu_constraint,
        cpu_constraint=cfg.cpu_constraint,
        timeout=cfg.timeout,
        num_days=cfg.num_days,
        extract3d=cfg.extract3d,
        class_selection=cfg.class_selection,
        manual_masking=cfg.manual_masking,
    )
    if cfg.prepare_only:
        logger.info("Prepared %s. Submit with:  (cd %s && sbatch %s)", pipeline_sh, output_dir, PIPELINE_FILE)
        return pipeline_sh
    _run(["sbatch", PIPELINE_FILE], output_dir)
    logger.info("Submitted %s.", pipeline_sh)
    return pipeline_sh


def orchestrate_data_portal(
    output_dir: Path,
    cfg: Py2RelyConfig,
    pixel_size: float | None = None,
    pixel_size_tol: float = DEFAULT_PIXEL_SIZE_TOL,
    no_orientations: bool = False,
    **data_portal_args,
) -> Path:
    """Resolve portal annotations, generate star files, prepare + (optionally) submit the py2rely STA pipeline."""
    assert_preflight()
    output_dir = Path(output_dir)
    (output_dir / "input" / "tiltseries").mkdir(parents=True, exist_ok=True)

    # resolve annotations once, derive+verify pixel size from the same tilt series, then reuse for star gen
    annotation_files = cdp_generate.resolve_annotation_files(**data_portal_args)
    tiltseries = resolve_selected_tiltseries(annotation_files)
    pixel_size = derive_and_verify_pixel_size(tiltseries, tol=pixel_size_tol, override=pixel_size)

    particles_star, tomograms_star, _ = cdp_generate.generate_starfiles_from_annotation_files(
        annotation_files, output_dir / "input", no_orientations
    )
    return _prepare_and_submit(output_dir, particles_star, tomograms_star, pixel_size, cfg)


def orchestrate_copick_data_portal(
    output_dir: Path,
    cfg: Py2RelyConfig,
    copick_config: Path,
    copick_name: str,
    copick_session_id: str,
    copick_user_id: str,
    copick_run_names: list[str] = None,
    copick_dataset_ids: list[int] = None,
    pixel_size: float | None = None,
    pixel_size_tol: float = DEFAULT_PIXEL_SIZE_TOL,
) -> Path:
    """Resolve a Data Portal-backed copick project, generate star files, prepare + (optionally) submit."""
    assert_preflight(require_copick=True)
    output_dir = Path(output_dir)
    (output_dir / "input" / "tiltseries").mkdir(parents=True, exist_ok=True)

    particles_star, tomograms_star, _, _, filtered_run_ids = copick_generate.generate_copick_data_portal_starfiles(
        output_dir=output_dir / "input",
        copick_config=copick_config,
        copick_name=copick_name,
        copick_session_id=copick_session_id,
        copick_user_id=copick_user_id,
        copick_run_names=copick_run_names,
        copick_dataset_ids=copick_dataset_ids,
    )
    tiltseries = resolve_copick_tiltseries(filtered_run_ids)
    pixel_size = derive_and_verify_pixel_size(tiltseries, tol=pixel_size_tol, override=pixel_size)
    return _prepare_and_submit(output_dir, particles_star, tomograms_star, pixel_size, cfg)


def science_options():
    opts = [
        click.option(
            "--protein-diameter",
            type=float,
            required=True,
            help="Particle diameter in Angstroms (py2rely --protein-diameter).",
        ),
        click.option(
            "--reference-template",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="Reference template .mrc for preliminary refinement (required unless --run-denovo-generation).",
        ),
        click.option("--symmetry", type=str, default="C1", show_default=True, help="Protein symmetry."),
        click.option("--low-pass", type=float, default=50.0, show_default=True, help="Reference low-pass [A]."),
        click.option(
            "--box-scaling", type=float, default=2.0, show_default=True, help="Sub-tomogram box padding factor."
        ),
        click.option(
            "--binning-list",
            type=str,
            default="4,2,1",
            show_default=True,
            help="Refinement binning factors (coarse->fine); '1' present => ctf-refine + polish stages run.",
        ),
    ]
    return cli_options.compose_options(opts)


def workflow_options():
    opts = [
        click.option(
            "--run-denovo-generation", is_flag=True, help="Generate an initial model de novo (no reference template)."
        ),
        click.option("--run-class3d", is_flag=True, help="Run 3D classification after refinement."),
        click.option("--nclasses", type=int, default=None, help="Number of classes for 3D classification."),
        click.option("--ninit-models", type=int, default=None, help="Number of initial (de novo) models."),
        click.option("--max-dose", type=float, default=None, help="Maximum dose for extraction [e-/A^2]."),
        click.option("--extract3d", is_flag=True, help="Extract 3D particles before initial model generation."),
        click.option(
            "--class-selection",
            type=click.Choice(["auto", "manual"]),
            default=None,
            help="Class selection method after 3D classification.",
        ),
        click.option("--manual-masking", is_flag=True, help="Apply manual masking after the first refinement."),
    ]
    return cli_options.compose_options(opts)


def compute_options():
    opts = [
        click.option("--nthreads", type=int, default=5, show_default=True, help="RELION threads per MPI rank (--j)."),
        click.option("--num-gpus", type=int, default=2, show_default=True, help="GPUs for GPU jobs (Refine3D)."),
        click.option(
            "--gpu-constraint",
            type=str,
            default=None,
            help="SLURM GPU architecture constraint; one arch (a100) or several meaning OR, comma- or "
            "pipe-separated (a100,h100 or 'a100|h100'). py2rely checks these against the cluster's SLURM "
            "features and drops any that are unavailable. Default: no constraint.",
        ),
        click.option(
            "--cpu-constraint",
            type=str,
            default="16,12",
            show_default=True,
            help="CPUs and memory per CPU in GB, as 'CPUs,mem-per-cpu-GB' (e.g. 16,12 = 16 CPUs, 12 GB each).",
        ),
        click.option("--timeout", type=int, default=24, show_default=True, help="submitit per-trial timeout [hours]."),
        click.option("--num-days", type=int, default=14, show_default=True, help="SLURM walltime request [days]."),
    ]
    return cli_options.compose_options(opts)


def control_options(pixel_size_tol: bool = True):
    opts = [
        click.option(
            "--output-dir",
            type=click.Path(file_okay=False, path_type=Path),
            required=True,
            help="py2rely project directory (star files land in <output-dir>/input).",
        ),
        click.option(
            "--pixel-size", type=float, default=None, help="Override the auto-derived tilt-series pixel size [A]."
        ),
    ]
    # only the portal paths cross-check the derived pixel size against tilt-series MRC headers
    if pixel_size_tol:
        opts.append(
            click.option(
                "--pixel-size-tol",
                type=float,
                default=DEFAULT_PIXEL_SIZE_TOL,
                show_default=True,
                help="Relative tolerance for the MRC-header pixel-size cross-check.",
            )
        )
    opts += [
        click.option("--prepare-only", is_flag=True, help="Prepare params + pipeline.sh but do not sbatch."),
        click.option("--debug", is_flag=True, help="Enable debug logging."),
    ]
    return cli_options.compose_options(opts)


# CLI option names (minus run_denovo_generation, handled separately) that map onto Py2RelyConfig fields
_CONFIG_KEYS = (
    "protein_diameter",
    "reference_template",
    "symmetry",
    "low_pass",
    "box_scaling",
    "binning_list",
    "nthreads",
    "run_class3d",
    "nclasses",
    "ninit_models",
    "max_dose",
    "extract3d",
    "class_selection",
    "manual_masking",
    "num_gpus",
    "gpu_constraint",
    "cpu_constraint",
    "timeout",
    "num_days",
    "prepare_only",
)


def _pop_config(kwargs: dict) -> Py2RelyConfig:
    """Pull the py2rely-facing options out of the flat click kwargs into a Py2RelyConfig (mutates kwargs)."""
    denovo = kwargs.pop("run_denovo_generation")
    return Py2RelyConfig(denovo_generation=denovo, **{k: kwargs.pop(k) for k in _CONFIG_KEYS})


@click.group(help="Drive the full CryoET Data Portal -> zarr STA pipeline via py2rely.")
def cli():
    pass


@cli.command("preflight", help="Check py2rely, RELION binaries, deps, and pipeliner entry points are available.")
@click.option("--copick", is_flag=True, help="Also require copick (for the copick-data-portal path).")
def cmd_preflight(copick):
    setup_logging(False)
    problems = preflight_problems(require_copick=copick)
    if problems:
        raise click.ClickException("Preflight checks failed:\n  - " + "\n  - ".join(problems))
    click.echo("Preflight OK: py2rely + RELION binaries on PATH, ccpem-pipeliner importable, zarr jobs registered.")


@cli.command("data-portal", help="Orchestrate the full STA pipeline from a CryoET Data Portal dataset + annotation.")
@cli_options.data_portal_options()
@science_options()
@workflow_options()
@compute_options()
@control_options()
def cmd_data_portal(**kwargs):
    setup_logging(kwargs.pop("debug", False))
    kwargs = cli_options.flatten_data_portal_args(kwargs)
    cfg = _pop_config(kwargs)
    output_dir = kwargs.pop("output_dir")
    orchestrate_data_portal(output_dir, cfg, **kwargs)


@cli.command(
    "copick-data-portal",
    help="Orchestrate the full STA pipeline from a Data Portal-backed copick project (copick picks + portal runs).",
)
@cli_options.copick_options()
@cli_options.data_portal_copick_options()
@science_options()
@workflow_options()
@compute_options()
@control_options()
def cmd_copick_data_portal(**kwargs):
    setup_logging(kwargs.pop("debug", False))
    kwargs["copick_run_names"] = cli_options.flatten(kwargs["copick_run_names"])
    kwargs["copick_dataset_ids"] = cli_options.flatten(kwargs["copick_dataset_ids"])
    cfg = _pop_config(kwargs)
    output_dir = kwargs.pop("output_dir")
    orchestrate_copick_data_portal(output_dir, cfg, **kwargs)


@cli.command(
    "local",
    help="Orchestrate the full STA pipeline from RELION stars you already have (both must live inside "
    "--output-dir). Zarr jobs are used only if the tomograms.star carries tomoTiltSeriesURI.",
)
@click.option(
    "--particles-starfile",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Your particles.star (inside --output-dir).",
)
@click.option(
    "--tomograms-starfile",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Your tomograms.star (inside --output-dir); pixel size is read from rlnTomoTiltSeriesPixelSize.",
)
@science_options()
@workflow_options()
@compute_options()
@control_options(pixel_size_tol=False)
def cmd_local(**kwargs):
    setup_logging(kwargs.pop("debug", False))
    cfg = _pop_config(kwargs)
    output_dir = kwargs.pop("output_dir")
    orchestrate_local(output_dir, cfg, **kwargs)


@cli.command(
    "copick-local",
    help="Orchestrate the full STA pipeline from copick picks plus your own tomograms.star (inside "
    "--output-dir); particles.star is generated into <output-dir>/input.",
)
@click.option(
    "--tomograms-starfile",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Your tomograms.star (inside --output-dir); supplies optics and the pixel size.",
)
@cli_options.copick_options()
@science_options()
@workflow_options()
@compute_options()
@control_options(pixel_size_tol=False)
def cmd_copick_local(**kwargs):
    setup_logging(kwargs.pop("debug", False))
    kwargs["copick_run_names"] = cli_options.flatten(kwargs["copick_run_names"])
    cfg = _pop_config(kwargs)
    output_dir = kwargs.pop("output_dir")
    orchestrate_copick_local(output_dir, cfg, **kwargs)


if __name__ == "__main__":
    cli()
