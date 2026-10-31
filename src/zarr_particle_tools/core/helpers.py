import logging
import math
import os
import shutil
from collections.abc import Callable, Collection
from pathlib import Path

import pandas as pd
import starfile
from rich.logging import RichHandler

from zarr_particle_tools.core.constants import (
    NOISY_LOGGERS,
    OPTICS_DF_COLUMNS,
    PARTICLE_ANGLES,
    SUBTOMOGRAM_ANGLES,
    TILTSERIES_URI_RELION_COLUMN,
)
from zarr_particle_tools.core.deformation import declaration_problems, parse_coefficients

logger = logging.getLogger(__name__)

# ====================== global helpers ======================


def _mem_budget_bytes() -> int | None:
    """This process's memory budget: SLURM allocation, then cgroup limit, then physical RAM, else None."""
    mpn = os.environ.get("SLURM_MEM_PER_NODE")  # MB
    if mpn and mpn.isdigit():
        return int(mpn) * 1024**2
    mpc = os.environ.get("SLURM_MEM_PER_CPU")  # MB
    ncpu = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_JOB_CPUS_PER_NODE", "")
    if mpc and mpc.isdigit():
        try:
            return int(mpc) * int(str(ncpu).split("(")[0]) * 1024**2
        except (ValueError, IndexError):
            pass
    try:
        rel = ""
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            fields = line.split(":")
            if len(fields) == 3 and ("memory" in fields[1] or fields[1] == ""):
                rel = fields[2]
                if "memory" in fields[1]:
                    break
        for p in (
            Path("/sys/fs/cgroup/memory") / rel.lstrip("/") / "memory.limit_in_bytes",
            Path("/sys/fs/cgroup") / rel.lstrip("/") / "memory.max",
        ):
            if p.exists():
                v = p.read_text().strip()
                if v.isdigit() and int(v) < (1 << 62):
                    return int(v)
    except OSError:
        pass
    try:  # e.g. a workstation or CI runner: no allocation, so the machine's RAM is the limit
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def auto_worker_count(cpu_cap: int, per_worker_gb: float = 10.0) -> int:
    """Worker count bounded by memory (each worker loads a full tilt series ~per_worker_gb).

    Prevents cgroup OOM from spawning too many workers. Override with env ZARR_N_WORKERS;
    falls back to cpu_cap when the memory budget can't be determined.
    """
    env = os.environ.get("ZARR_N_WORKERS")
    if env and env.isdigit():
        return max(1, min(cpu_cap, int(env)))
    budget = _mem_budget_bytes()
    if not budget:
        return cpu_cap
    mem_cap = int((budget * 0.75) / (per_worker_gb * 1024**3))
    return max(1, min(cpu_cap, mem_cap))


def suppress_noisy_loggers(loggers, level=logging.ERROR):
    for name in loggers:
        logging.getLogger(name).setLevel(level)


def setup_logging(debug: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_level=True, show_path=False, markup=True)],
        format="%(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
        force=True,
    )

    suppress_noisy_loggers(NOISY_LOGGERS)

    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).propagate = True


def get_filter(values, field, inexact_match, label=""):
    """Helper to append filters depending on inexact_match."""
    if not values:
        return None
    if inexact_match:
        if len(values) > 1:
            raise ValueError(
                f"Cannot use inexact match with multiple values ({values}) for {label}. Please provide a single value."
            )
        logger.info(f"Finding similar {label}: {values} (case insensitive, includes partial matches)")
        return field.ilike(f"%{values[0]}%")
    else:
        logger.info(f"Filtering by EXACT {label}: {values}")
        return field._in(values)


# ====================== starfile generation helpers ======================


def get_optics_group_name(run_id: int, tiltseries_id: int) -> str:
    return f"run_{run_id}_tiltseries_{tiltseries_id}"


def get_tomo_name(run_id: int, tiltseries_id: int, alignment_id: int, voxel_spacing_id: int) -> str:
    return f"run_{run_id}_tiltseries_{tiltseries_id}_alignment_{alignment_id}_spacing_{voxel_spacing_id}"


def get_unbinned_tomo_size(binned_size: int, binning: float, is_z: bool = False) -> int:
    """
    Bin-1 tomogram size in tilt-series pixels (rlnTomoSize*) for a tomogram dimension at the given binning: the even
    size that RELION's reconstruct bins back to ``binned_size`` (X/Y round, Z truncates).
    """
    unbinned_size = binned_size * binning
    if is_z:
        return 2 * math.ceil(unbinned_size / 2 - 1e-6)
    return 2 * round(unbinned_size / 2)


def project_relative(path: str | Path, project_dir: str | Path | None = None) -> str:
    """``path`` relative to ``project_dir`` (default: the working directory) when it is inside it, else absolute."""
    path = Path(path).absolute()
    try:
        return str(path.relative_to(Path(project_dir or Path.cwd()).absolute()))
    except ValueError:
        return str(path)


# ====================== file validation, reading, data extraction helpers ======================

# names are opaque strings in RELION; starfile would otherwise parse e.g. "007" as 7
STAR_NAME_COLUMNS = ["rlnTomoName", "rlnTomoParticleName", "rlnOpticsGroupName"]

OPTIMISATION_SET_KEYS = ("rlnTomoParticlesFile", "rlnTomoTomogramsFile", "rlnTomoTrajectoriesFile")
ZERNIKE = ("rlnEvenZernike", "rlnOddZernike")


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip().strip('"')


def resolve_path(value, base: str | Path | None = None) -> Path | None:
    """A path from a STAR file: absolute, else relative to the working directory if it exists there, else to ``base``.

    A blank value is ``None``.
    """
    if _blank(value):
        return None
    path = Path(str(value))
    if path.is_absolute() or path.exists() or base is None:
        return path
    return Path(base) / path


def read_optimisation_set(path: str | Path) -> tuple[Path, Path, Path | None]:
    """(particles, tomograms, trajectories) named by an optimisation set, each found by :func:`resolve_path`.

    Reads RELION's key-value block and a one-row table alike; an empty trajectories entry is ``None``.
    """
    data = starfile.read(path)
    if isinstance(data, dict) and not any(isinstance(v, pd.DataFrame) for v in data.values()):
        entries = data
    else:
        table = next(iter(data.values())) if isinstance(data, dict) else data
        if len(table) != 1:
            raise ValueError(f"{path}: an optimisation set has one row, found {len(table)}")
        entries = table.iloc[0].to_dict()
    paths = [resolve_path(entries.get(key), Path(path).parent) for key in OPTIMISATION_SET_KEYS]
    for key, resolved in zip(OPTIMISATION_SET_KEYS, paths, strict=True):
        if resolved is None and key != "rlnTomoTrajectoriesFile":
            raise ValueError(f"{key} in {path}: missing")
        if resolved is not None and not resolved.exists():
            raise FileNotFoundError(f"{key} in {path}: {resolved} does not exist")
    return tuple(paths)


def validate_and_setup(
    box_size: int,
    output_dir: str | Path,
    particles_starfile: str | Path | None = None,
    trajectories_starfile: str | Path | None = None,
    tiltseries_relative_dir: str | Path | None = None,
    tomograms_starfile: str | Path | None = None,
    optimisation_set_starfile: str | Path | None = None,
    crop_size: int = None,
    overwrite: bool = False,
    dry_run: bool = False,
    copick_data_portal: bool = False,
    check: Callable[[Path | None, Path | None, Path | None, Path | None], None] | None = None,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Validate and resolve the inputs and prepare ``output_dir``. ``check`` is called with the resolved
    (particles, tomograms, trajectories, tiltseries_relative_dir) before anything is written, dry runs included."""
    if not dry_run and box_size is None:
        raise ValueError("Box size must be specified.")

    if box_size is not None and box_size % 2 != 0 and box_size > 0:
        raise ValueError(f"Box size must be an even number greater than 0, got {box_size}.")

    if crop_size is not None:
        if crop_size % 2 != 0:
            raise ValueError(f"Crop size must be an even number, got {crop_size}.")
        if crop_size > box_size:
            raise ValueError(
                f"Crop size cannot be greater than box size, got crop size {crop_size} and box size {box_size}."
            )

    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
    particles_starfile = Path(particles_starfile) if isinstance(particles_starfile, str) else particles_starfile
    if particles_starfile is not None and not particles_starfile.exists():
        raise FileNotFoundError(f"Particles star file {particles_starfile} does not exist.")
    trajectories_starfile = (
        Path(trajectories_starfile) if isinstance(trajectories_starfile, str) else trajectories_starfile
    )
    if trajectories_starfile is not None and not trajectories_starfile.exists():
        raise FileNotFoundError(f"Trajectories star file {trajectories_starfile} does not exist.")
    tiltseries_relative_dir = (
        Path(tiltseries_relative_dir) if isinstance(tiltseries_relative_dir, str) else tiltseries_relative_dir
    )
    if tiltseries_relative_dir is not None and not Path(tiltseries_relative_dir).exists():
        raise FileNotFoundError(f"Tilseries relative directory {tiltseries_relative_dir} does not exist.")
    tomograms_starfile = Path(tomograms_starfile) if isinstance(tomograms_starfile, str) else tomograms_starfile
    if tomograms_starfile is not None and not tomograms_starfile.exists():
        raise FileNotFoundError(f"Tomograms star file {tomograms_starfile} does not exist.")
    optimisation_set_starfile = (
        Path(optimisation_set_starfile) if isinstance(optimisation_set_starfile, str) else optimisation_set_starfile
    )
    if optimisation_set_starfile is not None and not optimisation_set_starfile.exists():
        raise FileNotFoundError(f"Optimisation set star file {optimisation_set_starfile} does not exist.")

    if optimisation_set_starfile and (particles_starfile or tomograms_starfile or trajectories_starfile):
        raise ValueError(
            "Cannot specify both optimisation set star file and individual star files. Please provide only one of them."
        )

    if not copick_data_portal and not optimisation_set_starfile and not particles_starfile and not tomograms_starfile:
        raise ValueError("Either optimisation set star file or particles and tomograms star files must be provided.")

    if optimisation_set_starfile:
        particles_starfile, tomograms_starfile, trajectories_starfile = read_optimisation_set(optimisation_set_starfile)

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} already exists and is not empty. Use --overwrite to overwrite existing files."
        )

    if check is not None:
        check(particles_starfile, tomograms_starfile, trajectories_starfile, tiltseries_relative_dir)

    if not dry_run:
        if (output_dir / "Subtomograms").exists():
            shutil.rmtree(output_dir / "Subtomograms")

        (output_dir / "Subtomograms").mkdir(parents=True)

    return (
        output_dir,
        particles_starfile,
        trajectories_starfile,
        tiltseries_relative_dir,
        tomograms_starfile,
        optimisation_set_starfile,
    )


def check_unique_tomograms(tomograms_df: pd.DataFrame, tomograms_starfile: Path) -> None:
    """Each tomograms.star row is processed on its own, so a repeated rlnTomoName would be processed twice."""
    duplicated = sorted(set(tomograms_df.loc[tomograms_df["rlnTomoName"].duplicated(), "rlnTomoName"]))
    if duplicated:
        raise ValueError(f"Tomograms listed more than once in {tomograms_starfile}: {', '.join(duplicated)}")


def read_tomograms_starfile(tomograms_starfile: Path) -> tuple[dict | pd.DataFrame, pd.DataFrame]:
    """Reads a tomograms star file, returning its raw contents and its global table."""
    tomograms_data = starfile.read(tomograms_starfile, parse_as_string=STAR_NAME_COLUMNS)
    tomograms_df = tomograms_data["global"] if isinstance(tomograms_data, dict) else tomograms_data
    missing = [c for c in ("rlnTomoName", "rlnTomoTiltSeriesStarFile") if c not in tomograms_df.columns]
    if missing:
        raise ValueError(
            f"Tomograms star file {tomograms_starfile} is missing required column(s): {', '.join(missing)}"
        )
    check_unique_tomograms(tomograms_df, tomograms_starfile)
    return tomograms_data, tomograms_df


def read_single_table(path: str | Path) -> pd.DataFrame:
    """Read a star file that holds a single (optionally named) data block as a DataFrame."""
    data = starfile.read(str(path), parse_as_string=STAR_NAME_COLUMNS)
    if isinstance(data, dict):
        if len(data) != 1:
            raise ValueError(f"Expected a single data block in {path}, found {list(data)}.")
        return next(iter(data.values()))
    return data


def read_tiltseries_table(
    tiltseries_row_entry: pd.Series,
    tomograms_starfile: Path,
    tomograms_data: dict | pd.DataFrame | None = None,
    tiltseries_relative_dir: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """A tomogram's tilt-series table and the file it was read from.

    The table is the tomogram's block of a multi-block ``tomograms_data``, else the file named by
    ``rlnTomoTiltSeriesStarFile``: absolute, under ``tiltseries_relative_dir`` when given, else found by
    :func:`resolve_path` from the tomograms star's directory. A tomoTiltSeriesURI in the tomogram row is copied into
    a table without one.
    """
    name = str(tiltseries_row_entry["rlnTomoName"])
    if isinstance(tomograms_data, dict):
        if name not in tomograms_data:
            raise ValueError(f"rlnTomoName in {tomograms_starfile}: no tilt-series block for tomogram {name}")
        table, source = tomograms_data[name], Path(tomograms_starfile)
    else:
        star = tiltseries_row_entry.get("rlnTomoTiltSeriesStarFile")
        if tiltseries_relative_dir is not None and not _blank(star) and not Path(str(star)).is_absolute():
            source = Path(tiltseries_relative_dir) / str(star)
        else:
            source = resolve_path(star, Path(tomograms_starfile).parent)
        if source is None or not source.exists():
            raise FileNotFoundError(
                f"rlnTomoTiltSeriesStarFile in {tomograms_starfile}: tomogram {name}: {source} does not exist"
            )
        table = read_single_table(source)
    if table.empty:
        raise ValueError(f"{source}: tomogram {name} has no tilts")
    uri = tiltseries_row_entry.get(TILTSERIES_URI_RELION_COLUMN)
    if TILTSERIES_URI_RELION_COLUMN not in table.columns and not _blank(uri):
        table = table.assign(**{TILTSERIES_URI_RELION_COLUMN: str(uri)})
    return table, source


def get_tiltseries_data_locator(individual_tiltseries_df: pd.DataFrame, tiltseries_relative_dir: Path) -> str:
    """Return the single tilt-series locator referenced by an individual tilt-series table."""
    if TILTSERIES_URI_RELION_COLUMN in individual_tiltseries_df.columns:
        tiltseries_data_locators = individual_tiltseries_df[TILTSERIES_URI_RELION_COLUMN].to_list()
    else:
        tiltseries_data_locators = (
            individual_tiltseries_df["rlnMicrographName"].apply(lambda x: x.split("@", 1)[1]).to_list()
        )
    if len(set(tiltseries_data_locators)) != 1:
        raise ValueError(
            f"Multiple tiltseries data locators found: {set(tiltseries_data_locators)}. This is not supported."
        )
    tiltseries_data_locator = tiltseries_data_locators[0]
    if not tiltseries_data_locator.startswith("s3://") and not tiltseries_data_locator.startswith("/"):
        tiltseries_data_locator = tiltseries_relative_dir / tiltseries_data_locator
    return str(tiltseries_data_locator)


def get_tiltseries_data(
    particles_df: pd.DataFrame,
    optics_df: pd.DataFrame,
    trajectories_dict: dict[str, pd.DataFrame],
    tiltseries_row_entry: pd.Series,
    tiltseries_relative_dir: Path,
    tomograms_starfile: Path,
    tomograms_data: pd.DataFrame,
) -> dict[str, any] | None:
    """Builds the arguments for processing a single tiltseries and its particles."""
    individual_tiltseries_df, _ = read_tiltseries_table(
        tiltseries_row_entry, tomograms_starfile, tomograms_data, tiltseries_relative_dir
    )

    filtered_particles_df = particles_df[particles_df["rlnTomoName"] == tiltseries_row_entry["rlnTomoName"]]
    if filtered_particles_df.empty:
        logger.warning(
            f"No particles found for tomogram {tiltseries_row_entry['rlnTomoName']}. Please check the particles star file."
        )
        return None

    filtered_trajectories_dict = None
    if trajectories_dict:
        particle_names = filtered_particles_df["rlnTomoParticleName"].tolist()
        filtered_trajectories_dict = {k: v for k, v in trajectories_dict.items() if k in particle_names}
    group_name = tiltseries_row_entry.get("rlnOpticsGroupName", tiltseries_row_entry["rlnTomoName"])
    optics_row = optics_df[optics_df["rlnOpticsGroupName"] == group_name]
    return {
        "filtered_particles_df": filtered_particles_df,
        "filtered_trajectories_dict": filtered_trajectories_dict,
        "tiltseries_row_entry": tiltseries_row_entry,
        "individual_tiltseries_df": individual_tiltseries_df,
        "optics_row": optics_row,
    }


def optics_lookup(tomograms_df: pd.DataFrame) -> pd.DataFrame:
    """Each tomogram's optics group: ``rlnOpticsGroupName``, else the tomogram's name, numbered in tomogram order."""
    names = tomograms_df["rlnOpticsGroupName"] if "rlnOpticsGroupName" in tomograms_df else tomograms_df["rlnTomoName"]
    return pd.DataFrame(
        {
            "rlnOpticsGroup": range(1, len(tomograms_df) + 1),
            "rlnOpticsGroupName": names.astype(str).to_list(),
            "rlnTomoName": tomograms_df["rlnTomoName"].astype(str).to_list(),
        }
    )


def particles_and_optics(particles_data, tomograms_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(particles, optics). Without an optics block, build it from tomograms.star as RELION's ParticleSet does
    (particle_set.cpp:41-121)."""
    if isinstance(particles_data, dict):
        if "particles" not in particles_data:
            raise ValueError(f"no particles block in the particle STAR (blocks: {sorted(particles_data)})")
        if "optics" in particles_data:
            return particles_data["particles"], particles_data["optics"]
        particles_df = particles_data["particles"]
    else:
        particles_df = particles_data
    values = [c for c in OPTICS_DF_COLUMNS if c not in ("rlnOpticsGroup", "rlnOpticsGroupName", "rlnTomoName")]
    missing = [c for c in values if c not in tomograms_df.columns]
    if missing:
        raise ValueError(f"the particle STAR has no optics block and tomograms.star lacks {missing} to build one")
    lookup = optics_lookup(tomograms_df)
    group_of = dict(zip(lookup["rlnTomoName"], lookup["rlnOpticsGroupName"], strict=True))
    tomo_names = particles_df["rlnTomoName"].astype(str)
    groups = tomo_names.map(group_of)
    first_tomogram = {}
    for tomo_name, group in zip(tomo_names, groups, strict=True):
        if isinstance(group, str):
            first_tomogram.setdefault(group, tomo_name)
    number = {group: i for i, group in enumerate(first_tomogram, 1)}
    particles_df = particles_df.copy()
    particles_df["rlnOpticsGroup"] = groups.map(number).fillna(0).astype(int)
    rows = tomograms_df.set_index(tomograms_df["rlnTomoName"].astype(str))
    optics_df = pd.DataFrame(
        {
            "rlnOpticsGroup": list(number.values()),
            "rlnOpticsGroupName": list(number),
            **{c: [rows.at[first_tomogram[group], c] for group in number] for c in values},
        }
    )
    return particles_df, optics_df


# ====================== input checks ======================


def _nonzero_vector(value) -> bool:
    return not _blank(value) and bool((parse_coefficients(value) != 0).any())


def particle_problems(
    program: str,
    particles_df: pd.DataFrame,
    optics_df: pd.DataFrame,
    source: Path,
    apply_ctf: bool = True,
) -> list[str]:
    """Particle and optics metadata the Python ``extract`` or ``reconstruct`` would not apply as RELION does."""
    problems = []
    for column in SUBTOMOGRAM_ANGLES + (PARTICLE_ANGLES if program == "reconstruct" else ()):
        blank = int(pd.to_numeric(particles_df[column], errors="coerce").isna().sum()) if column in particles_df else 0
        if blank:
            problems.append(
                f"{column} in {source}: {blank} of {len(particles_df)} particles have no value; fill them in "
                "(0 is no rotation) or drop the column"
            )
    if program == "extract":
        if (
            apply_ctf
            and "rlnEvenZernike" in optics_df.columns
            and optics_df["rlnEvenZernike"].map(_nonzero_vector).any()
        ):
            problems.append(f"rlnEvenZernike in {source} (optics): even higher-order aberrations are not applied")
    elif program == "reconstruct":
        for block, table in (("optics", optics_df), ("particles", particles_df)):
            for column in ZERNIKE:
                if column in table.columns and table[column].map(_nonzero_vector).any():
                    problems.append(f"{column} in {source} ({block}): higher-order aberrations are not applied")
        if "rlnTomoTiltSeriesPixelSize" in optics_df.columns:
            sizes = sorted(set(optics_df["rlnTomoTiltSeriesPixelSize"].astype(float)))
            if len(sizes) > 1:
                problems.append(
                    f"rlnTomoTiltSeriesPixelSize in {source} (optics): reconstruction needs one tilt-series pixel "
                    f"size, found {sizes}"
                )
    return problems


def input_problems(
    program: str,
    particles: Path | None,
    tomograms: Path,
    trajectories: Path | None = None,
    tiltseries_relative_dir: Path | None = None,
    apply_ctf: bool = True,
    ignore_deformations: bool = False,
    tomo_names: Collection[str] | None = None,
) -> list[tuple[str, str]]:
    """``(input, message)`` for what ``program`` cannot honor in its inputs; ``input`` is ``particles``,
    ``tomograms`` or ``trajectories``.

    ``program`` is ``extract`` or ``reconstruct`` (tilt series of tomograms with particles, tilt stars under
    ``tiltseries_relative_dir``), or ``relion`` (CTF refinement and polishing: every tomogram's tilt series is staged,
    tilt stars found by :func:`resolve_path`). ``particles`` is ``None`` for copick picks; ``tomo_names`` then names
    the tomograms that have picks.
    """
    try:
        tomograms_data, tomograms_df = read_tomograms_starfile(tomograms)
    except (OSError, ValueError, KeyError) as exc:
        return [("tomograms", f"{tomograms}: {exc}")]
    problems = []
    with_particles = None if tomo_names is None else {str(name) for name in tomo_names}
    if particles is not None and program != "relion":
        try:
            particles_data = starfile.read(particles, parse_as_string=STAR_NAME_COLUMNS)
            particles_df, optics_df = particles_and_optics(particles_data, tomograms_df)
        except (OSError, ValueError, KeyError) as exc:
            problems.append(("particles", f"{particles}: {exc}"))
        else:
            with_particles = set(particles_df["rlnTomoName"].astype(str))
            messages = particle_problems(program, particles_df, optics_df, particles, apply_ctf)
            problems += [("particles", message) for message in messages]

    relative_dir = Path(tiltseries_relative_dir or "./")
    for _, row in tomograms_df.iterrows():
        name = str(row["rlnTomoName"])
        if with_particles is not None and name not in with_particles:
            continue
        try:
            if program == "relion":
                tilts, source = read_tiltseries_table(row, tomograms)
            else:
                tilts, source = read_tiltseries_table(row, tomograms, tomograms_data, relative_dir)
        except (OSError, ValueError, KeyError) as exc:
            problems.append(("tomograms", str(exc)))
            continue
        column = TILTSERIES_URI_RELION_COLUMN if TILTSERIES_URI_RELION_COLUMN in tilts else "rlnMicrographName"
        try:
            locator = get_tiltseries_data_locator(tilts, relative_dir)
        except (ValueError, IndexError) as exc:
            problems.append(("tomograms", f"{column} in {source}: tomogram {name}: {exc}"))
            continue
        if not locator.startswith("s3://") and not Path(locator).exists():
            problems.append(("tomograms", f"{column} in {source}: tomogram {name}: {locator} does not exist"))
        if program != "relion" and not ignore_deformations:
            for column, message in declaration_problems(row, tilts):
                where = source if column == "rlnTomoDeformationCoefficients" else tomograms
                problems.append(("tomograms", f"{column} in {where}: {message}"))

    if trajectories is not None and not Path(trajectories).exists():
        problems.append(("trajectories", f"{trajectories} does not exist"))
    return problems


def check_inputs(*args, **kwargs) -> None:
    """Raise :func:`input_problems` as one ``ValueError``."""
    problems = input_problems(*args, **kwargs)
    if problems:
        raise ValueError("Inputs cannot be honored:\n" + "\n".join(f"  - {message}" for _, message in problems))
