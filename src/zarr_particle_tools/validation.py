"""
What zarr-particle-tools can honour, checked before a job runs.

The Python extraction and reconstruction reimplement part of RELION. An option they do not implement used to raise
from inside a pipeliner wrapper; input metadata they do not apply (subtomogram orientations, higher-order
aberrations, anisotropic magnification, 2D deformations) was silently ignored, so a job succeeded and produced a
different result. :func:`check` reports both, as :class:`Problem` s naming the option or column and the file it was
found in, so a caller can refuse a job before it is submitted and a program can refuse it before it streams.

Imports only the standard library and ``starfile``, so a process that plans jobs can call it. The programs call
:func:`check_inputs` on their resolved inputs; the pipeliner wrappers call :func:`check_options` in
``get_commands``; a planner calls :func:`check` with the job's options and lets it resolve the inputs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import starfile

EXTRACT = "zarrparticletools.pseudosubtomo"
RECONSTRUCT = "zarrparticletools.reconstruct"
CTFREFINE = "zarrparticletools.ctfrefine"
POLISH = "zarrparticletools.polish"

#: Jobs whose pixels are produced by the Python reimplementation (not by stock RELION).
PYTHON_IMPLEMENTED = (EXTRACT, RECONSTRUCT)

TILTSERIES_URI_COLUMN = "tomoTiltSeriesURI"
SUBTOMOGRAM_ORIENTATION = ("rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt", "rlnTomoSubtomogramPsi")
ZERNIKE = ("rlnEvenZernike", "rlnOddZernike")
MAGNIFICATION = {"rlnMagMat00": 1.0, "rlnMagMat01": 0.0, "rlnMagMat10": 0.0, "rlnMagMat11": 1.0}
DEFORMATION = ("rlnTomoDeformationType", "rlnTomoDeformationCoefficients")


@dataclass(frozen=True)
class Problem:
    subject: str  # a job option or a STAR column
    message: str
    file: str | None = None

    def __str__(self) -> str:
        where = f" ({self.file})" if self.file else ""
        return f"{self.subject}: {self.message}{where}"


class UnsupportedError(ValueError):
    """Raised by the programs and wrappers when :func:`check` found problems."""

    def __init__(self, job_type: str, problems: list[Problem]):
        self.problems = problems
        super().__init__(f"{job_type} cannot honour this job:\n" + "\n".join(f"  - {p}" for p in problems))


def raise_if(job_type: str, problems: list[Problem]) -> None:
    if problems:
        raise UnsupportedError(job_type, problems)


# ====================== reading ======================


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("yes", "true", "1")


def _number(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip().strip('"')


def resolve_path(value, base: Path | None = None) -> Path | None:
    """A path written by RELION: absolute, else relative to the working directory (the project), else to ``base``."""
    if _blank(value):
        return None
    p = Path(str(value))
    if p.is_absolute() or p.exists() or base is None:
        return p
    return base / p


def project_relative(path: str | Path) -> str:
    """How RELION writes a path: relative to the working directory (the project) when inside it, else absolute."""
    absolute = Path(path).resolve()
    try:
        return str(absolute.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(absolute)


def read_optimisation_set(path: str | Path) -> tuple[Path | None, Path | None, Path | None]:
    """(particles, tomograms, trajectories) named by an optimisation set.

    RELION and zarr-particle-extract write the set as one block of key-value pairs (``starfile`` reads a dict of
    values); a single-row table is accepted too. An empty entry (RELION writes ``""``) is absent.
    """
    data = starfile.read(str(path))
    if isinstance(data, Mapping) and not any(hasattr(v, "iloc") for v in data.values()):
        row = data
    else:
        table = next(iter(data.values())) if isinstance(data, Mapping) else data
        if len(table) != 1:
            raise ValueError(f"{path}: an optimisation set holds one row, found {len(table)}")
        row = table.iloc[0].to_dict()
    base = Path(path).parent
    return tuple(resolve_path(row.get(k), base) for k in _SET_KEYS)  # type: ignore[return-value]


_SET_KEYS = ("rlnTomoParticlesFile", "rlnTomoTomogramsFile", "rlnTomoTrajectoriesFile")


def _blocks(path: Path) -> dict:
    data = starfile.read(str(path), always_dict=True)
    return data


def _nontrivial_vector(value) -> bool:
    """A RELION vector column (``[0.1,0,...]``) with any non-zero entry."""
    if _blank(value):
        return False
    text = str(value).strip().strip("[]")
    return any(abs(_number(x, 0.0)) > 0 for x in text.replace(" ", ",").split(",") if x)


# ====================== checks ======================


def check_options(job_type: str, options: Mapping[str, object]) -> list[Problem]:
    """Job option values the program does not implement, by pipeliner option name."""
    get = options.get
    problems: list[Problem] = []
    if job_type in PYTHON_IMPLEMENTED:
        # RELION reads the direct entries when do_use_direct_entries is Yes (pipeliner's default for particle
        # reconstruction); the program reads an optimisation set whenever one is named. Only naming both
        # disagrees, so only that is refused.
        direct_given = any(not _blank(get(k)) for k in ("in_particles", "in_tomograms"))
        if _truthy(get("do_use_direct_entries", "No")) and direct_given and not _blank(get("in_optimisation")):
            problems.append(
                Problem(
                    "do_use_direct_entries",
                    "is Yes and both an optimisation set and direct entries are named; RELION would read the "
                    "entries, the program reads the optimisation set, so name one input form",
                )
            )
    if job_type == EXTRACT:
        if _number(get("max_dose", -1), -1) > 0:
            problems.append(Problem("max_dose", "is not implemented (keep it <= 0)"))
        if int(_number(get("min_nr_frames", 1), 1)) != 1:
            problems.append(Problem("min_nr_frames", "is not implemented (keep it 1)"))
        if not _truthy(get("do_output_2dstacks", "Yes")):
            problems.append(Problem("do_output_2dstacks", "3D subtomograms are not implemented; only 2D stacks"))
        if _truthy(get("do_extract_reproject", "No")):
            problems.append(Problem("do_extract_reproject", "reprojected 2D extraction is not implemented"))
    if job_type == POLISH and _truthy(get("do_shift_align", "No")) == _truthy(get("do_motion", "No")):
        problems.append(
            Problem(
                "do_shift_align/do_motion", "exactly one of shift-only alignment and per-particle motion must be Yes"
            )
        )
    return problems


def check_inputs(
    job_type: str,
    particles: str | Path | None = None,
    tomograms: str | Path | None = None,
    trajectories: str | Path | None = None,
    tiltseries_relative_dir: str | Path | None = None,
) -> list[Problem]:
    """Input metadata the program would ignore, and tilt series it could not read.

    Per-tilt STAR paths resolve as the program resolves them: against ``tiltseries_relative_dir`` when one is
    given, else against the working directory and then the tomograms STAR's directory. Particles of tomograms the
    set does not list are the programs' concern (warned about and skipped, as RELION does), not a refusal here.
    """
    problems: list[Problem] = []

    if tomograms is not None:
        tomograms = Path(tomograms)
        if not tomograms.exists():
            return [Problem("rlnTomoTomogramsFile", "does not exist", str(tomograms))]
        blocks = _blocks(tomograms)
        global_df = blocks.get("global", next(iter(blocks.values())))
        for _, row in global_df.iterrows():
            problems += _check_series(job_type, row, blocks, tomograms, tiltseries_relative_dir)

    if particles is not None:
        particles = Path(particles)
        if not particles.exists():
            return problems + [Problem("rlnTomoParticlesFile", "does not exist", str(particles))]
        blocks = _blocks(particles)
        parts = blocks.get("particles", next(iter(blocks.values())))
        optics = blocks.get("optics")
        if job_type in PYTHON_IMPLEMENTED:
            for column in SUBTOMOGRAM_ORIENTATION:
                if column in parts.columns and (parts[column].astype(float).abs() > 1e-6).any():
                    problems.append(
                        Problem(column, "subtomogram orientations are not applied by the Python implementation")
                    )
                    break
            tables = [t for t in (optics, parts) if t is not None]
            for column in ZERNIKE:
                if any(column in t.columns and t[column].map(_nontrivial_vector).any() for t in tables):
                    problems.append(Problem(column, "higher-order aberrations are not applied", str(particles)))
            for column, identity in MAGNIFICATION.items():
                if any(
                    column in t.columns and (t[column].astype(float) - identity).abs().gt(1e-6).any() for t in tables
                ):
                    problems.append(Problem(column, "anisotropic magnification is not applied", str(particles)))
                    break
        if job_type == RECONSTRUCT and optics is not None:
            geometry = [c for c in ("rlnImagePixelSize", "rlnImageSize", "rlnTomoSubtomogramBinning") if c in optics]
            if geometry and len(optics[geometry].drop_duplicates()) > 1:
                problems.append(
                    Problem("/".join(geometry), "reconstruction needs one pixel size, box and binning", str(particles))
                )

    if trajectories is not None and not Path(trajectories).exists():
        problems.append(Problem("rlnTomoTrajectoriesFile", "does not exist", str(trajectories)))
    return problems


def _as_program_reads(value, tomograms: Path, tiltseries_relative_dir: str | Path | None) -> Path:
    """A per-tilt STAR or stack path as the programs open it: under ``tiltseries_relative_dir`` when given."""
    if tiltseries_relative_dir is None or Path(str(value)).is_absolute():
        return resolve_path(value, tomograms.parent)
    return Path(tiltseries_relative_dir) / str(value)


def _check_series(
    job_type: str, row, blocks: Mapping, tomograms: Path, tiltseries_relative_dir: str | Path | None = None
) -> list[Problem]:
    name = str(row["rlnTomoName"])
    star = row.get("rlnTomoTiltSeriesStarFile")
    if _blank(star):
        tilts = blocks.get(name)
        where = str(tomograms)
    else:
        path = _as_program_reads(star, tomograms, tiltseries_relative_dir)
        if not path.exists():
            return [Problem("rlnTomoTiltSeriesStarFile", f"tomogram {name}: {star} does not exist", str(tomograms))]
        tilts = next(iter(_blocks(path).values()))
        where = str(path)
    if tilts is None:
        return [Problem("rlnTomoName", f"tomogram {name}: no tilt-series table", str(tomograms))]
    problems = []
    if TILTSERIES_URI_COLUMN in tilts.columns:
        locators = set(tilts[TILTSERIES_URI_COLUMN].astype(str))
    elif TILTSERIES_URI_COLUMN in row and not _blank(row[TILTSERIES_URI_COLUMN]):
        locators = {str(row[TILTSERIES_URI_COLUMN])}
    else:
        locators = {str(m).split("@", 1)[-1] for m in tilts["rlnMicrographName"]}
        missing = sorted(
            loc for loc in locators if not _as_program_reads(loc, tomograms, tiltseries_relative_dir).exists()
        )
        if missing:
            problems.append(
                Problem(
                    "rlnMicrographName",
                    f"tomogram {name}: no {TILTSERIES_URI_COLUMN} and the stack {missing[0]} does not exist",
                    where,
                )
            )
    if len(locators) > 1:
        problems.append(Problem(TILTSERIES_URI_COLUMN, f"tomogram {name}: several tilt-series locators", where))
    if job_type in PYTHON_IMPLEMENTED:
        for column in DEFORMATION:
            if column in tilts.columns and not tilts[column].map(_blank).all():
                problems.append(Problem(column, f"tomogram {name}: 2D deformations are not applied", where))
                break
    return problems


def inputs_from_options(job_type: str, options: Mapping[str, object]) -> tuple[Path | None, Path | None, Path | None]:
    """(particles, tomograms, trajectories) the job's wrapper would hand its program, from its job options.

    The extraction and reconstruction wrappers read an optimisation set whenever one is named (see
    :func:`check_options`); CTF refinement and polishing honour ``use_direct_entries``.
    """
    optimisation = options.get("in_optimisation")
    if job_type in PYTHON_IMPLEMENTED:
        use_set = not _blank(optimisation)
    else:
        use_set = not _blank(optimisation) and not _truthy(options.get("use_direct_entries", "No"))
    if use_set:
        return read_optimisation_set(optimisation)
    return tuple(resolve_path(options.get(k)) for k in ("in_particles", "in_tomograms", "in_trajectories"))  # type: ignore[return-value]


def check(job_type: str, options: Mapping[str, object], resolve_inputs: bool = True) -> list[Problem]:
    """Everything a planner should refuse before submitting ``job_type`` with ``options``."""
    problems = check_options(job_type, options)
    if resolve_inputs:
        try:
            particles, tomograms, trajectories = inputs_from_options(job_type, options)
        except (OSError, ValueError, KeyError) as exc:
            return problems + [Problem("in_optimisation", f"cannot be read: {exc}")]
        problems += check_inputs(job_type, particles, tomograms, trajectories)
    return problems
