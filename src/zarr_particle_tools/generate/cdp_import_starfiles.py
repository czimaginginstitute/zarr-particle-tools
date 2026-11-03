"""
Build and write a RELION 5 tomogram set from resolved portal runs (see ``portal_selection``): ``tomograms.star``,
one tilt-series star per tomogram and sparse placeholder stacks. Tilt-series pixels are read from
``tomoTiltSeriesURI``; the placeholders hold none.
"""

import datetime
import json
import logging
import os
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cryoet_data_portal as cdp
import mrcfile
import numpy as np
import pandas as pd
import starfile

import zarr_particle_tools
import zarr_particle_tools.generate.cdp_cache as cdp_cache
from zarr_particle_tools.core.constants import (
    DEFAULT_AMPLITUDE_CONTRAST,
    INDIVIDUAL_TOMOGRAM_ALN_COLUMNS,
    INDIVIDUAL_TOMOGRAM_COLUMNS,
    INDIVIDUAL_TOMOGRAM_CTF_COLUMNS,
    OPTICS_DF_COLUMNS,
    TILTSERIES_MRCS_PLACEHOLDER,
    TILTSERIES_URI_RELION_COLUMN,
    TOMO_HAND_DEFAULT_VALUE,
    TOMOGRAM_URI_COLUMN,
)
from zarr_particle_tools.core.forwardprojection import in_plane_rotation_to_tilt_axis_rotation
from zarr_particle_tools.core.helpers import get_optics_group_name, project_relative
from zarr_particle_tools.generate.portal_selection import SELECTION_FILENAME, RunSelection, Selection, resolve

logger = logging.getLogger(__name__)

TOMOGRAMS_STAR = "tomograms.star"
TOMOGRAMS_STAR_COLUMNS = OPTICS_DF_COLUMNS + [
    "rlnMicrographOriginalPixelSize",
    "rlnTomoHand",
    "rlnTomoTiltSeriesStarFile",
    "rlnTomoSizeX",
    "rlnTomoSizeY",
    "rlnTomoSizeZ",
    TILTSERIES_URI_RELION_COLUMN,
    "rlnTomoImportFractionalDose",
    "rlnTomoTomogramBinning",
    TOMOGRAM_URI_COLUMN,
]


class PortalImportError(RuntimeError):
    """The runs cannot be imported, with the reason."""


@dataclass
class TomogramSet:
    tomograms: pd.DataFrame  # the global table of tomograms.star
    tilt_series: dict[str, pd.DataFrame]  # rlnTomoName -> per-tilt table
    placeholders: dict[str, tuple[tuple[int, int, int], float]]  # path in the output dir -> (z, y, x) shape, pixel size


def per_section_ctf_df(per_section_parameters: list[cdp.PerSectionParameters], frames: list[cdp.Frame]) -> pd.DataFrame:
    """Per-tilt CTF and pre-exposure rows of one tilt series, keyed by RELION's 1-based ``z_index``."""
    per_section_parameters_df = pd.DataFrame(columns=INDIVIDUAL_TOMOGRAM_CTF_COLUMNS)
    for param in per_section_parameters:
        frame = next((f for f in frames if f.id == param.frame_id), None)
        per_section_parameters_df.loc[len(per_section_parameters_df)] = {
            "z_index": param.z_index + 1,  # match RELION's 1-based indexing
            "rlnDefocusU": param.major_defocus,
            "rlnDefocusV": param.minor_defocus,
            "rlnDefocusAngle": param.astigmatic_angle,
            "rlnPhaseShift": param.phase_shift * 180.0 / np.pi,  # match RELION deg convention
            "rlnCtfMaxResolution": param.max_resolution,
            "rlnMicrographPreExposure": frame.accumulated_dose,
        }
    return per_section_parameters_df


def per_section_alignment_df(
    per_section_alignment_parameters: list[cdp.PerSectionAlignmentParameters], tiltseries_pixel_size: float
) -> pd.DataFrame:
    """Per-tilt projection geometry rows of one alignment, keyed by RELION's 1-based ``z_index``."""
    return pd.DataFrame(
        columns=INDIVIDUAL_TOMOGRAM_ALN_COLUMNS,
        data=[
            {
                "z_index": param.z_index + 1,  # match RELION's 1-based indexing
                "rlnTomoXTilt": param.volume_x_rotation,  # param.x_rotation offset (AreTomo3 beta) is not applied, as per AreTomo3 convention
                "rlnTomoYTilt": param.tilt_angle,  # already accounts for any tilt offset (AreTomo3 alpha)
                "rlnTomoZRot": in_plane_rotation_to_tilt_axis_rotation(np.array(param.in_plane_rotation)),
                # the portal stores x/y_offset in pixels, though its schema says A (cryoet-data-portal-backend#848)
                "rlnTomoXShiftAngst": param.x_offset * tiltseries_pixel_size,
                "rlnTomoYShiftAngst": param.y_offset * tiltseries_pixel_size,
            }
            for param in per_section_alignment_parameters
        ],
    )


def write_tiltseries_placeholder(
    output_dir: Path, shape: tuple[int, int, int], pixel_size: float, name: str = TILTSERIES_MRCS_PLACEHOLDER
) -> Path:
    """Write a sparse all-zero MRC stack of ``shape`` (z, y, x) for ``rlnMicrographName`` to point to."""
    path = Path(output_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new_mmap(path, shape=shape, mrc_mode=2, overwrite=True) as mrc:
        mrc.voxel_size = (pixel_size, pixel_size, 1.0)
    return path


def _placeholders(runs: list[RunSelection], tiltseries: dict[int, cdp.TiltSeries]) -> tuple[dict[str, str], dict]:
    """One placeholder per distinct tilt-image width, height and pixel size, because RELION reads the image size from
    the header. Returns ``rlnTomoName -> placeholder`` and ``placeholder -> (shape, pixel size)``."""
    key_of = {
        r.tomo_name: (tiltseries[r.tiltseries_id].size_x, tiltseries[r.tiltseries_id].size_y, r.tiltseries_pixel_size)
        for r in runs
    }
    depth: dict[tuple, int] = {}
    for r in runs:
        key = key_of[r.tomo_name]
        depth[key] = max(depth.get(key, 0), tiltseries[r.tiltseries_id].size_z)
    stem = TILTSERIES_MRCS_PLACEHOLDER.removesuffix(".mrcs")
    names = {
        key: TILTSERIES_MRCS_PLACEHOLDER if len(depth) == 1 else f"{stem}_{key[0]}x{key[1]}_{key[2]:g}A.mrcs"
        for key in depth
    }
    shapes = {names[key]: ((depth[key], key[1], key[0]), key[2]) for key in depth}
    return {tomo_name: names[key] for tomo_name, key in key_of.items()}, shapes


def get_optics_df(runs: list[RunSelection], amplitude_contrast: float = DEFAULT_AMPLITUDE_CONTRAST) -> pd.DataFrame:
    """One row per tomogram; one optics group per tilt series (``get_optics_group_name``), numbered in row order."""
    optics_df = pd.DataFrame(
        [
            {
                "rlnOpticsGroup": 0,
                "rlnOpticsGroupName": get_optics_group_name(r.run_id, r.tiltseries_id),
                "rlnSphericalAberration": r.spherical_aberration_mm,
                "rlnVoltage": r.voltage_kv,
                "rlnAmplitudeContrast": amplitude_contrast,
                "rlnTomoTiltSeriesPixelSize": r.tiltseries_pixel_size,
                "rlnTomoName": r.tomo_name,
            }
            for r in runs
        ],
        columns=OPTICS_DF_COLUMNS,
    )
    numbers = {name: i for i, name in enumerate(dict.fromkeys(optics_df["rlnOpticsGroupName"]), start=1)}
    optics_df["rlnOpticsGroup"] = optics_df["rlnOpticsGroupName"].map(numbers)
    return optics_df


def get_tomograms_df(
    runs: list[RunSelection],
    output_dir: Path,
    project_dir: Path | None = None,
    amplitude_contrast: float = DEFAULT_AMPLITUDE_CONTRAST,
    hand: int = TOMO_HAND_DEFAULT_VALUE,
) -> pd.DataFrame:
    """The global table of ``tomograms.star``; paths are relative to ``project_dir`` (default: the working
    directory) when inside it, else absolute."""
    tomograms_df = get_optics_df(runs, amplitude_contrast)
    tomograms_df["rlnMicrographOriginalPixelSize"] = [r.tiltseries_pixel_size for r in runs]
    tomograms_df["rlnTomoHand"] = hand
    tomograms_df["rlnTomoTiltSeriesStarFile"] = [
        project_relative(Path(output_dir) / "tiltseries" / f"{r.tomo_name}.star", project_dir) for r in runs
    ]
    for i, axis in enumerate("XYZ"):
        tomograms_df[f"rlnTomoSize{axis}"] = [int(r.rln_tomo_size[i]) for r in runs]
    tomograms_df[TILTSERIES_URI_RELION_COLUMN] = [r.tiltseries_uri for r in runs]
    tomograms_df["rlnTomoImportFractionalDose"] = [r.fractional_dose for r in runs]
    tomograms_df["rlnTomoTomogramBinning"] = [r.voxel_spacing / r.tiltseries_pixel_size for r in runs]
    tomograms_df[TOMOGRAM_URI_COLUMN] = [r.tomogram_uri for r in runs]
    if tomograms_df.isna().any().any():
        bad = tomograms_df.loc[tomograms_df.isna().any(axis=1), "rlnTomoName"].tolist()
        raise PortalImportError(
            f"missing values in {sorted(tomograms_df.columns[tomograms_df.isna().any()])} for {bad}"
        )
    return tomograms_df[TOMOGRAMS_STAR_COLUMNS]


def build(
    runs: list[RunSelection],
    output_dir: str | Path,
    project_dir: str | Path | None = None,
    amplitude_contrast: float = DEFAULT_AMPLITUDE_CONTRAST,
    hand: int = TOMO_HAND_DEFAULT_VALUE,
) -> TomogramSet:
    """The tables and placeholders for ``runs``, checked, without writing anything. Records come from ``cdp_cache``,
    which still holds what ``portal_selection`` fetched."""
    if not runs:
        raise PortalImportError("there are no runs to import")
    if hand not in (-1, 1):
        raise PortalImportError(f"hand must be -1 or 1, not {hand}")
    duplicated = sorted(name for name, n in Counter(r.tomo_name for r in runs).items() if n > 1)
    if duplicated:
        raise PortalImportError(f"tomograms listed more than once: {duplicated}")
    output_dir = Path(output_dir)
    tiltseries_ids = sorted({r.tiltseries_id for r in runs})
    alignment_ids = sorted({r.alignment_id for r in runs})
    tiltseries = {ts.id: ts for ts in cdp_cache.get_tiltseries(tiltseries_ids)}
    ctf_by_ts = cdp_cache.get_all_per_section_parameters_by_tiltseries_id(tiltseries_ids)
    aln_by_alignment = cdp_cache.get_per_section_alignments_by_alignment_id(alignment_ids)
    frames_by_run = cdp_cache.get_frames_by_run_id(sorted({r.run_id for r in runs}))
    placeholder_of, placeholders = _placeholders(runs, tiltseries)

    tilt_series: dict[str, pd.DataFrame] = {}
    for run in runs:
        written = set(run.written_z_indices)
        ctf = per_section_ctf_df(
            [p for p in ctf_by_ts.get(run.tiltseries_id, []) if p.z_index in written], frames_by_run.get(run.run_id, [])
        )
        aln = per_section_alignment_df(
            [p for p in aln_by_alignment.get(run.alignment_id, []) if p.z_index in written], run.tiltseries_pixel_size
        )
        df = pd.merge(ctf, aln, on="z_index", how="inner").sort_values("z_index")
        if sorted(df["z_index"].astype(int) - 1) != sorted(written):
            raise PortalImportError(
                f"[Run {run.run_id}] {len(df)} tilts built for {len(written)} writable sections; "
                "the portal's per-section records changed since the selection was resolved"
            )
        placeholder = project_relative(output_dir / placeholder_of[run.tomo_name], project_dir)
        df["rlnMicrographName"] = [f"{int(z)}@{placeholder}" for z in df["z_index"]]
        df[TILTSERIES_URI_RELION_COLUMN] = run.tiltseries_uri
        df = df.drop(columns=["z_index"])[INDIVIDUAL_TOMOGRAM_COLUMNS]
        if df.isna().any().any():
            raise PortalImportError(f"[Run {run.run_id}] missing values in {sorted(df.columns[df.isna().any()])}")
        tilt_series[run.tomo_name] = df
    tomograms_df = get_tomograms_df(runs, output_dir, project_dir, amplitude_contrast, hand)
    return TomogramSet(tomograms_df, tilt_series, placeholders)


def write(tomogram_set: TomogramSet, output_dir: str | Path, extra_files: dict[str, str] | None = None) -> Path:
    """
    Write ``tomogram_set`` (and ``extra_files``, name -> text) into ``output_dir``, replacing the tilt-series stars
    and placeholders of an earlier write. Everything is written to a temporary directory first and then renamed into
    place, ``tomograms.star`` last. Returns the path of ``tomograms.star``.
    """
    output_dir = Path(output_dir)
    staging_parent = output_dir if output_dir.is_dir() else output_dir.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = staging_parent / f".import-{uuid.uuid4().hex}"  # mkdir, unlike mkdtemp, honors the umask
    try:
        (staging / "tiltseries").mkdir(parents=True)
        for tomo_name, df in tomogram_set.tilt_series.items():
            starfile.write({tomo_name: df}, staging / "tiltseries" / f"{tomo_name}.star", overwrite=True)
        for name, (shape, pixel_size) in tomogram_set.placeholders.items():
            write_tiltseries_placeholder(staging, shape, pixel_size, name)
        for name, text in (extra_files or {}).items():
            (staging / name).write_text(text)
        starfile.write({"global": tomogram_set.tomograms}, staging / TOMOGRAMS_STAR, overwrite=True)

        if not output_dir.exists():
            staging.rename(output_dir)
        else:
            new = {p.relative_to(staging) for p in staging.rglob("*") if p.is_file()}
            tiltseries_dir = output_dir / "tiltseries"
            placeholders = tiltseries_dir.glob(Path(TILTSERIES_MRCS_PLACEHOLDER).stem + "*.mrcs")
            for stale in [*tiltseries_dir.glob("*.star"), *placeholders]:
                if stale.relative_to(output_dir) not in new:
                    stale.unlink()
            for rel in sorted(new, key=lambda p: p == Path(TOMOGRAMS_STAR)):
                (output_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / rel, output_dir / rel)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return output_dir / TOMOGRAMS_STAR


def verify_unchanged(expected: Selection) -> Selection:
    """
    Re-resolve a stored selection by its ids and fail if the portal changed underneath it. Returns the current
    selection under the stored criteria and timestamps. A per-run value the stored selection predates is not compared.
    """
    if expected.problems or not expected.runs:
        raise PortalImportError(f"the stored selection is {expected.status}:\n{expected.describe_problems()}")
    api_url = cdp_cache.get_client().url
    if expected.api_url and expected.api_url != api_url:
        raise PortalImportError(
            f"the selection was resolved against {expected.api_url}, not {api_url}; "
            "--staging selects the staging portal"
        )
    current = resolve(run_ids=[r.run_id for r in expected.runs], tomogram_ids=[r.tomogram_id for r in expected.runs])
    if current.problems:
        raise PortalImportError(f"the stored selection no longer resolves:\n{current.describe_problems()}")
    now = {r.run_id: r.__dict__ for r in current.runs}
    changes = []
    for run in expected.runs:
        was = run.__dict__
        diff = sorted(k for k, v in was.items() if v is not None and v != now.get(run.run_id, {}).get(k))
        if diff:
            changes.append(f"run {run.run_id}: {', '.join(diff)}")
    if changes:
        raise PortalImportError("the portal changed since the selection was resolved:\n" + "\n".join(changes))
    current.criteria, current.resolved_at = expected.criteria, expected.resolved_at
    return current


def import_selection(
    selection: Selection,
    output_dir: str | Path,
    amplitude_contrast: float = DEFAULT_AMPLITUDE_CONTRAST,
    hand: int = TOMO_HAND_DEFAULT_VALUE,
) -> Path:
    """Write the set for a settled selection, with paths relative to the working directory, plus the selection and
    what was written from it as ``portal_selection.json``. Returns the path of ``tomograms.star``."""
    if selection.problems or not selection.runs:
        raise PortalImportError(f"the selection is {selection.status}:\n{selection.describe_problems()}")
    output_dir = Path(output_dir)
    tomogram_set = build(selection.runs, output_dir, None, amplitude_contrast, hand)
    record = selection.to_dict()
    record["import"] = {
        "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "zarr_particle_tools_version": zarr_particle_tools.__version__,
        "amplitude_contrast": amplitude_contrast,
        "hand": hand,
        "tomograms_star": project_relative(output_dir / TOMOGRAMS_STAR),
        "runs": {
            run.tomo_name: {
                "tilt_series_star": project_relative(output_dir / "tiltseries" / f"{run.tomo_name}.star"),
                "sections_supplied": len(run.sections),
                "sections_written": len(tomogram_set.tilt_series[run.tomo_name]),
                "sections_excluded": [
                    {"z_index": s.z_index, "reason": s.reason} for s in run.sections if not s.written
                ],
            }
            for run in selection.runs
        },
    }
    tomograms_path = write(tomogram_set, output_dir, {SELECTION_FILENAME: json.dumps(record, indent=2) + "\n"})
    n_tilts = sum(len(df) for df in tomogram_set.tilt_series.values())
    logger.info(f"Imported {len(selection.runs)} run(s), {n_tilts} tilt(s) into {output_dir}; pixels stay on S3.")
    return tomograms_path
