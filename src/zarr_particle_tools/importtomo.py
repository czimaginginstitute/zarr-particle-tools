"""
Import CryoET Data Portal runs as a RELION 5 tomography set whose pixels stay on S3.

Writes, into ``--output-dir``:

- ``tomograms.star``: one row per run, ``rlnTomoName`` = the portal run id (the run name of a portal-backed copick
  project), ``rlnTomoSizeX/Y/Z`` in unbinned tilt-series pixels (RELION's unit), ``tomoTiltSeriesURI`` = the tilt
  series' OME-Zarr on S3, and one optics group per dataset (``dataset_<id>``), never one per tomogram.
- ``tiltseries/<run id>.star``: per-tilt CTF, projection geometry and pre-exposure. ``rlnMicrographName`` is
  ``N@.../tiltseries_placeholder.mrcs`` where ``N`` is the **source** section (``z_index + 1``), so a section the
  alignment excluded leaves a gap rather than renumbering the rest.
- ``tiltseries/tiltseries_placeholder.mrcs``: one sparse MRC that satisfies RELION's schema and pipeliner's node
  checks. It holds no pixels; readers stream from ``tomoTiltSeriesURI``.
- ``portal_selection.json``: the resolved selection (see :mod:`zarr_particle_tools.portal_selection`) plus what was
  written from it.

Every path written into a STAR file is relative to the working directory (the RELION project), not absolute and
not relative to the output directory. The import either writes all of this or nothing: every row is built and
checked before the first file is written.
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import click
import cryoet_data_portal as cdp
import mrcfile
import pandas as pd
import starfile

import zarr_particle_tools.cli.options as cli_options
import zarr_particle_tools.generate.cdp_cache as cdp_cache
import zarr_particle_tools.portal_selection as portal_selection
from zarr_particle_tools.cli.types import INT_LIST
from zarr_particle_tools.core.constants import (
    DEFAULT_AMPLITUDE_CONTRAST,
    INDIVIDUAL_TOMOGRAM_COLUMNS,
    TILTSERIES_MRCS_PLACEHOLDER,
    TILTSERIES_URI_RELION_COLUMN,
    TOMO_HAND_DEFAULT_VALUE,
)
from zarr_particle_tools.core.helpers import setup_logging
from zarr_particle_tools.generate.cdp_generate_starfiles import per_section_alignment_df, per_section_ctf_df

logger = logging.getLogger(__name__)

TOMOGRAMS_STAR = "tomograms.star"
TOMOGRAM_URI_COLUMN = "tomoTomogramURI"


class ImportError_(RuntimeError):  # noqa: N801 - not the builtin ImportError
    """The selection cannot be imported, with the reason."""


def _project_relative(path: Path) -> str:
    """``path`` relative to the working directory when it is inside it, else absolute."""
    absolute = path.resolve()
    try:
        return str(absolute.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(absolute)


def verify_unchanged(expected: portal_selection.Selection) -> portal_selection.Selection:
    """Re-resolve a stored selection by its explicit ids and fail if the portal changed underneath it."""
    if expected.problems:
        raise ImportError_(f"the stored selection is {expected.status}:\n{expected.describe_problems()}")
    current = portal_selection.resolve(
        run_ids=[r.run_id for r in expected.runs], tomogram_ids=[r.tomogram_id for r in expected.runs]
    )
    if current.problems:
        raise ImportError_(f"the stored selection no longer resolves:\n{current.describe_problems()}")
    was = {r.run_id: asdict(r) for r in expected.runs}
    now = {r.run_id: asdict(r) for r in current.runs}
    changes = []
    for run_id in sorted(was):
        diff = sorted(k for k in was[run_id] if was[run_id][k] != now.get(run_id, {}).get(k))
        if diff:
            changes.append(f"run {run_id}: {', '.join(diff)}")
    if changes:
        raise ImportError_("the portal changed since the selection was resolved:\n" + "\n".join(changes))
    # keep what the caller resolved (criteria, timestamps) as the record
    return expected


def build(
    selection: portal_selection.Selection,
    output_dir: Path,
    amplitude_contrast: float = DEFAULT_AMPLITUDE_CONTRAST,
    hand: int = TOMO_HAND_DEFAULT_VALUE,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The global table and each run's per-tilt table, checked, without writing anything."""
    if selection.problems or not selection.runs:
        raise ImportError_(f"the selection is {selection.status}:\n{selection.describe_problems()}")
    if hand not in (-1, 1):
        raise ImportError_(f"hand must be -1 or 1, not {hand}")
    client = cdp_cache.get_client()
    tiltseries_ids = [r.tiltseries_id for r in selection.runs]
    alignment_ids = [r.alignment_id for r in selection.runs]
    ctf_by_ts: dict[int, list] = {i: [] for i in tiltseries_ids}
    for p in cdp.PerSectionParameters.find(client, [cdp.PerSectionParameters.tiltseries_id._in(tiltseries_ids)]):
        ctf_by_ts[p.tiltseries_id].append(p)
    aln_by_alignment: dict[int, list] = {i: [] for i in alignment_ids}
    for p in cdp.PerSectionAlignmentParameters.find(
        client, [cdp.PerSectionAlignmentParameters.alignment_id._in(alignment_ids)]
    ):
        aln_by_alignment[p.alignment_id].append(p)
    frames_by_run = cdp_cache.get_frames_by_run_id([r.run_id for r in selection.runs])

    placeholder = _project_relative(output_dir / TILTSERIES_MRCS_PLACEHOLDER)
    globals_rows = []
    per_tilt: dict[str, pd.DataFrame] = {}
    for run in selection.runs:
        written = set(run.written_z_indices)
        ctf = per_section_ctf_df(
            [p for p in ctf_by_ts[run.tiltseries_id] if p.z_index in written], frames_by_run[run.run_id]
        )
        aln = per_section_alignment_df(
            [p for p in aln_by_alignment[run.alignment_id] if p.z_index in written], run.tiltseries_pixel_size
        )
        df = pd.merge(ctf, aln, on="z_index", how="inner").sort_values("z_index")
        if sorted(df["z_index"].astype(int) - 1) != sorted(written):
            raise ImportError_(
                f"run {run.run_id}: {len(df)} rows built for {len(written)} writable sections; "
                "the portal's per-section records changed since the selection was resolved"
            )
        df["rlnMicrographName"] = [f"{int(z)}@{placeholder}" for z in df["z_index"]]
        df[TILTSERIES_URI_RELION_COLUMN] = run.tiltseries_uri
        df = df.drop(columns=["z_index"])[INDIVIDUAL_TOMOGRAM_COLUMNS]
        if df.isna().any().any():
            bad = sorted(df.columns[df.isna().any()])
            raise ImportError_(f"run {run.run_id}: missing values in {bad}")
        per_tilt[run.tomo_name] = df
        size_x, size_y, size_z = run.rln_tomo_size
        globals_rows.append(
            {
                "rlnTomoName": run.tomo_name,
                "rlnVoltage": run.voltage_kv,
                "rlnSphericalAberration": run.spherical_aberration_mm,
                "rlnAmplitudeContrast": amplitude_contrast,
                "rlnMicrographOriginalPixelSize": run.tiltseries_pixel_size,
                "rlnTomoHand": hand,
                "rlnOpticsGroupName": None,  # one per dataset, assigned below
                "rlnTomoTiltSeriesPixelSize": run.tiltseries_pixel_size,
                "rlnTomoTiltSeriesStarFile": _project_relative(output_dir / "tiltseries" / f"{run.tomo_name}.star"),
                "rlnTomoSizeX": size_x,
                "rlnTomoSizeY": size_y,
                "rlnTomoSizeZ": size_z,
                TILTSERIES_URI_RELION_COLUMN: run.tiltseries_uri,
                TOMOGRAM_URI_COLUMN: run.tomogram_uri,
            }
        )
    tomograms = pd.DataFrame(globals_rows)
    names = optics_group_names(selection.runs, amplitude_contrast)
    tomograms["rlnOpticsGroupName"] = [names[run.run_id] for run in selection.runs]
    numbers = {name: i for i, name in enumerate(dict.fromkeys(tomograms["rlnOpticsGroupName"]), start=1)}
    tomograms.insert(0, "rlnOpticsGroup", [numbers[n] for n in tomograms["rlnOpticsGroupName"]])
    return tomograms, per_tilt


def optics_group_names(runs: list[portal_selection.RunSelection], amplitude_contrast: float) -> dict[int, str]:
    """One optics group per dataset: ``run_id -> rlnOpticsGroupName``.

    A dataset is one acquisition, and its runs share their optics; one group lets RELION estimate one noise model
    over all of them instead of one per tomogram from a few particles each. A dataset whose runs do *not* share
    voltage, spherical aberration and tilt-series pixel size cannot be one optics group; it is split by those
    values (``dataset_<id>_optics<k>``) and the split is logged.
    """
    by_dataset: dict[int, dict[tuple, list[int]]] = {}
    for run in runs:
        optics = (run.voltage_kv, run.spherical_aberration_mm, amplitude_contrast, run.tiltseries_pixel_size)
        by_dataset.setdefault(run.dataset_id, {}).setdefault(optics, []).append(run.run_id)
    names: dict[int, str] = {}
    for dataset_id, groups in by_dataset.items():
        if len(groups) > 1:
            logger.warning(
                f"dataset {dataset_id}: runs differ in optics (voltage, Cs, Q0, tilt pixel), so it is imported as "
                f"{len(groups)} optics groups: {sorted(groups)}"
            )
        for k, run_ids in enumerate(groups.values(), start=1):
            name = f"dataset_{dataset_id}" if len(groups) == 1 else f"dataset_{dataset_id}_optics{k}"
            names.update(dict.fromkeys(run_ids, name))
    return names


def _write_placeholder(path: Path, selection: portal_selection.Selection) -> None:
    """One sparse MRC deep enough for every run's largest source section; no pixels are ever written into it."""
    tiltseries = cdp_cache.get_tiltseries([r.tiltseries_id for r in selection.runs])
    shape = tuple(max(getattr(ts, f"size_{a}") for ts in tiltseries) for a in ("z", "y", "x"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new_mmap(path, shape=shape, mrc_mode=2, overwrite=True) as mrc:
        px = selection.runs[0].tiltseries_pixel_size
        mrc.voxel_size = (px, px, 1.0)


def write(
    selection: portal_selection.Selection,
    output_dir: str | Path,
    amplitude_contrast: float = DEFAULT_AMPLITUDE_CONTRAST,
    hand: int = TOMO_HAND_DEFAULT_VALUE,
) -> Path:
    """Build, then write the import into ``output_dir``. Returns the path of ``tomograms.star``."""
    output_dir = Path(output_dir)
    tomograms, per_tilt = build(selection, output_dir, amplitude_contrast, hand)
    (output_dir / "tiltseries").mkdir(parents=True, exist_ok=True)
    for tomo_name, df in per_tilt.items():
        starfile.write({tomo_name: df}, output_dir / "tiltseries" / f"{tomo_name}.star", overwrite=True)
    _write_placeholder(output_dir / TILTSERIES_MRCS_PLACEHOLDER, selection)
    tomograms_path = output_dir / TOMOGRAMS_STAR
    starfile.write({"global": tomograms}, tomograms_path, overwrite=True)
    record = selection.to_dict()
    record["import"] = {
        "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "zarr_particle_tools_version": version("zarr-particle-tools"),
        "amplitude_contrast": amplitude_contrast,
        "hand": hand,
        "tomograms_star": _project_relative(tomograms_path),
        "placeholder": _project_relative(output_dir / TILTSERIES_MRCS_PLACEHOLDER),
        "runs": {
            run.tomo_name: {
                "tilt_series_star": _project_relative(output_dir / "tiltseries" / f"{run.tomo_name}.star"),
                "sections_supplied": len(run.sections),
                "sections_written": len(per_tilt[run.tomo_name]),
                "sections_excluded": [
                    {"z_index": s.z_index, "reason": s.reason} for s in run.sections if not s.written
                ],
            }
            for run in selection.runs
        },
    }
    (output_dir / portal_selection.SELECTION_FILENAME).write_text(json.dumps(record, indent=2) + "\n")
    n_tilts = sum(len(df) for df in per_tilt.values())
    logger.info(f"Imported {len(per_tilt)} run(s), {n_tilts} tilt(s) into {output_dir}; pixels stay on S3.")
    return tomograms_path


@click.group(help="Import CryoET Data Portal runs as a RELION 5 tomography set whose tilt series stay on S3.")
def cli():
    pass


@cli.command(
    "data-portal",
    help=(
        "Write tomograms.star + per-tilt stars for portal runs. Resolves one tomogram per run (which fixes the "
        "alignment and voxel spacing), or takes a stored portal_selection.json and fails if the portal changed "
        "since it was resolved."
    ),
)
@click.option("--dataset-ids", type=INT_LIST, multiple=True, help="Every run of these datasets.")
@click.option("--run-ids", type=INT_LIST, multiple=True, help="These runs (narrows --dataset-ids).")
@click.option(
    "--tomogram-ids", type=INT_LIST, multiple=True, help="Pin these tomograms; the filters apply to other runs."
)
@click.option(
    "--tomogram-type",
    type=click.Choice(portal_selection.TOMOGRAM_TYPES),
    default="default",
    show_default=True,
    help="default = each run's visualization default; otherwise the portal's Tomogram.processing.",
)
@click.option("--tomogram-software", default="", help="Exact Tomogram.processing_software (e.g. a denoiser).")
@click.option("--reconstruction-method", default="", help="Tomogram.reconstruction_method, e.g. WBP or SART.")
@click.option("--voxel-spacing", type=float, default=0.0, show_default=True, help="Exact voxel spacing (A); 0 = any.")
@click.option(
    "--selection",
    "selection_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A stored portal_selection.json to import exactly; excludes the id and filter options.",
)
@click.option(
    "--amplitude-contrast",
    type=float,
    default=DEFAULT_AMPLITUDE_CONTRAST,
    show_default=True,
    help="rlnAmplitudeContrast (the portal does not record it).",
)
@click.option(
    "--hand",
    type=click.Choice(["-1", "1"]),
    default=str(TOMO_HAND_DEFAULT_VALUE),
    show_default=True,
    help="rlnTomoHand (the portal does not record the defocus handedness).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Where the import is written; STAR paths are written relative to the working directory.",
)
@click.option(
    "--staging",
    is_flag=True,
    help="Query the staging CryoET Data Portal (GraphQL + authenticated S3) instead of prod.",
)
@click.option("--debug", is_flag=True, help="Enable debug logging.")
def cmd_data_portal(
    dataset_ids,
    run_ids,
    tomogram_ids,
    tomogram_type,
    tomogram_software,
    reconstruction_method,
    voxel_spacing,
    selection_path,
    amplitude_contrast,
    hand,
    output_dir,
    staging,
    debug,
):
    setup_logging(debug)
    cli_options.configure_portal_endpoint(staging)
    dataset_ids, run_ids, tomogram_ids = (cli_options.flatten(v) for v in (dataset_ids, run_ids, tomogram_ids))
    filters = dataset_ids or run_ids or tomogram_ids or tomogram_software or reconstruction_method or voxel_spacing
    try:
        if selection_path:
            if filters or tomogram_type != "default":
                raise click.UsageError("--selection excludes the id and filter options")
            selection = verify_unchanged(portal_selection.Selection.read(selection_path))
        else:
            selection = portal_selection.resolve(
                dataset_ids=dataset_ids,
                run_ids=run_ids,
                tomogram_ids=tomogram_ids,
                tomogram_type=tomogram_type,
                tomogram_software=tomogram_software,
                reconstruction_method=reconstruction_method,
                voxel_spacing=voxel_spacing,
            )
        write(selection, output_dir, amplitude_contrast, int(hand))
    except (ImportError_, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    cli()
