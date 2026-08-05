"""
zarr-particle-export: materialize a self-contained on-disk STA project from the CryoET Data Portal.

Generates the same star files as the extract/orchestrator input step (particles.star + tomograms.star +
per-tomogram tilt stars), then downloads each tilt-series MRC from the portal to disk and rewrites the
tilt stars to point rlnMicrographName at the on-disk MRC (dropping the tomoTiltSeriesURI column). The
result is a standard, portable RELION/py2rely project that runs with stock jobs and needs no portal access.

This is mostly obviated by the zarr-native jobs (which read OME-Zarr directly); it exists for compatibility
and for handing off a portable on-disk project.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import starfile

import zarr_particle_tools.cli.options as cli_options
import zarr_particle_tools.generate.cdp_cache as cdp_cache
import zarr_particle_tools.generate.cdp_generate_starfiles as cdp_generate
import zarr_particle_tools.generate.copick_generate_starfiles as copick_generate
from zarr_particle_tools.core.constants import (
    THREAD_POOL_WORKER_COUNT,
    TILTSERIES_MRCS_PLACEHOLDER,
    TILTSERIES_URI_RELION_COLUMN,
)
from zarr_particle_tools.core.data import global_fs
from zarr_particle_tools.core.helpers import setup_logging

logger = logging.getLogger(__name__)


def _download_s3_file(s3_path: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    global_fs.get(s3_path.removeprefix("s3://"), str(dest))
    return dest


def _repoint_tilt_star(star_path: Path, mrc_path: Path) -> None:
    """Rewrite an individual tilt star: point rlnMicrographName at the on-disk MRC, drop the zarr URI column."""
    df = starfile.read(star_path)
    if isinstance(df, dict):
        df = next(iter(df.values()))
    df = df.copy()
    # keep the original 1-based tilt index, swap only the file the stack lives in
    df["rlnMicrographName"] = df["rlnMicrographName"].str.split("@").str[0] + "@" + str(mrc_path)
    df = df.drop(columns=[TILTSERIES_URI_RELION_COLUMN], errors="ignore")
    starfile.write({star_path.stem: df}, star_path, overwrite=True)


def _materialize_one(output_dir: Path, tomo_name: str) -> None:
    tiltseries_id = int(tomo_name.split("_")[3])  # get_tomo_name: run_R_tiltseries_T_alignment_A_spacing_V
    tiltseries = cdp_cache.get_tiltseries(tiltseries_id)[0]
    if not tiltseries.s3_mrc_file:
        raise click.ClickException(
            f"[TiltSeries {tiltseries_id}] has no MRC file on the portal; cannot export to disk."
        )
    mrc_path = output_dir / "tiltseries" / f"{tomo_name}.mrc"
    logger.info("Downloading %s -> %s", tiltseries.s3_mrc_file, mrc_path)
    _download_s3_file(tiltseries.s3_mrc_file, mrc_path)
    _repoint_tilt_star(output_dir / "tiltseries" / f"{tomo_name}.star", mrc_path)


def materialize_project_to_disk(output_dir: Path) -> None:
    """Download every tilt-series MRC referenced by output_dir/tomograms.star and repoint the tilt stars on disk."""
    output_dir = Path(output_dir)
    tomograms = starfile.read(output_dir / "tomograms.star")
    global_df = tomograms["global"] if isinstance(tomograms, dict) else tomograms
    tomo_names = list(global_df["rlnTomoName"])

    logger.info("Materializing %d tilt series to disk under %s ...", len(tomo_names), output_dir / "tiltseries")
    with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKER_COUNT) as pool:
        list(pool.map(lambda name: _materialize_one(output_dir, name), tomo_names))

    # the zarr placeholder stack is no longer referenced by any tilt star
    (output_dir / TILTSERIES_MRCS_PLACEHOLDER).unlink(missing_ok=True)
    logger.info("Exported on-disk project to %s", output_dir)


def export_data_portal_project(output_dir, no_orientations: bool = False, **data_portal_args) -> Path:
    """Generate the star files from a portal query, then materialize the tilt series to disk."""
    output_dir = Path(output_dir)
    cdp_generate.generate_starfiles(output_dir=output_dir, no_orientations=no_orientations, **data_portal_args)
    materialize_project_to_disk(output_dir)
    return output_dir


def export_copick_data_portal_project(
    output_dir,
    copick_config: Path,
    copick_name: str,
    copick_session_id: str,
    copick_user_id: str,
    copick_run_names: list[str] = None,
    copick_dataset_ids: list[int] = None,
) -> Path:
    """Generate the star files from a Data Portal-backed copick project, then materialize the tilt series to disk."""
    output_dir = Path(output_dir)
    copick_generate.generate_copick_data_portal_starfiles(
        output_dir=output_dir,
        copick_config=copick_config,
        copick_name=copick_name,
        copick_session_id=copick_session_id,
        copick_user_id=copick_user_id,
        copick_run_names=copick_run_names,
        copick_dataset_ids=copick_dataset_ids,
    )
    materialize_project_to_disk(output_dir)
    return output_dir


@click.group(help="Export a self-contained on-disk STA project (downloaded tilt series) from the CryoET Data Portal.")
def cli():
    pass


@cli.command("data-portal", help="Export an on-disk project from a CryoET Data Portal dataset + annotation.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Project directory (particles.star, tomograms.star, tiltseries/ land here).",
)
@cli_options.data_portal_options()
@click.option("--debug", is_flag=True, help="Enable debug logging.")
def cmd_data_portal(**kwargs):
    setup_logging(kwargs.pop("debug", False))
    kwargs = cli_options.flatten_data_portal_args(kwargs)
    output_dir = kwargs.pop("output_dir")
    export_data_portal_project(output_dir, **kwargs)


@cli.command("copick-data-portal", help="Export an on-disk project from a Data Portal-backed copick project.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Project directory (particles.star, tomograms.star, tiltseries/ land here).",
)
@cli_options.copick_options()
@cli_options.data_portal_copick_options()
@click.option("--debug", is_flag=True, help="Enable debug logging.")
def cmd_copick_data_portal(**kwargs):
    setup_logging(kwargs.pop("debug", False))
    kwargs["copick_run_names"] = cli_options.flatten(kwargs["copick_run_names"])
    kwargs["copick_dataset_ids"] = cli_options.flatten(kwargs["copick_dataset_ids"])
    output_dir = kwargs.pop("output_dir")
    export_copick_data_portal_project(output_dir, **kwargs)


if __name__ == "__main__":
    cli()
