"""
Import CryoET Data Portal runs as a RELION 5 tomogram set for zarr-particle-tools programs: ``tomograms.star``,
tilt-series stars and placeholder stacks, with the tilt series read from S3 (``generate/cdp_import_starfiles.py``).
"""

from pathlib import Path

import click

import zarr_particle_tools.cli.options as cli_options
import zarr_particle_tools.generate.cdp_import_starfiles as cdp_import
import zarr_particle_tools.generate.portal_selection as portal_selection
from zarr_particle_tools.core.constants import DEFAULT_AMPLITUDE_CONTRAST, TOMO_HAND_DEFAULT_VALUE, TOMOGRAM_TYPES
from zarr_particle_tools.core.helpers import setup_logging

_ZPT_ONLY = (
    "The set is for zarr-particle-tools programs: stock RELION programs that read pixels see the all-zero "
    "placeholder. Use zarr-particle-export for a project with the tilt series on disk."
)


@click.group(
    help=f"Import CryoET Data Portal runs as a RELION 5 tomogram set whose tilt series stay on S3. {_ZPT_ONLY}"
)
def cli():
    pass


@cli.command(
    "data-portal",
    help=(
        "Write tomograms.star and tilt-series stars for portal runs. Resolves one tomogram per run (which fixes the "
        "alignment and voxel spacing), or takes a stored portal_selection.json and fails if the portal changed "
        f"since it was resolved. STAR paths are relative to the working directory. {_ZPT_ONLY}"
    ),
)
@cli_options.portal_id_options("--dataset-ids", "--run-ids", "--tomogram-ids")
@click.option(
    "--tomogram-type",
    type=click.Choice(TOMOGRAM_TYPES),
    default="default",
    show_default=True,
    help="default = each run's visualization default; otherwise the portal's Tomogram.processing. Tomogram IDs pin "
    "their runs' tomograms; this and the filters below apply to the other runs.",
)
@click.option("--tomogram-software", default="", help="Exact Tomogram.processing_software (e.g. a denoiser).")
@click.option("--reconstruction-method", default="", help="Tomogram.reconstruction_method, e.g. WBP or SART.")
@click.option("--voxel-spacing", type=float, default=0.0, show_default=True, help="Exact voxel spacing (A); 0 = any.")
@click.option(
    "--selection",
    "selection_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A stored portal_selection.json to import exactly; excludes the ID and filter options.",
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
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Output directory.")
@cli_options.staging_option()
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
    if selection_path and (filters or tomogram_type != "default"):
        raise click.UsageError("--selection excludes the ID and filter options")
    try:
        if selection_path:
            selection = cdp_import.verify_unchanged(portal_selection.Selection.read(selection_path))
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
        cdp_import.import_selection(selection, output_dir, amplitude_contrast, int(hand))
    except (cdp_import.PortalImportError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    cli()
