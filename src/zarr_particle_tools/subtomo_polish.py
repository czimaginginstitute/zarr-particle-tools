"""
Run RELION's Bayesian polish / frame alignment (relion_tomo_align) on tilt series stored as OME-Zarr.

Thin wrapper over subtomo_relion_job: supplies the binary name and the align flag builder. align
reads the raw tilt series at the same seam as CTF refinement and is per-tomogram independent (its
finalise just concatenates per-tomogram results), so the two-phase per-tomogram mode is exact and
memory-bounded. Reference half-maps (--ref1/--ref2) from a prior Refine3D are required.
"""

import logging
from pathlib import Path

import click

import zarr_particle_tools.cli.options as cli_options
from zarr_particle_tools.core.helpers import setup_logging
from zarr_particle_tools.generate_tomograms import reject_optimisation_set, tomograms_star_for_job
from zarr_particle_tools.subtomo_relion_job import run_relion_tomo_job

logger = logging.getLogger(__name__)

RELION_BIN = "relion_tomo_align"


def build_align_cmd(relion_bin, opt_set, output_dir, box_size, ref1, ref2, mask, fsc, threads, opts) -> list:
    """Assemble the relion_tomo_align command line from the polish options dict."""
    cmd = [
        relion_bin,
        "--i",
        str(Path(opt_set).resolve()),
        "--ref1",
        str(Path(ref1).resolve()),
        "--ref2",
        str(Path(ref2).resolve()),
        "--b",
        str(box_size),
        "--o",
        str(Path(output_dir).resolve()) + "/",
        "--r",
        str(opts["range"]),
        "--j",
        str(threads),
    ]
    if mask is not None:
        cmd += ["--mask", str(Path(mask).resolve())]
    if fsc is not None:
        cmd += ["--fsc", str(Path(fsc).resolve())]
    if opts["shift_only"]:
        cmd += ["--shift_only"]
    if opts["do_motion"]:
        cmd += ["--motion", "--s_vel", str(opts["s_vel"]), "--s_div", str(opts["s_div"])]
    if opts["do_deformation"]:
        cmd += ["--deformation", "--def_model", str(opts["def_model"])]
    return cmd


def run_polish(
    output_dir: str | Path,
    box_size: int,
    ref1: str | Path,
    ref2: str | Path,
    particles_starfile: Path | None = None,
    tomograms_starfile: Path | None = None,
    trajectories_starfile: Path | None = None,
    optimisation_set_starfile: Path | None = None,
    tiltseries_relative_dir: Path | None = None,
    mask: Path | None = None,
    fsc: Path | None = None,
    do_motion: bool = True,
    s_vel: float = 0.2,
    s_div: float = 5000.0,
    do_deformation: bool = False,
    def_model: str = "spline",
    shift_only: bool = False,
    align_range: int = 20,
    threads: int = 6,
    relion_bin: str = RELION_BIN,
    shm_dir: str | Path = "/dev/shm",
    keep_shm: bool = False,
    per_tomogram: bool = True,
    n_workers: int = 0,
) -> Path:
    """
    Orchestrate a RELION polish (relion_tomo_align) run against zarr tilt series. Returns output dir.

    per_tomogram=True (default): two-phase, <= n_workers tilt series staged at once. align is per-tomogram
    independent, so results match all-at-once. Outputs motion.star (trajectories) + updated
    tomograms.star / particles.star (RELION-native, so they feed a following CTF-refine).
    """
    opts = {
        "do_motion": do_motion,
        "s_vel": s_vel,
        "s_div": s_div,
        "do_deformation": do_deformation,
        "def_model": def_model,
        "shift_only": shift_only,
        "range": align_range,
    }
    return run_relion_tomo_job(
        build_align_cmd,
        relion_bin,
        output_dir,
        box_size,
        ref1,
        ref2,
        opts,
        particles_starfile=particles_starfile,
        tomograms_starfile=tomograms_starfile,
        trajectories_starfile=trajectories_starfile,
        optimisation_set_starfile=optimisation_set_starfile,
        tiltseries_relative_dir=tiltseries_relative_dir,
        mask=mask,
        fsc=fsc,
        threads=threads,
        shm_dir=shm_dir,
        keep_shm=keep_shm,
        per_tomogram=per_tomogram,
        n_workers=n_workers,
    )


@click.group(help="Run RELION Bayesian polish / frame alignment on zarr tilt series.")
def cli():
    pass


@cli.command("local", help="Polish using RELION stars (optimisation set or particles+tomograms) + references.")
@cli_options.local_options()
@cli_options.local_shared_options()
@cli_options.polish_options()
def cmd_local(**kwargs):
    setup_logging(kwargs.pop("debug", False))
    run_polish(**kwargs)


@cli.command(
    "data-portal",
    help="Polish with a tomograms.star generated from the CryoET Data Portal (still needs your refined "
    "--particles-starfile and --ref1/--ref2).",
)
@cli_options.local_options()
@cli_options.polish_options()
@cli_options.data_portal_options()
@cli_options.job_dry_run_option
def cmd_data_portal(**kwargs):
    setup_logging(kwargs.pop("debug", False))
    dry_run = kwargs.pop("dry_run", False)
    reject_optimisation_set(kwargs.pop("optimisation_set_starfile", None), "data-portal")
    portal_args, kwargs = cli_options.split_data_portal_args(kwargs)
    kwargs["tomograms_starfile"] = tomograms_star_for_job(kwargs["output_dir"], data_portal_args=portal_args)
    if dry_run:
        logger.info("Dry run: generated %s; skipping RELION.", kwargs["tomograms_starfile"])
        return kwargs["tomograms_starfile"]
    run_polish(**kwargs)


@cli.command(
    "copick-data-portal",
    help="Polish with a tomograms.star generated for a copick project's Data Portal runs (still needs "
    "your refined --particles-starfile and --ref1/--ref2).",
)
@cli_options.local_options()
@cli_options.polish_options()
@cli_options.copick_options()
@cli_options.data_portal_copick_options()
@cli_options.job_dry_run_option
def cmd_copick_data_portal(**kwargs):
    setup_logging(kwargs.pop("debug", False))
    dry_run = kwargs.pop("dry_run", False)
    reject_optimisation_set(kwargs.pop("optimisation_set_starfile", None), "copick-data-portal")
    copick_args, kwargs = cli_options.split_copick_args(kwargs)
    kwargs["tomograms_starfile"] = tomograms_star_for_job(kwargs["output_dir"], copick_args=copick_args)
    if dry_run:
        logger.info("Dry run: generated %s; skipping RELION.", kwargs["tomograms_starfile"])
        return kwargs["tomograms_starfile"]
    run_polish(**kwargs)


if __name__ == "__main__":
    cli()
