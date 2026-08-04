"""
Run RELION's CTF refinement (relion_tomo_refine_ctf) on tilt series stored as OME-Zarr.

Thin wrapper over subtomo_relion_job: it only supplies the binary name and the CTF-refine flag
builder. Reference half-maps (--ref1/--ref2) from a prior Refine3D are mandatory (RELION requires
them). See docs/audit/phase3_4_ctfrefine_polish_design.md.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import click

import zarr_particle_tools.cli.options as cli_options
from zarr_particle_tools.core.helpers import setup_logging
from zarr_particle_tools.generate_tomograms import reject_optimisation_set, tomograms_star_for_job
from zarr_particle_tools.subtomo_relion_job import run_relion_tomo_job

logger = logging.getLogger(__name__)

RELION_BIN = "relion_tomo_refine_ctf"


def build_ctf_cmd(relion_bin, opt_set, output_dir, box_size, ref1, ref2, mask, fsc, threads, opts) -> list:
    """Assemble the relion_tomo_refine_ctf command line from the CTF-refine options dict."""
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
        "--j",
        str(threads),
    ]
    if mask is not None:
        cmd += ["--mask", str(Path(mask).resolve())]
    if fsc is not None:
        cmd += ["--fsc", str(Path(fsc).resolve())]
    if opts["do_defocus"]:
        fr = abs(opts["focus_range"])
        cmd += ["--do_defocus", "--d0", str(-fr), "--d1", str(fr)]
        if opts["do_reg_defocus"]:
            cmd += ["--do_reg_defocus", "--lambda", str(opts["lambda_reg"])]
    elif opts["do_reg_defocus"]:
        logger.warning("--do-reg-defocus has no effect without --do-defocus; ignoring.")
    if opts["do_scale"]:
        cmd += ["--do_scale"]
        if opts["per_frame_scale"]:
            cmd += ["--per_frame_scale"]
        if opts["per_tomogram_scale"]:
            cmd += ["--per_tomogram_scale"]
    if opts["do_even_aberrations"]:
        cmd += ["--do_even_aberrations"]
    if opts["do_odd_aberrations"]:
        cmd += ["--do_odd_aberrations"]
    return cmd


def run_ctf_refine(
    output_dir: Union[str, Path],
    box_size: int,
    ref1: Union[str, Path],
    ref2: Union[str, Path],
    particles_starfile: Optional[Path] = None,
    tomograms_starfile: Optional[Path] = None,
    trajectories_starfile: Optional[Path] = None,
    optimisation_set_starfile: Optional[Path] = None,
    tiltseries_relative_dir: Optional[Path] = None,
    mask: Optional[Path] = None,
    fsc: Optional[Path] = None,
    do_defocus: bool = False,
    do_reg_defocus: bool = False,
    lambda_reg: float = 0.1,
    do_scale: bool = False,
    per_frame_scale: bool = False,
    per_tomogram_scale: bool = False,
    do_even_aberrations: bool = False,
    do_odd_aberrations: bool = False,
    focus_range: float = 3000.0,
    threads: int = 6,
    relion_bin: str = RELION_BIN,
    shm_dir: Union[str, Path] = "/dev/shm",
    keep_shm: bool = False,
    per_tomogram: bool = True,
    n_workers: int = 0,
) -> Path:
    """
    Orchestrate a RELION CTF-refinement run against zarr tilt series. Returns the output dir.

    per_tomogram=True (default): two-phase, <= n_workers tilt series in RAM (n_workers=0 auto ~1/4
    cores). Reuses RELION's joint finalise, so results match all-at-once for every fit type
    (defocus / scale / aberrations). per_tomogram=False = single all-at-once run.
    """
    opts = {
        "do_defocus": do_defocus,
        "do_reg_defocus": do_reg_defocus,
        "lambda_reg": lambda_reg,
        "do_scale": do_scale,
        "per_frame_scale": per_frame_scale,
        "per_tomogram_scale": per_tomogram_scale,
        "do_even_aberrations": do_even_aberrations,
        "do_odd_aberrations": do_odd_aberrations,
        "focus_range": focus_range,
    }
    return run_relion_tomo_job(
        build_ctf_cmd,
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


@click.group(help="Run RELION CTF refinement on zarr tilt series.")
def cli():
    pass


@cli.command("local", help="CTF-refine using RELION stars (optimisation set or particles+tomograms) + references.")
@cli_options.local_options()
@cli_options.local_shared_options()
@cli_options.ctfrefine_options()
def cmd_local(**kwargs):
    setup_logging(kwargs.pop("debug", False))
    run_ctf_refine(**kwargs)


@cli.command(
    "data-portal",
    help="CTF-refine with a tomograms.star generated from the CryoET Data Portal (still needs your "
    "refined --particles-starfile and --ref1/--ref2).",
)
@cli_options.local_options()
@cli_options.ctfrefine_options()
@cli_options.data_portal_options()
def cmd_data_portal(**kwargs):
    setup_logging(kwargs.pop("debug", False))
    reject_optimisation_set(kwargs.pop("optimisation_set_starfile", None), "data-portal")
    portal_args, kwargs = cli_options.split_data_portal_args(kwargs)
    kwargs["tomograms_starfile"] = tomograms_star_for_job(kwargs["output_dir"], data_portal_args=portal_args)
    run_ctf_refine(**kwargs)


@cli.command(
    "copick-data-portal",
    help="CTF-refine with a tomograms.star generated for a copick project's Data Portal runs (still "
    "needs your refined --particles-starfile and --ref1/--ref2).",
)
@cli_options.local_options()
@cli_options.ctfrefine_options()
@cli_options.copick_options()
@cli_options.data_portal_copick_options()
def cmd_copick_data_portal(**kwargs):
    setup_logging(kwargs.pop("debug", False))
    reject_optimisation_set(kwargs.pop("optimisation_set_starfile", None), "copick-data-portal")
    copick_args, kwargs = cli_options.split_copick_args(kwargs)
    kwargs["tomograms_starfile"] = tomograms_star_for_job(kwargs["output_dir"], copick_args=copick_args)
    run_ctf_refine(**kwargs)


if __name__ == "__main__":
    cli()
