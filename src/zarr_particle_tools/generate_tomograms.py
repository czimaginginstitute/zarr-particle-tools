"""
Generate a RELION tomograms.star (S3-zarr tilt series + per-tilt CTF/geometry/dose) from the CryoET
Data Portal or a copick project, for feeding `zarr-particle-ctfrefine`/`zarr-particle-polish local`.

CTF-refine / polish need the tomograms.star (portal/copick-derived) plus your OWN refined particles.star
and reference half-maps (from a prior Refine3D) — so this emits only the tomograms.star (+ per-tomogram
tilt stars), not particles. The job driver hard-errors if the generated rlnTomoName scheme doesn't
overlap your particles, so a mismatch surfaces immediately instead of silently processing nothing.
"""

import logging
from pathlib import Path

import click
import starfile

import zarr_particle_tools.cli.options as cli_options
import zarr_particle_tools.generate.cdp_generate_starfiles as cdp_generate
from zarr_particle_tools.core.helpers import setup_logging
from zarr_particle_tools.generate.copick_generate_starfiles import get_copick_picks

logger = logging.getLogger(__name__)

_NEXT_STEPS = (
    "Feed this to `zarr-particle-ctfrefine local` / `zarr-particle-polish local` with "
    "--tomograms-starfile <this>, your refined --particles-starfile, and --ref1/--ref2."
)


def generate_data_portal_tomograms(output_dir, **data_portal_args) -> Path:
    """Emit tomograms.star from a CryoET Data Portal query (reuses the extract generator)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _particles, tomograms_path, _tiltseries = cdp_generate.generate_starfiles(output_dir=output_dir, **data_portal_args)
    logger.info("Wrote %s. %s", tomograms_path, _NEXT_STEPS)
    return tomograms_path


def generate_copick_data_portal_tomograms(
    output_dir,
    copick_config,
    copick_name,
    copick_session_id,
    copick_user_id,
    copick_run_names=None,
    copick_dataset_ids=None,
) -> Path:
    """Emit tomograms.star for the runs referenced by a copick project (Data Portal-backed)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not copick_run_names:
        picks = get_copick_picks(copick_config, copick_name, copick_session_id, copick_user_id, copick_run_names)
        copick_run_names = [p.run.name for p in picks]
    run_ids = [int(s) for s in copick_run_names if s.isdigit()]
    if len(run_ids) != len(copick_run_names):
        raise ValueError("All copick runs must be nonnegative integers (Data Portal run IDs).")
    _ids, _optics, tomograms_path, _tiltseries = cdp_generate.generate_tomograms_from_runs(
        run_ids=run_ids, dataset_ids=copick_dataset_ids, output_dir=output_dir
    )
    logger.info("Wrote %s. %s", tomograms_path, _NEXT_STEPS)
    return tomograms_path


JOB_INPUT_SUBDIR = "input"


def tomograms_star_for_job(output_dir, data_portal_args=None, copick_args=None) -> Path:
    """
    Generate a tomograms.star under <output_dir>/input for a CTF-refine / polish run, so the
    generated inputs sit alongside (not inside) the RELION results. Returns its path.
    """
    input_dir = Path(output_dir) / JOB_INPUT_SUBDIR
    if copick_args is not None:
        path = generate_copick_data_portal_tomograms(output_dir=input_dir, **copick_args)
    else:
        path = generate_data_portal_tomograms(output_dir=input_dir, **(data_portal_args or {}))
    return _absolutize_tiltseries_paths(path)


def _absolutize_tiltseries_paths(tomograms_star: Path) -> Path:
    """
    Rewrite rlnTomoTiltSeriesStarFile to absolute paths.

    The generator writes it relative to the *project root* (<output_dir>.parent), which is right for
    the pipeline (py2rely runs with cwd=<output_dir>). The CTF-refine / polish jobs get no such cwd:
    they try the value cwd-relative and then relative to the tomograms.star's own directory, so a
    relative value either misses entirely (landing on <output_dir>/input/input/tiltseries/...) or, if
    a same-named tiltseries/ exists under the cwd, silently resolves to the wrong job's tilt star.
    Absolute paths remove both hazards; the job driver rewrites this column for its own outputs
    anyway, so nothing downstream depends on it staying relative.
    """
    data = starfile.read(tomograms_star)
    blocks = data if isinstance(data, dict) else {getattr(data, "name", "global") or "global": data}
    base = tomograms_star.parent
    changed = False
    for key, df in blocks.items():
        if df is None or "rlnTomoTiltSeriesStarFile" not in getattr(df, "columns", []):
            continue

        def absolutize(value, base=base):
            p = Path(str(value))
            if p.is_absolute():
                return str(p)
            # the generator prefixes the output dir's own name, so that prefix is relative to base.parent
            root = base.parent if p.parts and p.parts[0] == base.name else base
            return str((root / p).resolve())

        blocks[key] = df.assign(rlnTomoTiltSeriesStarFile=df["rlnTomoTiltSeriesStarFile"].map(absolutize))
        changed = True
    if changed:
        # write back through the block dict so the RELION-required `global` block name survives
        starfile.write(blocks, tomograms_star)
    return tomograms_star


def reject_optimisation_set(optimisation_set_starfile, subcommand: str) -> None:
    """An optimisation set already names its own tomograms.star, so it cannot take a generated one."""
    if optimisation_set_starfile is not None:
        raise click.UsageError(
            f"--optimisation-set-starfile is not supported by `{subcommand}`, because it already "
            "references its own tomograms.star. Use the `local` subcommand for an optimisation set."
        )


@click.group(help="Generate a tomograms.star (S3-zarr tilt series) for CTF-refine / polish.")
def cli():
    pass


@cli.command("data-portal", help="Generate tomograms.star from the CryoET Data Portal.")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Output directory.")
@cli_options.data_portal_options()
@click.option("--debug", is_flag=True, help="Enable debug logging.")
def cmd_data_portal(**kwargs):
    setup_logging(kwargs.pop("debug", False))
    kwargs = cli_options.flatten_data_portal_args(kwargs)
    generate_data_portal_tomograms(**kwargs)


@cli.command("copick-data-portal", help="Generate tomograms.star for a copick project's Data Portal runs.")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Output directory.")
@cli_options.copick_options()
@cli_options.data_portal_copick_options()
@click.option("--debug", is_flag=True, help="Enable debug logging.")
def cmd_copick_data_portal(**kwargs):
    setup_logging(kwargs.pop("debug", False))
    kwargs["copick_run_names"] = cli_options.flatten(kwargs["copick_run_names"])
    kwargs["copick_dataset_ids"] = cli_options.flatten(kwargs["copick_dataset_ids"])
    generate_copick_data_portal_tomograms(**kwargs)


if __name__ == "__main__":
    cli()
