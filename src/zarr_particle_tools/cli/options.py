# TODO: globbing for all fields?
from pathlib import Path
from typing import Any

import click

from zarr_particle_tools.cli.types import INT_LIST, PARAM_TYPE_FOR_TYPE, STR_LIST


def compose_options(opts: list[click.Option]) -> callable:
    def _compose_options(f):
        for opt in reversed(opts):
            f = opt(f)
        return f

    return _compose_options


def common_options():
    opts = [
        click.option("--box-size", type=int, help="Box size of the extracted subtomograms in pixels."),
        click.option(
            "--crop-size",
            type=int,
            default=None,
            help="Crop size of the extracted subtomograms in pixels. If not specified, defaults to box-size.",
        ),
        click.option("--bin", type=int, default=1, show_default=True, help="Binning factor for the subtomograms."),
        click.option("--no-ctf", is_flag=True, help="Disable CTF premultiplication."),
        click.option(
            "--output-dir",
            type=click.Path(file_okay=False, path_type=Path),
            required=True,
            help="Path to the output directory where the extracted subtomograms will be saved.",
        ),
        click.option(
            "--overwrite", is_flag=True, help="If set, existing output files will be overwritten. Default is False."
        ),
        click.option("--debug", is_flag=True, help="Enable debug logging."),
    ]

    return compose_options(opts)


def extract_options():
    opts = [
        click.option(
            "--float16",
            is_flag=True,
            help="Use float16 precision for the output mrcs files. Default is False (float32).",
        ),
        click.option("--circle-precrop", is_flag=True, help="Enable circular precropping of the subtomograms."),
        click.option("--no-circle-crop", is_flag=True, help="Disable circular cropping of the subtomograms."),
        click.option("--no-ic", is_flag=True, help="Do not invert contrast of the subtomograms."),
        click.option(
            "--write-fourier", is_flag=True, help="Write Fourier space stacks (.npy) in addition to real space (.mrcs)."
        ),
        click.option(
            "--dont-apply-offsets",
            is_flag=True,
            help="Do not fold rlnOriginX/Y/ZAngst into the particle coordinates (RELION "
            "--dont_apply_offsets); by default they are applied.",
        ),
    ]
    return compose_options(opts)


def local_options():
    opts = [
        click.option(
            "--optimisation-set-starfile",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="Path to the optimisation set star file for optimisation set generation.",
        ),
        click.option(
            "--particles-starfile",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="Path to the particles *.star file.",
        ),
        click.option(
            "--trajectories-starfile",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="Path to the trajectories motion.star file for motion correction",
        ),
    ]
    return compose_options(opts)


def local_shared_options():
    opts = [
        click.option(
            "--tiltseries-relative-dir",
            type=click.Path(file_okay=True, path_type=Path),
            default=Path("./"),
            show_default=True,
            help="The directory in which the tiltseries file paths are relative to (not needed if absolute paths are used in the starfile or the tiltseries are in the tomograms.star file).",
        ),
        click.option(
            "--tomograms-starfile",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="Path to the tomograms.star file (containing all tiltseries entries, with entries as tiltseries).",
        ),
    ]

    return compose_options(opts)


def dry_run_option(f):
    return click.option(
        "--dry-run",
        is_flag=True,
        help="If set, do not extract subtomograms, only generate the starfiles needed for extraction.",
    )(f)


def job_dry_run_option(f):
    return click.option(
        "--dry-run",
        is_flag=True,
        help="If set, only generate the tomograms.star and stop before running RELION.",
    )(f)


def copick_options():
    opts = [
        click.option(
            "--copick-config",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            required=True,
            help="Path to the copick configuration file.",
        ),
        click.option("--copick-name", type=str, required=True, help="copick particle (object) name"),
        click.option("--copick-session-id", type=str, required=True, help="copick session ID"),
        click.option("--copick-user-id", type=str, required=True, help="copick user ID"),
        click.option(
            "--copick-run-names",
            "--copick-run-name",
            type=STR_LIST,
            multiple=True,
            help="copick run names (default: all runs)",
        ),
    ]
    return compose_options(opts)


def data_portal_copick_options():
    opts = [
        click.option(
            "--copick-dataset-ids",
            "--copick-dataset-id",
            type=INT_LIST,
            multiple=True,
            help="filter copick runs on corresponding CryoET Data Portal dataset IDs (default: all datasets)",
        ),
    ]

    return compose_options(opts)


DATA_PORTAL_ARGS = [
    ("--deposition-ids", int),
    ("--deposition-titles", str),
    ("--dataset-ids", int),
    ("--dataset-titles", str),
    ("--organism-names", str),
    ("--cell-names", str),
    ("--run-ids", int),
    ("--run-names", str),
    ("--tiltseries-ids", int),
    ("--alignment-ids", int),
    ("--tomogram-ids", int),
    ("--annotation-ids", int),
    ("--annotation-names", str),
]

DATA_PORTAL_ARG_REFS = [arg.removeprefix("--").replace("-", "_") for arg, _ in DATA_PORTAL_ARGS] + ["inexact_match"]


# NOTE: not robust since it assumes the plural form is just the singular form with an 's' at the end, which is currently the case but may not always be true
def arg_flags(plural: str) -> tuple[str, str]:
    """Given a plural form of a field, return the argument flags for both plural and singular forms."""
    return plural, plural[:-1]


def help_text(field_name: str, field_type: str, arg_type: type) -> str:
    return f"CryoET Data Portal {field_name} {field_type}(s) to filter picks (comma or space separated). \
        {' If --inexact-match is specified, filtering is case insensitive, contains search is used. NOTE: Not necessarily a unique identifier, results can span different datasets.' if arg_type is str else ''}"


def data_portal_options():
    options: list = []
    options.append(
        click.option(
            "--inexact-match",
            is_flag=True,
            help="Filter using case-insensitive 'contains' search for string fields.",
        )
    )
    options.append(
        click.option(
            "--ground-truth",
            is_flag=True,
            help="If set, only particles from annotations marked as ground truth will be extracted.",
        )
    )
    options.append(
        click.option(
            "--automated-only",
            is_flag=True,
            help="If set, only particles from automated (method_type=automated) annotations are pulled.",
        )
    )
    options.append(
        click.option(
            "--no-orientations",
            is_flag=True,
            help="Zero particle orientations (rlnAngleRot/Tilt/Psi) so poses are determined de novo.",
        )
    )
    options.append(
        click.option(
            "--staging",
            is_flag=True,
            help="Query the staging CryoET Data Portal (GraphQL + authenticated S3) instead of prod. "
            "Omit for prod. Fails hard with a 403 message if staging S3 access is denied.",
        )
    )

    for arg, py_type in DATA_PORTAL_ARGS:
        field_name = arg.removeprefix("--").split("-")[0]
        field_type = arg.removeprefix("--").split("-")[1].rstrip("s")
        help_msg = help_text(field_name, field_type, py_type)

        plural_flag, singular_flag = arg_flags(arg)
        param_type = PARAM_TYPE_FOR_TYPE[py_type]

        options.append(
            click.option(
                plural_flag,
                singular_flag,
                type=param_type,
                multiple=True,
                help=help_msg,
            )
        )

    return compose_options(options)


def configure_portal_endpoint(staging: bool) -> None:
    """Point the shared client + S3 at staging (GraphQL + authenticated S3) or prod."""
    import zarr_particle_tools.core.data as data
    import zarr_particle_tools.generate.cdp_cache as cdp_cache

    if staging:
        cdp_cache.set_api_url(cdp_cache.STAGING_GRAPHQL_URL)
        data.set_s3_anon(False)
    else:
        cdp_cache.set_api_url(None)
        data.set_s3_anon(True)


def flatten(val: Any) -> list:
    "Flattens a list of lists to a single list."
    if isinstance(val, (list, tuple)) and val and isinstance(val[0], (list, tuple)):
        return [item for chunk in val for item in chunk]
    else:
        return val


def flatten_data_portal_args(kwargs: dict) -> dict:
    "Flatten the data portal list args and point client/S3 at staging or prod (--staging)."
    configure_portal_endpoint(kwargs.pop("staging", False))
    for ref in DATA_PORTAL_ARG_REFS:
        if val := kwargs.get(ref):
            kwargs[ref] = flatten(val)

    return kwargs


# Everything generate_starfiles() accepts, i.e. the portal query + how to render the picks.
DATA_PORTAL_GENERATION_KEYS = DATA_PORTAL_ARG_REFS + ["ground_truth", "automated_only", "no_orientations"]
COPICK_GENERATION_KEYS = [
    "copick_config",
    "copick_name",
    "copick_session_id",
    "copick_user_id",
    "copick_run_names",
    "copick_dataset_ids",
]


def split_data_portal_args(kwargs: dict) -> tuple[dict, dict]:
    """Split a command's kwargs into (portal generation args, everything else)."""
    kwargs = flatten_data_portal_args(dict(kwargs))
    portal = {k: kwargs.pop(k) for k in DATA_PORTAL_GENERATION_KEYS if k in kwargs}
    return portal, kwargs


def split_copick_args(kwargs: dict) -> tuple[dict, dict]:
    """Split a command's kwargs into (copick generation args, everything else)."""
    kwargs = dict(kwargs)
    copick = {k: kwargs.pop(k) for k in COPICK_GENERATION_KEYS if k in kwargs}
    for k in ("copick_run_names", "copick_dataset_ids"):
        if k in copick:
            copick[k] = flatten(copick[k])
    return copick, kwargs


def ctfrefine_options():
    opts = [
        click.option(
            "--output-dir",
            type=click.Path(file_okay=False, path_type=Path),
            required=True,
            help="Output directory for RELION CTF-refinement results.",
        ),
        click.option("--box-size", type=int, required=True, help="Box size in pixels (RELION --b, compulsory)."),
        click.option(
            "--ref1",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            required=True,
            help="Reference half-map 1 (.mrc) from a prior Refine3D (RELION --ref1).",
        ),
        click.option(
            "--ref2",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            required=True,
            help="Reference half-map 2 (.mrc) (RELION --ref2).",
        ),
        click.option(
            "--mask",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="Reference mask (.mrc), optional (RELION --mask).",
        ),
        click.option(
            "--fsc",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="PostProcess FSC star, optional (RELION --fsc).",
        ),
        click.option("--do-defocus", is_flag=True, help="Refine per-tilt astigmatic defocus."),
        click.option("--do-reg-defocus", is_flag=True, help="Regularise defocus across tilts (needs --do-defocus)."),
        click.option("--lambda-reg", type=float, default=0.1, show_default=True, help="Defocus regularisation weight."),
        click.option("--do-scale", is_flag=True, help="Refine contrast scale."),
        click.option("--per-frame-scale", is_flag=True, help="Scale per frame (no Lambert model)."),
        click.option("--per-tomogram-scale", is_flag=True, help="Scale per tomogram."),
        click.option("--do-even-aberrations", is_flag=True, help="Refine even higher-order aberrations."),
        click.option("--do-odd-aberrations", is_flag=True, help="Refine odd higher-order aberrations."),
        click.option(
            "--focus-range", type=float, default=3000.0, show_default=True, help="Defocus search range [A] (--d0/--d1)."
        ),
        click.option("--threads", "-j", type=int, default=6, show_default=True, help="OMP threads (RELION --j)."),
        click.option(
            "--relion-bin",
            type=str,
            default="relion_tomo_refine_ctf",
            show_default=True,
            help="Path/name of the relion_tomo_refine_ctf binary.",
        ),
        click.option(
            "--shm-dir",
            type=click.Path(file_okay=False, path_type=Path),
            default=Path("/dev/shm"),
            show_default=True,
            help="RAM-backed dir for the materialized tilt-series MRCs.",
        ),
        click.option(
            "--per-tomogram/--all-at-once",
            default=True,
            show_default=True,
            help="Two-phase per-tomogram (bounded RAM); all-at-once keeps all tilt series in RAM.",
        ),
        click.option(
            "--n-workers",
            type=int,
            default=0,
            show_default=True,
            help="Parallel tomograms in phase 1 (0=auto: ~1/4 cores, capped 16; peak RAM ~ n-workers x tilt series).",
        ),
        click.option("--keep-shm", is_flag=True, help="Keep the materialized /dev/shm MRCs (debug)."),
        click.option("--debug", is_flag=True, help="Enable debug logging."),
    ]
    return compose_options(opts)


def polish_options():
    opts = [
        click.option(
            "--output-dir",
            type=click.Path(file_okay=False, path_type=Path),
            required=True,
            help="Output directory for RELION polish results.",
        ),
        click.option("--box-size", type=int, required=True, help="Box size in pixels (RELION --b, compulsory)."),
        click.option(
            "--ref1",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            required=True,
            help="Reference half-map 1 (.mrc) from a prior Refine3D (RELION --ref1).",
        ),
        click.option(
            "--ref2",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            required=True,
            help="Reference half-map 2 (.mrc) (RELION --ref2).",
        ),
        click.option(
            "--mask",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="Reference mask (.mrc), optional.",
        ),
        click.option(
            "--fsc",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="PostProcess FSC star, optional.",
        ),
        click.option(
            "--do-motion/--no-motion",
            default=True,
            show_default=True,
            help="Estimate per-particle motion trajectories (Bayesian polish).",
        ),
        click.option(
            "--s-vel", type=float, default=0.2, show_default=True, help="Motion velocity sigma [A/dose] (--s_vel)."
        ),
        click.option(
            "--s-div", type=float, default=5000.0, show_default=True, help="Motion divergence sigma [A] (--s_div)."
        ),
        click.option("--do-deformation", is_flag=True, help="Estimate 2D deformations (--deformation)."),
        click.option(
            "--def-model",
            type=click.Choice(["linear", "spline", "Fourier"]),
            default="spline",
            show_default=True,
            help="Deformation model.",
        ),
        click.option(
            "--shift-only", is_flag=True, help="Only apply a rigid shift per frame (no iterative optimisation)."
        ),
        click.option(
            "--align-range", type=int, default=20, show_default=True, help="Max particle shift [px] (RELION --r)."
        ),
        click.option("--threads", "-j", type=int, default=6, show_default=True, help="OMP threads (RELION --j)."),
        click.option(
            "--relion-bin",
            type=str,
            default="relion_tomo_align",
            show_default=True,
            help="Path/name of relion_tomo_align.",
        ),
        click.option(
            "--shm-dir",
            type=click.Path(file_okay=False, path_type=Path),
            default=Path("/dev/shm"),
            show_default=True,
            help="RAM-backed dir for materialized tilt series.",
        ),
        click.option(
            "--per-tomogram/--all-at-once",
            default=True,
            show_default=True,
            help="Two-phase per-tomogram (bounded RAM); all-at-once keeps all in RAM.",
        ),
        click.option(
            "--n-workers",
            type=int,
            default=0,
            show_default=True,
            help="Parallel tomograms in phase 1 (0=auto: ~1/4 cores, capped 16).",
        ),
        click.option("--keep-shm", is_flag=True, help="Keep the materialized /dev/shm MRCs (debug)."),
        click.option("--debug", is_flag=True, help="Enable debug logging."),
    ]
    return compose_options(opts)


def reconstruct_options():
    opts = [
        click.option(
            "--cutoff-fraction",
            type=float,
            default=0.01,
            show_default=True,
            help="Ignore shells for which the dose weight falls below this value.",
        ),
        click.option(
            "--snr",
            type=float,
            default=None,
            help="Assumed signal-to-noise ratio (RELION --SNR). Given, CTF correction uses a Wiener "
            "offset of 1/SNR; omitted, it uses RELION's radial-average heuristic.",
        ),
        click.option(
            "--taper",
            type=float,
            default=10.0,
            show_default=True,
            help="Spherical soft-mask falloff in pixels for the final volume (RELION --taper).",
        ),
        click.option(
            "--symmetry",
            type=str,
            default="C1",
            show_default=True,
            help="Symmetry group to apply during reconstruction (e.g. C1, C2, D2, etc).",
        ),
    ]

    return compose_options(opts)
