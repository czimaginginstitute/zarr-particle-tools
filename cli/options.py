# TODO: fuzzy matching for name fields?
# TODO: globbing for all fields?
from pathlib import Path

import click

from cli.types import PARAM_TYPE_FOR_TYPE, STR_LIST


def compose_options(opts: list[click.Option]) -> callable:
    def _compose_options(f):
        for opt in reversed(opts):
            f = opt(f)
        return f

    return _compose_options


def common_options():
    opts = [
        click.option("--box-size", type=int, required=True, help="Box size of the extracted subtomograms in pixels."),
        click.option(
            "--crop-size",
            type=int,
            default=None,
            help="Crop size of the extracted subtomograms in pixels. If not specified, defaults to box-size.",
        ),
        click.option("--bin", type=int, default=1, show_default=True, help="Binning factor for the subtomograms."),
        click.option(
            "--float16",
            is_flag=True,
            help="Use float16 precision for the output mrcs files. Default is False (float32).",
        ),
        click.option("--no-ctf", is_flag=True, help="Disable CTF premultiplication."),
        click.option("--no-circle-crop", is_flag=True, help="Disable circular cropping of the subtomograms."),
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


def local_options():
    opts = [
        click.option(
            "--particles-starfile",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="Path to the particles *.star file.",
        ),
        click.option(
            "--tiltseries-relative-dir",
            type=click.Path(file_okay=True, path_type=Path),
            default=Path("./"),
            show_default=True,
            help="The directory in which the tiltseries file paths are relative to.",
        ),
        click.option(
            "--tomograms-starfile",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="Path to the tomograms.star file (containing all tiltseries entries, with entries as tiltseries).",
        ),
        click.option(
            "--trajectories-starfile",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="Path to the trajectories motion.star file for motion correction",
        ),
        click.option(
            "--optimisation-set-starfile",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="Path to the optimisation set star file for optimisation set generation.",
        ),
    ]
    return compose_options(opts)


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
            "--dry-run",
            is_flag=True,
            help="If set, do not extract subtomograms, only generate the starfiles needed for extraction.",
        )
    )
    options.append(
        click.option(
            "--no-particles-starfile",
            is_flag=True,
            help="If set, do not generate the particles.star file.",
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


def flatten_data_portal_args(kwargs: dict) -> dict:
    "Flattens the data portal arguments from lists of lists to a single list."
    for ref in DATA_PORTAL_ARG_REFS:
        val = kwargs.get(ref)
        if isinstance(val, (list, tuple)) and val and isinstance(val[0], list):
            kwargs[ref] = [item for chunk in val for item in chunk]

    return kwargs
