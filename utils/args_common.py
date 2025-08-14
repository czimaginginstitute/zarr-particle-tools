# TODO: fuzzy matching for name fields?
# TODO: globbing for all fields?
import argparse

DATA_PORTAL_ARGS = [
    ("--deposition-ids", int, True),
    ("--deposition-titles", str, True),
    ("--dataset-ids", int, True),
    ("--dataset-titles", str, True),
    ("--organism-names", str, True),
    ("--cell-names", str, True),
    ("--run-ids", int, True),
    ("--run-names", str, True),
    ("--tiltseries-ids", int, True),
    ("--alignment-ids", int, True),
    ("--tomogram-ids", int, True),
    ("--tomogram-names", str, True),
    ("--annotation-ids", int, True),
    ("--annotation-names", str, True),
]

DATA_PORTAL_ARG_REFS = [arg.lstrip("--").replace("-", "_") for arg, _, _ in DATA_PORTAL_ARGS] + ["inexact_match"]


# NOTE: not robust since it assumes the plural form is just the singular form with an 's' at the end, which is currently the case but may not always be true
def arg_flags(plural: str) -> tuple[str, str]:
    """Given a plural form of a field, return the argument flags for both plural and singular forms."""
    return plural, plural[:-1]


def help_text(field_name: str, field_type: str, arg_type: type, is_unique: bool = True) -> str:
    return f"CryoET Data Portal {field_name} {field_type}(s) to filter picks (space separated). \
        {f' If --inexact-match is specified, filtering is case insensitive, "contains" search is used.' if arg_type is str else ''} \
        {f' NOTE: Not neccessarily a unique identifier across the portal. Results can span different datasets.' if not is_unique else ''}"


def add_data_portal_args(parser: argparse.ArgumentParser):
    parser.add_argument("--inexact-match", action="store_true", help="Filter using case insensitive, 'contains' search for string fields.")
    for arg, arg_type, is_unique in DATA_PORTAL_ARGS:
        field_name = arg.lstrip("--").split("-")[0]
        field_type = arg.lstrip("--").split("-")[1].rstrip("s")
        help_message = help_text(field_name, field_type, arg_type, is_unique)
        parser.add_argument(*arg_flags(arg), type=arg_type, nargs="*", help=help_message)
