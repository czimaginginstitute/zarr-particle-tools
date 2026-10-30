"""Shared parts of the zarr-particle-tools pipeliner jobs: job option and input checks.

Imports only pipeliner at module level; the input checks import zarr_particle_tools.core.helpers when they run.
"""

from __future__ import annotations

from pathlib import Path

from pipeliner.job_options import JobOptionValidationResult

INPUT_OPTIONS = {"particles": "in_particles", "tomograms": "in_tomograms", "trajectories": "in_trajectories"}


def _number(job, name: str, default: float) -> float:
    try:
        return float(job.joboptions[name].get_string())
    except (TypeError, ValueError):
        return default


def uses_optimisation_set(job) -> bool:
    """Whether the job's command reads ``in_optimisation`` rather than the direct entries."""
    if not job.joboptions["in_optimisation"].get_string():
        return False
    return "use_direct_entries" not in job.joboptions or not job.joboptions["use_direct_entries"].get_boolean()


def option_problems(job, program: str) -> list[tuple[list[str], str]]:
    """``(options, message)`` for each job option value ``program`` would not honor."""
    jo = job.joboptions
    problems = []
    if program in ("extract", "reconstruct"):
        direct = jo["in_particles"].get_string() or jo["in_tomograms"].get_string()
        if jo["do_use_direct_entries"].get_boolean() and direct and jo["in_optimisation"].get_string():
            problems.append(
                (
                    ["do_use_direct_entries", "in_optimisation"],
                    "name either an optimisation set or direct entries, not both",
                )
            )
        if jo["other_args"].get_string().strip():
            problems.append((["other_args"], "other_args is not passed on: the program is not RELION"))
    if program == "extract":
        if _number(job, "max_dose", -1) > 0:
            problems.append((["max_dose"], "max_dose > 0 is not implemented"))
        if _number(job, "min_nr_frames", 1) != 1:
            problems.append((["min_nr_frames"], "min_nr_frames other than 1 is not implemented"))
        if not jo["do_output_2dstacks"].get_boolean():
            problems.append((["do_output_2dstacks"], "do_output_2dstacks = No (3D subtomograms) is not implemented"))
        if jo["do_extract_reproject"].get_boolean():
            problems.append(
                (["do_extract_reproject"], "do_extract_reproject = Yes (reprojected 2D extraction) is not implemented")
            )
    if program == "polish":
        shift, motion = jo["do_shift_align"].get_boolean(), jo["do_motion"].get_boolean()
        if shift == motion:
            problems.append(
                (["do_shift_align", "do_motion"], "exactly one of do_shift_align and do_motion must be Yes")
            )
        if shift and not motion and jo["shift_align_type"].get_string() == "Only particles":
            problems.append((["shift_align_type"], "shift_align_type 'Only particles' is not implemented"))
        if motion and not shift and jo["do_sq_exp_ker"].get_boolean():
            problems.append((["do_sq_exp_ker"], "do_sq_exp_ker is not implemented"))
    return problems


def check_options(job, program: str) -> None:
    """Raise what :func:`option_problems` reports, for ``get_commands``."""
    problems = option_problems(job, program)
    if problems:
        raise ValueError(f"{job.PROCESS_NAME} cannot honor: " + "; ".join(message for _, message in problems))


def input_problems(job, program: str) -> list[tuple[list[str], str]]:
    """``(options, message)`` for what ``program`` would not honor in the job's inputs, once they all exist."""
    from zarr_particle_tools.core import helpers

    jo = job.joboptions
    if uses_optimisation_set(job):
        optimisation_set = Path(jo["in_optimisation"].get_string())
        if not optimisation_set.exists():
            return []
        try:
            particles, tomograms, trajectories = helpers.read_optimisation_set(optimisation_set)
        except (OSError, ValueError, KeyError) as exc:
            return [(["in_optimisation"], str(exc))]
        options = dict.fromkeys(INPUT_OPTIONS, "in_optimisation")
    else:
        named = {source: jo[option].get_string() for source, option in INPUT_OPTIONS.items()}
        if not (named["particles"] and named["tomograms"]) or not all(Path(v).exists() for v in named.values() if v):
            return []
        particles, tomograms, trajectories = (Path(v) if v else None for v in named.values())
        options = INPUT_OPTIONS
    checked = "relion" if program in ("ctfrefine", "polish") else program
    problems = helpers.input_problems(checked, particles, tomograms, trajectories)
    return [([options[source]], message) for source, message in problems]


def validate(job, program: str) -> list[JobOptionValidationResult]:
    """``additional_joboption_validation``: :func:`option_problems` and :func:`input_problems` as errors."""
    problems = option_problems(job, program) + input_problems(job, program)
    return [
        JobOptionValidationResult("error", [job.joboptions[name] for name in names], message)
        for names, message in problems
    ]
