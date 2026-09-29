from __future__ import annotations

import copy
import re
from collections.abc import Sequence

from pipeliner.data_structure import TOMO_CTFREFINE_DIR
from pipeliner.job_options import BooleanJobOption, StringJobOption
from pipeliner.jobs.tomography.relion_tomo.tomo_ctfrefine_job import TomoRelionCtfRefine
from pipeliner.nodes import (
    NODE_TOMOGRAMGROUPMETADATA,
    NODE_TOMOOPTIMISATIONSET,
)
from pipeliner.pipeliner_job import ExternalProgram, PipelinerCommand, PipelinerJob
from pipeliner.results_display_objects import ResultsDisplayObject


def _add_staging_options(job) -> None:
    """Where the tilt series are staged for stock RELION, and whether that must be RAM."""
    job.joboptions["staging_dir"] = StringJobOption(
        label="Staging directory:",
        default_value="/dev/shm",
        help_text="Each tilt series is streamed here as a temporary MRC for RELION, in a job-owned subdirectory "
        "removed when the job ends.",
    )
    job.joboptions["do_require_ram_staging"] = BooleanJobOption(
        label="Require RAM-backed staging?",
        default_value=True,
        help_text="Fail before streaming unless the staging directory is tmpfs with room for the staged stacks, "
        "instead of falling back to the system temp directory.",
    )


def _staging_args(job) -> list[str]:
    args = ["--shm-dir", job.joboptions["staging_dir"].get_string()]
    if job.joboptions["do_require_ram_staging"].get_boolean():
        args.append("--require-ram-staging")
    return args


class PythonRelionSubtomoCtfRefineJob(PipelinerJob):
    PROCESS_NAME = "zarrparticletools.ctfrefine"
    OUT_DIR = TOMO_CTFREFINE_DIR
    CATEGORY_LABEL = "CTF Refinement"

    def __init__(self):
        super().__init__()
        self.jobinfo.programs = [ExternalProgram(command="zarr-particle-ctfrefine")]
        self.jobinfo.display_name = "Refine CTF parameters (Python)."
        self.jobinfo.short_desc = (
            "CTF-refine tilt series stored as OME-Zarr using zarr-particle-ctfrefine and stock RELION."
        )
        self.joboptions = copy.deepcopy(TomoRelionCtfRefine().joboptions)
        _add_staging_options(self)

    def create_output_nodes(self):
        self.add_output_node("tomograms.star", NODE_TOMOGRAMGROUPMETADATA, ["relion", "tomo", "ctfrefine", "python"])
        self.add_output_node(
            "optimisation_set.star", NODE_TOMOOPTIMISATIONSET, ["relion", "tomo", "ctfrefine", "python"]
        )

    def get_commands(self):
        cmd = ["zarr-particle-ctfrefine", "local"]

        optimisation_starfile = self.joboptions["in_optimisation"].get_string()
        if optimisation_starfile and not self.joboptions["use_direct_entries"].get_boolean():
            cmd += ["--optimisation-set-starfile", optimisation_starfile]
        else:
            cmd += ["--particles-starfile", self.joboptions["in_particles"].get_string()]
            cmd += ["--tomograms-starfile", self.joboptions["in_tomograms"].get_string()]
            if self.joboptions["in_trajectories"].get_string():
                cmd += ["--trajectories-starfile", self.joboptions["in_trajectories"].get_string()]

        # RELION supplies one half-map and derives the other by name
        in_ref = self.joboptions["in_halfmaps"].get_string()
        if re.search(r"half1", in_ref):
            ref1, ref2 = in_ref, re.sub(r"half1", r"half2", in_ref)
        elif re.search(r"half2", in_ref):
            ref1, ref2 = re.sub(r"half2", r"half1", in_ref), in_ref
        else:
            raise ValueError(f"Second halfmap corresponding to {in_ref} not found")
        cmd += ["--ref1", ref1, "--ref2", ref2]

        mask = self.joboptions["in_refmask"].get_string()
        if mask:
            cmd += ["--mask", mask]
        fsc = self.joboptions["in_post"].get_string()
        if fsc:
            cmd += ["--fsc", fsc]

        cmd += ["--output-dir", self.output_dir]
        cmd += ["--box-size", self.joboptions["box_size"].get_string()]

        if self.joboptions["do_defocus"].get_boolean():
            cmd += ["--do-defocus", "--focus-range", self.joboptions["focus_range"].get_string()]
            if self.joboptions["do_reg_def"].get_boolean():
                cmd += ["--do-reg-defocus", "--lambda-reg", self.joboptions["lambda"].get_string()]

        if self.joboptions["do_scale"].get_boolean():
            cmd += ["--do-scale"]
            per_frame = self.joboptions["do_frame_scale"].get_boolean()
            per_tomo = self.joboptions["do_tomo_scale"].get_boolean()
            if per_frame and not per_tomo:
                cmd += ["--per-frame-scale"]
            elif per_tomo and not per_frame:
                cmd += ["--per-tomogram-scale"]

        cmd += ["--threads", self.joboptions["nr_threads"].get_string()]
        cmd += _staging_args(self)

        return [PipelinerCommand(cmd)]

    def create_results_display(self) -> Sequence[ResultsDisplayObject]:
        return [
            n.default_results_display(self.output_dir)
            for n in self.output_nodes
            if "particles_ctf_refine.star" in n.name
        ]


if __name__ == "__main__":
    pass
