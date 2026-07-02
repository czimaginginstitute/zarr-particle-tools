from __future__ import annotations

import copy
from collections.abc import Sequence

from pipeliner.data_structure import BAYESPOLISH_DIR
from pipeliner.jobs.tomography.relion_tomo.tomo_bayesianpolish_job import TomoRelionBayesPolishJob
from pipeliner.nodes import (
    NODE_PARTICLEGROUPMETADATA,
    NODE_TOMOGRAMGROUPMETADATA,
    NODE_TOMOOPTIMISATIONSET,
    NODE_TOMOTRAJECTORYDATA,
)
from pipeliner.pipeliner_job import ExternalProgram, PipelinerCommand, PipelinerJob
from pipeliner.results_display_objects import ResultsDisplayObject


class PythonRelionSubtomoPolishJob(PipelinerJob):
    PROCESS_NAME = "zarrparticletools.polish"
    OUT_DIR = BAYESPOLISH_DIR
    CATEGORY_LABEL = "Particle Polishing"

    def __init__(self):
        super().__init__()
        self.jobinfo.programs = [ExternalProgram(command="zarr-particle-polish")]
        self.jobinfo.display_name = "Bayesian polishing / frame alignment (Python)."
        self.jobinfo.short_desc = (
            "Polish tilt series stored as OME-Zarr using zarr-particle-polish (stock RELION on /dev/shm)."
        )
        self.joboptions = copy.deepcopy(TomoRelionBayesPolishJob().joboptions)

        # remove options unsupported by this wrapper (parallelism is n_workers, not MPI)
        if "nr_mpi" in self.joboptions:
            del self.joboptions["nr_mpi"]

    def create_output_nodes(self):
        self.add_output_node("motion.star", NODE_TOMOTRAJECTORYDATA, ["relion", "tomo", "polish", "python"])
        self.add_output_node("tomograms.star", NODE_TOMOGRAMGROUPMETADATA, ["relion", "tomo", "polish", "python"])
        self.add_output_node("particles.star", NODE_PARTICLEGROUPMETADATA, ["relion", "tomo", "polish", "python"])
        self.add_output_node("optimisation_set.star", NODE_TOMOOPTIMISATIONSET, ["relion", "tomo", "polish", "python"])

    def get_commands(self):
        cmd = ["zarr-particle-polish", "local"]

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
        if "half1" in in_ref:
            ref1, ref2 = in_ref, in_ref.replace("half1", "half2")
        elif "half2" in in_ref:
            ref1, ref2 = in_ref.replace("half2", "half1"), in_ref
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
        cmd += [
            "--align-range",
            str(int(float(self.joboptions["max_error"].get_string()))),
        ]  # RELION --r is float; our CLI is int

        # RELION requires exactly one of shift-only / motion
        do_shift = self.joboptions["do_shift_align"].get_boolean()
        do_motion = self.joboptions["do_motion"].get_boolean()
        if do_shift and not do_motion:
            cmd += ["--shift-only", "--no-motion"]
        elif do_motion and not do_shift:
            cmd += [
                "--do-motion",
                "--s-vel",
                self.joboptions["sigma_vel"].get_string(),
                "--s-div",
                self.joboptions["sigma_div"].get_string(),
            ]
        else:
            raise AssertionError("Per-particle motion and shift-only corrections cannot be applied simultaneously")

        cmd += ["--threads", self.joboptions["nr_threads"].get_string()]

        return [PipelinerCommand(cmd)]

    def create_results_display(self) -> Sequence[ResultsDisplayObject]:
        return [
            n.default_results_display(self.output_dir) for n in self.output_nodes if "optimisation_set.star" in n.name
        ]


if __name__ == "__main__":
    pass
