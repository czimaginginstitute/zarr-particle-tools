from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

from pipeliner.data_structure import TOMO_RECONSTRUCT_DIR
from pipeliner.display_tools import make_maps_slice_montage_and_3d_display
from pipeliner.jobs.tomography.relion_tomo.tomo_reconstructparticle_job import (
    RelionReconstructParticleJob,
)
from pipeliner.nodes import (
    NODE_DENSITYMAP,
    NODE_TOMOOPTIMISATIONSET,
)
from pipeliner.pipeliner_job import ExternalProgram, PipelinerCommand, PipelinerJob
from pipeliner.results_display_objects import ResultsDisplayObject


class PythonRelionSubtomoReconstructJob(PipelinerJob):
    PROCESS_NAME = "zarrparticletools.reconstruct"
    OUT_DIR = TOMO_RECONSTRUCT_DIR
    CATEGORY_LABEL = "Reconstruct"

    def __init__(self):
        super().__init__()
        self.jobinfo.programs = [ExternalProgram(command="zarr-particle-reconstruct")]
        self.jobinfo.display_name = "Reconstruct 3D particle (Python)."
        self.jobinfo.short_desc = "Reconstruct a 3D particle density map using zarr-particle-reconstruct (Python reimplementation of RELION job)."
        self.joboptions = copy.deepcopy(RelionReconstructParticleJob().joboptions)

        # remove options not supported by the Python implementation
        for k in ("Wiener_SNR", "point_group", "do_helical"):
            if k in self.joboptions:
                del self.joboptions[k]

    def create_output_nodes(self):
        self.add_output_node("merged.mrc", NODE_DENSITYMAP, ["relion", "tomo", "reconstruct", "python"])
        self.add_output_node("half1.mrc", NODE_DENSITYMAP, ["relion", "halfmap", "reconstruct", "python"])
        self.add_output_node("half2.mrc", NODE_DENSITYMAP, ["relion", "halfmap", "reconstruct", "python"])
        self.add_output_node("optimisation_set.star", NODE_TOMOOPTIMISATIONSET, ["relion", "reconstruct", "python"])

    def get_commands(self):
        cmd = ["zarr-particle-reconstruct", "local", "--overwrite", "--debug"]

        optimisation_starfile = self.joboptions["in_optimisation"].get_string()
        if optimisation_starfile:
            cmd += ["--optimisation-set-starfile", optimisation_starfile]
        else:
            cmd += ["--tomograms-starfile", self.joboptions["in_tomograms"].get_string()]
            cmd += ["--particles-starfile", self.joboptions["in_particles"].get_string()]
            if self.joboptions["in_trajectories"].get_string():
                cmd += ["--trajectories-starfile", self.joboptions["in_trajectories"].get_string()]

        cmd += ["--output-dir", self.output_dir]
        cmd += ["--box-size", self.joboptions["box_size"].get_string()]
        cmd += ["--bin", self.joboptions["binfactor"].get_string()]
        crop_size = self.joboptions["crop_size"].get_string()
        if crop_size != "-1":
            cmd += ["--crop-size", crop_size]

        return [PipelinerCommand(cmd)]

    def create_results_display(self) -> Sequence[ResultsDisplayObject]:
        od = Path(self.output_dir)
        merged = str(od / "merged.mrc")
        half1 = str(od / "half1.mrc")
        half2 = str(od / "half2.mrc")

        return make_maps_slice_montage_and_3d_display(
            in_maps={
                merged: "Reconstructed map",
                half1: "Halfmap 1",
                half2: "Halfmap 2",
            },
            output_dir=self.output_dir,
            combine_montages=False,
        )


if __name__ == "__main__":
    pass
