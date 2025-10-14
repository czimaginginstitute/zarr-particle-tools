import copy
from collections.abc import Sequence

from pipeliner.data_structure import (
    TOMO_SUBTOMO_DIR,
)
from pipeliner.jobs.tomography.relion_tomo.tomo_pseudosubtomo_job import RelionPseudoSubtomoJob
from pipeliner.nodes import (
    NODE_PARTICLEGROUPMETADATA,
    NODE_TOMOOPTIMISATIONSET,
)
from pipeliner.pipeliner_job import ExternalProgram, PipelinerCommand, PipelinerJob
from pipeliner.results_display_objects import ResultsDisplayObject


class PythonRelionPseudoSubtomoJob(PipelinerJob):
    PROCESS_NAME = "zarrparticletools.pseudosubtomo"
    OUT_DIR = TOMO_SUBTOMO_DIR
    CATEGORY_LABEL = "PseudoSubTomogram Generation"

    def __init__(self):
        super().__init__()
        self.jobinfo.programs = [ExternalProgram(command="zarr-particle-extract")]
        self.jobinfo.display_name = "Extract subtomogram volumes (Python)."
        self.jobinfo.short_desc = (
            "Extract subtomograms using zarr-particle-extract (Python reimplementation of RELION job)."
        )
        self.joboptions = copy.deepcopy(RelionPseudoSubtomoJob().joboptions)

    def create_output_nodes(self):
        self.add_output_node("particles.star", NODE_PARTICLEGROUPMETADATA, ["relion", "tomo", "extract", "python"])
        self.add_output_node("optimisation_set.star", NODE_TOMOOPTIMISATIONSET, ["relion", "tomo", "extract", "python"])

    def get_commands(self):
        # TODO: remove --debug
        command = ["zarr-particle-extract", "local", "--overwrite", "--debug"]

        optimisation_starfile = self.joboptions["in_optimisation"].get_string()
        if optimisation_starfile:
            command += ["--optimisation-set-starfile", optimisation_starfile]
        else:
            command += ["--tomograms-starfile", self.joboptions["in_tomograms"].get_string()]
            command += ["--particles-starfile", self.joboptions["in_particles"].get_string()]
            if self.joboptions["in_trajectories"].get_string():
                command += ["--trajectories-starfile", self.joboptions["in_trajectories"].get_string()]

        command.extend(["--output-dir", self.output_dir])
        command.extend(["--bin", self.joboptions["binfactor"].get_string()])
        command.extend(["--box-size", self.joboptions["box_size"].get_string()])
        crop_size = self.joboptions["crop_size"].get_string()
        if crop_size != "-1":
            command.extend(["--crop-size", crop_size])

        max_dose = float(self.joboptions["max_dose"].get_string())
        if max_dose > 0.0:
            raise NotImplementedError("Max dose handling not implemented in the Python implementation")
        if self.joboptions["min_nr_frames"].get_string() != "1":
            raise NotImplementedError("Min frames handling not implemented in the Python implementation")

        if self.joboptions["do_float16"].get_boolean():
            command.extend(["--float16"])

        if not self.joboptions["do_output_2dstacks"].get_boolean():
            raise NotImplementedError("3D subtomogram extraction not implemented in the Python implementation")

        if self.joboptions["do_extract_reproject"].get_boolean():
            raise NotImplementedError(
                "Reprojected 2D (real subtomogram) extraction not implemented in the Python implementation"
            )

        return [PipelinerCommand(command)]

    def create_results_display(self) -> Sequence[ResultsDisplayObject]:
        return [n.default_results_display(self.output_dir) for n in self.output_nodes if "particles.star" in n.name]


if __name__ == "__main__":
    pass
