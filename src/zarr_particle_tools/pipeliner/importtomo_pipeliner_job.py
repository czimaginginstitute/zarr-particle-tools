"""
``zarrparticletools.importtomo``: CryoET Data Portal runs imported as a RELION 5 tomography set whose tilt series stay
on S3 (``zarr-particle-importtomo``).

Imports only pipeliner and the standard library (plus zpt's literal-only constants), so a process that plans jobs
can load it without the extraction stack.
"""

from collections.abc import Sequence

from pipeliner.data_structure import NODE_PROCESSDATA, NODE_TOMOGRAMGROUPMETADATA, TOMO_IMPORT_DIR
from pipeliner.job_options import (
    ExternalFileJobOption,
    FloatJobOption,
    MultipleChoiceJobOption,
    StringJobOption,
)
from pipeliner.pipeliner_job import ExternalProgram, PipelinerCommand, PipelinerJob
from pipeliner.results_display_objects import ResultsDisplayObject

from zarr_particle_tools.core.constants import DEFAULT_AMPLITUDE_CONTRAST, TOMO_HAND_DEFAULT_VALUE, TOMOGRAM_TYPES

_KEYWORDS = ["relion", "tomo", "portal", "zarr"]


class PythonPortalImportTomoJob(PipelinerJob):
    PROCESS_NAME = "zarrparticletools.importtomo"
    OUT_DIR = TOMO_IMPORT_DIR
    CATEGORY_LABEL = "Tomography Import"

    def __init__(self):
        super().__init__()
        self.jobinfo.programs = [ExternalProgram(command="zarr-particle-importtomo")]
        self.jobinfo.display_name = "Import CryoET Data Portal runs (S3, Python)."
        self.jobinfo.short_desc = (
            "RELION 5 tomograms.star and tilt-series stars from the CryoET Data Portal; tilt series stay on S3."
        )
        self.is_tomo = True

        self.joboptions["dataset_ids"] = StringJobOption(
            label="Dataset IDs:",
            default_value="",
            help_text="Comma-separated portal dataset IDs; every run is imported unless run IDs narrow it.",
        )
        self.joboptions["run_ids"] = StringJobOption(
            label="Run IDs:",
            default_value="",
            help_text="Comma-separated portal run IDs (narrows the datasets).",
        )
        self.joboptions["tomogram_ids"] = StringJobOption(
            label="Tomogram IDs (pins):",
            default_value="",
            help_text="Pin these tomograms; the type/software/method/spacing filters apply to the other runs.",
        )
        self.joboptions["tomogram_type"] = MultipleChoiceJobOption(
            label="Tomogram type:",
            choices=list(TOMOGRAM_TYPES),
            default_value_index=0,
            help_text=(
                "default = each run's visualization default; otherwise the portal's Tomogram.processing. The "
                "tomogram fixes the alignment and voxel spacing the import uses."
            ),
        )
        self.joboptions["tomogram_software"] = StringJobOption(
            label="Tomogram processing software:",
            default_value="",
            help_text="Exact Tomogram.processing_software, when a type matches several (e.g. two denoisers).",
        )
        self.joboptions["reconstruction_method"] = StringJobOption(
            label="Reconstruction method:",
            default_value="",
            help_text="Tomogram.reconstruction_method (e.g. WBP or SART), when a type matches several.",
        )
        self.joboptions["voxel_spacing"] = FloatJobOption(
            label="Voxel spacing (A):",
            default_value=0.0,
            hard_min=0.0,
            help_text="An exact deposited voxel spacing; 0 accepts any. Never a resampling.",
        )
        self.joboptions["in_selection"] = ExternalFileJobOption(
            label="Stored portal selection (optional):",
            default_value="",
            help_text=(
                "A portal_selection.json to import exactly. The job re-resolves its IDs and fails if the portal "
                "changed since it was resolved. Excludes the ID and filter options."
            ),
        )
        self.joboptions["Q0"] = FloatJobOption(
            label="Amplitude contrast:",
            default_value=DEFAULT_AMPLITUDE_CONTRAST,
            hard_min=0.0,
            hard_max=1.0,
            help_text="rlnAmplitudeContrast; the portal does not record it.",
        )
        self.joboptions["hand"] = MultipleChoiceJobOption(
            label="Defocus handedness (rlnTomoHand):",
            choices=["-1", "1"],
            default_value_index=0 if TOMO_HAND_DEFAULT_VALUE == -1 else 1,
            help_text="The portal does not record the defocus handedness.",
        )
        self.get_runtab_options(addtl_args=True)

    def create_output_nodes(self):
        self.add_output_node("tomograms.star", NODE_TOMOGRAMGROUPMETADATA, _KEYWORDS)
        self.add_output_node("portal_selection.json", NODE_PROCESSDATA, ["portal", "selection"])

    def get_commands(self):
        cmd = ["zarr-particle-importtomo", "--output-dir", self.output_dir]
        selection = self.joboptions["in_selection"].get_string().strip()
        if selection:
            cmd += ["--selection", selection]
        else:
            for option, flag in (
                ("dataset_ids", "--dataset-ids"),
                ("run_ids", "--run-ids"),
                ("tomogram_ids", "--tomogram-ids"),
                ("tomogram_software", "--tomogram-software"),
                ("reconstruction_method", "--reconstruction-method"),
            ):
                value = self.joboptions[option].get_string().strip()
                if value:
                    cmd += [flag, value]
            cmd += ["--tomogram-type", self.joboptions["tomogram_type"].get_string()]
            cmd += ["--voxel-spacing", self.joboptions["voxel_spacing"].get_string()]
        cmd += ["--amplitude-contrast", self.joboptions["Q0"].get_string()]
        cmd += ["--hand", self.joboptions["hand"].get_string()]
        cmd += self.parse_additional_args()
        return [PipelinerCommand(cmd)]

    def create_results_display(self) -> Sequence[ResultsDisplayObject]:
        return [n.default_results_display(self.output_dir) for n in self.output_nodes if "tomograms.star" in n.name]


if __name__ == "__main__":
    pass
