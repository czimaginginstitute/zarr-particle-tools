import logging
import shutil
import sys
from pathlib import Path
from typing import Union

import pandas as pd
import starfile
from tqdm import tqdm

from portal_particle_extraction.core.constants import NOISY_LOGGERS

logger = logging.getLogger(__name__)

# ====================== global helpers ======================


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stdout)
            self.flush()
        except Exception:
            self.handleError(record)


def suppress_noisy_loggers(loggers, level=logging.ERROR):
    for name in loggers:
        logging.getLogger(name).setLevel(level)


def setup_logging(debug: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        handlers=[TqdmLoggingHandler()],
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    suppress_noisy_loggers(NOISY_LOGGERS)

    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).propagate = True


def get_filter(values, field, inexact_match, label=""):
    """Helper to append filters depending on inexact_match."""
    if not values:
        return None
    if inexact_match:
        if len(values) > 1:
            raise ValueError(
                f"Cannot use inexact match with multiple values ({values}) for {label}. Please provide a single value."
            )
        logger.info(f"Finding similar {label}: {values} (case insensitive, includes partial matches)")
        return field.ilike(f"%{values[0]}%")
    else:
        logger.info(f"Filtering by EXACT {label}: {values}")
        return field._in(values)


# ====================== starfile generation helpers ======================


def get_optics_group_name(run_id: int, tiltseries_id: int) -> str:
    return f"run_{run_id}_tiltseries_{tiltseries_id}"


def get_tomo_name(run_id: int, tiltseries_id: int, alignment_id: int, voxel_spacing_id: int) -> str:
    return f"run_{run_id}_tiltseries_{tiltseries_id}_alignment_{alignment_id}_spacing_{voxel_spacing_id}"


# ====================== file validation, reading, data extraction helpers ======================


def validate_and_setup(
    box_size: int,
    output_dir: Union[str, Path],
    particles_starfile: Union[str, Path, None] = None,
    trajectories_starfile: Union[str, Path, None] = None,
    tiltseries_relative_dir: Union[str, Path, None] = None,
    tomograms_starfile: Union[str, Path, None] = None,
    optimisation_set_starfile: Union[str, Path, None] = None,
    crop_size: int = None,
    overwrite: bool = False,
    dry_run: bool = False,
    copick_data_portal: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    if not dry_run and box_size is None:
        raise ValueError("Box size must be specified.")

    if box_size is not None and box_size % 2 != 0 and box_size > 0:
        raise ValueError(f"Box size must be an even number greater than 0, got {box_size}.")

    if crop_size is not None:
        if crop_size % 2 != 0:
            raise ValueError(f"Crop size must be an even number, got {crop_size}.")
        if crop_size > box_size:
            raise ValueError(
                f"Crop size cannot be greater than box size, got crop size {crop_size} and box size {box_size}."
            )

    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
    particles_starfile = Path(particles_starfile) if isinstance(particles_starfile, str) else particles_starfile
    if particles_starfile is not None and not particles_starfile.exists():
        raise FileNotFoundError(f"Particles star file {particles_starfile} does not exist.")
    trajectories_starfile = (
        Path(trajectories_starfile) if isinstance(trajectories_starfile, str) else trajectories_starfile
    )
    if trajectories_starfile is not None and not trajectories_starfile.exists():
        raise FileNotFoundError(f"Trajectories star file {trajectories_starfile} does not exist.")
    tiltseries_relative_dir = (
        Path(tiltseries_relative_dir) if isinstance(tiltseries_relative_dir, str) else tiltseries_relative_dir
    )
    if tiltseries_relative_dir is not None and not Path(tiltseries_relative_dir).exists():
        raise FileNotFoundError(f"Tilseries relative directory {tiltseries_relative_dir} does not exist.")
    tomograms_starfile = Path(tomograms_starfile) if isinstance(tomograms_starfile, str) else tomograms_starfile
    if tomograms_starfile is not None and not tomograms_starfile.exists():
        raise FileNotFoundError(f"Tomograms star file {tomograms_starfile} does not exist.")
    optimisation_set_starfile = (
        Path(optimisation_set_starfile) if isinstance(optimisation_set_starfile, str) else optimisation_set_starfile
    )
    if optimisation_set_starfile is not None and not optimisation_set_starfile.exists():
        raise FileNotFoundError(f"Optimisation set star file {optimisation_set_starfile} does not exist.")

    if optimisation_set_starfile and (particles_starfile or tomograms_starfile or trajectories_starfile):
        raise ValueError(
            "Cannot specify both optimisation set star file and individual star files. Please provide only one of them."
        )

    if not copick_data_portal and not optimisation_set_starfile and not particles_starfile and not tomograms_starfile:
        raise ValueError("Either optimisation set star file or particles and tomograms star files must be provided.")

    if optimisation_set_starfile:
        optimisation_dict = starfile.read(optimisation_set_starfile)
        particles_starfile = Path(optimisation_dict["rlnTomoParticlesFile"])
        tomograms_starfile = Path(optimisation_dict["rlnTomoTomogramsFile"])
        trajectories_starfile = optimisation_dict.get("rlnTomoTrajectoriesFile")
        trajectories_starfile = trajectories_starfile and Path(trajectories_starfile)

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} already exists and is not empty. Use --overwrite to overwrite existing files."
        )

    if not dry_run:
        if (output_dir / "Subtomograms").exists():
            shutil.rmtree(output_dir / "Subtomograms")

        (output_dir / "Subtomograms").mkdir(parents=True)

    return (
        output_dir,
        particles_starfile,
        trajectories_starfile,
        tiltseries_relative_dir,
        tomograms_starfile,
        optimisation_set_starfile,
    )


def get_tiltseries_data(
    particles_df: pd.DataFrame,
    optics_df: pd.DataFrame,
    trajectories_dict: dict[str, pd.DataFrame],
    tiltseries_row_entry: pd.Series,
    tiltseries_relative_dir: Path,
    tomograms_starfile: Path,
    tomograms_data: pd.DataFrame,
) -> dict[str, any] | None:
    """Builds the arguments for processing a single tiltseries and its particles."""
    individual_tiltseries_df = None
    individual_tiltseries_path = None
    if isinstance(tomograms_data, dict):
        if tiltseries_row_entry["rlnTomoName"] not in tomograms_data:
            raise ValueError(
                f"Tiltseries {tiltseries_row_entry['rlnTomoName']} not found in tomograms star file. Please check the file."
            )
        individual_tiltseries_df = tomograms_data[tiltseries_row_entry["rlnTomoName"]]
    else:
        individual_tiltseries_path = Path(tiltseries_row_entry["rlnTomoTiltSeriesStarFile"])
        if not individual_tiltseries_path.is_absolute():
            individual_tiltseries_path = tiltseries_relative_dir / tiltseries_row_entry["rlnTomoTiltSeriesStarFile"]
        if not individual_tiltseries_path.exists():
            raise FileNotFoundError(
                f"Tiltseries file {individual_tiltseries_path} does not exist. Please check the path and try again."
            )
        individual_tiltseries_df = starfile.read(individual_tiltseries_path)

    if individual_tiltseries_df.empty:
        raise ValueError(f"Tiltseries data for {tiltseries_row_entry['rlnTomoName']} is empty.")

    filtered_particles_df = particles_df[particles_df["rlnTomoName"] == tiltseries_row_entry["rlnTomoName"]]
    if filtered_particles_df.empty:
        logger.warning(
            f"No particles found for tomogram {tiltseries_row_entry['rlnTomoName']}. Please check the particles star file."
        )
        return None

    filtered_trajectories_dict = None
    if trajectories_dict:
        particle_names = filtered_particles_df["rlnTomoParticleName"].tolist()
        filtered_trajectories_dict = {k: v for k, v in trajectories_dict.items() if k in particle_names}
    optics_row = optics_df[optics_df["rlnOpticsGroupName"] == tiltseries_row_entry["rlnOpticsGroupName"]]
    return {
        "filtered_particles_df": filtered_particles_df,
        "filtered_trajectories_dict": filtered_trajectories_dict,
        "tiltseries_row_entry": tiltseries_row_entry,
        "individual_tiltseries_df": individual_tiltseries_df,
        "individual_tiltseries_path": (
            individual_tiltseries_path if individual_tiltseries_path else tomograms_starfile
        ),
        "optics_row": optics_row,
    }
