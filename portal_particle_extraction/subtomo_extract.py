# TODO: Add support for a consolidated tiltseries star file (where all the tiltseries entries are just in the tomograms.star file)
"""
Primary entry point for extracting subtomograms from local files and the CryoET Data Portal.
Run portal-particle-extraction --help for usage instructions.
"""

import logging
import multiprocessing as mp
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Union

import click
import mrcfile
import numpy as np
import pandas as pd
import starfile
from scipy.ndimage import fourier_shift
from tqdm import tqdm

import portal_particle_extraction.cli.options as cli_options
import portal_particle_extraction.generate.cdp_generate_starfiles as cdp_generate
from portal_particle_extraction.core.constants import OPTICS_DF_COLUMNS, TILTSERIES_URI_RELION_COLUMN
from portal_particle_extraction.core.ctf import calculate_ctf
from portal_particle_extraction.core.data import DataReader
from portal_particle_extraction.core.dose import calculate_dose_weight_image
from portal_particle_extraction.core.helpers import setup_logging
from portal_particle_extraction.core.mask import circular_mask, circular_soft_mask
from portal_particle_extraction.core.projection import (
    calculate_projection_matrix_from_starfile_df,
    fourier_crop,
    get_particle_crop_and_visibility,
    get_particles_to_tiltseries_coordinates,
)
from portal_particle_extraction.generate.copick_generate_starfiles import copick_picks_to_starfile, get_copick_picks

logger = logging.getLogger(__name__)


def update_particles_df(
    particles_df: pd.DataFrame, output_folder: Path, all_visible_sections_relion_column: list, skipped_particles: set
) -> pd.DataFrame:
    """Updates the particles DataFrame to include the new columns and values for RELION format."""
    updated_particles_df = particles_df.copy()
    updated_particles_df = updated_particles_df.drop(
        columns=["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"], errors="ignore"
    )
    if "rlnTomoParticleName" not in updated_particles_df.columns:
        updated_particles_df = updated_particles_df.reset_index(drop=True)
        updated_particles_df.index += 1  # increment index by 1 to match RELION's 1-indexing
        updated_particles_df["rlnTomoParticleName"] = (
            updated_particles_df["rlnTomoName"] + "/" + updated_particles_df.index.astype(str)
        )
    # set index to be based on rlnTomoParticleName for easier processing
    updated_particles_df.index = updated_particles_df["rlnTomoParticleName"].str.split("/").str[-1].astype(int)
    updated_particles_df["rlnImageName"] = updated_particles_df.index.to_series().apply(
        lambda idx: (output_folder / f"{idx}_stack2d.mrcs").resolve()
    )
    # drop rows by particle_id that were skipped
    updated_particles_df = updated_particles_df.drop(
        updated_particles_df.index[
            updated_particles_df["rlnTomoParticleName"].str.split("/").str[-1].astype(int).isin(skipped_particles)
        ]
    )
    updated_particles_df["rlnTomoVisibleFrames"] = all_visible_sections_relion_column
    # offsets are applied to rlnCenteredCoordinateXAngst/YAngst/ZAngst beforehand if they exist, so they can be removed here
    updated_particles_df["rlnOriginXAngst"] = 0.0
    updated_particles_df["rlnOriginYAngst"] = 0.0
    updated_particles_df["rlnOriginZAngst"] = 0.0

    return updated_particles_df


def process_tiltseries(args) -> Union[None, tuple[pd.DataFrame, int]]:
    """
    Processes a single alignment file to extract subtomograms from the tiltseries.
    Does projection math to map from 3D coordinates to 2D tiltseries coordinates and then applies CTF premultiplication, dose weighting, and background subtraction.
    Writes resulting data to .mrcs files (2D stack) for each particle.

    Returns the updated particles DataFrame and the number of skipped particles.
    """
    (
        filtered_particles_df,
        filtered_trajectories_dict,
        box_size,
        crop_size,
        bin,
        individual_tiltseries_df,
        tiltseries_row_entry,
        individual_tiltseries_path,  # either the individual tiltseries star file or the tomograms star file if consolidated
        optics_row,
        float16,
        no_ctf,
        no_circle_crop,
        no_ic,
        output_dir,
    ) = args
    setup_logging()
    # following RELION convention
    pre_bin_box_size = int(round(box_size * bin))
    pre_bin_crop_size = crop_size * bin

    particles_tomo_name = tiltseries_row_entry["rlnTomoName"]
    output_folder = output_dir / "Subtomograms" / particles_tomo_name
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.debug(
        f"Extracting subtomograms for {len(filtered_particles_df)} particles (filtered by rlnTomoName: {particles_tomo_name})"
    )

    if TILTSERIES_URI_RELION_COLUMN in individual_tiltseries_df.columns:
        tiltseries_data_locators = individual_tiltseries_df[TILTSERIES_URI_RELION_COLUMN].to_list()
    else:
        tiltseries_data_locators = (
            individual_tiltseries_df["rlnMicrographName"].apply(lambda x: x.split("@")[1]).to_list()
        )
    if len(set(tiltseries_data_locators)) != 1:
        raise ValueError(
            f"Multiple tiltseries data locators found: {set(tiltseries_data_locators)}. This is not supported."
        )
    tiltseries_data_locator = tiltseries_data_locators[0]
    if not tiltseries_data_locator.startswith("s3://") and not tiltseries_data_locator.startswith("/"):
        # assume it's a local relative path, relative to the individual tiltseries star file or the tomograms star file if consolidated
        tiltseries_data_locator = individual_tiltseries_path.parent / tiltseries_data_locator
    tiltseries_data = DataReader(str(tiltseries_data_locator))

    # projection-relevant variables
    background_mask = circular_mask(box_size, crop_size) == 0.0
    soft_mask = circular_soft_mask(box_size, crop_size, falloff=5.0)
    tiltseries_pixel_size = tiltseries_row_entry["rlnTomoTiltSeriesPixelSize"]
    tiltseries_x = tiltseries_data.data.shape[2]
    tiltseries_y = tiltseries_data.data.shape[1]
    logger.debug(f"Tiltseries data shape: {tiltseries_data.data.shape}, pixel size: {tiltseries_pixel_size}")
    projection_matrices = calculate_projection_matrix_from_starfile_df(individual_tiltseries_df)
    particles_to_tiltseries_coordinates = get_particles_to_tiltseries_coordinates(
        filtered_particles_df, filtered_trajectories_dict, individual_tiltseries_df, projection_matrices
    )

    # ctf & dose-weighting parameters
    voltage = tiltseries_row_entry["rlnVoltage"]
    spherical_aberration = tiltseries_row_entry["rlnSphericalAberration"]
    amplitude_contrast = tiltseries_row_entry["rlnAmplitudeContrast"]
    handedness = tiltseries_row_entry["rlnTomoHand"]
    phase_shift = (
        tiltseries_row_entry["rlnPhaseShift"]
        if "rlnPhaseShift" in optics_row.columns
        else [0.0] * len(individual_tiltseries_df)
    )
    defocus_u = individual_tiltseries_df["rlnDefocusU"].values
    defocus_v = individual_tiltseries_df["rlnDefocusV"].values
    defocus_angle = individual_tiltseries_df["rlnDefocusAngle"].values
    doses = individual_tiltseries_df["rlnMicrographPreExposure"].values
    ctf_scalefactor = (
        individual_tiltseries_df["rlnCtfScalefactor"]
        if "rlnCtfScalefactor" in individual_tiltseries_df.columns
        else [1.0] * len(individual_tiltseries_df)
    )
    bfactor_per_electron_dose = (
        individual_tiltseries_df["rlnCtfBfactorPerElectronDose"]
        if "rlnCtfBfactorPerElectronDose" in individual_tiltseries_df.columns
        else [0.0] * len(individual_tiltseries_df)
    )
    dose_weights = np.stack(
        [
            calculate_dose_weight_image(dose, tiltseries_pixel_size * bin, box_size, bfactor)
            for dose, bfactor in zip(doses, bfactor_per_electron_dose)
        ],
        dtype=np.float32,
    )

    all_particle_data = []
    skipped_particles = set()
    # for rlnTomoVisibleFrames
    all_visible_sections_relion_column = []

    for particle_id, sections in particles_to_tiltseries_coordinates.items():
        particle_data, visible_sections = get_particle_crop_and_visibility(
            tiltseries_data,
            particle_id,
            sections,
            tiltseries_x,
            tiltseries_y,
            tiltseries_pixel_size,
            pre_bin_box_size,
            pre_bin_crop_size,
        )

        # RELION by default only requires one tilt to be visible, so only skip if all sections are out of bounds (and also don't append since the entry doesn't exist in the particles.star file)
        if len(particle_data) == 0:
            skipped_particles.add(particle_id)
            continue

        all_particle_data.append(particle_data)
        # modify the values to be in RELION format
        all_visible_sections_relion_column.append(str(visible_sections).replace(" ", ""))

    start_time = time.time()
    tiltseries_data.compute_crops()  # compute all cached slices
    end_time = time.time()
    logger.debug(f"Downloading crops for {tiltseries_data_locator} took {end_time - start_time:.2f} seconds.")

    def process_particle_data(particle_data):
        particle_id = particle_data[0]["particle_id"]

        tilt_stack = np.zeros((len(particle_data), pre_bin_box_size, pre_bin_box_size), dtype=np.float32)
        for tilt in range(len(particle_data)):
            tiltseries_key = particle_data[tilt]["tiltseries_key"]
            x_pre_padding = particle_data[tilt]["x_pre_padding"]
            y_pre_padding = particle_data[tilt]["y_pre_padding"]
            x_post_padding = particle_data[tilt]["x_post_padding"]
            y_post_padding = particle_data[tilt]["y_post_padding"]
            padded_crop = np.pad(
                tiltseries_data[tiltseries_key],
                ((y_pre_padding, y_post_padding), (x_pre_padding, x_post_padding)),
                mode="edge",
            )
            tilt_stack[tilt] = padded_crop
        fourier_tilt_stack = np.fft.rfft2(tilt_stack, norm="ortho", axes=(-2, -1))

        new_fourier_tilt_stack = np.zeros((len(particle_data), box_size, box_size // 2 + 1), dtype=np.complex64)
        for tilt in range(len(particle_data)):
            section: int = particle_data[tilt]["section"]
            subpixel_shift: tuple[int, int] = particle_data[tilt]["subpixel_shift"]
            coordinate: np.ndarray = particle_data[tilt]["coordinate"]

            fourier_tilt = fourier_tilt_stack[tilt]
            fourier_tilt = fourier_shift(fourier_tilt, subpixel_shift, n=fourier_tilt.shape[0], axis=1)

            if bin > 1:
                fourier_tilt = fourier_crop(fourier_tilt, bin)

            # TODO: look into gamma offset
            # TODO: implement spherical aberration correction
            if not no_ctf:
                ctf_weights = calculate_ctf(
                    coordinate=coordinate,
                    tilt_projection_matrix=projection_matrices[section - 1],
                    voltage=voltage,
                    spherical_aberration=spherical_aberration,
                    amplitude_contrast=amplitude_contrast,
                    handedness=handedness,
                    tiltseries_pixel_size=tiltseries_pixel_size,
                    phase_shift=phase_shift[section - 1],
                    defocus_u=defocus_u[section - 1],
                    defocus_v=defocus_v[section - 1],
                    defocus_angle=defocus_angle[section - 1],
                    dose=doses[section - 1],
                    ctf_scalefactor=ctf_scalefactor[section - 1],
                    bfactor=bfactor_per_electron_dose[section - 1],
                    box_size=box_size,
                    bin=bin,
                )
                fourier_tilt *= dose_weights[section - 1, :, :] * ctf_weights
            if not no_ic:
                fourier_tilt *= -1  # phase flip for RELION compatibility
            fourier_tilt /= float(bin) ** 2  # normalize by binning factor
            new_fourier_tilt_stack[tilt] = fourier_tilt

        new_tilt_stack = np.fft.irfft2(new_fourier_tilt_stack, norm="ortho", axes=(-2, -1))
        # remove noise via background subtraction and apply soft circular mask
        if not no_circle_crop:
            background_mean = new_tilt_stack[:, background_mask].mean(axis=1)
            new_tilt_stack -= background_mean[:, None, None]
            new_tilt_stack *= soft_mask

        # crop to final desired size
        cropped_tilt_stack = new_tilt_stack[
            :,
            (box_size - crop_size) // 2 : (box_size + crop_size) // 2,
            (box_size - crop_size) // 2 : (box_size + crop_size) // 2,
        ]

        output_path = output_folder / f"{particle_id}_stack2d.mrcs"
        with mrcfile.new(output_path) as mrc:
            mrc.set_data(cropped_tilt_stack.astype(np.float16 if float16 else np.float32))
            mrc.voxel_size = (tiltseries_pixel_size * bin, tiltseries_pixel_size * bin, 1.0)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:  # emperically determined 4 threads is optimal for this task
        futures = [executor.submit(process_particle_data, particle_data) for particle_data in all_particle_data]
        for future in as_completed(futures):
            future.result()

    updated_filtered_particles_df = update_particles_df(
        filtered_particles_df, output_folder, all_visible_sections_relion_column, skipped_particles
    )

    end_time = time.time()
    logger.debug(
        f"Extracted subtomograms for {particles_tomo_name} with {len(particles_to_tiltseries_coordinates)} particles, skipped {len(skipped_particles)} particles due to out-of-bounds coordinates, took {end_time - start_time:.2f} seconds."
    )

    return updated_filtered_particles_df, len(skipped_particles)


def write_starfiles(
    merged_particles_df: pd.DataFrame,
    particle_optics_df: pd.DataFrame,
    tomograms_starfile: str,
    crop_size: int,
    bin: int,
    no_ctf: bool,
    output_dir: Path,
    trajectories_starfile: str = None,
) -> None:
    """
    Writes the updated particles and optimisation set star files, as per RELION expected format & outputs.
    """
    merged_particles_df["ParticleID"] = merged_particles_df["rlnTomoParticleName"].str.split("/").str[-1].astype(int)
    merged_particles_df = merged_particles_df.sort_values(by=["rlnTomoName", "ParticleID"]).reset_index(drop=True)
    merged_particles_df = merged_particles_df.drop(columns="ParticleID")

    updated_optics_df = particle_optics_df.copy()
    updated_optics_df["rlnCtfDataAreCtfPremultiplied"] = 0 if no_ctf else 1
    updated_optics_df["rlnImageDimensionality"] = 2
    updated_optics_df["rlnTomoSubtomogramBinning"] = float(bin)
    updated_optics_df["rlnImagePixelSize"] = updated_optics_df["rlnTomoTiltSeriesPixelSize"] * bin
    updated_optics_df["rlnImageSize"] = crop_size

    general_df = {"rlnTomoSubTomosAre2DStacks": 1}
    starfile.write(
        {
            "general": general_df,
            "optics": updated_optics_df,
            "particles": merged_particles_df,
        },
        output_dir / "particles.star",
    )
    optimisation_set_dict = {
        "rlnTomoParticlesFile": (output_dir / "particles.star").resolve(),
        "rlnTomoTomogramsFile": tomograms_starfile.resolve(),
    }
    if trajectories_starfile:
        optimisation_set_dict["rlnTomoTrajectoriesFile"] = trajectories_starfile.resolve()

    starfile.write(optimisation_set_dict, output_dir / "optimisation_set.star")


def extract_subtomograms(
    box_size: int,
    output_dir: Union[str, Path],
    particles_starfile: Path,
    tomograms_starfile: Path,
    bin: int = 1,
    float16: bool = False,
    no_ctf: bool = False,
    no_circle_crop: bool = False,
    no_ic: bool = False,
    crop_size: int = None,
    tiltseries_relative_dir: Path = None,
    trajectories_starfile: Path = None,
) -> tuple[int, int, int]:
    """
    Extracts subtomograms from a provided particles *.star file, tiltseries *.star file, and set of *.aln files.
    Creates new *.mrcs files for each particle in the output directory, as well as updated particles and optimisation set star files.
    Uses multiprocessing to speed up the extraction process.

    Returns:
        tuple: Number of particles extracted, number of skipped particles, number of tiltseries processed.
    """
    if crop_size is None:
        crop_size = box_size

    logger.debug(f"Starting subtomogram extraction, reading file {particles_starfile} and {tomograms_starfile}")
    particles_star_file = starfile.read(particles_starfile)
    particles_df = particles_star_file["particles"]
    trajectories_dict = starfile.read(trajectories_starfile) if trajectories_starfile else None
    tomograms_data = starfile.read(tomograms_starfile)
    if isinstance(tomograms_data, dict):
        tomograms_df = tomograms_data["global"]
    else:
        tomograms_df = tomograms_data
    if not tiltseries_relative_dir:
        tiltseries_relative_dir = Path("./")  # Default to current directory

    # apply alignment rlnOriginXAngst/YAngst/ZAngst to rlnCenteredCoordinateXAngst/YAngst/ZAngst, subtract to follow RELION's convention
    if (
        "rlnOriginXAngst" in particles_df.columns
        and "rlnOriginYAngst" in particles_df.columns
        and "rlnOriginZAngst" in particles_df.columns
    ):
        particles_df["rlnCenteredCoordinateXAngst"] -= particles_df["rlnOriginXAngst"]
        particles_df["rlnCenteredCoordinateYAngst"] -= particles_df["rlnOriginYAngst"]
        particles_df["rlnCenteredCoordinateZAngst"] -= particles_df["rlnOriginZAngst"]

    def build_args(tiltseries_row_entry: pd.Series) -> tuple:
        """Builds the arguments for processing a single tiltseries and its particles (to be passed into process_tiltseries)."""
        individual_tiltseries_df = None
        individual_tiltseries_path: Path = None
        if isinstance(tomograms_data, dict):
            if tiltseries_row_entry["rlnTomoName"] not in tomograms_data:
                raise ValueError(
                    f"Tiltseries {tiltseries_row_entry['rlnTomoName']} not found in tomograms star file {tomograms_starfile}. Please check the file."
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
                f"No particles found for tomogram {tiltseries_row_entry['rlnTomoName']} in {particles_starfile}. Please check the particles star file."
            )
            return None

        filtered_trajectories_dict = None
        if trajectories_dict:
            particle_names = filtered_particles_df["rlnTomoParticleName"].tolist()
            filtered_trajectories_dict = {k: v for k, v in trajectories_dict.items() if k in particle_names}
        optics_row = particles_star_file["optics"][
            particles_star_file["optics"]["rlnOpticsGroupName"] == tiltseries_row_entry["rlnOpticsGroupName"]
        ]
        return (
            filtered_particles_df,
            filtered_trajectories_dict,
            box_size,
            crop_size,
            bin,
            individual_tiltseries_df,
            tiltseries_row_entry,
            individual_tiltseries_path if individual_tiltseries_path else tomograms_starfile,
            optics_row,
            float16,
            no_ctf,
            no_circle_crop,
            no_ic,
            output_dir,
        )

    if "rlnTomoTiltSeriesStarFile" not in tomograms_df.columns:
        raise ValueError(
            f"Tomograms star file {tomograms_starfile} does not contain the required column 'rlnTomoTiltSeriesStarFile'. Please check the file."
        )

    args_list = [build_args(tiltseries_row_entry) for _, tiltseries_row_entry in tomograms_df.iterrows()]
    # filter out empty or invalid entries
    args_list = [args for args in args_list if args is not None]

    # do actual subtomogram extraction & .mrcs file creation here
    total_skipped_count = 0
    particles_df_results = []
    cpu_count = min(32, mp.cpu_count(), len(tomograms_df))
    logger.info(f"Starting extraction of subtomograms from {len(tomograms_df)} tiltseries using {cpu_count} CPU cores.")
    with mp.Pool(processes=cpu_count) as pool:
        for updated_filtered_particles_df, skipped_count in tqdm(
            pool.imap_unordered(process_tiltseries, args_list, chunksize=1), total=len(args_list), file=sys.stdout
        ):
            if updated_filtered_particles_df is not None and not updated_filtered_particles_df.empty:
                particles_df_results.append(updated_filtered_particles_df)
            total_skipped_count += skipped_count

    if not particles_df_results:
        raise ValueError("No particles were extracted. Please check the input files and parameters.")

    merged_particles_df = pd.concat(particles_df_results, ignore_index=True)
    # update all the relevant star files
    write_starfiles(
        merged_particles_df,
        particles_star_file["optics"],
        tomograms_starfile,
        crop_size,
        bin,
        no_ctf,
        output_dir,
        trajectories_starfile,
    )

    return len(merged_particles_df), total_skipped_count, len(tomograms_df)


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
    trajectories_starfile = (
        Path(trajectories_starfile) if isinstance(trajectories_starfile, str) else trajectories_starfile
    )
    tiltseries_relative_dir = (
        Path(tiltseries_relative_dir) if isinstance(tiltseries_relative_dir, str) else tiltseries_relative_dir
    )
    tomograms_starfile = Path(tomograms_starfile) if isinstance(tomograms_starfile, str) else tomograms_starfile
    optimisation_set_starfile = (
        Path(optimisation_set_starfile) if isinstance(optimisation_set_starfile, str) else optimisation_set_starfile
    )

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


# TODO: compress all the common options into a model / kwargs?
def parse_extract_local_subtomograms(
    box_size: int,
    output_dir: Union[str, Path],
    bin: int = 1,
    float16: bool = False,
    no_ctf: bool = False,
    no_circle_crop: bool = False,
    no_ic: bool = False,
    crop_size: int = None,
    particles_starfile: Path = None,
    trajectories_starfile: Path = None,
    tiltseries_relative_dir: Path = None,
    tomograms_starfile: Path = None,
    optimisation_set_starfile: Path = None,
    overwrite: bool = False,
    no_logging: bool = False,
) -> None:
    start_time = time.time()
    (
        output_dir,
        particles_starfile,
        trajectories_starfile,
        tiltseries_relative_dir,
        tomograms_starfile,
        optimisation_set_starfile,
    ) = validate_and_setup(
        box_size=box_size,
        crop_size=crop_size,
        overwrite=overwrite,
        output_dir=output_dir,
        particles_starfile=particles_starfile,
        trajectories_starfile=trajectories_starfile,
        tiltseries_relative_dir=tiltseries_relative_dir,
        tomograms_starfile=tomograms_starfile,
        optimisation_set_starfile=optimisation_set_starfile,
    )

    if optimisation_set_starfile and (particles_starfile or tomograms_starfile or trajectories_starfile):
        raise ValueError(
            "Cannot specify both optimisation set star file and individual star files. Please provide only one of them."
        )

    if not optimisation_set_starfile and not particles_starfile and not tomograms_starfile:
        raise ValueError("Either optimisation set star file or particles and tomograms star files must be provided.")

    if optimisation_set_starfile:
        optimisation_dict = starfile.read(optimisation_set_starfile)
        particles_starfile = Path(optimisation_dict["rlnTomoParticlesFile"])
        tomograms_starfile = Path(optimisation_dict["rlnTomoTomogramsFile"])
        trajectories_starfile = optimisation_dict.get("rlnTomoTrajectoriesFile")
        trajectories_starfile = trajectories_starfile and Path(trajectories_starfile)

    particles_count, total_skipped_count, individual_tiltseries_count = extract_subtomograms(
        box_size=box_size,
        crop_size=crop_size,
        bin=bin,
        float16=float16,
        no_ctf=no_ctf,
        no_circle_crop=no_circle_crop,
        no_ic=no_ic,
        output_dir=output_dir,
        particles_starfile=particles_starfile,
        trajectories_starfile=trajectories_starfile,
        tiltseries_relative_dir=tiltseries_relative_dir,
        tomograms_starfile=tomograms_starfile,
    )
    end_time = time.time()
    if not no_logging:
        logger.info(
            f"Subtomogram extraction completed in {end_time - start_time:.2f} seconds. Extracted {particles_count} particles from {individual_tiltseries_count} tiltseries, skipped {total_skipped_count} particles due to out-of-bounds coordinates. Wrote to {output_dir}."
        )


# TODO: test that this actually works
def parse_extract_local_copick_subtomograms(
    box_size: int,
    output_dir: Union[str, Path],
    copick_config: Path,
    copick_name: str,
    copick_session_id: str,
    copick_user_id: str,
    bin: int = 1,
    float16: bool = False,
    no_ctf: bool = False,
    no_circle_crop: bool = False,
    no_ic: bool = False,
    copick_run_names: list[str] = None,
    crop_size: int = None,
    tiltseries_relative_dir: Path = None,
    tomograms_starfile: Path = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    start_time = time.time()
    output_dir, _, _, tiltseries_relative_dir, tomograms_starfile, _ = validate_and_setup(
        box_size=box_size,
        crop_size=crop_size,
        overwrite=overwrite,
        output_dir=output_dir,
        tiltseries_relative_dir=tiltseries_relative_dir,
        tomograms_starfile=tomograms_starfile,
        dry_run=dry_run,
    )

    if copick_run_names is None:
        picks = get_copick_picks(copick_config, copick_name, copick_session_id, copick_user_id, copick_run_names)
        copick_run_names = [p.run.name for p in picks]

    tomograms_df = starfile.read(tomograms_starfile)
    if isinstance(tomograms_df, dict):
        tomograms_df = tomograms_df["global"]
    optics_df = tomograms_df[OPTICS_DF_COLUMNS].drop_duplicates().reset_index(drop=True)

    # add copick particles to particles.star file
    particles_df = copick_picks_to_starfile(
        copick_config,
        copick_name,
        copick_session_id,
        copick_user_id,
        copick_run_names,
        optics_df,
        data_portal_runs=True,
    )
    particles_path = output_dir / "particles.star"
    starfile.write({"optics": optics_df, "particles": particles_df}, particles_path)
    logger.info(f"Generated particles star file at {particles_path} with {len(particles_df)} particles.")

    if dry_run:
        logger.info("Dry run enabled, skipping subtomogram extraction.")
        return

    parse_extract_local_subtomograms(
        box_size=box_size,
        bin=bin,
        float16=float16,
        no_ctf=no_ctf,
        no_circle_crop=no_circle_crop,
        no_ic=no_ic,
        output_dir=output_dir,
        crop_size=crop_size,
        particles_starfile=particles_path,
        trajectories_starfile=None,
        tiltseries_relative_dir=tiltseries_relative_dir,
        tomograms_starfile=tomograms_starfile,
        optimisation_set_starfile=None,
        overwrite=overwrite,
        no_logging=True,
    )

    end_time = time.time()
    logger.info(
        f"Subtomogram extraction completed in {end_time - start_time:.2f} seconds. Extracted {len(particles_df)} particles from {len(tomograms_df)} tiltseries. Wrote to {output_dir}."
    )


def parse_extract_data_portal_subtomograms(
    output_dir: Union[str, Path],
    box_size: int = None,
    bin: int = 1,
    float16: bool = False,
    no_ctf: bool = False,
    no_circle_crop: bool = False,
    no_ic: bool = False,
    crop_size: int = None,
    dry_run: bool = False,
    overwrite: bool = False,
    **data_portal_args,
) -> None:
    start_time = time.time()
    output_dir, _, _, _, _, _ = validate_and_setup(
        box_size=box_size,
        crop_size=crop_size,
        overwrite=overwrite,
        output_dir=output_dir,
        dry_run=dry_run,
    )

    particles_path, tomograms_path, tiltseries_folder = cdp_generate.generate_starfiles(
        output_dir=output_dir,
        **data_portal_args,
    )

    if not particles_path.exists():
        raise ValueError(
            f"Starfile generation failed. Expected particles star file at {particles_path} does not exist."
        )

    if not tomograms_path.exists():
        raise ValueError(
            f"Starfile generation failed. Expected tomograms star file at {tomograms_path} does not exist."
        )

    if not tiltseries_folder.exists() or not any(tiltseries_folder.glob("*.star")):
        raise ValueError(
            f"Starfile generation failed. Expected tiltseries star files in {tiltseries_folder} do not exist."
        )

    if dry_run:
        logger.info(
            f"Dry run enabled, skipping subtomogram extraction. Generated star files at {particles_path} and {tomograms_path}."
        )
        return

    particles_count, total_skipped_count, individual_tiltseries_count = extract_subtomograms(
        particles_starfile=particles_path,
        trajectories_starfile=None,  # No trajectories data in Data Portal
        tiltseries_relative_dir=output_dir,
        tomograms_starfile=tomograms_path,
        box_size=box_size,
        crop_size=crop_size,
        bin=bin,
        float16=float16,
        no_ctf=no_ctf,
        no_circle_crop=no_circle_crop,
        no_ic=no_ic,
        output_dir=output_dir,
    )

    end_time = time.time()
    logger.info(
        f"Subtomogram extraction completed in {end_time - start_time:.2f} seconds. Extracted {particles_count} particles from {individual_tiltseries_count} tiltseries, skipped {total_skipped_count} particles due to out-of-bounds coordinates. Wrote to {output_dir}."
    )


def parse_extract_data_portal_copick_subtomograms(
    output_dir: Union[str, Path],
    copick_config: Path,
    copick_name: str,
    copick_session_id: str,
    copick_user_id: str,
    copick_run_names: list[str] = None,
    copick_dataset_ids: list[int] = None,
    box_size: int = None,
    bin: int = 1,
    float16: bool = False,
    no_ctf: bool = False,
    no_circle_crop: bool = False,
    no_ic: bool = False,
    crop_size: int = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    start_time = time.time()
    output_dir, _, _, _, _, _ = validate_and_setup(
        box_size=box_size,
        crop_size=crop_size,
        overwrite=overwrite,
        output_dir=output_dir,
        dry_run=dry_run,
    )

    if not copick_run_names:
        picks = get_copick_picks(copick_config, copick_name, copick_session_id, copick_user_id, copick_run_names)
        copick_run_names = [p.run.name for p in picks]

    # convert copick_run_names to ints and fail if not possible
    copick_run_names = [int(s) for s in copick_run_names if s.isdigit()]
    if len(copick_run_names) != len(copick_run_names):
        raise ValueError("All copick runs must be nonnegative integers")

    # generate a tomograms starfile with cdp_generate
    optics_df, tomograms_path, tiltseries_folder = cdp_generate.generate_tomograms_from_runs(
        run_ids=copick_run_names,
        dataset_ids=copick_dataset_ids,
        output_dir=output_dir,
    )

    # add optics and copick particles to particles.star file
    particles_df = copick_picks_to_starfile(
        copick_config,
        copick_name,
        copick_session_id,
        copick_user_id,
        copick_run_names,
        optics_df,
        data_portal_runs=True,
    )
    # filter out particles that don't have a corresponding tomogram in the tomograms starfile
    particles_df = particles_df[particles_df["rlnTomoName"].isin(optics_df["rlnTomoName"])]
    particles_path = output_dir / "particles.star"
    starfile.write({"optics": optics_df, "particles": particles_df}, particles_path)
    logger.info(f"Generated particles star file at {particles_path} with {len(particles_df)} particles.")

    if not particles_path.exists():
        raise ValueError(
            f"Starfile generation failed. Expected particles star file at {particles_path} does not exist."
        )

    if not tomograms_path.exists():
        raise ValueError(
            f"Starfile generation failed. Expected tomograms star file at {tomograms_path} does not exist."
        )

    if not tiltseries_folder.exists() or not any(tiltseries_folder.glob("*.star")):
        raise ValueError(
            f"Starfile generation failed. Expected tiltseries star files in {tiltseries_folder} do not exist."
        )

    if dry_run:
        logger.info(
            f"Dry run enabled, skipping subtomogram extraction. Generated star files at {particles_path} and {tomograms_path}."
        )
        return

    particles_count, total_skipped_count, individual_tiltseries_count = extract_subtomograms(
        particles_starfile=particles_path,
        trajectories_starfile=None,  # No trajectories data in Data Portal
        tiltseries_relative_dir=output_dir,
        tomograms_starfile=tomograms_path,
        box_size=box_size,
        crop_size=crop_size,
        bin=bin,
        float16=float16,
        no_ctf=no_ctf,
        no_circle_crop=no_circle_crop,
        no_ic=no_ic,
        output_dir=output_dir,
    )

    end_time = time.time()
    logger.info(
        f"Subtomogram extraction completed in {end_time - start_time:.2f} seconds. Extracted {particles_count} particles from {individual_tiltseries_count} tiltseries, skipped {total_skipped_count} particles due to out-of-bounds coordinates. Wrote to {output_dir}."
    )


@click.group("Extract subtomograms.")
def cli():
    pass


@cli.command("local", help="Extract subtomograms from local files (particles and tiltseries).")
@cli_options.local_options()
@cli_options.local_shared_options()
@cli_options.common_options()
def cmd_local(**kwargs):
    setup_logging(debug=kwargs.pop("debug", False))
    parse_extract_local_subtomograms(**kwargs)


@cli.command("copick-local", help="Extract subtomograms from local files (tiltseries) with copick particles.")
@cli_options.local_shared_options()
@cli_options.copick_options()
@cli_options.common_options()
@cli_options.dry_run_option
def cmd_local_copick(**kwargs):
    setup_logging(debug=kwargs.pop("debug", False))
    kwargs["copick_run_names"] = cli_options.flatten(kwargs["copick_run_names"])
    parse_extract_local_copick_subtomograms(**kwargs)


@cli.command("data-portal", help="Extract subtomograms from the CryoET Data Portal.")
@cli_options.common_options()
@cli_options.data_portal_options()
@cli_options.dry_run_option
def cmd_data_portal(**kwargs):
    setup_logging(debug=kwargs.pop("debug", False))
    kwargs = cli_options.flatten_data_portal_args(kwargs)
    parse_extract_data_portal_subtomograms(**kwargs)


# TODO: write tests
@cli.command("copick-data-portal", help="Extract subtomograms from CryoET Data Portal runs with copick particles.")
@cli_options.common_options()
@cli_options.copick_options()
@cli_options.data_portal_copick_options()
@cli_options.dry_run_option
def cmd_data_portal_copick(**kwargs):
    setup_logging(debug=kwargs.pop("debug", False))
    kwargs["copick_run_names"] = cli_options.flatten(kwargs["copick_run_names"])
    kwargs["copick_dataset_ids"] = cli_options.flatten(kwargs["copick_dataset_ids"])
    parse_extract_data_portal_copick_subtomograms(**kwargs)


if __name__ == "__main__":
    cli()
