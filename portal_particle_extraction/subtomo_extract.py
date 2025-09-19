# TODO: Add support for a consolidated tiltseries star file (where all the tiltseries entries are just in the tomograms.star file)
"""
Primary entry point for extracting subtomograms from local files and the CryoET Data Portal.
Run portal-particle-extraction --help for usage instructions.
"""

import logging
import multiprocessing as mp
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
from portal_particle_extraction.core.constants import OPTICS_DF_COLUMNS
from portal_particle_extraction.core.ctf import calculate_ctf
from portal_particle_extraction.core.data import get_tiltseries_datareader
from portal_particle_extraction.core.dose import calculate_dose_weight_image
from portal_particle_extraction.core.forwardprojection import (
    apply_offsets_to_coordinates,
    calculate_projection_matrix_from_starfile_df,
    fourier_crop,
    get_particle_crop_and_visibility,
    get_particles_to_tiltseries_coordinates,
)
from portal_particle_extraction.core.helpers import get_tiltseries_data, setup_logging, validate_and_setup
from portal_particle_extraction.core.mask import circular_mask, circular_soft_mask
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


def process_tiltseries_wrapper(kwargs):
    return process_tiltseries(**kwargs)


def process_tiltseries(
    box_size: int,
    crop_size: int,
    bin: int,
    float16: bool,
    no_ctf: bool,
    circle_precrop: bool,
    no_circle_crop: bool,
    no_ic: bool,
    normalize_bin: bool,
    output_dir: Path,
    filtered_particles_df: pd.DataFrame,
    filtered_trajectories_dict: dict,
    tiltseries_row_entry: pd.Series,
    individual_tiltseries_df: pd.DataFrame,
    individual_tiltseries_path: Path,
    optics_row: pd.DataFrame,
) -> Union[None, tuple[pd.DataFrame, int]]:
    """
    Processes a single alignment file to extract subtomograms from the tiltseries.
    Does projection math to map from 3D coordinates to 2D tiltseries coordinates and then applies CTF premultiplication, dose weighting, and background subtraction.
    Writes resulting data to .mrcs files (2D stack) for each particle.

    individual_tiltseries_path is either the individual tiltseries star file or the tomograms star file if consolidated

    Returns the updated particles DataFrame and the number of skipped particles.
    """
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

    tiltseries_datareader = get_tiltseries_datareader(individual_tiltseries_df, individual_tiltseries_path)

    # projection-relevant variables
    pre_bin_background_mask = circular_mask(pre_bin_box_size, pre_bin_box_size) == 0.0
    pre_bin_soft_mask = circular_soft_mask(pre_bin_box_size, pre_bin_box_size, falloff=5.0)
    background_mask = circular_mask(box_size, crop_size) == 0.0
    soft_mask = circular_soft_mask(box_size, crop_size, falloff=5.0)
    tiltseries_pixel_size = tiltseries_row_entry["rlnTomoTiltSeriesPixelSize"]
    tiltseries_x = tiltseries_datareader.data.shape[2]
    tiltseries_y = tiltseries_datareader.data.shape[1]
    logger.debug(f"Tiltseries data shape: {tiltseries_datareader.data.shape}, pixel size: {tiltseries_pixel_size}")
    projection_matrices = calculate_projection_matrix_from_starfile_df(individual_tiltseries_df)
    particles_to_tiltseries_coordinates = get_particles_to_tiltseries_coordinates(
        filtered_particles_df, filtered_trajectories_dict, individual_tiltseries_df, projection_matrices
    )
    sections = individual_tiltseries_df["rlnMicrographName"].str.split("@").str[0].astype(int).tolist()
    section_to_section_index = {section: idx for idx, section in enumerate(sections)}

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
        dtype=np.complex64,
    )

    all_particle_data = []
    skipped_particles = set()
    # for rlnTomoVisibleFrames
    all_visible_sections_relion_column = []

    for particle_id, particle_sections in particles_to_tiltseries_coordinates.items():
        particle_data, visible_sections = get_particle_crop_and_visibility(
            tiltseries_datareader,
            particle_id,
            particle_sections,
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

    tiltseries_datareader.compute_crops()  # compute all cached slices

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
                tiltseries_datareader[tiltseries_key],
                ((y_pre_padding, y_post_padding), (x_pre_padding, x_post_padding)),
                mode="edge",
            )
            tilt_stack[tilt] = padded_crop

        if circle_precrop:
            pre_bin_background_mean = tilt_stack[:, pre_bin_background_mask].mean(axis=1)
            tilt_stack -= pre_bin_background_mean[:, None, None]
            tilt_stack *= pre_bin_soft_mask

        fourier_tilt_stack = np.fft.rfft2(tilt_stack, norm="ortho", axes=(-2, -1))

        new_fourier_tilt_stack = np.zeros((len(particle_data), box_size, box_size // 2 + 1), dtype=np.complex64)
        for tilt in range(len(particle_data)):
            section_index: int = section_to_section_index[particle_data[tilt]["section"]]
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
                    tilt_projection_matrix=projection_matrices[section_index],
                    voltage=voltage,
                    spherical_aberration=spherical_aberration,
                    amplitude_contrast=amplitude_contrast,
                    handedness=handedness,
                    tiltseries_pixel_size=tiltseries_pixel_size,
                    phase_shift=phase_shift[section_index],
                    defocus_u=defocus_u[section_index],
                    defocus_v=defocus_v[section_index],
                    defocus_angle=defocus_angle[section_index],
                    dose=doses[section_index],
                    ctf_scalefactor=ctf_scalefactor[section_index],
                    bfactor=bfactor_per_electron_dose[section_index],
                    box_size=box_size,
                    bin=bin,
                )
                fourier_tilt *= dose_weights[section_index] * ctf_weights
            if not no_ic:
                fourier_tilt *= -1  # phase flip for RELION compatibility
            fourier_tilt /= float(bin) ** (2 if normalize_bin else 1)  # normalize by binning factor
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
    box_size: int,
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
    updated_optics_df["BoxSize"] = box_size

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
    circle_precrop: bool = False,
    no_circle_crop: bool = False,
    no_ic: bool = False,
    normalize_bin: bool = True,
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
    particles_data = starfile.read(particles_starfile)
    particles_df = apply_offsets_to_coordinates(particles_data["particles"])
    optics_df = particles_data["optics"]
    trajectories_dict = starfile.read(trajectories_starfile) if trajectories_starfile else None
    tomograms_data = starfile.read(tomograms_starfile)
    tomograms_df = tomograms_data["global"] if isinstance(tomograms_data, dict) else tomograms_data
    if "rlnTomoTiltSeriesStarFile" not in tomograms_df.columns:
        raise ValueError(
            f"Tomograms star file {tomograms_starfile} does not contain the required column 'rlnTomoTiltSeriesStarFile'. Please check the file."
        )
    if not tiltseries_relative_dir:
        tiltseries_relative_dir = Path("./")

    args_list = [
        get_tiltseries_data(
            particles_df=particles_df,
            optics_df=optics_df,
            trajectories_dict=trajectories_dict,
            tiltseries_row_entry=tiltseries_row_entry,
            tiltseries_relative_dir=tiltseries_relative_dir,
            tomograms_starfile=tomograms_starfile,
            tomograms_data=tomograms_data,
        )
        for _, tiltseries_row_entry in tomograms_df.iterrows()
    ]
    # filter out empty or invalid entries and add constant args
    constant_args = {
        "box_size": box_size,
        "crop_size": crop_size,
        "bin": bin,
        "float16": float16,
        "no_ctf": no_ctf,
        "circle_precrop": circle_precrop,
        "no_circle_crop": no_circle_crop,
        "no_ic": no_ic,
        "normalize_bin": normalize_bin,
        "output_dir": output_dir,
    }
    args_list = [{**args, **constant_args} for args in args_list if args is not None]

    # do actual subtomogram extraction & .mrcs file creation here
    total_skipped_count = 0
    particles_df_results = []
    cpu_count = min(32, mp.cpu_count(), len(tomograms_df))
    logger.info(f"Starting extraction of subtomograms from {len(tomograms_df)} tiltseries using {cpu_count} CPU cores.")

    with mp.Pool(processes=cpu_count) as pool:
        for updated_filtered_particles_df, skipped_count in tqdm(
            pool.imap_unordered(process_tiltseries_wrapper, args_list, chunksize=1),
            total=len(args_list),
            file=sys.stdout,
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
        particles_data["optics"],
        tomograms_starfile,
        box_size,
        crop_size,
        bin,
        no_ctf,
        output_dir,
        trajectories_starfile,
    )

    return len(merged_particles_df), total_skipped_count, len(tomograms_df)


# TODO: compress all the common options into a model / kwargs?
def parse_extract_local_subtomograms(
    box_size: int,
    output_dir: Union[str, Path],
    bin: int = 1,
    float16: bool = False,
    no_ctf: bool = False,
    circle_precrop: bool = False,
    no_circle_crop: bool = False,
    no_ic: bool = False,
    normalize_bin: bool = True,
    crop_size: int = None,
    particles_starfile: Path = None,
    trajectories_starfile: Path = None,
    tiltseries_relative_dir: Path = None,
    tomograms_starfile: Path = None,
    optimisation_set_starfile: Path = None,
    overwrite: bool = False,
    no_logging: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    """
    Extracts subtomograms from local files using the provided parameters.

    Returns:
        tuple: particles_starfile, trajectories_starfile, tiltseries_relative_dir, tomograms_starfile, optimisation_set_starfile
    """
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

    particles_count, total_skipped_count, individual_tiltseries_count = extract_subtomograms(
        box_size=box_size,
        crop_size=crop_size,
        bin=bin,
        float16=float16,
        no_ctf=no_ctf,
        circle_precrop=circle_precrop,
        no_circle_crop=no_circle_crop,
        no_ic=no_ic,
        normalize_bin=normalize_bin,
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

    return (
        output_dir / "particles.star",
        trajectories_starfile,
        tiltseries_relative_dir,
        tomograms_starfile,
        output_dir / "optimisation_set.star",
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
    circle_precrop: bool = False,
    no_circle_crop: bool = False,
    no_ic: bool = False,
    normalize_bin: bool = True,
    copick_run_names: list[str] = None,
    crop_size: int = None,
    tiltseries_relative_dir: Path = None,
    tomograms_starfile: Path = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    """
    Extracts subtomograms from local files using copick picks and the provided parameters.

    Returns:
        tuple: particles_starfile, trajectories_starfile, tiltseries_relative_dir, tomograms_starfile, optimisation_set_starfile
    """
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
        data_portal_runs=False,
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
        circle_precrop=circle_precrop,
        no_circle_crop=no_circle_crop,
        no_ic=no_ic,
        normalize_bin=normalize_bin,
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

    return (
        output_dir / "particles.star",
        None,
        tiltseries_relative_dir,
        tomograms_starfile,
        output_dir / "optimisation_set.star",
    )


def parse_extract_data_portal_subtomograms(
    output_dir: Union[str, Path],
    box_size: int = None,
    bin: int = 1,
    float16: bool = False,
    no_ctf: bool = False,
    circle_precrop: bool = False,
    no_circle_crop: bool = False,
    no_ic: bool = False,
    normalize_bin: bool = True,
    crop_size: int = None,
    overwrite: bool = False,
    dry_run: bool = False,
    **data_portal_args,
) -> tuple[Path, Path, Path, Path, Path]:
    """
    Extracts subtomograms from the CryoET Data Portal using the provided parameters.

    Returns:
        tuple: particles_starfile, trajectories_starfile, tiltseries_relative_dir, tomograms_starfile, optimisation_set_starfile
    """
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
        circle_precrop=circle_precrop,
        no_circle_crop=no_circle_crop,
        no_ic=no_ic,
        normalize_bin=normalize_bin,
        output_dir=output_dir,
    )

    end_time = time.time()
    logger.info(
        f"Subtomogram extraction completed in {end_time - start_time:.2f} seconds. Extracted {particles_count} particles from {individual_tiltseries_count} tiltseries, skipped {total_skipped_count} particles due to out-of-bounds coordinates. Wrote to {output_dir}."
    )

    return (
        output_dir / "particles.star",
        None,
        output_dir,
        tomograms_path,
        output_dir / "optimisation_set.star",
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
    circle_precrop: bool = False,
    no_circle_crop: bool = False,
    no_ic: bool = False,
    normalize_bin: bool = True,
    crop_size: int = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    """
    Extracts subtomograms from the CryoET Data Portal using copick picks and the provided parameters.

    Returns:
        tuple: particles_starfile, trajectories_starfile, tiltseries_relative_dir, tomograms_starfile, optimisation_set_starfile
    """
    start_time = time.time()
    output_dir, _, _, _, _, _ = validate_and_setup(
        box_size=box_size,
        crop_size=crop_size,
        overwrite=overwrite,
        output_dir=output_dir,
        dry_run=dry_run,
        copick_data_portal=True,
    )

    if not copick_run_names:
        picks = get_copick_picks(copick_config, copick_name, copick_session_id, copick_user_id, copick_run_names)
        copick_run_names = [p.run.name for p in picks]

    # convert copick_run_names to ints and fail if not possible
    copick_run_ids = [int(s) for s in copick_run_names if s.isdigit()]
    if len(copick_run_ids) != len(copick_run_names):
        raise ValueError("All copick runs must be nonnegative integers")

    # generate a tomograms starfile with cdp_generate
    filtered_copick_run_ids, optics_df, tomograms_path, tiltseries_folder = cdp_generate.generate_tomograms_from_runs(
        run_ids=copick_run_ids,
        dataset_ids=copick_dataset_ids,
        output_dir=output_dir,
    )

    filtered_copick_run_names = [str(run_id) for run_id in filtered_copick_run_ids]

    # add optics and copick particles to particles.star file
    particles_df = copick_picks_to_starfile(
        copick_config,
        copick_name,
        copick_session_id,
        copick_user_id,
        filtered_copick_run_names,
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
        circle_precrop=circle_precrop,
        no_circle_crop=no_circle_crop,
        no_ic=no_ic,
        normalize_bin=normalize_bin,
        output_dir=output_dir,
    )

    end_time = time.time()
    logger.info(
        f"Subtomogram extraction completed in {end_time - start_time:.2f} seconds. Extracted {particles_count} particles from {individual_tiltseries_count} tiltseries, skipped {total_skipped_count} particles due to out-of-bounds coordinates. Wrote to {output_dir}."
    )

    return (
        output_dir / "particles.star",
        None,
        output_dir,
        tomograms_path,
        output_dir / "optimisation_set.star",
    )


@click.group("Extract subtomograms.")
def cli():
    pass


@cli.command("local", help="Extract subtomograms from local files (RELION *.star files).")
@cli_options.local_options()
@cli_options.local_shared_options()
@cli_options.common_options()
@cli_options.extract_options()
def cmd_local(**kwargs):
    setup_logging(debug=kwargs.pop("debug", False))
    parse_extract_local_subtomograms(**kwargs)


# TODO: write tests
@cli.command("copick-local", help="Extract subtomograms from local files (tiltseries) with copick particles.")
@cli_options.local_shared_options()
@cli_options.copick_options()
@cli_options.common_options()
@cli_options.extract_options()
@cli_options.dry_run_option
def cmd_local_copick(**kwargs):
    setup_logging(debug=kwargs.pop("debug", False))
    kwargs["copick_run_names"] = cli_options.flatten(kwargs["copick_run_names"])
    parse_extract_local_copick_subtomograms(**kwargs)


# TODO: write full tests
@cli.command("data-portal", help="Extract subtomograms from the CryoET Data Portal.")
@cli_options.common_options()
@cli_options.extract_options()
@cli_options.data_portal_options()
@cli_options.dry_run_option
def cmd_data_portal(**kwargs):
    setup_logging(debug=kwargs.pop("debug", False))
    kwargs = cli_options.flatten_data_portal_args(kwargs)
    parse_extract_data_portal_subtomograms(**kwargs)


# TODO: write tests
@cli.command("copick-data-portal", help="Extract subtomograms from CryoET Data Portal runs with copick particles.")
@cli_options.common_options()
@cli_options.extract_options()
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
