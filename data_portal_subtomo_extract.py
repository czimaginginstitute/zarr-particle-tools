"""
Primary entry point for extracting subtomograms from a CryoET Data Portal run.
Run python data_portal_subtomo_extract.py --help for usage instructions.
"""

import argparse
import logging
import starfile
import numpy as np
import pandas as pd
import mrcfile
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from scipy.ndimage import fourier_shift
from typing import Union
from tqdm import tqdm

import utils.args_common as args_common
from core.projection import get_particles_to_tiltseries_coordinates, fourier_crop, get_particle_crop_and_visibility, calculate_projection_matrix_from_starfile_df
from core.mask import circular_mask, circular_soft_mask
from core.dose import calculate_dose_weight_image
from core.ctf import calculate_ctf
from core.data import DataReader
from generate.generate_starfile import generate_starfiles
from generate.constants import TILTSERIES_URI_RELION_COLUMN

logger = logging.getLogger(__name__)


def update_particles_df(particles_df: pd.DataFrame, output_folder: Path, all_visible_sections_relion_column: list, skipped_particles: set) -> pd.DataFrame:
    """Updates the particles DataFrame to include the new columns and values for RELION format."""
    updated_particles_df = particles_df.copy()
    updated_particles_df = updated_particles_df.drop(columns=["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"], errors="ignore")
    updated_particles_df = updated_particles_df.reset_index(drop=True)
    updated_particles_df.index += 1  # increment index by 1 to match RELION's 1-indexing
    updated_particles_df["rlnTomoParticleName"] = updated_particles_df["rlnTomoName"] + "/" + updated_particles_df.index.astype(str)
    updated_particles_df["rlnImageName"] = updated_particles_df.index.to_series().apply(lambda idx: (output_folder / f"{idx}_stack2d.mrcs").resolve())
    updated_particles_df = updated_particles_df.drop(index=skipped_particles, errors="ignore")  # drop rows by particle_id that were skipped
    updated_particles_df["rlnTomoVisibleFrames"] = all_visible_sections_relion_column
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
        box_size,
        bin,
        individual_tiltseries_path,
        tiltseries_row_entry,
        optics_row,
        float16,
        no_ctf,
        no_circle_crop,
        output_dir,
    ) = args
    pre_bin_box_size = box_size * bin

    if not individual_tiltseries_path.exists():
        raise FileNotFoundError(f"Tiltseries file {individual_tiltseries_path} does not exist. Please check the path and try again.")

    particles_tomo_name = tiltseries_row_entry["rlnTomoName"]
    output_folder = output_dir / "Subtomograms" / particles_tomo_name
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Extracting subtomograms for {len(filtered_particles_df)} particles (filtered by rlnTomoName: {particles_tomo_name}) from tiltseries {individual_tiltseries_path}")

    individual_tiltseries_df = starfile.read(individual_tiltseries_path)
    if individual_tiltseries_df.empty:
        raise ValueError(f"Tiltseries file {individual_tiltseries_path} is empty or does not contain the required columns. Please check the file.")
    
    if TILTSERIES_URI_RELION_COLUMN in individual_tiltseries_df.columns:
        tiltseries_data_locators = individual_tiltseries_df[TILTSERIES_URI_RELION_COLUMN].to_list()
    else:
        tiltseries_data_locators = individual_tiltseries_df["rlnMicrographName"].apply(lambda x: x.split("@")[1]).to_list()
    if len(set(tiltseries_data_locators)) != 1:
        raise ValueError(f"Multiple tiltseries data locators found: {set(tiltseries_data_locators)}. This is not supported.")
    tiltseries_data_locator = tiltseries_data_locators[0]
    if not tiltseries_data_locator.startswith("s3://") and not tiltseries_data_locator.startswith("/"): # assume it's a local relative path, relative to the tiltseries file
        tiltseries_data_locator = individual_tiltseries_path.parent / tiltseries_data_locator
    tiltseries_data = DataReader(str(tiltseries_data_locator))

    # projection-relevant variables
    background_mask = circular_mask(box_size) == 0.0
    soft_mask = circular_soft_mask(box_size, falloff=5.0)
    tiltseries_pixel_size = tiltseries_row_entry["rlnTomoTiltSeriesPixelSize"]
    tiltseries_x = tiltseries_data.data.shape[2]
    tiltseries_y = tiltseries_data.data.shape[1]
    logger.debug(f"Tiltseries data shape: {tiltseries_data.data.shape}, pixel size: {tiltseries_pixel_size}")
    projection_matrices = calculate_projection_matrix_from_starfile_df(individual_tiltseries_df)
    particles_to_tiltseries_coordinates = get_particles_to_tiltseries_coordinates(filtered_particles_df, individual_tiltseries_df, projection_matrices)

    # ctf & dose-weighting parameters
    voltage = tiltseries_row_entry["rlnVoltage"]
    spherical_aberration = tiltseries_row_entry["rlnSphericalAberration"]
    amplitude_contrast = tiltseries_row_entry["rlnAmplitudeContrast"]
    handedness = tiltseries_row_entry["rlnTomoHand"]
    phase_shift = tiltseries_row_entry["rlnPhaseShift"] if "rlnPhaseShift" in optics_row.columns else 0.0
    defocus_u = individual_tiltseries_df["rlnDefocusU"].values
    defocus_v = individual_tiltseries_df["rlnDefocusV"].values
    defocus_angle = individual_tiltseries_df["rlnDefocusAngle"].values
    doses = individual_tiltseries_df["rlnMicrographPreExposure"].values
    bfactor_per_electron_dose = individual_tiltseries_df["rlnCtfBfactorPerElectronDose"] if "rlnCtfBfactorPerElectronDose" in individual_tiltseries_df.columns else [0.0] * len(individual_tiltseries_df)
    dose_weights = np.stack([calculate_dose_weight_image(dose, tiltseries_pixel_size * bin, box_size, bfactor) for dose, bfactor in zip(doses, bfactor_per_electron_dose)], dtype=np.float32)

    all_particle_data = []
    skipped_particles = set()
    # for rlnTomoVisibleFrames
    all_visible_sections_relion_column = []

    for particle_id, sections in particles_to_tiltseries_coordinates.items():
        particle_data, visible_sections = get_particle_crop_and_visibility(tiltseries_data, particle_id, sections, tiltseries_x, tiltseries_y, tiltseries_pixel_size, pre_bin_box_size)

        # RELION by default only requires one tilt to be visible, so only skip if all sections are out of bounds (and also don't append since the entry doesn't exist in the particles.star file)
        if len(particle_data) == 0:
            # logger.debug(f"Particle {particle_id} in tomogram {particles_tomo_name} has {len(particle_data)} visible tilts. Skipping this particle.")
            skipped_particles.add(particle_id)
            continue

        all_particle_data.append(particle_data)
        # modify the values to be in RELION format
        all_visible_sections_relion_column.append(str(visible_sections).replace(" ", ""))

    start_time = time.time()
    tiltseries_data.compute_crops() # compute all cached slices
    end_time = time.time()
    logger.debug(f"Downloading crops took {end_time - start_time:.2f} seconds")

    def process_particle_data(particle_data):
        particle_id = particle_data[0]["particle_id"]
        tilt_data = np.zeros((len(particle_data), box_size, box_size))

        for tilt in range(len(particle_data)):
            section: int = particle_data[tilt]["section"]
            tiltseries_slice_key: tuple[int, slice, slice] = particle_data[tilt]["tiltseries_slice_key"]
            subpixel_shift: tuple[int, int] = particle_data[tilt]["subpixel_shift"]
            coordinate: np.ndarray = particle_data[tilt]["coordinate"]

            current_tilt = tiltseries_data[tiltseries_slice_key]
            fourier_tilt = np.fft.rfft2(current_tilt, norm="ortho")

            fourier_tilt = fourier_shift(fourier_tilt, subpixel_shift, n=current_tilt.shape[0], axis=1)

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
                    phase_shift=phase_shift,
                    defocus_u=defocus_u[section - 1],
                    defocus_v=defocus_v[section - 1],
                    defocus_angle=defocus_angle[section - 1],
                    dose=doses[section - 1],
                    bfactor=bfactor_per_electron_dose[section - 1],
                    box_size=box_size,
                    bin=bin,
                )
                fourier_tilt *= dose_weights[section - 1, :, :] * ctf_weights
            fourier_tilt *= -1  # phase flip for RELION compatibility
            fourier_tilt /= float(bin) ** 2  # normalize by binning factor
            current_tilt = np.fft.irfft2(fourier_tilt, norm="ortho")

            # remove noise via background subtraction and apply soft circular mask
            if not no_circle_crop:
                background_data_mean = np.mean(current_tilt, where=background_mask)
                current_tilt -= background_data_mean
                current_tilt *= soft_mask

            tilt_data[tilt] = current_tilt

        output_path = output_folder / f"{particle_id}_stack2d.mrcs"
        with mrcfile.new(output_path, overwrite=True) as mrc:
            mrc.set_data(tilt_data.astype(np.float16 if float16 else np.float32))
            mrc.voxel_size = (tiltseries_pixel_size * bin, tiltseries_pixel_size * bin, 1.0)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor: # emperically determined 4 threads is optimal for this task
        futures = [executor.submit(process_particle_data, particle_data) for particle_data in all_particle_data]
        for future in as_completed(futures):
            future.result()

    updated_filtered_particles_df = update_particles_df(filtered_particles_df, output_folder, all_visible_sections_relion_column, skipped_particles)

    end_time = time.time()
    logger.debug(
        f"Extracted subtomograms for {particles_tomo_name} with {len(particles_to_tiltseries_coordinates)} particles, skipped {len(skipped_particles)} particles due to out-of-bounds coordinates, took {end_time - start_time:.2f} seconds."
    )

    return updated_filtered_particles_df, len(skipped_particles)


def write_starfiles(merged_particles_df: pd.DataFrame, particle_optics_df: pd.DataFrame, tomograms_starfile: str, box_size: int, bin: int, no_ctf: bool, output_dir: Path) -> None:
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
    updated_optics_df["rlnImageSize"] = box_size

    general_df = {"rlnTomoSubTomosAre2DStacks": 1}
    starfile.write(
        {
            "general": general_df,
            "optics": updated_optics_df,
            "particles": merged_particles_df,
        },
        output_dir / "particles.star",
    )
    optimisation_set_df = pd.DataFrame(
        {
            "rlnTomoParticlesFile": [(output_dir / "particles.star").resolve()],
            "rlnTomoTomogramsFile": [tomograms_starfile.resolve()],
        }
    )
    starfile.write(optimisation_set_df, output_dir / "optimisation_set.star", overwrite=True)


def extract_subtomograms(
    particles_starfile: str,
    box_size: int,
    bin: int,
    tiltseries_relative_dir: Path,
    tomograms_starfile: str,
    float16: bool,
    no_ctf: bool,
    no_circle_crop: bool,
    output_dir: Path,
) -> tuple[int, int, int]:
    """
    Extracts subtomograms from a provided particles *.star file, tiltseries *.star file, and set of *.aln files.
    Creates new *.mrcs files for each particle in the output directory, as well as updated particles and optimisation set star files.
    Uses multiprocessing to speed up the extraction process.
    
    Returns:
        tuple: Number of particles extracted, number of skipped particles, number of tiltseries processed.
    """
    particles_star_file = starfile.read(particles_starfile)
    particles_df = particles_star_file["particles"]
    # tiltseries files with per-tilt / per-section entries

    # TODO: filter list by what is in the tomograms star file
    tomograms_df = starfile.read(tomograms_starfile)

    def build_args(tiltseries_row_entry: pd.Series) -> tuple:
        """Builds the arguments for processing a single tiltseries and its particles (to be passed into process_tiltseries)."""
        individual_tiltseries_path = tiltseries_relative_dir / tiltseries_row_entry["rlnTomoTiltSeriesStarFile"]
        filtered_particles_df = particles_df[particles_df["rlnTomoName"] == tiltseries_row_entry["rlnTomoName"]]
        if filtered_particles_df.empty:
            raise ValueError(f"No particles found for tomogram {tiltseries_row_entry["rlnTomoName"]} in {particles_starfile}. Please check the particles star file.")
        optics_row = particles_star_file["optics"][particles_star_file["optics"]["rlnOpticsGroupName"] == tiltseries_row_entry["rlnOpticsGroupName"]]
        return (
            filtered_particles_df,
            box_size,
            bin,
            individual_tiltseries_path,
            tiltseries_row_entry,
            optics_row,
            float16,
            no_ctf,
            no_circle_crop,
            output_dir,
        )
    
    if "rlnTomoTiltSeriesStarFile" not in tomograms_df.columns:
        raise ValueError(f"Tomograms star file {tomograms_starfile} does not contain the required column 'rlnTomoTiltSeriesStarFile'. Please check the file.")

    args_list = [build_args(tiltseries_row_entry) for _, tiltseries_row_entry in tomograms_df.iterrows()]

    # do actual subtomogram extraction & .mrcs file creation here
    total_skipped_count = 0
    particles_df_results = []
    cpu_count = min(64, mp.cpu_count(), len(tomograms_df))
    logger.info(f"Starting extraction of subtomograms from {len(tomograms_df)} tiltseries using {cpu_count} CPU cores.")
    with mp.Pool(processes=cpu_count) as pool:
        for updated_filtered_particles_df, skipped_count in tqdm(pool.imap_unordered(process_tiltseries, args_list, chunksize=1), total=len(args_list)):
            if updated_filtered_particles_df is not None and not updated_filtered_particles_df.empty:
                particles_df_results.append(updated_filtered_particles_df)
            total_skipped_count += skipped_count

    if not particles_df_results:
        raise ValueError("No particles were extracted. Please check the input files and parameters.")

    merged_particles_df = pd.concat(particles_df_results, ignore_index=True)
    # update all the relevant star files
    write_starfiles(merged_particles_df, particles_star_file["optics"], tomograms_starfile, box_size, bin, no_ctf, output_dir)

    return len(merged_particles_df), total_skipped_count, len(tomograms_df)


def parse_extract_local_subtomograms(args):
    start_time = time.time()
    if not args.particles_starfile.exists():
        raise FileNotFoundError(f"Particles star file '{args.particles_starfile}' does not exist.")
    if not args.tomograms_starfile.exists():
        raise FileNotFoundError(f"Tomograms star file '{args.tomograms_starfile}' does not exist.")

    particles_count, total_skipped_count, individual_tiltseries_count = extract_subtomograms(
        particles_starfile=args.particles_starfile,
        box_size=args.box_size,
        bin=args.bin,
        tiltseries_relative_dir=args.tiltseries_relative_dir,
        tomograms_starfile=args.tomograms_starfile,
        float16=args.float16,
        no_ctf=args.no_ctf,
        no_circle_crop=args.no_circle_crop,
        output_dir=args.output_dir,
    )
    end_time = time.time()
    logger.info(
        f"Subtomogram extraction completed in {end_time - start_time:.2f} seconds. Extracted {particles_count} particles from {individual_tiltseries_count} tiltseries, skipped {total_skipped_count} particles due to out-of-bounds coordinates."
    )


def parse_extract_data_portal_subtomograms(args):
    start_time = time.time()
    data_portal_args = {arg_ref: getattr(args, arg_ref) for arg_ref in args_common.DATA_PORTAL_ARG_REFS}

    particles_path, tomograms_path, _ = generate_starfiles(
        output_dir=args.output_dir,
        **data_portal_args,
        debug=args.debug,
    )

    if not particles_path.exists():
        raise ValueError(f"Starfile generation failed. Expected particles star file at {particles_path} does not exist.")
    
    if not tomograms_path.exists():
        raise ValueError(f"Starfile generation failed. Expected tomograms star file at {tomograms_path} does not exist.")

    particles_count, total_skipped_count, individual_tiltseries_count = extract_subtomograms(
        particles_starfile=particles_path,
        box_size=args.box_size,
        bin=args.bin,
        tiltseries_relative_dir=args.output_dir,
        tomograms_starfile=tomograms_path,
        float16=args.float16,
        no_ctf=args.no_ctf,
        no_circle_crop=args.no_circle_crop,
        output_dir=args.output_dir,
    )

    end_time = time.time()
    logger.info(
        f"Subtomogram extraction completed in {end_time - start_time:.2f} seconds. Extracted {particles_count} particles from {individual_tiltseries_count} tiltseries, skipped {total_skipped_count} particles due to out-of-bounds coordinates."
    )

def main():
    parser = argparse.ArgumentParser(description="Extract subtomograms from a provided particles *.star file, and tiltseries *.star files.")
    subparser = parser.add_subparsers(dest="command", required=True)

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--box-size", type=int, required=True, help="Box size of the extracted subtomograms in pixels.")
    common_parser.add_argument("--bin", type=int, default=1, help="Binning factor for the subtomograms. Default is 1 (no binning).")
    common_parser.add_argument("--float16", action="store_true", help="Use float16 precision for the output mrcs files. Default is False (float32).")
    common_parser.add_argument("--no-ctf", action="store_true", help="Disable CTF premultiplication.")
    common_parser.add_argument("--no-circle-crop", action="store_true", help="Disable circular cropping of the subtomograms")
    common_parser.add_argument("--output-dir", type=Path, required=True, help="Path to the output directory where the extracted subtomograms will be saved.")
    common_parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    local = subparser.add_parser("local", parents=[common_parser], help="Extract subtomograms from local files (particles, tiltseries, and alignment files).")
    # TODO: support multiple starfiles
    local.add_argument("--particles-starfile", type=Path, required=True, help="Path to the particles *.star file.")
    local.add_argument("--tiltseries-relative-dir", type=Path, default="./", help="The directory in which the tiltseries file paths are relative to. Default is the current directory.")
    local.add_argument("--tomograms-starfile", type=Path, required=True, help="Path to the tomograms.star file (containing all tiltseries entries, with entries as tiltseries).")
    # TODO: Make this a filter for only running extraction on specific tiltseries
    # parser.add_argument("--tiltseries-pixel-size", type=float, required=True, help="Pixel size of the tiltseries in Angstroms.")

    data_portal = subparser.add_parser("data-portal", parents=[common_parser], help="Extract subtomograms from a CryoET Data Portal run.")
    args_common.add_data_portal_args(data_portal)

    args = parser.parse_args()

    if args.box_size % 2 != 0:
        raise ValueError(f"Box size must be an even number, got {args.box_size}.")

    (args.output_dir / "Subtomograms").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.command == "local":
        parse_extract_local_subtomograms(args)
    elif args.command == "data-portal":
        parse_extract_data_portal_subtomograms(args)


if __name__ == "__main__":
    main()
