"""
Primary entry point for extracting subtomograms from a CryoET Data Portal run.
Run python data_portal_subtomo_extract.py --help for usage instructions.
"""
import os
import argparse
import logging
import starfile
import numpy as np
import pandas as pd
import mrcfile
import time
from scipy.ndimage import fourier_shift
from typing import Union
from tqdm import tqdm
from multiprocessing import Pool
from cryoet_data_portal import Client, Run
from cryoet_alignment.io.aretomo3 import AreTomo3ALN
from cryoet_alignment.io.cryoet_data_portal import Alignment

from core.projection import calculate_projection_matrix_from_aretomo_aln, get_particles_to_tiltseries_coordinates, fourier_crop, get_particle_crop_and_visibility
from core.mask import circular_mask, circular_soft_mask
from core.dose import calculate_dose_weight_image
from core.ctf import calculate_ctf

logger = logging.getLogger(__name__)


# TODO: Implement subtomomgram extraction from a CryoET Data Portal run
def extract_subtomogram_from_run(run: Run, output_dir: str):
    pass


def update_particles_df(particles_df: pd.DataFrame, output_folder: str, all_visible_sections_column: list, skipped_particles: set) -> pd.DataFrame:
    """Updates the particles DataFrame to include the new columns and values for RELION format."""
    updated_particles_df = particles_df.copy()
    updated_particles_df = updated_particles_df.drop(columns=["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"], errors="ignore")
    updated_particles_df = updated_particles_df.reset_index(drop=True)
    updated_particles_df.index += 1  # increment index by 1 to match RELION's 1-indexing
    updated_particles_df["rlnTomoParticleName"] = updated_particles_df["rlnTomoName"] + "/" + updated_particles_df.index.astype(str)
    updated_particles_df["rlnImageName"] = updated_particles_df.index.to_series().apply(lambda idx: os.path.abspath(os.path.join(output_folder, f"{idx}_stack2d.mrcs")))
    updated_particles_df = updated_particles_df.drop(index=skipped_particles, errors="ignore")  # drop rows by particle_id that were skipped
    updated_particles_df["rlnTomoVisibleFrames"] = all_visible_sections_column
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
    aln_file_path, particles_tomo_name, filtered_particles_df, box_size, bin, tiltseries_dir, tiltseries_x, tiltseries_y, tiltseries_row_entry, optics_row, float16, no_ctf, no_circle_crop, output_dir = args
    pre_bin_box_size = box_size * bin


    # TODO: can also load from the tiltseries file, aln not required? but might be less reliable?
    tiltseries_file = os.path.basename(aln_file_path).replace(".aln", ".star")
    tiltseries_path = os.path.join(tiltseries_dir, tiltseries_file)
    if not os.path.exists(tiltseries_path):
        logger.warning(f"Tiltseries file {tiltseries_file} not found for alignment file {aln_file_path}. Skipping.")
        return

    output_folder = os.path.join(output_dir, "Subtomograms", particles_tomo_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    logger.debug(
        f"Extracting subtomograms for {len(filtered_particles_df)} particles (filtered by rlnTomoName: {particles_tomo_name}) from tiltseries {tiltseries_file} with alignment {os.path.basename(aln_file_path)}."
    )

    tiltseries_df = starfile.read(tiltseries_path)
    if tiltseries_df.empty or tiltseries_row_entry.empty:
        raise ValueError(f"Tiltseries file {tiltseries_file} is empty or does not contain the required columns. Please check the file.")
    tiltseries_mrc_file = tiltseries_df["rlnMicrographName"].iloc[0].split("@")[1]
    # if a relative path, consider it relative to the tiltseries path
    if not tiltseries_mrc_file.startswith("/"):
        tiltseries_mrc_file = os.path.join(tiltseries_dir, tiltseries_mrc_file)

    # projection-relevant variables
    background_mask = circular_mask(box_size) == 0.0
    soft_mask = circular_soft_mask(box_size, falloff=5.0)
    tiltseries_pixel_size = tiltseries_row_entry["rlnTomoTiltSeriesPixelSize"].values[0]
    projection_matrices = calculate_projection_matrix_from_aretomo_aln(AreTomo3ALN.from_file(aln_file_path), tiltseries_pixel_size=tiltseries_pixel_size)
    particles_to_tiltseries_coordinates = get_particles_to_tiltseries_coordinates(filtered_particles_df, tiltseries_df, projection_matrices)

    # ctf & dose-weighting parameters
    voltage = tiltseries_row_entry["rlnVoltage"].values[0]
    spherical_aberration = tiltseries_row_entry["rlnSphericalAberration"].values[0]
    amplitude_contrast = tiltseries_row_entry["rlnAmplitudeContrast"].values[0]
    handedness = tiltseries_row_entry["rlnTomoHand"].values[0]
    phase_shift = tiltseries_row_entry["rlnPhaseShift"].values[0] if "rlnPhaseShift" in optics_row.columns else 0.0
    defocus_u = tiltseries_df["rlnDefocusU"].values
    defocus_v = tiltseries_df["rlnDefocusV"].values
    defocus_angle = tiltseries_df["rlnDefocusAngle"].values
    doses = tiltseries_df["rlnMicrographPreExposure"].values
    bfactor_per_electron_dose = tiltseries_df["rlnCtfBfactorPerElectronDose"] if "rlnCtfBfactorPerElectronDose" in tiltseries_df.columns else [0.0] * len(tiltseries_df)
    dose_weights = np.stack([calculate_dose_weight_image(dose, tiltseries_pixel_size * bin, box_size, bfactor) for dose, bfactor in zip(doses, bfactor_per_electron_dose)], dtype=np.float32)

    # after mapping all particles to tiltseries coordinates for this tomogram, we can extract the subtomograms
    with mrcfile.open(tiltseries_mrc_file) as tiltseries_mrc:
        data = tiltseries_mrc.data

    # for rlnTomoVisibleFrames data column
    all_visible_sections_column = []
    skipped_particles = set()
    # Do it particle by particle, so that we can write the particle mrcs in one go
    for particle_id, sections in particles_to_tiltseries_coordinates.items():
        particle_data, visible_sections = get_particle_crop_and_visibility(particle_id, sections, tiltseries_x, tiltseries_y, tiltseries_pixel_size, pre_bin_box_size)

        # RELION by default only requires one tilt to be visible, so only skip if all sections are out of bounds (and also don't append since the entry doesn't exist in the particles.star file)
        if len(particle_data) == 0:
            # logger.debug(f"Particle {particle_id} in tomogram {particles_tomo_name} has {len(particle_data)} visible tilts. Skipping this particle.")
            skipped_particles.add(particle_id)
            continue

        # modify the values to be in RELION format
        all_visible_sections_column.append(str(visible_sections).replace(" ", ""))

        tilt_data = np.zeros((len(particle_data), box_size, box_size))
        for tilt in range(len(particle_data)):
            section = particle_data[tilt]["section"]
            x_start_px = particle_data[tilt]["x_start_px"]
            x_end_px = particle_data[tilt]["x_end_px"]
            y_start_px = particle_data[tilt]["y_start_px"]
            y_end_px = particle_data[tilt]["y_end_px"]
            subpixel_shift = particle_data[tilt]["subpixel_shift"]
            coordinate = particle_data[tilt]["coordinate"]

            # Section is 1 indexed in RELION, so subtract 1 for 0-indexed Python arrays
            current_tilt = data[section - 1, y_start_px:y_end_px, x_start_px:x_end_px]
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
            fourier_tilt *= -1 # phase flip for RELION compatibility
            fourier_tilt /= float(bin) ** 2 # normalize by binning factor
            current_tilt = np.fft.irfft2(fourier_tilt, norm="ortho")

            # remove noise via background subtraction and apply soft circular mask
            if not no_circle_crop:
                background_data_mean = np.mean(current_tilt, where=background_mask)
                current_tilt -= background_data_mean
                current_tilt *= soft_mask

            current_tilt = current_tilt
            tilt_data[tilt] = current_tilt

        with mrcfile.new(os.path.join(output_folder, f"{particle_id}_stack2d.mrcs"), overwrite=True) as mrc:
            mrc.set_data(tilt_data.astype(np.float16 if float16 else np.float32))
            mrc.voxel_size = (tiltseries_pixel_size * bin, tiltseries_pixel_size * bin, 1.0)

    updated_filtered_particles_df = update_particles_df(filtered_particles_df, output_folder, all_visible_sections_column, skipped_particles)

    logger.debug(
        f"Extracted subtomograms for {particles_tomo_name} with {len(particles_to_tiltseries_coordinates)} particles, skipped {len(skipped_particles)} particles due to out-of-bounds coordinates."
    )

    return updated_filtered_particles_df, len(skipped_particles)


def write_starfiles(merged_particles_df: pd.DataFrame, particle_optics_df: pd.DataFrame, tiltseries_starfile: str, box_size: int, bin: int, no_ctf: bool, output_dir: str) -> None:
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
        os.path.join(output_dir, "particles.star"),
    )
    optimisation_set_df = pd.DataFrame(
        {
            "rlnTomoParticlesFile": [os.path.abspath(os.path.join(output_dir, "particles.star"))],
            "rlnTomoTomogramsFile": [os.path.abspath(tiltseries_starfile)],
        }
    )
    starfile.write(optimisation_set_df, os.path.join(output_dir, "optimisation_set.star"), overwrite=True)


def extract_local_subtomograms(
    particles_starfile: str,
    particles_tomo_name_prefix: str,
    box_size: int,
    bin: int,
    tiltseries_dir: str,
    tiltseries_starfile: str,
    tiltseries_x: int,
    tiltseries_y: int,
    aln_dir: str,
    float16: bool,
    no_ctf: bool,
    no_circle_crop: bool,
    output_dir: str,
) -> None:
    """
    Extracts subtomograms from a provided particles *.star file, tiltseries *.star file, and set of *.aln files.
    Creates new *.mrcs files for each particle in the output directory, as well as updated particles and optimisation set star files.
    Uses multiprocessing to speed up the extraction process.
    """
    start_time = time.time()
    particles_star_file = starfile.read(particles_starfile)
    particles_df = particles_star_file["particles"]
    # TODO: make this more adaptable? Look in other common directories? make it based on the tiltseries file? have to figure out pathing issues though
    tiltseries_files = [f for f in os.listdir(tiltseries_dir) if f.endswith(".star")]
    aln_files = [f for f in os.listdir(aln_dir) if f.endswith(".aln")]
    if len(tiltseries_files) == 0:
        raise ValueError(f"No tiltseries files found in {tiltseries_dir}. Expected files with .star extension.")
    if len(aln_files) == 0:
        raise ValueError(f"No alignment files found in {aln_dir}. Expected files with .aln extension.")

    if len(aln_files) != len(tiltseries_files):
        logger.warning(f"Number of alignment files ({len(aln_files)}) does not match number of tiltseries files ({len(tiltseries_files)}). This may lead to missing subtomograms.")

    aln_files.sort()
    tiltseries_files.sort()

    logger.debug(f"Found alignment files: {aln_files}")
    logger.debug(f"Found tiltseries files: {tiltseries_files}")

    # TODO: filter list by what is in the tiltseries star file
    tiltseries_df = starfile.read(tiltseries_starfile)

    def build_args(aln_file):
        """Builds the arguments for processing a single tiltseries and its particles (to be passed into process_tiltseries)."""
        aln_file_path = os.path.join(aln_dir, aln_file)
        particles_tomo_name = f"{particles_tomo_name_prefix}{aln_file.replace('.aln', '')}"
        filtered_particles_df = particles_df[particles_df["rlnTomoName"] == particles_tomo_name]
        tiltseries_row_entry = tiltseries_df[tiltseries_df["rlnTomoName"] == particles_tomo_name]
        optics_row = particles_star_file["optics"][particles_star_file["optics"]["rlnOpticsGroupName"] == tiltseries_row_entry["rlnOpticsGroupName"].values[0]]
        return (
            aln_file_path,
            particles_tomo_name,
            filtered_particles_df,
            box_size,
            bin,
            tiltseries_dir,
            tiltseries_x,
            tiltseries_y,
            tiltseries_row_entry,
            optics_row,
            float16,
            no_ctf,
            no_circle_crop,
            output_dir,
        )

    args_list = [build_args(aln_file) for aln_file in aln_files]

    # do actual subtomogram extraction & .mrcs file creation here
    total_skipped_count = 0
    particles_df_results = []
    cpu_count = min(64, os.cpu_count(), len(aln_files))
    logger.info(f"Starting extraction of subtomograms from {len(aln_files)} alignments using {cpu_count} CPU cores.")
    with Pool(processes=cpu_count) as pool:
        try:
            for updated_filtered_particles_df, skipped_count in tqdm(pool.imap_unordered(process_tiltseries, args_list, chunksize=1), total=len(args_list)):
                if updated_filtered_particles_df is not None and not updated_filtered_particles_df.empty:
                    particles_df_results.append(updated_filtered_particles_df)
                total_skipped_count += skipped_count
        except Exception as e:
            end_time = time.time()
            logger.error(f"Subtomogram extraction failed after {end_time - start_time:.2f} seconds.")
            pool.terminate()
            raise e

    if not particles_df_results:
        raise ValueError("No particles were extracted. Please check the input files and parameters.")

    merged_particles_df = pd.concat(particles_df_results, ignore_index=True)
    # update all the relevant star files
    write_starfiles(merged_particles_df, particles_star_file["optics"], tiltseries_starfile, box_size, bin, no_ctf, output_dir)

    end_time = time.time()
    logger.info(
        f"Subtomogram extraction completed in {end_time - start_time:.2f} seconds. Extracted {len(merged_particles_df)} particles from {len(aln_files)} alignments, skipped {total_skipped_count} particles due to out-of-bounds coordinates."
    )


# TODO: might have to refactor this to use subparsers and permit multiple input (copick projects?, CryoET Data Portal)
def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Extract subtomograms from a provided particles *.star file, tiltseries *.star file, and set of *.aln files.")
    subparser = parser.add_subparsers(dest="command", required=True)
    
    local = subparser.add_parser("local", help="Extract subtomograms from local files (particles, tiltseries, and alignment files).")
    # TODO: support multiple starfiles, dirs
    local.add_argument("--particles-starfile", type=str, required=True, help="Path to the particles *.star file.")
    local.add_argument(
        "--particles-tomo-name-prefix",
        type=str,
        default="",
        help="An added prefix to the tomogram names in the particles star file. Used to properly match the particles to the tiltseries and alignment files.",
    )
    # TODO: consolidate the two to one? just find the tiltseries star file in the tiltseries dir (can be done if we restrain to just pyrelion format)
    local.add_argument("--tiltseries-dir", type=str, required=True, help="Path to the tiltseries directory containing *.star files (individual tiltseries, with entries as individual tilts).")
    local.add_argument("--tiltseries-starfile", type=str, required=True, help="Path to the tiltseries star file (containing all tiltseries entries, with entries as tiltseries).")
    # TODO: Make this a filter for only running extraction on specific tiltseries
    # parser.add_argument("--tiltseries-pixel-size", type=float, required=True, help="Pixel size of the tiltseries in Angstroms.")
    local.add_argument("--tiltseries-x", type=int, required=True, help="X dimension of the tiltseries in pixels.")
    local.add_argument("--tiltseries-y", type=int, required=True, help="Y dimension of the tiltseries in pixels.")
    local.add_argument("--aln-dir", type=str, required=True, help="Path to the directory containing the *.aln files (should be named the same as the tiltseries).")
    
    data_portal = subparser.add_parser("data_portal", help="Extract subtomograms from a CryoET Data Portal run.")
    data_portal.add_argument("--run-id", type=str, required=True, help="ID of the CryoET Data Portal run to extract subtomograms from.")

    parser.add_argument("--box-size", type=int, required=True, help="Box size of the extracted subtomograms in pixels.")
    parser.add_argument("--bin", type=int, default=1, help="Binning factor for the subtomograms. Default is 1 (no binning).")
    parser.add_argument("--float16", action="store_true", help="Use float16 precision for the output mrcs files. Default is False (float32).")
    parser.add_argument("--no-ctf", action="store_true", help="Disable CTF premultiplication.")
    parser.add_argument("--no-circle-crop", action="store_true", help="Disable circular cropping of the subtomograms")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory where the extracted subtomograms will be saved.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if not os.path.exists(args.particles_starfile):
        raise FileNotFoundError(f"Particles star file '{args.particles_starfile}' does not exist.")
    if not os.path.exists(args.tiltseries_dir):
        raise FileNotFoundError(f"Tiltseries directory '{args.tiltseries_dir}' does not exist.")
    if not os.path.exists(args.tiltseries_starfile):
        raise FileNotFoundError(f"Tiltseries star file '{args.tiltseries_starfile}' does not exist.")
    if not os.path.exists(args.aln_dir):
        raise FileNotFoundError(f"Alignment directory '{args.aln_dir}' does not exist.")
    if args.box_size % 2 != 0:
        raise ValueError(f"Box size must be an even number, got {args.box_size}.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(os.path.join(args.output_dir, "Subtomograms")):
        os.makedirs(os.path.join(args.output_dir, "Subtomograms"))

    if args.debug:
        logger.setLevel(logging.DEBUG)

    extract_local_subtomograms(
        particles_starfile=args.particles_starfile,
        particles_tomo_name_prefix=args.particles_tomo_name_prefix,
        box_size=args.box_size,
        bin=args.bin,
        tiltseries_dir=args.tiltseries_dir,
        tiltseries_starfile=args.tiltseries_starfile,
        tiltseries_x=args.tiltseries_x,
        tiltseries_y=args.tiltseries_y,
        aln_dir=args.aln_dir,
        float16=args.float16,
        no_ctf=args.no_ctf,
        no_circle_crop=args.no_circle_crop,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
