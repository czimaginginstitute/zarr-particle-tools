# NOTE: This extraction is a grossly simplified version of the RELION subtomogram extraction and is missing support for several flags & data elements. Known limitations:
# - Does not support gamma offset
# - Does not support spherical aberration correction
# - Does not support subpixel shifts
# - Does not support circle precrop
# - Does not support grid precorrection
# - Does not support whitening (power spectral flattening)
# - Does not support 3D volume extraction
# - Does not support min_frames or max_dose flags
# - Does not write any other *.mrcs files other than the 2D stacks themselves
# - Does not support defocus slope (rlnTomoDefocusSlope)
# - Does not (currently) support binning
# - Does not support float16

# TODO: List of TODO (in order of priority):
# TODO: Clean up code
# TODO: - Fix path issues & reduce needed paths
# TODO: - Modularize this code to different files
# TODO: Determine why the extracted subtomograms are slightly different (even without)
#       - could possibly be due to the fourier transform they do? investigate in fourier space?
# TODO: (In tandem with above) Write tests (using synthetic data this should be pretty easy, just compare against RELION output, serving also as a tracker for if RELION output ever changes)
# TODO: Implement binning
# TODO: Support multiple optics groups
# TODO: Incorporate alpha and beta offset parameters from AreTomo .aln file (for additional rotation) (and from CryoET Data Portal? if it exists?)
# TODO: Notify aretomo of this work and possible integration into their codebase
# TODO: Data Portal, copick support 

import os
import argparse
import logging
import starfile
import numpy as np
import pandas as pd
import mrcfile
import time
from projection import project_3d_point_to_2d, calculate_projection_matrix_from_aretomo_aln, circular_mask, circular_soft_mask
from ctf import calculate_dose_weight_image, calculate_ctf
from tqdm import tqdm
from multiprocessing import Pool
from cryoet_data_portal import Client, Run
from cryoet_alignment.io.aretomo3 import AreTomo3ALN
from cryoet_alignment.io.cryoet_data_portal import Alignment

logger = logging.getLogger(__name__)


# TODO: Refactor this to take in PerSectionAlignmentParameters instead of Run?
def calculate_projection_matrix_from_run(run: Run) -> dict[int, np.ndarray]:
    """
    Calculates the projection matrices for each section in the given Run object.

    Args:
        run (Run): A Run object containing the alignment parameters.

    Returns:
        dict[int, np.ndarray]: A dictionary of 4x4 projection matrices for each section. (section ID as key)
    """
    pass

def extract_subtomogram_from_run(run: Run, output_dir: str):
    pass


def process_aln_file(args):
    aln_file_path, particles_tomo_name, filtered_particles_df, box_size, bin, tiltseries_dir, tiltseries_x, tiltseries_y, tiltseries_row_entry, optics_row, output_dir, debug = args

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
    tiltseries_mrc_file = tiltseries_df["rlnMicrographName"].iloc[0].split("@")[1]

    # particle data
    particles_to_tiltseries_coordinates = {}
    skipped_particles = set()

    # projection-relevant variables
    background_mask = circular_mask(box_size) == 0.0
    soft_mask = circular_soft_mask(box_size, falloff=5.0)
    tilt_angles = tiltseries_df["rlnTomoYTilt"].values
    tiltseries_pixel_size = tiltseries_row_entry["rlnTomoTiltSeriesPixelSize"].values[0]
    projection_matrices = calculate_projection_matrix_from_aretomo_aln(AreTomo3ALN.from_file(aln_file_path), tiltseries_pixel_size=tiltseries_pixel_size)

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
    dose_weights = np.stack([calculate_dose_weight_image(dose, tiltseries_pixel_size, box_size, bfactor) for dose, bfactor in zip(doses, bfactor_per_electron_dose)], dtype=np.float32)


    # loop over each tilt in the tiltseries (idea is this outer loop so that if we do end up doing speedup, we can parallelize this)
    for _, tilt in tiltseries_df.iterrows():
        section = int(tilt["rlnMicrographName"].split("@")[0])
        projection_matrix = projection_matrices[section - 1]

        # match 1-indexing of RELION
        for particle_count, particle in enumerate(filtered_particles_df.itertuples(), start=1):
            coordinate = np.array([particle.rlnCenteredCoordinateXAngst, particle.rlnCenteredCoordinateYAngst, particle.rlnCenteredCoordinateZAngst])
            projected_point = project_3d_point_to_2d(coordinate, projection_matrix)[:2]

            if particle_count not in particles_to_tiltseries_coordinates:
                particles_to_tiltseries_coordinates[particle_count] = {}

            particles_to_tiltseries_coordinates[particle_count][section] = (coordinate, projected_point)

    # after mapping all particles to tiltseries coordinates for this tomogram, we can extract the subtomograms
    with mrcfile.open(tiltseries_mrc_file) as tiltseries_mrc:
        data = tiltseries_mrc.data

        # for rlnTomoVisibleFrames data column
        all_visible_sections_column = []
        # Do it particle by particle, so that we can write the particle mrcs in one go
        for particle_count, sections in particles_to_tiltseries_coordinates.items():
            particle_data = []
            # individual particle entry's visible sections
            visible_sections = []
            for section, coords in sections.items():
                coordinate, projected_point = coords
                x, y = projected_point
                # convert back from centered angstrom to pixel coordinates
                x_px = round((x + tiltseries_x * tiltseries_pixel_size / 2.0) / tiltseries_pixel_size)
                y_px = round((y + tiltseries_y * tiltseries_pixel_size / 2.0) / tiltseries_pixel_size)
                x_start_px = x_px - box_size // 2
                y_start_px = y_px - box_size // 2
                x_end_px = x_start_px + box_size
                y_end_px = y_start_px + box_size
                
                if x_start_px < 0 or x_end_px > tiltseries_x or y_start_px < 0 or y_end_px > tiltseries_y:
                    visible_sections.append(0)
                    # logger.debug(f"{particles_tomo_name} Subtomogram extraction for particle {particle_count} in section {section} at {tilt_angles[section - 1]} would exceed tiltseries bounds. x_start={x_start_px}, x_end={x_end_px}, y_start={y_start_px}, y_end={y_end_px}. Original 3D coordinates: {coordinate}. Skipping this section.")
                    continue

                visible_sections.append(1)
                particle_data.append(
                    {
                        "particle_count": particle_count,
                        "coordinate": coordinate,
                        "section": section,
                        "x_start_px": x_start_px,
                        "y_start_px": y_start_px,
                        "x_end_px": x_end_px,
                        "y_end_px": y_end_px,
                    }
                )

            # RELION by default only requires one tilt to be visible, so only skip if all sections are out of bounds (and also don't append since the entry doesn't exist in the particles.star file)
            if len(particle_data) == 0:
                # logger.debug(f"Particle {particle_count} in tomogram {particles_tomo_name} has {len(particle_data)} visible tilts. Skipping this particle.")
                skipped_particles.add(particle_count)
                continue

            all_visible_sections_column.append(str(visible_sections).replace(" ", ""))

            tilt_data = np.zeros((len(particle_data), box_size, box_size), dtype=np.float32)
            for tilt in range(len(particle_data)):
                section = particle_data[tilt]["section"]
                x_start_px = particle_data[tilt]["x_start_px"]
                x_end_px = particle_data[tilt]["x_end_px"]
                y_start_px = particle_data[tilt]["y_start_px"]
                y_end_px = particle_data[tilt]["y_end_px"]
                coordinate= particle_data[tilt]["coordinate"]

                # TODO: handle binning here
                # Section is 1 indexed in RELION, so subtract 1 for 0-indexed Python arrays
                current_tilt = tilt_data[tilt, :, :]
                current_tilt[:] = data[section - 1, y_start_px:y_end_px, x_start_px:x_end_px]
                
                # TODO: look into subpixel shift?
                fourier_tilt = np.fft.rfft2(current_tilt)
                # TODO: look into gamma offset
                # TODO: implement spherical aberration correction
                # TODO: implement binning for CTF application
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
                )
                fourier_tilt *= -1 * dose_weights[section - 1, :, :] * ctf_weights

                current_tilt[:] = np.fft.irfft2(fourier_tilt).real
                
                # remove noise via background subtraction and apply soft circular mask
                background_data_mean = np.mean(current_tilt, where=background_mask)
                current_tilt -= background_data_mean
                current_tilt *= soft_mask
                

            with mrcfile.new(os.path.join(output_folder, f"{particle_count}_stack2d.mrcs"), overwrite=True) as mrc:
                mrc.set_data(tilt_data)
                mrc.voxel_size = (tiltseries_pixel_size, tiltseries_pixel_size, 1)

    # Return an updated dataframe
    updated_filtered_particles_df = filtered_particles_df.copy()
    updated_filtered_particles_df = updated_filtered_particles_df.drop(columns=["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"], errors="ignore")
    updated_filtered_particles_df = updated_filtered_particles_df.reset_index(drop=True)
    updated_filtered_particles_df.index += 1 # increment index by 1 to match RELION's 1-indexing
    updated_filtered_particles_df["rlnTomoParticleName"] = updated_filtered_particles_df["rlnTomoName"] + "/" + updated_filtered_particles_df.index.astype(str)
    updated_filtered_particles_df["rlnImageName"] = updated_filtered_particles_df.index.to_series().apply(lambda idx: os.path.abspath(os.path.join(output_folder, f"{idx}_stack2d.mrcs")))
    updated_filtered_particles_df = updated_filtered_particles_df.drop(index=skipped_particles, errors="ignore") # drop rows by particle_count that were skipped
    updated_filtered_particles_df["rlnTomoVisibleFrames"] = all_visible_sections_column
    updated_filtered_particles_df["rlnOriginXAngst"] = 0.0
    updated_filtered_particles_df["rlnOriginYAngst"] = 0.0
    updated_filtered_particles_df["rlnOriginZAngst"] = 0.0

    logger.debug(f"Extracted subtomograms for {particles_tomo_name} with {len(particles_to_tiltseries_coordinates)} particles, skipped {len(skipped_particles)} particles due to out-of-bounds coordinates.")

    return updated_filtered_particles_df, len(skipped_particles)


def extract_local_aln_subtomograms(
    particles_starfile: str,
    particles_tomo_name_prefix: str,
    box_size: int,
    bin: int,
    tiltseries_dir: str,
    tiltseries_starfile: str,
    tiltseries_x: int,
    tiltseries_y: int,
    aln_dir: str,
    output_dir: str,
    debug: bool,
):
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
            output_dir,
            debug,
        )

    args_list = [build_args(aln_file) for aln_file in aln_files]

    total_skipped_count = 0
    merged_particles_df = pd.DataFrame()
    cpu_count = min(64, os.cpu_count(), len(aln_files))
    logger.info(f"Starting extraction of subtomograms from {len(aln_files)} alignments using {cpu_count} CPU cores.")
    with Pool(processes=cpu_count) as pool:
        try:
            for updated_filtered_particles_df, skipped_count in tqdm(pool.imap_unordered(process_aln_file, args_list, chunksize=1), total=len(args_list)):
                if updated_filtered_particles_df is not None and not updated_filtered_particles_df.empty:
                    merged_particles_df = pd.concat([merged_particles_df, updated_filtered_particles_df], ignore_index=True)
                total_skipped_count += skipped_count
        except Exception as e:
            end_time = time.time()
            logger.error(f"Subtomogram extraction failed after {end_time - start_time:.2f} seconds.")
            pool.terminate()
            raise e

    merged_particles_df["ParticleCount"] = merged_particles_df["rlnTomoParticleName"].str.split("/").str[-1].astype(int)
    merged_particles_df = merged_particles_df.sort_values(by=["rlnTomoName", "ParticleCount"]).reset_index(drop=True)
    merged_particles_df = merged_particles_df.drop(columns="ParticleCount")

    updated_optics_df = particles_star_file["optics"].copy()
    updated_optics_df["rlnCtfDataAreCtfPremultiplied"] = 1
    updated_optics_df["rlnImageDimensionality"] = 2
    updated_optics_df["rlnTomoSubtomogramBinning"] = bin
    updated_optics_df["rlnImagePixelSize"] = updated_optics_df["rlnTomoTiltSeriesPixelSize"]
    updated_optics_df["rlnImageSize"] = box_size

    general_df = pd.DataFrame({
        "rlnTomoSubTomosAre2DStacks": [1],
    })

    starfile.write(
        {
            "general": general_df,
            "optics": updated_optics_df,
            "particles": merged_particles_df,
        },
        os.path.join(output_dir, "particles.star"),
    )

    optimisation_set_df = pd.DataFrame({
        "rlnTomoParticlesFile": [os.path.abspath(os.path.join(output_dir, "particles.star"))],
        "rlnTomoTomogramsFile": [os.path.abspath(tiltseries_starfile)],
    })

    starfile.write(optimisation_set_df, os.path.join(output_dir, "optimisation_set.star"), overwrite=True)

    end_time = time.time()
    logger.info(f"Subtomogram extraction completed in {end_time - start_time:.2f} seconds. Extracted {len(merged_particles_df)} particles from {len(aln_files)} alignments, skipped {total_skipped_count} particles due to out-of-bounds coordinates.")

# TODO: might have to refactor this to use subparsers and permit multiple input (copick projects?, CryoET Data Portal)
def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Extract subtomograms from a provided particles *.star file, tiltseries *.star file, and set of *.aln files.")
    # TODO: support multiple starfiles, dirs
    parser.add_argument("--particles-starfile", type=str, required=True, help="Path to the particles *.star file.")
    parser.add_argument(
        "--particles-tomo-name-prefix",
        type=str,
        default="",
        help="An added prefix to the tomogram names in the particles star file. Used to properly match the particles to the tiltseries and alignment files.",
    )
    parser.add_argument("--box-size", type=int, required=True, help="Box size of the extracted subtomograms in pixels.")
    parser.add_argument("--bin", type=int, default=1, help="Binning factor for the subtomograms. Default is 1 (no binning).")
    # TODO: consolidate the two to one? just find the tiltseries star file in the tiltseries dir (can be done if we restrain to just pyrelion format)
    parser.add_argument("--tiltseries-dir", type=str, required=True, help="Path to the tiltseries directory containing *.star files (individual tiltseries, with entries as individual tilts).")
    parser.add_argument("--tiltseries-starfile", type=str, required=True, help="Path to the tiltseries star file (containing all tiltseries entries, with entries as tiltseries).")
    # TODO: Make this a filter for only running extraction on specific tiltseries
    # parser.add_argument("--tiltseries-pixel-size", type=float, required=True, help="Pixel size of the tiltseries in Angstroms.")
    parser.add_argument("--tiltseries-x", type=int, required=True, help="X dimension of the tiltseries in pixels.")
    parser.add_argument("--tiltseries-y", type=int, required=True, help="Y dimension of the tiltseries in pixels.")
    parser.add_argument("--aln-dir", type=str, required=True, help="Path to the directory containing the *.aln files (should be named the same as the tiltseries).")
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

    extract_local_aln_subtomograms(
        particles_starfile=args.particles_starfile,
        particles_tomo_name_prefix=args.particles_tomo_name_prefix,
        box_size=args.box_size,
        bin=args.bin,  # TODO: implement binning
        tiltseries_dir=args.tiltseries_dir,
        tiltseries_starfile=args.tiltseries_starfile,
        tiltseries_x=args.tiltseries_x,
        tiltseries_y=args.tiltseries_y,
        aln_dir=args.aln_dir,
        output_dir=args.output_dir,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()