"""
A module for reconstructing from extracted particles. A numerically-precise reimplementation of RELION's Reconstruct particle job (relion_tomo_reconstruct_particle).

Regarding output files:
- data_merged.mrc: the uncorrected reconstruction (after gridding correction)
- merged_full.mrc: the CTF-corrected reconstruction (before spherical masking and mean subtraction)
- merged.mrc: the final reconstruction (after spherical masking and mean subtraction)

The same applies for half1 and half2 if random subsets are present.
"""

# TODO: write tests (at least for local)
import ast
import logging
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path
from typing import Union

import click
import mrcfile
import numpy as np
import pandas as pd
import starfile
from tqdm import tqdm

import zarr_particle_tools.cli.options as cli_options
from zarr_particle_tools.core.backprojection import (
    backproject_slice_backward,
    ctf_correct_3d_heuristic,
    get_rotation_matrix_from_euler,
    gridding_correct_3d_sinc2,
)
from zarr_particle_tools.core.ctf import calculate_ctf
from zarr_particle_tools.core.dose import calculate_dose_weight_image
from zarr_particle_tools.core.forwardprojection import (
    apply_offsets_to_coordinates,
    calculate_projection_matrix_from_starfile_df,
    get_particles_to_tiltseries_coordinates,
)
from zarr_particle_tools.core.helpers import get_tiltseries_data, setup_logging
from zarr_particle_tools.core.mask import spherical_soft_mask
from zarr_particle_tools.core.symmetry import (
    get_transforms_from_symmetry,
    symmetrise_fs_complex,
    symmetrise_fs_real,
)
from zarr_particle_tools.subtomo_extract import (
    parse_extract_data_portal_copick_subtomograms,
    parse_extract_data_portal_subtomograms,
    parse_extract_local_copick_subtomograms,
    parse_extract_local_subtomograms,
)

logger = logging.getLogger(__name__)


def process_particle_wrapper(args):
    return process_particle(**args)


def process_particle(
    particle: pd.Series,
    particle_projection_matrices: np.ndarray,
    particle_coordinates: dict[int, tuple[float, float]],
    box_size: int,
    sections: list[int],
    tiltseries_projection_matrices: np.ndarray,
    len_individual_tiltseries_df: int,
    no_ctf: bool,
    voltage: float,
    spherical_aberration: float,
    amplitude_contrast: float,
    handedness: int,
    tiltseries_pixel_size: float,
    phase_shift: list[float],
    defocus_u: list[float],
    defocus_v: list[float],
    defocus_angle: list[float],
    doses: list[float],
    ctf_scalefactor: list[float],
    bfactor_per_electron_dose: list[float],
    bin: int,
    dose_weights: np.ndarray,
    freq_cutoff_idx: list[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Read an extracted particle and backproject it into a Fourier volume.

    Returns:
        particle_data_fourier_volume (np.ndarray): The Fourier volume of the particle data.
        particle_weight_fourier_volume (np.ndarray): The Fourier volume of the particle weights.
        random_subset (int): The assigned random subset of the particle (for half-set reconstructions). If not present, returns 0.
    """
    visible_sections = ast.literal_eval(particle["rlnTomoVisibleFrames"])
    assert (
        len(visible_sections) == len_individual_tiltseries_df
    ), f"Mismatch between visible sections and tiltseries for {particle['rlnTomoParticleName']}"
    particle_path = Path(particle["rlnImageName"])
    if not particle_path.exists():
        raise FileNotFoundError(
            f"Particle file {particle_path} does not exist. Please check the path (and current working directory) and try again."
        )

    # Reading fourier data prevents loss of precision due to FFT and inverse FFT in float32
    particle_data = np.load(particle_path.with_suffix(".npy"))
    assert (
        sum(visible_sections) == particle_data.shape[0]
    ), f"Mismatch between visible sections and particle data for {particle['rlnTomoParticleName']}"
    assert (
        particle_data.shape[1] == box_size and particle_data.shape[2] == box_size // 2 + 1
    ), f"Mismatch between box size and particle data for {particle['rlnTomoParticleName']}"

    weight_data = np.ones((len(sections), box_size, box_size // 2 + 1), dtype=np.complex128)
    particle_data_fourier_volume = np.zeros((box_size, box_size, box_size // 2 + 1), dtype=np.complex128)
    particle_weight_fourier_volume = np.zeros((box_size, box_size, box_size // 2 + 1), dtype=np.complex128)

    particle_section_index = 0
    for section_index, section in enumerate(sections):
        if not visible_sections[section_index]:
            continue

        coordinate, _ = particle_coordinates[section]
        if not no_ctf:
            weight_data[section_index] = (
                calculate_ctf(
                    coordinate=coordinate,
                    tilt_projection_matrix=tiltseries_projection_matrices[section_index],
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
                * dose_weights[section_index]
            )
            particle_data[particle_section_index] *= weight_data[section_index]
            weight_data[section_index] **= 2

        backproject_slice_backward(
            particle_data_slice=particle_data[particle_section_index],
            particle_weight_slice=weight_data[section_index],
            particle_data_fourier_volume=particle_data_fourier_volume,
            particle_weight_fourier_volume=particle_weight_fourier_volume,
            particle_projection_matrix=particle_projection_matrices[section_index],
            freq_cutoff=freq_cutoff_idx[section_index] if not no_ctf else box_size // 2 + 1,
        )
        particle_section_index += 1

    return (
        particle_data_fourier_volume,
        particle_weight_fourier_volume,
        particle["rlnRandomSubset"] if "rlnRandomSubset" in particle else 1,
    )


def reconstruct_single_tiltseries(
    no_ctf: bool,
    cutoff_fraction: float,
    filtered_particles_df: pd.DataFrame,
    filtered_trajectories_dict: pd.DataFrame,
    tiltseries_row_entry: pd.Series,
    individual_tiltseries_df: pd.DataFrame,
    optics_row: pd.DataFrame,
    progress_bar: tqdm = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct particles from a single tiltseries.
    Returns:
        data_fourier_volume_half1 (np.ndarray): The Fourier volume of the particle data for random subset 1.
        weight_fourier_volume_half1 (np.ndarray): The Fourier volume of the particle weights for random subset 1.
        data_fourier_volume_half2 (np.ndarray): The Fourier volume of the particle data for random subset 2.
        weight_fourier_volume_half2 (np.ndarray): The Fourier volume of the particle weights for random subset 2.
    """
    filtered_particles_df = filtered_particles_df.reset_index(drop=True)

    # particle variables
    box_size = int(optics_row["rlnImageSize"].iloc[0])
    pixel_size = float(optics_row["rlnImagePixelSize"].iloc[0])
    bin = int(optics_row["rlnTomoSubtomogramBinning"].iloc[0])
    ctf_premultiplied = bool(optics_row["rlnCtfDataAreCtfPremultiplied"].iloc[0])
    if ctf_premultiplied:
        raise ValueError("CTF premultiplied particles are not supported for reconstruction.")
    if {"rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"}.issubset(filtered_particles_df.columns):
        particle_rotation_matrices = get_rotation_matrix_from_euler(
            filtered_particles_df[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy()
        )
    else:
        particle_rotation_matrices = np.tile(np.eye(3), (len(filtered_particles_df), 1, 1))

    # tiltseries variables
    tiltseries_pixel_size = tiltseries_row_entry["rlnTomoTiltSeriesPixelSize"]
    assert np.isclose(
        tiltseries_pixel_size * bin, pixel_size
    ), f"Mismatch between tiltseries pixel size and optics pixel size for {tiltseries_row_entry['rlnTomoName']}"
    tiltseries_projection_matrices = calculate_projection_matrix_from_starfile_df(individual_tiltseries_df)
    particles_to_tiltseries_coordinates = get_particles_to_tiltseries_coordinates(
        filtered_particles_df,
        filtered_trajectories_dict,
        individual_tiltseries_df,
        tiltseries_projection_matrices,
        use_tomo_particle_name_for_id=False,
    )
    sections = individual_tiltseries_df["rlnMicrographName"].str.split("@").str[0].astype(int).to_list()
    all_particle_projection_matrices = (
        np.asarray(tiltseries_projection_matrices)[:, :3, :3][None, :, :, :]
        @ np.asarray(particle_rotation_matrices)[:, None, :, :]
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
            calculate_dose_weight_image(dose, tiltseries_pixel_size * bin, box_size, bfactor, cutoff_fraction)
            for dose, bfactor in zip(doses, bfactor_per_electron_dose)
        ],
        dtype=np.complex128,
    )
    freq_cutoff = dose_weights[:, 0, :] < cutoff_fraction
    freq_cutoff_idx = freq_cutoff.shape[1] - np.argmax(freq_cutoff[:, ::-1], axis=1)

    args_list = [
        {
            "particle": particle,
            "particle_projection_matrices": all_particle_projection_matrices[particle_index],
            "particle_coordinates": particles_to_tiltseries_coordinates[particle_index + 1],  # RELION 1-based indexing
        }
        for particle_index, particle in filtered_particles_df.iterrows()
    ]
    constant_args = {
        "box_size": box_size,
        "sections": sections,
        "tiltseries_projection_matrices": tiltseries_projection_matrices,
        "len_individual_tiltseries_df": len(individual_tiltseries_df),
        "no_ctf": no_ctf,
        "voltage": voltage,
        "spherical_aberration": spherical_aberration,
        "amplitude_contrast": amplitude_contrast,
        "handedness": handedness,
        "tiltseries_pixel_size": tiltseries_pixel_size,
        "phase_shift": phase_shift,
        "defocus_u": defocus_u,
        "defocus_v": defocus_v,
        "defocus_angle": defocus_angle,
        "doses": doses,
        "ctf_scalefactor": ctf_scalefactor,
        "bfactor_per_electron_dose": bfactor_per_electron_dose,
        "bin": bin,
        "dose_weights": dose_weights,
        "freq_cutoff_idx": freq_cutoff_idx,
    }
    args_list = [{**args, **constant_args} for args in args_list]

    data_fourier_volume_half1 = np.zeros((box_size, box_size, box_size // 2 + 1), dtype=np.complex128)
    data_fourier_volume_half2 = np.zeros((box_size, box_size, box_size // 2 + 1), dtype=np.complex128)
    weight_fourier_volume_half1 = np.zeros((box_size, box_size, box_size // 2 + 1), dtype=np.complex128)
    weight_fourier_volume_half2 = np.zeros((box_size, box_size, box_size // 2 + 1), dtype=np.complex128)

    cpu_count = min(32, mp.cpu_count(), len(filtered_particles_df))
    with mp.Pool(processes=cpu_count) as pool:
        for particle_data_fourier_volume, particle_weight_fourier_volume, random_subset in pool.imap_unordered(
            process_particle_wrapper, args_list
        ):
            if random_subset == 1:
                data_fourier_volume_half1 += particle_data_fourier_volume
                weight_fourier_volume_half1 += particle_weight_fourier_volume
            elif random_subset == 2:
                data_fourier_volume_half2 += particle_data_fourier_volume
                weight_fourier_volume_half2 += particle_weight_fourier_volume
            else:
                raise ValueError(f"Invalid random subset {random_subset} found! Should be 1 or 2.")
            if progress_bar is not None:
                progress_bar.update(1)

    return (
        data_fourier_volume_half1,
        weight_fourier_volume_half1,
        data_fourier_volume_half2,
        weight_fourier_volume_half2,
    )


def finalise_volume(
    data_fourier_volume: np.ndarray,
    weight_fourier_volume: np.ndarray,
    output_dir: Union[str, Path],
    voxel_size: float,
    crop_size: int,
    symmetry: str,
    tag: str,
) -> tuple[Path, Path, Path]:
    """
    Finalise the volume by applying symmetry, gridding correction, CTF correction, and spherical masking with mean subtraction.
    Writes out data_*.mrc, *_full.mrc, and *.mrc files.
    Returns:
        data_path (Path): The path to the non-CTF-corrected reconstruction, after gridding correction (data_*.mrc).
        full_path (Path): The path to the non-masked reconstruction, after CTF correction (*_full.mrc).
        final_path (Path): The path to the final reconstruction, after masking and mean subtraction (*.mrc).
    """
    if symmetry.upper() != "C1":
        transforms = get_transforms_from_symmetry(symmetry)
        data_fourier_volume = symmetrise_fs_complex(data_fourier_volume, transforms)
        weight_fourier_volume = symmetrise_fs_real(weight_fourier_volume, transforms)

    gridding_corrected_volume = gridding_correct_3d_sinc2(particle_fourier_volume=data_fourier_volume)
    ctf_corrected_real_volume = ctf_correct_3d_heuristic(
        real_space_volume=gridding_corrected_volume, weights_fourier_volume=weight_fourier_volume
    )

    data_path = Path(output_dir) / f"data_{tag}.mrc"
    with mrcfile.new(data_path, overwrite=True) as mrc:
        mrc.set_data(gridding_corrected_volume.astype(np.float32))
        mrc.voxel_size = voxel_size

    full_path = Path(output_dir) / f"{tag}_full.mrc"
    with mrcfile.new(full_path, overwrite=True) as mrc:
        mrc.set_data(ctf_corrected_real_volume.astype(np.float32))
        mrc.voxel_size = voxel_size

    final_volume = ctf_corrected_real_volume

    if crop_size < ctf_corrected_real_volume.shape[0]:
        start = (ctf_corrected_real_volume.shape[0] - crop_size) // 2
        end = start + crop_size
        final_volume = ctf_corrected_real_volume[start:end, start:end, start:end]

    soft_mask = spherical_soft_mask(box_size=crop_size, crop_size=crop_size, falloff=10.0)
    inner_mask = soft_mask > 0
    inner_mean = (final_volume[inner_mask] * soft_mask[inner_mask]).sum() / soft_mask[inner_mask].sum()
    final_volume[~inner_mask] = 0.0
    final_volume[inner_mask] = soft_mask[inner_mask] * (final_volume[inner_mask] - inner_mean)

    final_path = Path(output_dir) / f"{tag}.mrc"
    with mrcfile.new(final_path, overwrite=True) as mrc:
        mrc.set_data(final_volume.astype(np.float32))
        mrc.voxel_size = voxel_size

    logger.info(f"Wrote out {data_path}, {full_path}, and {final_path}.")
    return data_path, full_path, final_path


# TODO: write out weight*.mrc files
# TODO: implement tiltseries relative dir but for particles
# TODO: support no_circle_crop
# TODO: support multiple box sizes / crop sizes / pixel sizes
def reconstruct(
    output_dir: Union[str, Path],
    box_size: int,
    crop_size: int = None,
    symmetry: str = "C1",
    no_ctf: bool = False,
    cutoff_fraction: float = 0.01,
    particles_starfile: Union[str, Path] = None,
    trajectories_starfile: Union[str, Path] = None,
    tiltseries_relative_dir: Union[str, Path] = None,
    tomograms_starfile: Union[str, Path] = None,
) -> None:
    """
    Reconstruct a particle map from particles and tiltseries.
    """
    start_time = time.time()

    if not crop_size:
        crop_size = box_size

    particles_metadata = starfile.read(particles_starfile)
    particles_df = apply_offsets_to_coordinates(particles_metadata["particles"])
    optics_df = particles_metadata["optics"]
    trajectories_dict = starfile.read(trajectories_starfile) if trajectories_starfile else None
    tomograms_data = starfile.read(tomograms_starfile)
    tomograms_df = tomograms_data["global"] if isinstance(tomograms_data, dict) else tomograms_data
    if "rlnTomoTiltSeriesStarFile" not in tomograms_df.columns:
        raise ValueError(
            f"Tomograms star file {tomograms_starfile} does not contain the required column 'rlnTomoTiltSeriesStarFile'. Please check the file."
        )
    if not tiltseries_relative_dir:
        tiltseries_relative_dir = Path("./")

    assert optics_df["rlnImageDimensionality"].unique() == [2], "Input particles must be 2D"
    # TODO: move this into a multiproc process support multiple
    assert optics_df["rlnImageSize"].nunique() == 1, "Currently only supports one crop size"
    assert optics_df["rlnImagePixelSize"].nunique() == 1, "Currently only supports one pixel size"
    assert optics_df["rlnTomoSubtomogramBinning"].nunique() == 1, "Currently only supports one binning"

    voxel_size = float(optics_df["rlnImagePixelSize"].iloc[0])

    logger.info(f"Starting particle reconstruction from {len(particles_df)} particles...")

    if "rlnRandomSubset" in particles_df.columns:
        logger.info("Random subsets exist for the particles. Reconstructing half-maps.")
    else:
        logger.info("No random subsets exist for the particles. Reconstructing full map only.")

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

    progress_bar = tqdm(total=len(particles_df), file=sys.stdout, desc="Reconstructing particles", unit="particles")
    constant_args = {
        "no_ctf": no_ctf,
        "cutoff_fraction": cutoff_fraction,
        "progress_bar": progress_bar,
    }
    args_list = [{**args, **constant_args} for args in args_list if args is not None]

    output_data_fourier_volume_half1 = np.zeros((box_size, box_size, box_size // 2 + 1), dtype=np.complex128)
    output_data_fourier_volume_half2 = np.zeros((box_size, box_size, box_size // 2 + 1), dtype=np.complex128)
    output_weight_fourier_volume_half1 = np.zeros((box_size, box_size, box_size // 2 + 1), dtype=np.complex128)
    output_weight_fourier_volume_half2 = np.zeros((box_size, box_size, box_size // 2 + 1), dtype=np.complex128)

    for arg in args_list:
        (
            data_fourier_volume_half1,
            weight_fourier_volume_half1,
            data_fourier_volume_half2,
            weight_fourier_volume_half2,
        ) = reconstruct_single_tiltseries(**arg)

        output_data_fourier_volume_half1 += data_fourier_volume_half1
        output_data_fourier_volume_half2 += data_fourier_volume_half2
        output_weight_fourier_volume_half1 += weight_fourier_volume_half1
        output_weight_fourier_volume_half2 += weight_fourier_volume_half2

    progress_bar.close()

    output_data_fourier_volume = output_data_fourier_volume_half1 + output_data_fourier_volume_half2
    output_weight_fourier_volume = output_weight_fourier_volume_half1 + output_weight_fourier_volume_half2

    logger.info("Finalising volumes and writing to disk...")

    if "rlnRandomSubset" in particles_df.columns:
        finalise_volume(
            output_data_fourier_volume_half1,
            output_weight_fourier_volume_half1,
            output_dir,
            voxel_size,
            crop_size,
            symmetry,
            tag="half1",
        )
        finalise_volume(
            output_data_fourier_volume_half2,
            output_weight_fourier_volume_half2,
            output_dir,
            voxel_size,
            crop_size,
            symmetry,
            tag="half2",
        )
    finalise_volume(
        output_data_fourier_volume,
        output_weight_fourier_volume,
        output_dir,
        voxel_size,
        crop_size,
        symmetry,
        tag="merged",
    )

    # delete Subtomograms directory if it exists
    subtomos_dir = Path(output_dir) / "Subtomograms"
    if subtomos_dir.exists() and subtomos_dir.is_dir():
        shutil.rmtree(subtomos_dir)

    # delete particles.star if it exists
    particles_star_path = Path(output_dir) / "particles.star"
    if particles_star_path.exists() and particles_star_path.is_file():
        particles_star_path.unlink()

    # delete optimisation_set.star if it exists
    optimisation_set_star_path = Path(output_dir) / "optimisation_set.star"
    if optimisation_set_star_path.exists() and optimisation_set_star_path.is_file():
        optimisation_set_star_path.unlink()

    end_time = time.time()
    logger.info(f"Reconstructing particles took {end_time - start_time:.2f} seconds.")


def reconstruct_local(
    box_size: int,
    output_dir: Union[str, Path],
    bin: int = 1,
    crop_size: int = None,
    symmetry: str = "C1",
    no_ctf: bool = False,
    cutoff_fraction: float = 0.01,
    particles_starfile: Union[str, Path] = None,
    trajectories_starfile: Union[str, Path] = None,
    tiltseries_relative_dir: Union[str, Path] = None,
    tomograms_starfile: Union[str, Path] = None,
    optimisation_set_starfile: Union[str, Path] = None,
    overwrite: bool = False,
):
    """
    Reconstruct a particle map from local tiltseries using RELION particles.
    """
    (
        new_particles_starfile,
        trajectories_starfile,
        tiltseries_relative_dir,
        tomograms_starfile,
        optimisation_set_starfile,
    ) = parse_extract_local_subtomograms(
        box_size=box_size,
        output_dir=output_dir,
        bin=bin,
        float16=False,
        no_ctf=True,
        circle_precrop=True,
        no_circle_crop=True,
        no_ic=False,
        normalize_bin=False,
        write_fourier=True,
        crop_size=box_size,  # extracted particles must be the same size as box size for reconstruction (due to how cropping is done)
        particles_starfile=particles_starfile,
        trajectories_starfile=trajectories_starfile,
        tiltseries_relative_dir=tiltseries_relative_dir,
        tomograms_starfile=tomograms_starfile,
        optimisation_set_starfile=optimisation_set_starfile,
        overwrite=overwrite,
    )

    reconstruct(
        output_dir=output_dir,
        box_size=box_size,
        crop_size=crop_size if crop_size is not None else box_size,
        symmetry=symmetry,
        no_ctf=no_ctf,
        cutoff_fraction=cutoff_fraction,
        particles_starfile=new_particles_starfile,
        trajectories_starfile=trajectories_starfile,
        tiltseries_relative_dir=tiltseries_relative_dir,
        tomograms_starfile=tomograms_starfile,
    )


def reconstruct_local_copick(
    box_size: int,
    output_dir: Union[str, Path],
    copick_config: Path,
    copick_name: str,
    copick_session_id: str,
    copick_user_id: str,
    bin: int = 1,
    crop_size: int = None,
    symmetry: str = "C1",
    no_ctf: bool = False,
    cutoff_fraction: float = 0.01,
    copick_run_names: list[str] = None,
    tiltseries_relative_dir: Path = None,
    tomograms_starfile: Path = None,
    overwrite: bool = False,
):
    """
    Reconstruct a particle map from local tiltseries using copick particles.
    """
    (
        particles_starfile,
        trajectories_starfile,
        tiltseries_relative_dir,
        tomograms_starfile,
        _,
    ) = parse_extract_local_copick_subtomograms(
        box_size=box_size,
        output_dir=output_dir,
        copick_config=copick_config,
        copick_name=copick_name,
        copick_session_id=copick_session_id,
        copick_user_id=copick_user_id,
        bin=bin,
        float16=False,
        no_ctf=True,
        circle_precrop=True,
        no_circle_crop=True,
        no_ic=False,
        normalize_bin=False,
        write_fourier=True,
        copick_run_names=copick_run_names,
        crop_size=box_size,
        tiltseries_relative_dir=tiltseries_relative_dir,
        tomograms_starfile=tomograms_starfile,
        overwrite=overwrite,
    )

    reconstruct(
        output_dir=output_dir,
        box_size=box_size,
        crop_size=crop_size if crop_size is not None else box_size,
        symmetry=symmetry,
        no_ctf=no_ctf,
        cutoff_fraction=cutoff_fraction,
        particles_starfile=particles_starfile,
        trajectories_starfile=trajectories_starfile,
        tiltseries_relative_dir=tiltseries_relative_dir,
        tomograms_starfile=tomograms_starfile,
    )


def reconstruct_data_portal(
    output_dir: Union[str, Path],
    box_size: int = None,
    bin: int = 1,
    crop_size: int = None,
    symmetry: str = "C1",
    no_ctf: bool = False,
    cutoff_fraction: float = 0.01,
    overwrite: bool = False,
    **data_portal_args,
):
    """
    Reconstruct a particle map using picks and tiltseries from the CryoET Data Portal.
    """
    (
        particles_starfile,
        trajectories_starfile,
        tiltseries_relative_dir,
        tomograms_starfile,
        _,
    ) = parse_extract_data_portal_subtomograms(
        output_dir=output_dir,
        box_size=box_size,
        bin=bin,
        float16=False,
        no_ctf=True,
        circle_precrop=True,
        no_circle_crop=True,
        no_ic=False,
        normalize_bin=False,
        write_fourier=True,
        crop_size=box_size,
        overwrite=overwrite,
        **data_portal_args,
    )

    reconstruct(
        output_dir=output_dir,
        box_size=box_size,
        crop_size=crop_size if crop_size is not None else box_size,
        symmetry=symmetry,
        no_ctf=no_ctf,
        cutoff_fraction=cutoff_fraction,
        particles_starfile=particles_starfile,
        trajectories_starfile=trajectories_starfile,
        tiltseries_relative_dir=tiltseries_relative_dir,
        tomograms_starfile=tomograms_starfile,
    )


def reconstruct_data_portal_copick(
    output_dir: Union[str, Path],
    copick_config: Path,
    copick_name: str,
    copick_session_id: str,
    copick_user_id: str,
    copick_run_names: list[str] = None,
    copick_dataset_ids: list[int] = None,
    box_size: int = None,
    bin: int = 1,
    crop_size: int = None,
    symmetry: str = "C1",
    no_ctf: bool = False,
    cutoff_fraction: float = 0.01,
    overwrite: bool = False,
    **extra_kwargs,
):
    """
    Reconstruct a particle map using copick picks and tiltseries from the CryoET Data Portal.
    """
    (
        particles_starfile,
        trajectories_starfile,
        tiltseries_relative_dir,
        tomograms_starfile,
        _,
    ) = parse_extract_data_portal_copick_subtomograms(
        output_dir=output_dir,
        copick_config=copick_config,
        copick_name=copick_name,
        copick_session_id=copick_session_id,
        copick_user_id=copick_user_id,
        copick_run_names=copick_run_names,
        copick_dataset_ids=copick_dataset_ids,
        box_size=box_size,
        bin=bin,
        float16=False,
        no_ctf=True,
        circle_precrop=True,
        no_circle_crop=True,
        no_ic=False,
        normalize_bin=False,
        write_fourier=True,
        crop_size=box_size,
        overwrite=overwrite,
        **extra_kwargs,
    )

    reconstruct(
        output_dir=output_dir,
        box_size=box_size,
        crop_size=crop_size if crop_size is not None else box_size,
        symmetry=symmetry,
        no_ctf=no_ctf,
        cutoff_fraction=cutoff_fraction,
        particles_starfile=particles_starfile,
        trajectories_starfile=trajectories_starfile,
        tiltseries_relative_dir=tiltseries_relative_dir,
        tomograms_starfile=tomograms_starfile,
    )


@click.group("Reconstruct a particle map from particles and tiltseries.")
def cli():
    pass


@cli.command("local", help="Reconstruct a particle map from local tiltseries using RELION particles.")
@cli_options.local_options()
@cli_options.local_shared_options()
@cli_options.common_options()
@cli_options.reconstruct_options()
def cmd_local(**kwargs):
    setup_logging(kwargs.pop("debug", False))
    reconstruct_local(**kwargs)


@cli.command("local-copick", help="Reconstruct a particle map from local tiltseries with copick particles.")
@cli_options.local_shared_options()
@cli_options.copick_options()
@cli_options.common_options()
@cli_options.reconstruct_options()
def cmd_local_copick(**kwargs):
    setup_logging(debug=kwargs.pop("debug", False))
    kwargs["copick_run_names"] = cli_options.flatten(kwargs["copick_run_names"])
    reconstruct_local_copick(**kwargs)


@cli.command("data-portal", help="Reconstruct a particle map using picks and tiltseries from the CryoET Data Portal.")
@cli_options.common_options()
@cli_options.reconstruct_options()
@cli_options.data_portal_options()
def cmd_data_portal(**kwargs):
    setup_logging(debug=kwargs.pop("debug", False))
    kwargs = cli_options.flatten_data_portal_args(kwargs)
    reconstruct_data_portal(**kwargs)


@cli.command(
    "copick-data-portal",
    help="Reconstruct a particle map using copick picks and tiltseries from the CryoET Data Portal.",
)
@cli_options.common_options()
@cli_options.reconstruct_options()
@cli_options.copick_options()
@cli_options.data_portal_copick_options()
def cmd_data_portal_copick(**kwargs):
    setup_logging(debug=kwargs.pop("debug", False))
    kwargs["copick_run_names"] = cli_options.flatten(kwargs["copick_run_names"])
    kwargs["copick_dataset_ids"] = cli_options.flatten(kwargs["copick_dataset_ids"])
    reconstruct_data_portal_copick(**kwargs)


if __name__ == "__main__":
    cli()
