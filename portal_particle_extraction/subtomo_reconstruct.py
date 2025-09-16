"""
A module for reconstructing from extracted particles. A numerically-precise reimplementation of RELION's Reconstruct particle job (relion_tomo_reconstruct_particle).
"""

# TODO: write tests (at least for local)
import ast
import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

import click
import mrcfile
import numpy as np
import pandas as pd
import starfile
from tqdm import tqdm

import portal_particle_extraction.cli.options as cli_options
from portal_particle_extraction.core.backprojection import (
    backproject_slice_backward,
    ctf_correct_3d_heuristic,
    get_rotation_matrix_from_euler,
    gridding_correct_3d_sinc2,
)
from portal_particle_extraction.core.ctf import calculate_ctf
from portal_particle_extraction.core.dose import calculate_dose_weight_image
from portal_particle_extraction.core.forwardprojection import (
    apply_offsets_to_coordinates,
    calculate_projection_matrix_from_starfile_df,
    get_particles_to_tiltseries_coordinates,
)
from portal_particle_extraction.core.helpers import get_tiltseries_data, setup_logging
from portal_particle_extraction.subtomo_extract import (
    parse_extract_data_portal_copick_subtomograms,
    parse_extract_data_portal_subtomograms,
    parse_extract_local_copick_subtomograms,
    parse_extract_local_subtomograms,
)

logger = logging.getLogger(__name__)


def reconstruct_single_tiltseries_wrapper(kwargs):
    return reconstruct_single_tiltseries(**kwargs)


# TODO: validate box, crop, pixel size, and bin here
# TODO: do frequency cutoff here
def reconstruct_single_tiltseries(
    no_ctf: bool,
    crop_size: int,
    cutoff_fraction: float,
    filtered_particles_df: pd.DataFrame,
    filtered_trajectories_dict: pd.DataFrame,
    tiltseries_row_entry: pd.Series,
    individual_tiltseries_df: pd.DataFrame,
    individual_tiltseries_path: Path,
    optics_row: pd.DataFrame,
):
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

    data_fourier_volume = np.zeros((box_size, box_size, box_size // 2 + 1), dtype=np.complex64)
    weight_fourier_volume = np.zeros((box_size, box_size, box_size // 2 + 1), dtype=np.complex64)

    # tiltseries variables
    tiltseries_pixel_size = tiltseries_row_entry["rlnTomoTiltSeriesPixelSize"]
    assert np.isclose(
        tiltseries_pixel_size * bin, pixel_size
    ), f"Mismatch between tiltseries pixel size and optics pixel size for {tiltseries_row_entry['rlnTomoName']}"
    projection_matrices = calculate_projection_matrix_from_starfile_df(individual_tiltseries_df)
    particles_to_tiltseries_coordinates = get_particles_to_tiltseries_coordinates(
        filtered_particles_df,
        filtered_trajectories_dict,
        individual_tiltseries_df,
        projection_matrices,
        use_tomo_particle_name_for_id=False,
    )
    sections = individual_tiltseries_df["rlnMicrographName"].str.split("@").str[0].astype(int).to_list()

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
        dtype=np.complex64,
    )

    def process_particle(particle_index, particle):
        visible_sections = ast.literal_eval(particle["rlnTomoVisibleFrames"])
        assert len(visible_sections) == len(
            individual_tiltseries_df
        ), f"Mismatch between visible sections and tiltseries for {particle['rlnTomoParticleName']}"
        particle_path = Path(particle["rlnImageName"])
        if not particle_path.exists():
            raise FileNotFoundError(
                f"Particle file {particle_path} does not exist. Please check the path (and current working directory) and try again."
            )
        with mrcfile.open(particle_path, permissive=True) as mrc:
            mrc_data = mrc.data
        assert (
            sum(visible_sections) == mrc_data.shape[0]
        ), f"Mismatch between visible sections and particle data for {particle['rlnTomoParticleName']}"
        assert (
            mrc_data.shape[1] == box_size and mrc_data.shape[2] == box_size
        ), f"Mismatch between box size and particle data for {particle['rlnTomoParticleName']}"
        assert mrc_data.dtype == np.float32, f"Particle data must be float32 for {particle['rlnTomoParticleName']}"

        particle_projection_matrices = (
            np.asarray(projection_matrices)[:, :3, :3] @ particle_rotation_matrices[particle_index]
        )

        particle_data = np.fft.rfft2(mrc_data, norm="ortho", axes=(-2, -1))

        weight_data = np.ones((len(sections), box_size, box_size // 2 + 1), dtype=np.complex64)
        particle_data_fourier_volume = np.zeros((box_size, box_size, box_size // 2 + 1), dtype=np.complex64)
        particle_weight_fourier_volume = np.zeros((box_size, box_size, box_size // 2 + 1), dtype=np.complex64)

        # adapt to RELION 1-based indexing
        particle_coordinates = particles_to_tiltseries_coordinates[particle_index + 1]
        particle_section_index = 0
        for section_index, section in enumerate(sections):
            if not visible_sections[section_index]:
                continue

            coordinate, _ = particle_coordinates[section]
            if not no_ctf:
                weight_data[section_index] = (
                    calculate_ctf(
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
                freq_cutoff=np.argmax(dose_weights[section_index][0] < cutoff_fraction) if not no_ctf else box_size + 1,
            )
            particle_section_index += 1

        logger.debug(
            f"Processed particle {particle['rlnTomoParticleName']} from tiltseries {tiltseries_row_entry['rlnTomoName']}"
        )
        return particle_data_fourier_volume, particle_weight_fourier_volume

    with ThreadPoolExecutor(max_workers=4) as executor:
        for particle_data_fourier_volume, particle_weight_fourier_volume in executor.map(
            lambda t: process_particle(*t), filtered_particles_df.iterrows()
        ):
            data_fourier_volume += particle_data_fourier_volume
            weight_fourier_volume += particle_weight_fourier_volume

    return data_fourier_volume, weight_fourier_volume


# TODO: implement tiltseries relative dir but for particles
# TODO: support no_circle_crop
# TODO: also list out things that are not supported (helical symmetry, etc)
# TODO: figure out what's going on with (implement) taper
# TODO: implement particle splitting? (with random seed?)
# TODO: implement multiprocessing & multithreading
# TODO: support multiple box sizes / crop sizes / pixel sizes
def reconstruct_particle(
    output_dir: Union[str, Path],
    crop_size: int = None,
    no_ctf: bool = False,
    cutoff_fraction: float = 0.01,
    particles_starfile: Union[str, Path] = None,
    trajectories_starfile: Union[str, Path] = None,
    tiltseries_relative_dir: Union[str, Path] = None,
    tomograms_starfile: Union[str, Path] = None,
):
    start_time = time.time()

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

    constant_args = {
        "no_ctf": no_ctf,
        "crop_size": crop_size,
        "cutoff_fraction": cutoff_fraction,
    }

    args_list = [{**args, **constant_args} for args in args_list if args is not None]

    output_data_fourier_volume = None
    output_weight_fourier_volume = None

    cpu_count = min(32, mp.cpu_count(), len(tomograms_df))
    with mp.Pool(processes=cpu_count) as pool:
        for data_fourier_volume, weight_fourier_volume in tqdm(
            pool.imap_unordered(reconstruct_single_tiltseries_wrapper, args_list),
            total=len(args_list),
            desc="Reconstructing particles",
            file=sys.stdout,
        ):
            if output_data_fourier_volume is None:
                output_data_fourier_volume = np.zeros((data_fourier_volume.shape), dtype=np.complex64)
            output_data_fourier_volume += data_fourier_volume

            if output_weight_fourier_volume is None:
                output_weight_fourier_volume = np.zeros((weight_fourier_volume.shape), dtype=np.complex64)
            output_weight_fourier_volume += weight_fourier_volume

    grid_corrected_real_volume = gridding_correct_3d_sinc2(particle_fourier_volume=output_data_fourier_volume)
    ctf_corrected_real_volume = ctf_correct_3d_heuristic(
        real_space_volume=grid_corrected_real_volume, weights_fourier_volume=output_weight_fourier_volume
    )

    with mrcfile.new(Path(output_dir) / "merged.mrc", overwrite=True) as mrc:
        mrc.set_data(ctf_corrected_real_volume.astype(np.float32))
        mrc.voxel_size = float(optics_df["rlnImagePixelSize"].iloc[0])

    end_time = time.time()
    logger.info(f"Reconstructing particles took {end_time - start_time:.2f} seconds.")


def reconstruct_local_particle(
    box_size: int,
    output_dir: Union[str, Path],
    bin: int = 1,
    crop_size: int = None,
    no_ctf: bool = False,
    cutoff_fraction: float = 1,
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
        circle_precrop=False,
        no_circle_crop=True,
        no_ic=False,
        normalize_bin=False,
        crop_size=box_size,  # extracted particles must be the same size as box size for reconstruction (due to how cropping is done)
        particles_starfile=particles_starfile,
        trajectories_starfile=trajectories_starfile,
        tiltseries_relative_dir=tiltseries_relative_dir,
        tomograms_starfile=tomograms_starfile,
        optimisation_set_starfile=optimisation_set_starfile,
        overwrite=overwrite,
    )

    reconstruct_particle(
        output_dir=output_dir,
        crop_size=crop_size if crop_size is not None else box_size,
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
        copick_run_names=copick_run_names,
        crop_size=box_size,
        tiltseries_relative_dir=tiltseries_relative_dir,
        tomograms_starfile=tomograms_starfile,
        overwrite=overwrite,
    )

    reconstruct_particle(
        output_dir=output_dir,
        crop_size=crop_size if crop_size is not None else box_size,
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
        crop_size=box_size,
        overwrite=overwrite,
        **data_portal_args,
    )

    reconstruct_particle(
        output_dir=output_dir,
        crop_size=crop_size if crop_size is not None else box_size,
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
        crop_size=box_size,
        overwrite=overwrite,
        **extra_kwargs,
    )

    reconstruct_particle(
        output_dir=output_dir,
        crop_size=crop_size if crop_size is not None else box_size,
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
    reconstruct_local_particle(**kwargs)


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
