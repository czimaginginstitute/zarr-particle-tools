"""
A module for reconstructing from extracted particles. A numerically-precise reimplementation of RELION's Reconstruct particle job (relion_tomo_reconstruct_particle).
"""

# TODO: write tests (at least for local)
import ast
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
from tqdm import tqdm

import portal_particle_extraction.cli.options as cli_options
from portal_particle_extraction.core.ctf import calculate_ctf
from portal_particle_extraction.core.dose import calculate_dose_weight_image
from portal_particle_extraction.core.helpers import get_tiltseries_data, setup_logging
from portal_particle_extraction.core.projection import (
    apply_offsets_to_coordinates,
    calculate_projection_matrix_from_starfile_df,
    get_particles_to_tiltseries_coordinates,
)
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
    # in fourier space
    particle_data = np.zeros(
        (len(filtered_particles_df), len(individual_tiltseries_df), box_size, box_size // 2 + 1), dtype=np.complex64
    )
    ctf_data = np.zeros(
        (len(filtered_particles_df), len(individual_tiltseries_df), box_size, box_size // 2 + 1), dtype=np.complex64
    )

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
            calculate_dose_weight_image(dose, tiltseries_pixel_size * bin, box_size, bfactor)
            for dose, bfactor in zip(doses, bfactor_per_electron_dose)
        ],
        dtype=np.complex64,
    )

    def process_particle(particle_index, particle):
        visible_sections = ast.literal_eval(particle["rlnTomoVisibleFrames"])
        assert len(visible_sections) == len(
            individual_tiltseries_df
        ), f"Mismatch between visible sections and tilt series for {particle['rlnTomoParticleName']}"
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
        fourier_data = np.fft.rfft2(mrc_data, norm="ortho", axes=(-2, -1))

        visible_index = np.asarray(visible_sections, dtype=bool)

        particle_data[particle_index, visible_index] = fourier_data
        # adapt to RELION 1-based indexing
        particle_coordinates = particles_to_tiltseries_coordinates[particle_index + 1]
        for section_index, section in enumerate(sections):
            if not visible_sections[section_index]:
                continue

            # phase flipping was already done, so no need to do it
            coordinate, _ = particle_coordinates[section]
            ctf_data[particle_index, section_index] = (
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

            if no_ctf and ctf_premultiplied:
                logger.warning(
                    "Particles have been CTF premultiplied but no_ctf is set to True. Reversing CTF premultiplication (not recommended)."
                )
                particle_data[particle_index, section_index] /= np.clip(
                    ctf_data[particle_index, section_index], a_min=1e-9, a_max=None
                )
            elif not no_ctf and not ctf_premultiplied:
                logger.warning(
                    "Particles have not been CTF premultiplied but no_ctf is set to False. Applying CTF premultiplication."
                )
                particle_data[particle_index, section_index] *= ctf_data[particle_index, section_index]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_particle, particle_index, particle)
            for particle_index, particle in filtered_particles_df.iterrows()
        ]
        for future in as_completed(futures):
            future.result()

    ctf_data = ctf_data**2


# TODO: implement tiltseries relative dir but for particles
# TODO: modify the subtomo extract calls to output both (ic, float32) box size and crop size mrcs files if reconstruct is set to true (or handle this case smartly somehow)
# TODO: also list out things that are not supported (helical symmetry, etc)
# TODO: figure out what's going on with (implement) taper
# TODO: implement particle splitting? (with random seed?)
# TODO: implement multiprocessing & multithreading
# TODO: support multiple box sizes / crop sizes / pixel sizes
def reconstruct_particle(
    crop_size: int = None,
    no_ctf: bool = False,
    cutoff_fraction: float = 0.01,  # TODO: implement
    particles_starfile: Union[str, Path] = None,
    trajectories_starfile: Union[str, Path] = None,  # TODO: Implement
    tiltseries_relative_dir: Union[str, Path] = None,
    tomograms_starfile: Union[str, Path] = None,
):
    start_time = time.time()

    particles_metadata = starfile.read(particles_starfile)
    particles_df = apply_offsets_to_coordinates(particles_metadata["particles"])
    optics_df = particles_metadata["optics"]
    trajectories_dict = starfile.read(trajectories_starfile) if trajectories_starfile else None
    tomograms_df = starfile.read(tomograms_starfile)
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

    constant_args = {"no_ctf": no_ctf, "crop_size": crop_size}

    args_list = [{**args, **constant_args} for args in args_list if args is not None]

    cpu_count = min(32, mp.cpu_count(), len(tomograms_df))
    with mp.Pool(processes=cpu_count) as pool:
        for _ in tqdm(
            pool.imap_unordered(reconstruct_single_tiltseries_wrapper, args_list),
            total=len(args_list),
            desc="Reconstructing particles",
            file=sys.stdout,
        ):
            pass

    end_time = time.time()
    logger.info(f"Reconstructing particles took {end_time - start_time:.2f} seconds.")


def reconstruct_local_particle(
    box_size: int,
    output_dir: Union[str, Path],
    bin: int = 1,
    crop_size: int = None,
    no_ctf: bool = False,
    cutoff_fraction: float = 0.01,  # TODO: implement
    particles_starfile: Union[str, Path] = None,
    trajectories_starfile: Union[str, Path] = None,  # TODO: Implement
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
        no_ctf=no_ctf,
        no_circle_crop=False,
        no_ic=False,
        crop_size=box_size,  # extracted particles must be the same size as box size for reconstruction (due to how cropping is done)
        particles_starfile=particles_starfile,
        trajectories_starfile=trajectories_starfile,
        tiltseries_relative_dir=tiltseries_relative_dir,
        tomograms_starfile=tomograms_starfile,
        optimisation_set_starfile=optimisation_set_starfile,
        overwrite=overwrite,
    )

    reconstruct_particle(
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
    cutoff_fraction: float = 0.01,  # TODO: implement
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
        no_ctf=no_ctf,
        no_circle_crop=False,
        no_ic=False,
        copick_run_names=copick_run_names,
        crop_size=box_size,
        tiltseries_relative_dir=tiltseries_relative_dir,
        tomograms_starfile=tomograms_starfile,
        overwrite=overwrite,
    )

    reconstruct_particle(
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
    cutoff_fraction: float = 0.01,  # TODO: implement
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
        no_ctf=no_ctf,
        no_circle_crop=False,
        no_ic=False,
        crop_size=box_size,
        overwrite=overwrite,
        **data_portal_args,
    )

    reconstruct_particle(
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
    cutoff_fraction: float = 0.01,  # TODO: implement
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
        no_ctf=no_ctf,
        no_circle_crop=False,
        no_ic=False,
        crop_size=box_size,  # reconstruction expects extracted size == box_size
        overwrite=overwrite,
        **extra_kwargs,
    )

    reconstruct_particle(
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
