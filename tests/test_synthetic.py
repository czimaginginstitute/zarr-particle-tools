# TODO: create more focused tests for individual functions?
# TODO: don't store tiltseries / RELION *.mrcs data in the repository, but host them on zenodo
import os
import shutil
from pathlib import Path
from data_portal_subtomo_extract import extract_local_subtomograms


def test_extract_local_subtomograms_synthetic_baseline(synthetic_data_root, validate_optimisation_set_starfile, validate_particles_starfile, compare_mrcs_dirs):
    output_dir = Path("tests/output/synthetic_baseline/")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    extract_local_subtomograms(
        particles_starfile=synthetic_data_root / "particles.star",
        particles_tomo_name_prefix="session1_",
        box_size=64,
        bin=1,
        tiltseries_dir=synthetic_data_root / "tiltSeries/",
        tiltseries_starfile=synthetic_data_root / "tiltSeries/tomograms.star",
        tiltseries_x=630,
        tiltseries_y=630,
        aln_dir=synthetic_data_root / "aln/",
        float16=False,
        output_dir=output_dir,
    )

    subtomo_dir = os.path.join(output_dir, "Subtomograms/session1_TS_1")
    relion_subtomo_dir = "tests/data/relion_project_synthetic/relion_output_baseline/Subtomograms/session1_TS_1"
    validate_optimisation_set_starfile(output_dir / "optimisation_set.star")
    validate_particles_starfile(output_dir / "particles.star", synthetic_data_root / "relion_output_baseline/particles.star")
    compare_mrcs_dirs(relion_subtomo_dir, subtomo_dir)


def test_extract_local_subtomograms_synthetic_float16(synthetic_data_root, validate_optimisation_set_starfile, validate_particles_starfile, compare_mrcs_dirs):
    output_dir = Path("tests/output/synthetic_float16/")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    extract_local_subtomograms(
        particles_starfile=synthetic_data_root / "particles.star",
        particles_tomo_name_prefix="session1_",
        box_size=64,
        bin=1,
        tiltseries_dir=synthetic_data_root / "tiltSeries/",
        tiltseries_starfile=synthetic_data_root / "tiltSeries/tomograms.star",
        tiltseries_x=630,
        tiltseries_y=630,
        aln_dir=synthetic_data_root / "aln/",
        float16=True,
        output_dir=output_dir,
    )

    subtomo_dir = os.path.join(output_dir, "Subtomograms/session1_TS_1")
    relion_subtomo_dir = "tests/data/relion_project_synthetic/relion_output_float16/Subtomograms/session1_TS_1"
    validate_optimisation_set_starfile(output_dir / "optimisation_set.star")
    validate_particles_starfile(output_dir / "particles.star", synthetic_data_root / "relion_output_float16/particles.star")
    compare_mrcs_dirs(relion_subtomo_dir, subtomo_dir, tol=1e-4)
