# TODO: create more focused tests for individual functions?
# TODO: don't store tiltseries / RELION *.mrcs data in the repository, but host them on zenodo
import pytest
import shutil
from pathlib import Path
from data_portal_subtomo_extract import extract_local_subtomograms

@pytest.mark.parametrize(
    "output_suffix,relion_suffix,extract_arguments",
    [
        ("synthetic_baseline", "relion_output_baseline", {"box_size": 64, "bin": 1}),
        ("synthetic_float16", "relion_output_float16", {"box_size": 64, "bin": 1, "float16": True}),
        ("synthetic_box16_bin4", "relion_output_box16_bin4", {"box_size": 16, "bin": 4}),
        ("synthetic_box16_bin6", "relion_output_box16_bin6", {"box_size": 16, "bin": 6}),
        ("synthetic_box32_bin2", "relion_output_box32_bin2", {"box_size": 32, "bin": 2}),
        ("synthetic_box32_bin4", "relion_output_box32_bin4", {"box_size": 32, "bin": 4}),
        ("synthetic_noctf", "relion_output_noctf", {"box_size": 64, "bin": 1, "no_ctf": True}),
        ("synthetic_nocirclecrop", "relion_output_nocirclecrop", {"box_size": 64, "bin": 1, "no_circle_crop": True}),
        ("synthetic_noctf_nocirclecrop", "relion_output_noctf_nocirclecrop", {"box_size": 64, "bin": 1, "no_ctf": True, "no_circle_crop": True}),
        ("synthetic_box16_bin4_noctf", "relion_output_box16_bin4_noctf", {"box_size": 16, "bin": 4, "no_ctf": True}),
        ("synthetic_box16_bin4_nocirclecrop", "relion_output_box16_bin4_nocirclecrop", {"box_size": 16, "bin": 4, "no_circle_crop": True}),
        ("synthetic_box16_bin4_noctf_nocirclecrop", "relion_output_box16_bin4_noctf_nocirclecrop", {"box_size": 16, "bin": 4, "no_ctf": True, "no_circle_crop": True}),
    ]
)
def test_extract_local_subtomograms_parametrized(
    synthetic_data_root,
    validate_optimisation_set_starfile,
    validate_particles_starfile,
    compare_mrcs_dirs,
    output_suffix,
    relion_suffix,
    extract_arguments,
):
    float16 = extract_arguments.get("float16", False)

    output_dir = Path(f"tests/output/{output_suffix}/")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    extract_local_subtomograms(
        particles_starfile=synthetic_data_root / "particles.star",
        particles_tomo_name_prefix="session1_",
        box_size=extract_arguments.get("box_size"),
        bin=extract_arguments.get("bin"),
        tiltseries_dir=synthetic_data_root / "tiltSeries/",
        tiltseries_starfile=synthetic_data_root / "tiltSeries/tomograms.star",
        tiltseries_x=630,
        tiltseries_y=630,
        aln_dir=synthetic_data_root / "aln/",
        float16=float16,
        no_ctf=extract_arguments.get("no_ctf", False),
        no_circle_crop=extract_arguments.get("no_circle_crop", False),
        output_dir=output_dir,
    )

    validate_optimisation_set_starfile(output_dir / "optimisation_set.star")
    validate_particles_starfile(
        output_dir / "particles.star",
        synthetic_data_root / f"{relion_suffix}/particles.star",
    )

    subtomo_dir = output_dir / "Subtomograms/session1_TS_1"
    relion_dir = f"tests/data/relion_project_synthetic/{relion_suffix}/Subtomograms/session1_TS_1"
    if float16:
        compare_mrcs_dirs(relion_dir, subtomo_dir, tol=1e-4)
    else:
        compare_mrcs_dirs(relion_dir, subtomo_dir)