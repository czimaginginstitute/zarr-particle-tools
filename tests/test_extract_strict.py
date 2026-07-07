"""
Strict, unmasked extraction comparison vs RELION (the float64-as-oracle policy).

Unlike `test_extract.py` (which masks the worst 0.5% of voxels and uses fixed absolute tolerances),
this tier compares EVERY voxel against a magnitude-aware tolerance pinned to the float32 *storage*
floor (`ulp_factor * float32_ulp(max|values|)`). Extraction is algorithmically correct (see PLAN.md
Phase 0) and the residual vs RELION is exactly that float32 storage floor — except the no-CTF path,
where RELION rounds the IFFT to float32 before the cropCircle mean-subtraction (we keep float64),
leaving a small documented DC residual covered by `extra_atol`.
"""

import shutil
from pathlib import Path

import pytest

from tests.helpers.compare import mrc_close_unmasked
from zarr_particle_tools.subtomo_extract import extract_subtomograms

DATA_ROOTS = {
    "synthetic": Path("tests/data/relion_project_synthetic"),
    "unroofing": Path("tests/data/relion_project_unroofing"),
}

# Default 16x the float32 ULP (~2e-6 relative) is the storage floor: per-file worst voxels reach
# ~8-10x ULP across these cases, far below any algorithmic error. The no-CTF case carries an extra
# ~3e-5 DC residual from RELION's float32-before-cropCircle mean-subtraction ordering.
STRICT_CASES = {
    "synthetic_baseline": {"dataset": "synthetic", "suffix": "baseline", "box_size": 64, "bin": 1},
    "synthetic_box16_bin4": {"dataset": "synthetic", "suffix": "box16_bin4", "box_size": 16, "bin": 4},
    "unroofing_baseline": {"dataset": "unroofing", "suffix": "baseline", "box_size": 64, "bin": 1},
    "unroofing_box64_bin2_crop32": {
        "dataset": "unroofing",
        "suffix": "box64_bin2_crop32",
        "box_size": 64,
        "bin": 2,
        "crop_size": 32,
    },
    "unroofing_noctf": {
        "dataset": "unroofing",
        "suffix": "noctf",
        "box_size": 64,
        "bin": 1,
        "no_ctf": True,
        "extra_atol": 5e-5,  # RELION float32-before-cropCircle mean-subtraction DC residual
    },
}


@pytest.mark.parametrize("case", STRICT_CASES, ids=list(STRICT_CASES))
def test_extract_strict_unmasked(case):
    cfg = STRICT_CASES[case]
    data_root = DATA_ROOTS[cfg["dataset"]]
    if not (data_root / "particles.star").exists():
        pytest.skip(f"local test data not available: {data_root} (requires the large Zenodo tarball)")
    output_dir = Path(f"tests/output/strict_{case}")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    extract_subtomograms(
        box_size=cfg["box_size"],
        crop_size=cfg.get("crop_size"),
        bin=cfg["bin"],
        float16=False,
        no_ctf=cfg.get("no_ctf", False),
        no_circle_crop=cfg.get("no_circle_crop", False),
        output_dir=output_dir,
        particles_starfile=data_root / "particles.star",
        tiltseries_relative_dir=data_root,
        tomograms_starfile=data_root / "tomograms.star",
    )

    relion_dir = data_root / f"Extract/relion_output_{cfg['suffix']}/Subtomograms/"
    out_dir = output_dir / "Subtomograms/"
    ulp_factor = cfg.get("ulp_factor", 16.0)
    extra_atol = cfg.get("extra_atol", 0.0)

    mrcs = sorted(relion_dir.rglob("*.mrcs"))
    assert mrcs, f"no reference .mrcs found under {relion_dir}"
    for ref in mrcs:
        got = out_dir / ref.relative_to(relion_dir)
        assert got.exists(), f"Expected output missing: {got}"
        assert mrc_close_unmasked(ref, got, ulp_factor=ulp_factor, extra_atol=extra_atol)
