#!/usr/bin/env python
"""HPC verification helper: run one reconstruct case and report the UNMASKED diff
vs a RELION reference (committed or freshly regenerated).

Usage:
    python scripts/hpc_reconstruct_case.py <case> <out_dir> <ref_merged.mrc>

<case> is a key in SYNTHETIC_RECONSTRUCT_PARAMETERS (e.g. baseline, baseline_I2).
Prints a JSON line with max/median/RMS abs-diff, ULP multiple, and argmax radius.
"""
import json
import sys
from pathlib import Path

import numpy as np

from tests.helpers.compare import mrc_unmasked_report
from tests.test_reconstruct import SYNTHETIC_RECONSTRUCT_PARAMETERS
from zarr_particle_tools.subtomo_reconstruct import reconstruct_local

DATA_ROOT = Path("tests/data/relion_project_synthetic")


def main():
    case = sys.argv[1]
    out_dir = Path(sys.argv[2])
    ref = Path(sys.argv[3])
    args = SYNTHETIC_RECONSTRUCT_PARAMETERS[case]

    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)

    reconstruct_local(
        box_size=args.get("box_size"),
        crop_size=args.get("crop_size"),
        bin=args.get("bin", 1),
        symmetry=args.get("symmetry", "C1"),
        output_dir=out_dir,
        particles_starfile=DATA_ROOT / "particles.star",
        trajectories_starfile=None,
        tiltseries_relative_dir=DATA_ROOT,
        tomograms_starfile=DATA_ROOT / "tomograms.star",
        no_ctf=args.get("no_ctf", False),
    )

    rep = mrc_unmasked_report(ref, out_dir / "merged.mrc")
    # radius of worst voxel from box center
    box = args.get("crop_size") or args.get("box_size")
    center = (box - 1) / 2.0
    argmax = np.array(rep["argmax"], dtype=float)
    rep["argmax_radius"] = float(np.sqrt(((argmax - center) ** 2).sum()))
    rep["box"] = box
    rep["case"] = case
    rep["ref"] = str(ref)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
