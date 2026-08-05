# zarr-particle-tools
Subtomogram extraction and reconstruction in Python from local [OME-Zarr files](https://ngff.openmicroscopy.org/0.4/index.html) 
and the [CryoET Data Portal](https://cryoetdataportal.czscience.com/). A reimplementation of the [RELION](https://github.com/3dem/relion) subtomogram 
extraction and particle reconstruction jobs, but designed to work on ZARR-based tiltseries with the CryoET Data Portal
API and remove the need for downloading the entire tiltseries.

In addition to particle extraction and reconstruction, this package is built in a modular way to allow for use of
individual functions (see `core/`), such as:
- Projection matrix generation & point projection
- CTF premultiplication
- Dose weighting
- Fourier cropping
- Masking
- S3 & Zarr Data I/O
- Backprojection
- Interpolation
- Symmetry operations
- and more!


Primary steps in subtomogram extraction are:
- 3D affine transformation matrix calculation
- Projection of 3D coordinates to 2D tiltseries coordinates
- CTF premultiplication
- Dose weighting
- Background masking and subtraction
- Writing of MRC stacks to disk
- Writing of updated STAR files

Primary steps in subtomogram reconstruction are:
- Subtomogram extraction
- Backprojection into 3D Fourier space with interpolation
- Symmetry application
- Gridding correction
- CTF correction

This package also runs RELION's CTF refinement (`relion_tomo_refine_ctf`) and Bayesian polishing / frame
alignment (`relion_tomo_align`) directly against OME-Zarr tilt series, without modifying RELION: each tilt
series is streamed from zarr into a RAM-backed (`/dev/shm`) MRC that stock RELION reads. Both run in a
memory-bounded two-phase per-tomogram mode by default (bounded RAM), or all-at-once. A companion tool emits
a `tomograms.star` (S3-zarr tilt series with per-tilt CTF/geometry/dose) from the CryoET Data Portal or a
copick project to feed these jobs.

## Installation

Create a new conda environment if you'd like to keep this separate from your other Python environments:
```bash
conda create -n zarr-particle-tools python=3.12
conda activate zarr-particle-tools
pip install uv
```

And then install:
```bash
uv pip install zarr-particle-tools
```
> [!NOTE]  
> [CCPEM pipeliner](https://ccpem-pipeliner.readthedocs.io/en/latest/) is not yet released on PyPI. In order to use this
> package with pipeliner, please install pipeliner manually from the [pipeliner repository](https://gitlab.com/ccpem/ccpem-pipeliner).

## Example runs
### See full options with `zarr-particle-extract --help`, `zarr-particle-reconstruct --help`, `zarr-particle-ctfrefine --help`, `zarr-particle-polish --help`, and `zarr-particle-tomograms --help`.

For RELION projects, a `--tiltseries-relative-dir` is not needed if this script is run from the RELION project directory root.

#### Subtomogram extraction

```
zarr-particle-extract local \
  --particles-starfile tests/data/relion_project_synthetic/particles.star \
  --tomograms-starfile tests/data/relion_project_synthetic/tomograms.star \
  --tiltseries-relative-dir tests/data/relion_project_synthetic/ \
  --output-dir tests/output/sample_local_test/ \
  --box-size 16 --bin 4
```

```
zarr-particle-extract local \
  --particles-starfile tests/data/relion_project_unroofing/particles.star \
  --tomograms-starfile tests/data/relion_project_unroofing/tomograms.star \
  --tiltseries-relative-dir tests/data/relion_project_unroofing/ \
  --output-dir tests/output/sample_local_test/ \
  --box-size 64 --bin 1 --no-ctf --no-circle-crop
```

```
zarr-particle-extract local \
  --particles-starfile tests/data/relion_project_synthetic/particles.star \
  --tomograms-starfile tests/data/relion_project_synthetic/tomograms.star \
  --tiltseries-relative-dir tests/data/relion_project_synthetic/ \
  --output-dir tests/output/sample_local_test/ \
  --box-size 128 --crop-size 64 --bin 1 --overwrite
```

```
zarr-particle-extract data-portal \
  --dataset-id "10426" \
  --annotation-names "ribosome" \
  --inexact-match \
  --output-dir tests/output/sample_data_portal_test/ \
  --box-size 128 --bin 2
```

```
zarr-particle-extract data-portal \
  --run-id "16848, 16851" \
  --annotation-names "ribosome" \
  --ground-truth \
  --inexact-match \
  --output-dir tests/output/sample_data_portal_test/ \
  --box-size 128 --bin 2
```

```
zarr-particle-extract data-portal \
  --run-id 17700 \
  --annotation-names "ferritin complex" \
  --ground-truth \
  --output-dir tests/output/sample_data_portal_test/ \
  --box-size 32
```

```
zarr-particle-extract local-copick --help
```

```
zarr-particle-extract copick-data-portal --help
```

#### Subtomogram reconstruction (WIP, EXPERIMENTAL)

```
zarr-particle-reconstruct local \
  --particles-starfile tests/data/relion_project_unroofing/reconstruct_particles.star \
  --tiltseries-relative-dir tests/data/relion_project_unroofing/ \
  --tomograms-starfile tests/data/relion_project_unroofing/tomograms.star \
  --output-dir tests/output/sample_local_reconstruct_test/ \
  --box-size 384 --crop-size 256
```

```
zarr-particle-reconstruct data-portal --help
```

```
zarr-particle-reconstruct local-copick --help
```

```
zarr-particle-reconstruct copick-data-portal --help
```

#### Generate a tomograms.star for CTF-refine / polish

Emits only a `tomograms.star` (S3-zarr tilt series with per-tilt CTF/geometry/dose) — supply your own refined `particles.star` and reference half-maps.

```
zarr-particle-tomograms data-portal \
  --dataset-ids 10426 \
  --output-dir tests/output/sample_tomograms_test/
```

```
zarr-particle-tomograms copick-data-portal --help
```

#### CTF refinement (stock RELION on OME-Zarr)

Runs `relion_tomo_refine_ctf` against zarr tilt series. Takes RELION stars (an optimisation set, or particles + tomograms) plus reference half-maps from a prior Refine3D.

```
zarr-particle-ctfrefine local \
  --particles-starfile tests/data/relion_project_unroofing/refined_particles.star \
  --tomograms-starfile tests/data/relion_project_unroofing/tomograms.star \
  --ref1 tests/data/relion_project_unroofing/half1.mrc \
  --ref2 tests/data/relion_project_unroofing/half2.mrc \
  --box-size 384 --do-defocus --do-scale \
  --output-dir tests/output/sample_ctfrefine_test/
```

#### Bayesian polishing / frame alignment (stock RELION on OME-Zarr)

Runs `relion_tomo_align` against zarr tilt series (same input shape as CTF-refine); writes `motion.star` plus updated stars.

```
zarr-particle-polish local \
  --particles-starfile tests/data/relion_project_unroofing/refined_particles.star \
  --tomograms-starfile tests/data/relion_project_unroofing/tomograms.star \
  --ref1 tests/data/relion_project_unroofing/half1.mrc \
  --ref2 tests/data/relion_project_unroofing/half2.mrc \
  --box-size 384 --do-motion \
  --output-dir tests/output/sample_polish_test/
```

Both default to `--per-tomogram` (memory-bounded two-phase, `--n-workers 0` auto); pass `--all-at-once` to keep all tilt series in RAM.

## Pytest
To ensure that the subtomogram extraction matches RELION's subtomogram extraction, we have a set of tests that compare the output of this script with RELION 5.0's output and ensure that they match within reasonable numerical precision. float16 data has a more relaxed tolerance due to the reduced precision of the data type, and the real experimental data has a more relaxed tolerance due to the noisier nature of the data.

Comparisons use a magnitude-aware, unmasked (per-voxel) float32-storage-floor comparator (`tests/helpers/compare.py`): every voxel is checked against a tolerance of `ulp_factor * float32_ulp(max|values|)`, and the worst voxel is reported as a multiple of a float32 ULP. Test coverage:
- Extraction and reconstruction (`test_extract_strict.py`, `test_reconstruct.py`): strict per-voxel equivalence vs RELION across synthetic and real (unroofing) data, with binning, cropping, and no-CTF cases, plus reconstruction half-map parity and self-consistency checks.
- CTF refinement and polishing (`test_ctfrefine.py`, `test_polish.py`, RELION-binary-gated): zarr→`/dev/shm` results match stock RELION on real MRC tilt series, and two-phase per-tomogram matches all-at-once across fit variants (defocus, scale, aberrations, motion).
- Unit tests (`tests/unit/`, no RELION binary required): CTF B-factor envelope and phase shift, per-row / scalar dose frequency cutoff vs RELION's `findDoseXRanges`, the zarr full-stack reader and zarr→MRC streamer, and the `/dev/shm` preflight / cleanup safeguards.

To download the test data and run it yourself:

```
pip install -e .[dev] # Install development dependencies
mkdir -p tests/data
cd tests/data
# Download both files with retries and resume
curl -L --fail --retry 5 --retry-delay 5 --continue-at - \
  -o zarr_particle_tools_test_data_large.tar.gz \
  "https://zenodo.org/records/17338016/files/zarr_particle_tools_test_data_large.tar.gz?download=1"
curl -L --fail --retry 5 --retry-delay 5 --continue-at - \
  -o zarr_particle_tools_test_data_small.tar.gz \
  "https://zenodo.org/records/17338016/files/zarr_particle_tools_test_data_small.tar.gz?download=1"
# Extract
for f in *.tar.gz; do tar -xzf "$f"; done
```

> [!NOTE]
> On shared/login nodes, avoid `pytest -n auto`: it spawns a worker per core and each worker's BLAS pool adds threads, oversubscribing the CPU. Pin BLAS threads and use a modest worker count instead:
> ```bash
> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -n 4 -q
> ```

## Known Limitations
If you would like to see a feature added (on or off this limitation list), please open an issue!

### Extraction (and reconstruction) limitations
- Does not write any other *.mrcs files other than the 2D stacks themselves
- Does not (yet) support particle subtomogram orientation (rlnTomoSubtomogramRot, rlnTomoSubtomogramTilt, rlnTomoSubtomogramPsi)
- Does not support gamma offset
- Does not support spherical aberration correction
- Does not support grid precorrection
- Does not support whitening (power spectral flattening)
- Does not support 3D volume extraction
- Does not support min_frames or max_dose flags
- Does not support defocus slope (rlnTomoDefocusSlope)
- Does not support --apply_orientations
- Does not support --dont_apply_offsets
- Does not support cone flags (--cone_weight, --cone_angle, --cone_sig0)
- Does not support Anisotropic magnification matrix (EMDL_IMAGE_MAG_MATRIX_00, EMDL_IMAGE_MAG_MATRIX_01, EMDL_IMAGE_MAG_MATRIX_10, EMDL_IMAGE_MAG_MATRIX_11)
- Does not support 2D deformations (EMDL_TOMO_DEFORMATION_GRID_SIZE_X, EMDL_TOMO_DEFORMATION_GRID_SIZE_Y, EMDL_TOMO_DEFORMATION_TYPE, EMDL_TOMO_DEFORMATION_COEFFICIENTS)

### Reconstruction limitations
- Does not (yet) support a SNR value (`--snr`) flag
- Does not support weight_*.mrc output files
- Does not support helical symmetry
- Does not support backup / only do unfinished features 

## Project roadmap
- [ ] Write tests for generating star files and pulling from the CryoET Data Portal
- [ ] Support multiple optics groups 
- [ ] Support features that have (yet) to be implemented
- [ ] Add starfile generation from CryoET Data Portal into cryoet-alignment package

## Development
To set up a development environment, run the following commands:

```bash
conda create -n zarr-particle-tools python=3.12
conda activate zarr-particle-tools
pip install uv

git clone git@github.com:czimaginginstitute/zarr-particle-tools.git
cd zarr-particle-tools
uv pip install -e .[dev]
```

## License

`zarr-particle-tools` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Reporting Security Issues

If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).
