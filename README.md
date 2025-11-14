# zarr-particle-tools
Subtomogram extraction and reconstruction in Python from local files and the CryoET Data Portal. A reimplementation of the RELION subtomogram extraction and particle reconstruction jobs, but designed to work on ZARR-based tiltseries with the CryoET Data Portal API and remove the need for downloading the entire tiltseries.

In addition to particle extraction and reconstruction, this package is built in a modular way to allow for use of individual functions (see `core/`), such as:
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

## Example runs
### See full options with `zarr-particle-extract --help` and `zarr-particle-reconstruct --help`.

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
  --run-id "16848, 16851, 16861" \
  --annotation-names "ribosome" \
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

## Pytest
To ensure that the subtomogram extraction matches RELION's subtomogram extraction, we have a set of tests that compare the output of this script with RELION 5.0's output and ensure that they match within reasonable numerical precision. float16 data has a more relaxed tolerance due to the reduced precision of the data type, and the real experimental data has a more relaxed tolerance due to the noisier nature of the data.

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
- Does not support CTF_BFACTOR (rlnCtfBfactor) or CTF_BFACTOR_PERELECTRONDOSE (rlnCtfBfactorPerElectronDose)
- Does not support Anisotropic magnification matrix (EMDL_IMAGE_MAG_MATRIX_00, EMDL_IMAGE_MAG_MATRIX_01, EMDL_IMAGE_MAG_MATRIX_10, EMDL_IMAGE_MAG_MATRIX_11)
- Does not support 2D deformations (EMDL_TOMO_DEFORMATION_GRID_SIZE_X, EMDL_TOMO_DEFORMATION_GRID_SIZE_Y, EMDL_TOMO_DEFORMATION_TYPE, EMDL_TOMO_DEFORMATION_COEFFICIENTS)

### Reconstruction limitations
- Does not (yet) support a SNR value (`--snr`) flag
- Does not (yet) support no_ctf
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
