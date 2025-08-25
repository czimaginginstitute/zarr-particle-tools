# portal-particle-extraction
Subtomogram extraction from CryoET Data Portal datasets. A reimplementation of the RELION subtomogram extraction job, but designed to work on ZARR-based tiltseries with the CryoET Data Portal API and reduce the need for downloading entire tiltseries.

Primary steps in subtomogram extraction are:
- 3D affine transformation matrix calculation
- Projection of 3D coordinates to 2D tiltseries coordinates
- CTF premultiplication
- Dose weighting
- Background masking and subtraction
- Writing of MRC stacks to disk
- Writing of updated STAR files

## Installation

Create a new conda environment if you'd like to keep this separate from your other Python environments:
```bash
conda create -n portal-particle-extraction python=3.12
conda activate portal-particle-extraction
```

And then install via poetry:
```bash
poetry install
```

## Example runs
### See full options with `python subtomo_extract.py local --help` and `python subtomo_extract.py data-portal --help`. 

For RELION projects, a `--tiltseries-relative-dir` is not needed if this script is run from the RELION project directory root.

```
python subtomo_extract.py local \
  --particles-starfile tests/data/relion_project_synthetic/particles.star \
  --tomograms-starfile tests/data/relion_project_synthetic/tomograms.star \
  --tiltseries-relative-dir tests/data/relion_project_synthetic/ \
  --output-dir tests/output/sample_local_test/ \
  --box-size 16 --bin 4
```

```
python subtomo_extract.py local \
  --particles-starfile tests/data/relion_project_unroofing/particles.star \
  --tomograms-starfile tests/data/relion_project_unroofing/tomograms.star \
  --tiltseries-relative-dir tests/data/relion_project_unroofing/ \
  --output-dir tests/output/sample_local_test/ \
  --box-size 64 --bin 1 --no-ctf --no-circle-crop
```

```
python subtomo_extract.py local \
  --particles-starfile tests/data/relion_project_synthetic/particles.star \
  --tomograms-starfile tests/data/relion_project_synthetic/tomograms.star \
  --tiltseries-relative-dir tests/data/relion_project_synthetic/ \
  --output-dir tests/output/sample_local_test/ \
  --box-size 128 --crop-size 64 --bin 1 --overwrite
```

```
python subtomo_extract.py data-portal \
  --run-id 16463 \
  --annotation-names "ribosome" \
  --inexact-match \
  --output-dir tests/output/sample_data_portal_test/ \
  --box-size 64
```

```
python subtomo_extract.py data-portal \
  --run-id 17700 \
  --annotation-names "ferritin complex" \
  --ground-truth \
  --output-dir tests/output/sample_data_portal_test/ \
  --box-size 32
```

## Pytest
To ensure that the subtomogram extraction matches RELION's subtomogram extraction, we have a set of tests that compare the output of this script with RELION 5.0's output and ensure that they match within reasonable numerical precision. float16 data has a more relaxed tolerance due to the reduced precision of the data type, and the real experimental data has a more relaxed tolerance due to the noisier nature of the data.

## Known Limitations
- Does not support gamma offset
- Does not support spherical aberration correction
- Does not support circle precrop
- Does not support grid precorrection
- Does not support whitening (power spectral flattening)
- Does not support 3D volume extraction
- Does not support min_frames or max_dose flags
- Does not write any other *.mrcs files other than the 2D stacks themselves
- Does not support defocus slope (rlnTomoDefocusSlope)
- Does not support particle subtomogram orientation (rlnTomoSubtomogramRot, rlnTomoSubtomogramTilt, rlnTomoSubtomogramPsi)
- Does not support --apply_orientations
- Does not support --dont_apply_offsets
- Does not support CTF_BFACTOR (rlnCtfBfactor) or CTF_BFACTOR_PERELECTRONDOSE (rlnCtfBfactorPerElectronDose)
- Does not support Anisotropic magnification matrix (EMDL_IMAGE_MAG_MATRIX_00, EMDL_IMAGE_MAG_MATRIX_01, EMDL_IMAGE_MAG_MATRIX_10, EMDL_IMAGE_MAG_MATRIX_11)

## Project roadmap
- [ ] TEST: Support multiple optics groups 
- [ ] Write tests for generating star files and pulling from the CryoET Data Portal
- [ ] Add copick support
- [ ] Notify teamtomo of this work and possible integration into their codebase
- [ ] Data Portal support (not downloading the entire tiltseries & using the API to pull relevant alignment & CTF information)