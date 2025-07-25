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

And then install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Example run
```
python data_portal_subtomo_extract.py \
  --particles-starfile tests/data/relion_project_synthetic/particles.star \
  --tiltseries-dir tests/data/relion_project_synthetic/tiltSeries \
  --tiltseries-starfile tests/data/relion_project_synthetic/tiltSeries/tomograms.star \
  --aln-dir tests/data/relion_project_synthetic/aln/ \
  --output-dir tests/output/sample_test/ \
  --particles-tomo-name-prefix "session1_" \
  --tiltseries-x 630 --tiltseries-y 630 --box-size 16 --bin 4 --debug
```

## Pytest
To ensure that the subtomogram extraction matches RELION's subtomogram extraction, we have a set of tests that compare the output of this script with RELION 5.0's output and ensure that they match within reasonable numerical precision (1e-6 for float32, 1e-4 for float16).

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
- Does not (currently) support particle orientations / offsets (rlnAngleRot, rlnAngleTilt, rlnAnglePsi, rlnOriginXAngst, rlnOriginYAngst, rlnOriginZAngst)
- Does not (currently) support multiple optics groups
- Does not support CTF_BFACTOR (rlnCtfBfactor) or CTF_BFACTOR_PERELECTRONDOSE (rlnCtfBfactorPerElectronDose)

## Project roadmap
- [ ] Support multiple optics groups
- [ ] Write more tests (with variations in extract job flags - no ctf, no cropping, experimental data, different binning, different box size (bin and box that don't have clean divides), different paths etc.)
- [ ] Incorporate alpha and beta offset parameters from AreTomo .aln file (for additional rotation) (and from CryoET Data Portal? if it exists?)
- [ ] Notify teamtomo of this work and possible integration into their codebase
- [ ] Data Portal support (not downloading the entire tiltseries & using the API to pull relevant alignment & CTF information)