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
time python /hpc/projects/group.czii/daniel.ji/cryoet-data-portal-pick-extract/portal-particle-extraction/data_portal_subtomo_extract.py \
  --particles-starfile /hpc/projects/group.czii/daniel.ji/cryoet-data-portal-pick-extract/pyrelion-runs/polnet/input/session1_polnet_0_ribosome.star \
  --tiltseries-dir /hpc/projects/group.czii/daniel.ji/cryoet-data-portal-pick-extract/pyrelion-runs/polnet/input/tiltSeries \
  --tiltseries-starfile /hpc/projects/group.czii/daniel.ji/cryoet-data-portal-pick-extract/pyrelion-runs/polnet/input/tiltSeries/aligned_tilt_series.star \
  --aln-dir /hpc/projects/group.czii/daniel.ji/cryoet-data-portal-pick-extract/pyrelion-runs/polnet/aretomo_mock/session1/run001/ \
  --output-dir /hpc/projects/group.czii/daniel.ji/cryoet-data-portal-pick-extract/pyrelion-runs/polnet/relion_mock/Extract/mockjob001/ \
  --particles-tomo-name-prefix "session1_" \
  --tiltseries-x 630 --tiltseries-y 630 --box-size 64 --bin 1 --debug \
  &> /hpc/projects/group.czii/daniel.ji/cryoet-data-portal-pick-extract/pyrelion-runs/polnet/relion_mock/relion_extract_manual.log
```

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
- Does not (currently) support binning
- Does not (currently) support particle orientations / offsets (rlnAngleRot, rlnAngleTilt, rlnAnglePsi, rlnOriginXAngst, rlnOriginYAngst, rlnOriginZAngst)
- Does not (currently) support multiple optics groups

## Project roadmap
- [ ] Implement binning
- [ ] Support multiple optics groups
- [ ] Write more tests (with variations in extract job flags - no ctf, no cropping, experimental data, different binning, different box size, different paths etc.)
- [ ] Incorporate alpha and beta offset parameters from AreTomo .aln file (for additional rotation) (and from CryoET Data Portal? if it exists?)
- [ ] Notify teamtomo of this work and possible integration into their codebase
- [ ] Data Portal support (not downloading the entire tiltseries & using the API to pull relevant alignment & CTF information)