# zarr-particle-tools

Subtomogram averaging on [OME-Zarr](https://ngff.openmicroscopy.org/0.4/index.html) tilt series, straight
from the [CryoET Data Portal](https://cryoetdataportal.czscience.com/). It reimplements
[RELION](https://github.com/3dem/relion)'s subtomogram extraction and reconstruction in Python, and runs
RELION's own CTF refinement and Bayesian polishing against zarr — so **nothing is downloaded to disk**;
pixels stream from S3 as needed.

`zarr-particle-pipeline` ties this together: point it at a dataset and an annotation, and it drives a full
STA run end to end via [py2rely](https://github.com/chanzuckerberg/py2rely) (RELION 5).

## Quickstart

```bash
conda create -n zarr-particle-tools python=3.12 && conda activate zarr-particle-tools
pip install uv && uv pip install zarr-particle-tools

zarr-particle-pipeline preflight          # checks py2rely, RELION binaries, zarr job registration
```

Then a full STA run from the portal (see [Prerequisites](#prerequisites-for-the-pipeline) first):

```bash
zarr-particle-pipeline data-portal \
  --dataset-id 10426 \
  --annotation-name ribosome --inexact-match --ground-truth \
  --output-dir 10426_sta \
  --protein-diameter 330 \
  --reference-template ribo80s_emd_3883_866_128_resized.mrc \
  --num-gpus 4 --prepare-only
```

`--prepare-only` writes `all_sta_parameters.json` + `pipeline.sh` without submitting; inspect them, then
`cd 10426_sta && sbatch pipeline.sh`.

## Installation

```bash
conda create -n zarr-particle-tools python=3.12
conda activate zarr-particle-tools
pip install uv
uv pip install zarr-particle-tools
```

> [!NOTE]
> [CCPEM pipeliner](https://ccpem-pipeliner.readthedocs.io/en/latest/) is not on PyPI. To use this package
> with pipeliner, install it from the [pipeliner repository](https://gitlab.com/ccpem/ccpem-pipeliner).

### Prerequisites for the pipeline

Extraction and reconstruction need only this package; ctfrefine and polish additionally need the
`relion_tomo_*` binaries on `PATH`. The **orchestrator** needs all of:

- **py2rely** on your `PATH`
  ([README](https://github.com/chanzuckerberg/py2rely/blob/main/README.md)).
- **RELION 5** with the `relion_*` binaries on `PATH` (source its `setup-env.sh`, or add its `build/bin`).
- This package installed with pip (editable or not) so its `ccpem_pipeliner.jobs` entry points register —
  that registration is what lets py2rely pick the zarr jobs over stock RELION.
- **[copick](https://github.com/copick/copick)**, for the `copick-*` variants only.
- A **SLURM** cluster: py2rely submits via `sbatch`.

`zarr-particle-pipeline preflight` verifies py2rely, the RELION binaries, ccpem-pipeliner and the zarr job
registration (add `--copick` to include copick; SLURM availability is not checked). The orchestrator also
runs it automatically.

## Commands

Every command takes `--help`. Sources are named consistently: `local` (your own star files), `copick-local`,
`data-portal`, `copick-data-portal` (copick picks + portal tilt series; copick run names must be portal run
IDs).

| Command | Purpose | Sources |
|---|---|---|
| `zarr-particle-pipeline` | Full STA pipeline via py2rely | `local`, `copick-local`, `data-portal`, `copick-data-portal`, plus `preflight` |
| `zarr-particle-extract` | Subtomogram extraction (2D stacks) | `local`, `copick-local`, `data-portal`, `copick-data-portal` |
| `zarr-particle-reconstruct` | Particle map reconstruction (experimental) | `local`, `copick-local`, `data-portal`, `copick-data-portal` |
| `zarr-particle-ctfrefine` | RELION `relion_tomo_refine_ctf` on zarr | `local`, `data-portal`, `copick-data-portal` |
| `zarr-particle-polish` | RELION `relion_tomo_align` on zarr | `local`, `data-portal`, `copick-data-portal` |
| `zarr-particle-tomograms` | Emit just a `tomograms.star` | `data-portal`, `copick-data-portal` |
| `zarr-particle-export` | Self-contained on-disk project (downloads tilt series) | `data-portal`, `copick-data-portal` |

`core/` is also usable directly: projection matrices and point projection, CTF premultiplication, dose
weighting, Fourier cropping, masking, backprojection, interpolation, symmetry, and S3/zarr I/O.

## Usage

### Full pipeline (STA)

`zarr-particle-pipeline` resolves a dataset + annotation, derives and verifies the tilt-series pixel size
(portal metadata cross-checked against every tilt-series MRC header), writes star files into
`<output-dir>/input`, then runs `py2rely prepare relion5-parameters` → `prepare relion5-pipeline` →
`sbatch pipeline.sh`.

The generated `input/tomograms.star` carries a `tomoTiltSeriesURI` column, and that column is what makes
py2rely auto-select the four zarr jobs (extract / reconstruct / ctf-refine / polish) instead of stock
RELION. Refine3D / Class3D / MaskCreate / PostProcess stay stock.

```bash
zarr-particle-pipeline data-portal \
  --dataset-id 10426 \
  --annotation-name ribosome --inexact-match --ground-truth \
  --output-dir 10426_sta \
  --protein-diameter 330 \
  --reference-template ribo80s_emd_3883_866_128_resized.mrc \
  --symmetry C1 --low-pass 50 --binning-list 4,2,1 \
  --num-gpus 4 --cpu-constraint 16,8 --timeout 120
```

`--protein-diameter` is required, and `--reference-template` is required unless
`--run-denovo-generation`. Add `--run-ids 16848,16851` to restrict to a subset of runs, `--pixel-size` to
override the derived pixel size, or `--pixel-size-tol` to loosen the header check.

From a copick project instead:

```bash
zarr-particle-pipeline copick-data-portal \
  --copick-config pick-unroofing.json \
  --copick-name ribosome --copick-user-id user0 --copick-session-id 19 \
  --output-dir 10426_sta_copick \
  --protein-diameter 330 \
  --reference-template ribo80s_emd_3883_866_128_resized.mrc
```

Or from star files you already have. Both must live inside `--output-dir`, because py2rely resolves star
paths relative to the project directory; the pixel size is read from `rlnTomoTiltSeriesPixelSize`. Zarr jobs
are used only if your `tomograms.star` carries `tomoTiltSeriesURI` — otherwise py2rely runs stock RELION
against whatever the tilt stars point at.

```bash
zarr-particle-pipeline local \
  --output-dir my_sta \
  --particles-starfile my_sta/input/particles.star \
  --tomograms-starfile my_sta/input/tomograms.star \
  --protein-diameter 330 \
  --reference-template ref.mrc
```

`copick-local` is the same, but builds `particles.star` from copick picks (using your `tomograms.star`
for optics) into `<output-dir>/input`:

```bash
zarr-particle-pipeline copick-local \
  --output-dir my_sta \
  --tomograms-starfile my_sta/input/tomograms.star \
  --copick-config pick.json \
  --copick-name ribosome --copick-user-id user0 --copick-session-id 1 \
  --protein-diameter 330 \
  --reference-template ref.mrc
```

### Extraction

For RELION projects, `--tiltseries-relative-dir` can be omitted if you run from the project root.

```bash
zarr-particle-extract local \
  --particles-starfile tests/data/relion_project_synthetic/particles.star \
  --tomograms-starfile tests/data/relion_project_synthetic/tomograms.star \
  --tiltseries-relative-dir tests/data/relion_project_synthetic/ \
  --output-dir tests/output/sample_local_test/ \
  --box-size 16 --bin 4
```

Extract a larger box and crop it back down with `--crop-size` (which must be even and no larger than
`--box-size`), `--no-ctf` to skip CTF premultiplication, and `--no-circle-crop` for noisier real data:

```bash
zarr-particle-extract local \
  --particles-starfile tests/data/relion_project_unroofing/particles.star \
  --tomograms-starfile tests/data/relion_project_unroofing/tomograms.star \
  --tiltseries-relative-dir tests/data/relion_project_unroofing/ \
  --output-dir tests/output/sample_local_test/ \
  --box-size 64 --bin 1 --no-ctf --no-circle-crop
```

Straight from the portal, no download:

```bash
zarr-particle-extract data-portal \
  --run-id "16848, 16851" \
  --annotation-names "ribosome" \
  --ground-truth --inexact-match \
  --output-dir tests/output/sample_data_portal_test/ \
  --box-size 128 --bin 2
```

### Reconstruction (experimental)

```bash
zarr-particle-reconstruct local \
  --particles-starfile tests/data/relion_project_unroofing/reconstruct_particles.star \
  --tomograms-starfile tests/data/relion_project_unroofing/tomograms.star \
  --tiltseries-relative-dir tests/data/relion_project_unroofing/ \
  --output-dir tests/output/sample_local_reconstruct_test/ \
  --box-size 384 --crop-size 256
```

### CTF refinement and polishing

These run stock RELION (`relion_tomo_refine_ctf` / `relion_tomo_align`) against zarr: each tilt series is
streamed into a RAM-backed (`/dev/shm`) MRC that unmodified RELION reads. Both need a **refined**
`particles.star` and reference half-maps from a prior Refine3D — which is why there is no `copick-local`
variant: raw picks are not refined.

Both default to `--per-tomogram` (memory-bounded two-phase, `--n-workers 0` auto); `--all-at-once` keeps
every tilt series in RAM.

```bash
zarr-particle-ctfrefine local \
  --particles-starfile refined_particles.star \
  --tomograms-starfile tomograms.star \
  --ref1 half1.mrc --ref2 half2.mrc \
  --box-size 384 --do-defocus --do-scale \
  --output-dir tests/output/sample_ctfrefine_test/
```

```bash
zarr-particle-polish local \
  --particles-starfile refined_particles.star \
  --tomograms-starfile tomograms.star \
  --ref1 half1.mrc --ref2 half2.mrc \
  --box-size 384 --do-motion \
  --output-dir tests/output/sample_polish_test/
```

The `data-portal` and `copick-data-portal` variants generate the `tomograms.star` for you into
`<output-dir>/input`, so you only supply the refined particles and half-maps:

```bash
zarr-particle-ctfrefine data-portal \
  --dataset-ids 10426 \
  --particles-starfile refined_particles.star \
  --ref1 half1.mrc --ref2 half2.mrc \
  --box-size 384 --do-defocus \
  --output-dir tests/output/sample_ctfrefine_portal/
```

To generate only the `tomograms.star` (and feed it to the `local` variants yourself):

```bash
zarr-particle-tomograms data-portal --dataset-ids 10426 --output-dir tests/output/sample_tomograms_test/
```

### Export an on-disk project

Mostly optional, since the zarr jobs read OME-Zarr directly — it exists to hand off a portable project, or
to run stock RELION with no portal access. This downloads the **full** tilt-series stacks (large) and
repoints the tilt stars at the on-disk MRCs, dropping `tomoTiltSeriesURI`.

```bash
zarr-particle-export data-portal \
  --dataset-id 10426 \
  --annotation-name ribosome --inexact-match --ground-truth \
  --output-dir 10426_ondisk
```

## Reproducing a full run

This is the exact invocation behind a completed 3-class ribosome run on portal dataset 10476 (~12,676
copick picks from Octopi → ~6.7k after automatic class selection → **8.22 Å**). Set the four paths, then
the command is verbatim. Run `zarr-particle-pipeline copick-data-portal --help` for what each flag does.

```bash
export RELION_BIN=/path/to/relion/build/bin
export COPICK_CONFIG=/path/to/copick-picks/10476/config_ribosome.json
export TEMPLATE=/path/to/ribo80s_emd_3883_925_128_resized.mrc
export OUT=/path/to/zarr-pyrelion-runs/10476_ribosome_clean

conda activate zarr-particle-tools
export PATH="$RELION_BIN:$PATH"

zarr-particle-pipeline copick-data-portal \
  --copick-config "$COPICK_CONFIG" \
  --copick-name ribosome \
  --copick-user-id octopi \
  --copick-session-id 1 \
  --copick-dataset-ids 10476 \
  --output-dir "$OUT" \
  --protein-diameter 330 \
  --low-pass 50 \
  --box-scaling 1.65 \
  --binning-list 4,2,1 \
  --symmetry C1 \
  --run-class3d \
  --nclasses 3 \
  --class-selection auto \
  --reference-template "$TEMPLATE" \
  --num-gpus 4 \
  --gpu-constraint a40,a6000,l40s,a100,h100,h200 \
  --cpu-constraint 16,12 \
  --nthreads 4 \
  --timeout 12 \
  --num-days 14
```

This submits to SLURM and takes days. Star files land in `$OUT/input`; RELION job directories
(`Extract/`, `Refine3D/`, `Class3D/`, `CtfRefine/`, `Polish/`, `PostProcess/`, …) are created alongside
them. Resume or relaunch with `cd "$OUT" && sbatch pipeline.sh`. `--gpu-constraint` accepts one
architecture or several meaning OR, comma- or pipe-separated; py2rely checks them against the cluster's
SLURM features and drops any that are unavailable.

## Testing

Extraction and reconstruction are checked against RELION 5.0 output using a magnitude-aware, unmasked
per-voxel comparator (`tests/helpers/compare.py`): every voxel must fall within
`ulp_factor * float32_ulp(max|values|)`, and the worst voxel is reported as a multiple of a float32 ULP.
float16 and real experimental data get looser tolerances.

- `test_extract_strict.py`, `test_reconstruct.py` — strict per-voxel equivalence vs RELION on synthetic and
  real data, across binning, cropping, and no-CTF cases.
- `test_ctfrefine.py`, `test_polish.py` (need RELION binaries) — zarr→`/dev/shm` matches stock RELION, and
  two-phase per-tomogram matches all-at-once across every fit variant.
- `tests/unit/` (no RELION needed) — CTF envelope and phase shift, dose frequency cutoff vs RELION's
  `findDoseXRanges`, the zarr readers, and the `/dev/shm` preflight/cleanup safeguards.

```bash
uv pip install -e .[dev]
mkdir -p tests/data && cd tests/data
for f in zarr_particle_tools_test_data_large zarr_particle_tools_test_data_small; do
  curl -L --fail --retry 5 --retry-delay 5 --continue-at - -o "$f.tar.gz" \
    "https://zenodo.org/records/21780175/files/$f.tar.gz?download=1"
done
for f in *.tar.gz; do tar -xzf "$f"; done
```

The record ID is also set as `ZENODO_RECORD` in `.github/workflows/pytest.yml`; check there if this drifts.

> [!NOTE]
> On shared/login nodes avoid `pytest -n auto`: it spawns a worker per core and each worker's BLAS pool adds
> threads, oversubscribing the CPU. Pin BLAS threads and use a modest worker count:
> ```bash
> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -n 4 -q
> ```

## Known limitations

If you would like to see a feature added, on or off this list, please open an issue.

### Extraction (and reconstruction)
- Does not write any `*.mrcs` files other than the 2D stacks themselves
- Does not (yet) support particle subtomogram orientation (rlnTomoSubtomogramRot, rlnTomoSubtomogramTilt, rlnTomoSubtomogramPsi)
- Does not support gamma offset
- Does not support spherical aberration correction
- Does not support grid precorrection
- Does not support whitening (power spectral flattening)
- Does not support 3D volume extraction
- Does not support min_frames or max_dose flags
- Does not support --apply_orientations
- Does not support --dont_apply_offsets for reconstruction (extraction does, via `--dont-apply-offsets`)
- Does not support cone flags (--cone_weight, --cone_angle, --cone_sig0)
- Does not support anisotropic magnification matrix (EMDL_IMAGE_MAG_MATRIX_00, EMDL_IMAGE_MAG_MATRIX_01, EMDL_IMAGE_MAG_MATRIX_10, EMDL_IMAGE_MAG_MATRIX_11)
- Does not support 2D deformations (EMDL_TOMO_DEFORMATION_GRID_SIZE_X, EMDL_TOMO_DEFORMATION_GRID_SIZE_Y, EMDL_TOMO_DEFORMATION_TYPE, EMDL_TOMO_DEFORMATION_COEFFICIENTS)

### Reconstruction
- Only reproduces RELION's `--no_circle_crop` mode; its default circle cropping is not implemented
- Does not support `weight_*.mrc` output files
- Does not support helical symmetry
- Does not support backup / only-do-unfinished features

## Project roadmap
- [ ] Support multiple optics groups
- [ ] Support features that have (yet) to be implemented
- [ ] Add star file generation from the CryoET Data Portal into the cryoet-alignment package

## Development

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
