# zarr-particle-tools

`zarr-particle-tools` provides a full subtomogram averaging (STA) pipeline and standalone jobs for
subtomogram extraction, reconstruction, CTF refinement, and Bayesian polishing. Tilt series can be read
from local or S3-backed MRC files and [OME-Zarr](https://ngff.openmicroscopy.org/0.4/index.html),
including datasets in the [CryoET Data Portal](https://cryoetdataportal.czscience.com/).

Extraction and reconstruction are Python implementations of the corresponding
[RELION 5](https://github.com/3dem/relion) jobs and stream Zarr pixels without first downloading the
complete tilt series. CTF refinement and polishing run RELION against Zarr through temporary MRC
files, preferring `/dev/shm` and falling back to the system temp directory. `zarr-particle-pipeline`
connects the jobs through [py2rely](https://github.com/chanzuckerberg/py2rely) for SLURM-based STA
workflows.

## Contents

- [Quickstart (container)](#quickstart-container)
- [Installation](#installation)
- [Pipeline prerequisites](#pipeline-prerequisites)
- [Commands](#commands)
- [Usage](#usage)
  - [Full pipeline](#full-pipeline)
  - [Extraction](#extraction)
  - [Reconstruction](#reconstruction)
  - [CTF refinement and polishing](#ctf-refinement-and-polishing)
  - [Export an on-disk project](#export-an-on-disk-project)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Development](#development)

## Quickstart (container)

The [relion-zarr-sta](https://github.com/czimaginginstitute/relion-docker) Docker/Apptainer image
bundles RELION, py2rely, and zarr-particle-tools together, no local installation needed. On a
SLURM cluster, this is enough to run the full pipeline:

```bash
apptainer pull relion-zarr-sta.sif oras://ghcr.io/czimaginginstitute/relion-zarr-sta-sif:5.0-cuda12.8
pip install git+https://github.com/czimaginginstitute/relion-docker.git#subdirectory=shims
relion-docker-shims --sif relion-zarr-sta.sif --out ~/relion-shims/bin --wire-py2rely
export PATH="$HOME/relion-shims/bin:$PATH"
```

`--wire-py2rely` points py2rely's SLURM job scripts at the container automatically; the `PATH`
export lets you also run any of the commands below directly. See relion-docker's
[`shims/`](https://github.com/czimaginginstitute/relion-docker/tree/main/shims) for details.
Otherwise, see [Installation](#installation) below for a standalone/manual setup.

## Installation

```bash
conda create -n zarr-particle-tools python=3.12
conda activate zarr-particle-tools
pip install uv
uv pip install zarr-particle-tools
```

- Extraction and reconstruction need no RELION installation.
- CTF refinement and polishing require the corresponding `relion_tomo_*` binaries.
- The full pipeline has the [additional prerequisites](#pipeline-prerequisites) below.

The `[pipeliner]` extra installs [CCPEM pipeliner](https://ccpem-pipeliner.readthedocs.io/en/latest/)
for the four registered job wrappers:

```bash
uv pip install "zarr-particle-tools[pipeliner]"
```

Installing py2rely in the same environment also installs ccpem-pipeliner.

## Pipeline prerequisites

The [Quickstart container](#quickstart-container) already has all of the below; skip this
section if you're using it.

`zarr-particle-pipeline` requires:

- [py2rely](https://github.com/chanzuckerberg/py2rely) on `PATH`.
- RELION 5 binaries on `PATH`. Source RELION's `setup-env.sh` or add its `build/bin` directory.
- This package installed in the same environment as py2rely so its `ccpem_pipeliner.jobs` entry points
  are registered.
- A SLURM cluster with `sbatch` available.

Run `zarr-particle-pipeline preflight` to check py2rely, RELION, ccpem-pipeliner, and job registration.
SLURM availability is not checked. The same preflight runs automatically before pipeline preparation.

## Commands

Every command takes `--help`. Sources are named consistently: `local` (your own star files), `copick-local`,
`data-portal`, `copick-data-portal` (copick picks + portal tilt series; copick run names must be portal run
IDs).

| Command | Purpose | Sources |
|---|---|---|
| `zarr-particle-pipeline` | Full STA pipeline via py2rely | `local`, `copick-local`, `data-portal`, `copick-data-portal`, plus `preflight` |
| `zarr-particle-extract` | Subtomogram extraction (2D stacks) | `local`, `copick-local`, `data-portal`, `copick-data-portal` |
| `zarr-particle-reconstruct` | Particle map reconstruction | `local`, `copick-local`, `data-portal`, `copick-data-portal` |
| `zarr-particle-ctfrefine` | RELION `relion_tomo_refine_ctf` on zarr | `local`, `data-portal`, `copick-data-portal` |
| `zarr-particle-polish` | RELION `relion_tomo_align` on zarr | `local`, `data-portal`, `copick-data-portal` |
| `zarr-particle-tomograms` | Write a `tomograms.star` | `data-portal`, `copick-data-portal` |
| `zarr-particle-export` | Self-contained on-disk project (downloads tilt series) | `data-portal`, `copick-data-portal` |

`core/` can also be used directly for projection matrices and point projection, CTF premultiplication,
dose weighting, Fourier cropping, masking, backprojection, interpolation, symmetry, and Zarr/S3 I/O.

## Usage

### Full pipeline

`zarr-particle-pipeline`:

- Resolves the selected tilt series and particles.
- Derives the pixel size and cross-checks portal metadata against each tilt-series MRC header.
- Writes star files to `<output-dir>/input`.
- Runs `py2rely prepare relion5-parameters` and `py2rely prepare relion5-pipeline`, then submits
  `pipeline.sh` unless `--prepare-only` is set.

The generated `tomograms.star` carries a `tomoTiltSeriesURI` column. py2rely uses that column to select
the Zarr extraction, reconstruction, CTF-refinement, and polishing jobs. Refine3D, Class3D, MaskCreate,
and PostProcess continue to use stock RELION jobs.

The following portal example uses dataset 10426 and the `cytosolic ribosome` picks from deposition
10358: 5,246 oriented picks across 38 tilt series at 2.165 Å/px, with per-tilt alignment and CTF
metadata.

```bash
export TEMPLATE=tests/templates/ribo80s_emd_3883_866_64_resized.mrc
export OUT=10426_ribosome_sta

zarr-particle-pipeline data-portal \
  --dataset-ids 10426 \
  --deposition-ids 10358 \
  --annotation-names "cytosolic ribosome" \
  --output-dir "$OUT" \
  --reference-template "$TEMPLATE" \
  --protein-diameter 330 \
  --box-scaling 1.65 \
  --num-gpus 4 \
  --nthreads 4 \
  --prepare-only
```

- `--protein-diameter` is required.
- `--reference-template` is required unless `--run-denovo-generation` is set.
- `--run-ids` limits the selection to specific runs.
- The portal form also accepts deposition, dataset, organism, run, tilt-series, alignment, tomogram, and
  annotation filters by ID or name.
- `--pixel-size` overrides the derived pixel size; `--pixel-size-tol` adjusts the MRC-header check.
- `--timeout` is the per-job limit in hours; `--num-days` sets the SLURM walltime request.
- `--run-class3d --nclasses N --class-selection auto` enables 3D classification.
- `--gpu-constraint` accepts one SLURM GPU feature or a comma- or pipe-separated set interpreted as OR.
  py2rely drops requested features that are unavailable on the cluster.
- `--prepare-only` writes `all_sta_parameters.json` and `pipeline.sh` without submitting. Submit later
  with `cd "$OUT" && sbatch pipeline.sh`.

Run `zarr-particle-pipeline data-portal --help` for the complete option list and current defaults.

Star files are written under `$OUT/input`. RELION job directories such as `Extract/`, `Refine3D/`,
`Class3D/`, `CtfRefine/`, `Polish/`, and `PostProcess/` are created alongside them as the pipeline runs.
End-to-end runs can take days on SLURM.

The repository includes `tests/templates/ribo80s_emd_3883_866_64_resized.mrc`, an 80S ribosome map
([EMD-3883](https://www.ebi.ac.uk/emdb/EMD-3883)) resampled to 8.66 Å/px in a box of 64
(`330 × 1.65 / 8.66 = 62.9`, rounded up). The path above assumes a source checkout; otherwise set
`TEMPLATE` to your own reference. Match its voxel size to the dataset's coarsest requested binning.

When the picks live in copick instead, `copick-data-portal` takes them from there and the tilt series
from the portal (the copick run names must be portal run IDs):

```bash
zarr-particle-pipeline copick-data-portal \
  --copick-config config.json \
  --copick-name ribosome --copick-user-id picker --copick-session-id 1 \
  --copick-dataset-ids 10426 \
  --output-dir ribosome_sta \
  --reference-template template.mrc \
  --protein-diameter 330 \
  --box-scaling 1.65 \
  --num-gpus 4 --nthreads 4
```

The science options are the same as for `data-portal`.

For existing star files, both files must be inside `--output-dir` because py2rely resolves their paths
relative to the project directory. The pixel size is read from `rlnTomoTiltSeriesPixelSize`. Zarr jobs
are selected only when `tomograms.star` contains `tomoTiltSeriesURI`; otherwise py2rely uses stock
RELION jobs.

```bash
zarr-particle-pipeline local \
  --output-dir my_sta \
  --particles-starfile my_sta/input/particles.star \
  --tomograms-starfile my_sta/input/tomograms.star \
  --protein-diameter 330 \
  --reference-template ref.mrc
```

`copick-local` builds `particles.star` from copick picks and uses the supplied `tomograms.star` for
optics. It writes the particles file to `<output-dir>/input`:

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

From the portal without first materializing the tilt series locally:

```bash
zarr-particle-extract data-portal \
  --run-id "16848, 16851" \
  --annotation-names "ribosome" \
  --ground-truth --inexact-match \
  --output-dir tests/output/sample_data_portal_test/ \
  --box-size 128 --bin 2
```

### Reconstruction

```bash
zarr-particle-reconstruct local \
  --particles-starfile tests/data/relion_project_unroofing/reconstruct_particles.star \
  --tomograms-starfile tests/data/relion_project_unroofing/tomograms.star \
  --tiltseries-relative-dir tests/data/relion_project_unroofing/ \
  --output-dir tests/output/sample_local_reconstruct_test/ \
  --box-size 384 --crop-size 256
```

### CTF refinement and polishing

- CTF refinement runs `relion_tomo_refine_ctf`; polishing runs `relion_tomo_align`.
- Zarr tilt series are streamed into temporary MRC files for RELION to read. `/dev/shm` is used when
  it is writable and has enough space; otherwise the system temp directory is used.
- Both jobs require a refined `particles.star` and reference half-maps from a prior Refine3D job.
- There is no `copick-local` variant because raw copick picks are not refined particles.
- `--per-tomogram` is the default and uses a staging-bounded two-phase workflow. `--all-at-once`
  stages every selected tilt series together.
- `--n-workers 0` selects the worker count automatically.

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

The `data-portal` and `copick-data-portal` variants write `tomograms.star` to `<output-dir>/input`. Supply
the refined particles and half-maps:

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

`zarr-particle-export` creates a portable project for use without portal access or with stock RELION. It
downloads the complete tilt-series stacks, which can be large, updates the tilt stars to use local MRC
files, and removes `tomoTiltSeriesURI`.

```bash
zarr-particle-export data-portal \
  --dataset-id 10426 \
  --annotation-name ribosome --inexact-match --ground-truth \
  --output-dir 10426_ondisk
```

## Testing

Strict extraction and reconstruction comparisons against RELION 5.0 check every voxel with
magnitude-aware float32 tolerances and report the worst mismatch. Broader float16 and experimental
cases use looser comparisons appropriate to their precision and noise.

- `test_extract_strict.py`, `test_reconstruct.py` — strict per-voxel equivalence vs RELION on synthetic and
  real data, across binning, cropping, and no-CTF cases.
- `test_ctfrefine.py`, `test_polish.py` (need RELION binaries) — temporarily staged Zarr data
  matches stock RELION, with two-phase parity coverage for defocus, scale, aberration, and motion modes.
- `tests/unit/` (no RELION needed) — CTF envelope and phase shift, dose frequency cutoff vs RELION's
  `findDoseXRanges`, the zarr readers, and temporary-file preflight/fallback/cleanup safeguards.

```bash
uv sync --locked --extra dev
mkdir -p tests/data
(
cd tests/data
for f in zarr_particle_tools_test_data_large zarr_particle_tools_test_data_small; do
  curl -L --fail --retry 5 --retry-delay 5 --continue-at - -o "$f.tar.gz" \
    "https://zenodo.org/records/21797999/files/$f.tar.gz?download=1"
done
for f in *.tar.gz; do tar -xzf "$f"; done
)
```

The two archives are about 9.34 GB compressed and need additional space when extracted. The record ID
is also set as `ZENODO_RECORD` in `.github/workflows/pytest.yml` and `.github/workflows/pytest_full.yml`.

On shared or login nodes, avoid `pytest -n auto`; it starts one worker per core and can oversubscribe the
CPU through BLAS thread pools. Use a fixed worker count:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked pytest -n 4 -q
```

## Known limitations

If you would like to see a feature added, on or off this list, please open an issue.

### Extraction and reconstruction

- Does not apply particle subtomogram orientations (`rlnTomoSubtomogramRot`,
  `rlnTomoSubtomogramTilt`, `rlnTomoSubtomogramPsi`); extraction also lacks RELION's
  `--apply_orientations` mode
- Does not apply higher-order optical-aberration corrections from RELION optics metadata
  (even-Zernike gamma offsets or odd-Zernike phase corrections)
- Does not support whitening (power spectral flattening)
- Does not support anisotropic magnification matrices (`EMDL_IMAGE_MAG_MATRIX_00`,
  `EMDL_IMAGE_MAG_MATRIX_01`, `EMDL_IMAGE_MAG_MATRIX_10`, `EMDL_IMAGE_MAG_MATRIX_11`)
- Does not support 2D deformations (`EMDL_TOMO_DEFORMATION_GRID_SIZE_X`,
  `EMDL_TOMO_DEFORMATION_GRID_SIZE_Y`, `EMDL_TOMO_DEFORMATION_TYPE`,
  `EMDL_TOMO_DEFORMATION_COEFFICIENTS`)
- Does not support `--only_do_unfinished`

### Extraction

- Does not support 3D volume extraction or write any `*.mrcs` files other than the 2D stacks themselves
- Does not support `min_frames` or `max_dose` (`zarr-particle-pipeline` rejects `--max-dose` before
  extraction)
- Does not support grid precorrection
- Does not support cone flags (`--cone_weight`, `--cone_angle`, `--cone_sig0`)

### Reconstruction

- Does not support helical symmetry
- Only reproduces RELION's `--no_circle_crop` mode; its default circle cropping is not implemented
- Does not support `--dont_apply_offsets` (extraction supports `--dont-apply-offsets`)
- Requires a single image size, pixel size, and binning across optics groups
- Does not support `weight_*.mrc` output files
- Does not support RELION's reconstruction backup / `--no_backup` behavior

## Development

```bash
conda create -n zarr-particle-tools python=3.12
conda activate zarr-particle-tools
pip install uv

git clone git@github.com:czimaginginstitute/zarr-particle-tools.git
cd zarr-particle-tools
uv sync --locked --extra dev
```

## License

`zarr-particle-tools` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Reporting Security Issues

If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).
